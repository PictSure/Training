"""
MLP Baseline: Train MLPs on random 10-class subsets of datasets
stored in the hierarchical DuckDB.

Usage:
    python mlp_baseline/train_mlp.py --db /path/to/hierarchical.duckdb \
        --output mlp_baseline/results --epochs 100 --lr 1e-3 --multiclass-runs 70

Outputs:
    <output>/results.json                – aggregated results for all runs per dataset
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import duckdb
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import trange, tqdm


# ---------------------------------------------------------------------------
# Helpers – DuckDB schema navigation
# ---------------------------------------------------------------------------

def _quote(name: str) -> str:
    return f'"{name.replace(chr(34), chr(34)+chr(34))}"'


def _safe_name(value: str) -> str:
    sanitized = [c if c.isalnum() or c in ("-", "_") else "_" for c in value]
    return "".join(sanitized).strip("_") or "dataset"


def list_datasets(db_path: str) -> List[str]:
    """Return all dataset names in the DuckDB."""
    conn = duckdb.connect(database=db_path, read_only=True)
    rows = conn.execute('SELECT "dataset_name" FROM "datasets"').fetchall()
    conn.close()
    return [r[0] for r in rows]


def calculate_max_combinations(num_classes: int, subset_size: int = 10) -> int:
    """Calculate C(n, k) = n! / (k! * (n-k)!)"""
    if num_classes < subset_size:
        return 0
    return math.comb(num_classes, subset_size)


def generate_random_class_subsets(
    num_classes: int,
    subset_size: int = 10,
    num_subsets: int = 70,
    seed: int = 42
) -> List[List[int]]:
    """Generate random class subsets.

    If num_subsets exceeds the number of unique combinations, unique subsets are
    cycled so that every run still gets a class subset (the differing seed per run
    produces a different train/test split even for repeated class subsets).
    """
    rng = np.random.RandomState(seed)
    max_combos = calculate_max_combinations(num_classes, subset_size)

    if max_combos == 0:
        return []

    # Collect all unique subsets (up to the hard limit imposed by combinatorics).
    unique_count = min(num_subsets, max_combos)
    unique_subsets: List[List[int]] = []
    seen: set = set()

    while len(unique_subsets) < unique_count:
        subset = tuple(sorted(rng.choice(num_classes, size=subset_size, replace=False)))
        if subset not in seen:
            unique_subsets.append(list(subset))
            seen.add(subset)

    # Cycle through unique subsets to reach the requested num_subsets.
    return [unique_subsets[i % len(unique_subsets)] for i in range(num_subsets)]


def load_single_dataset(
    db_path: str, dataset_name: str
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load one dataset from the hierarchical DuckDB.

    Returns
    -------
    (embeddings [N, D], labels [N], class_names)
    """
    conn = duckdb.connect(database=db_path, read_only=True)

    row = conn.execute(
        'SELECT "label_table" FROM "datasets" WHERE "dataset_name" = ?',
        [dataset_name],
    ).fetchone()
    if row is None:
        conn.close()
        raise ValueError(f"Dataset '{dataset_name}' not found in DuckDB")
    label_table = row[0]

    label_rows = conn.execute(
        f'SELECT "label_name", "table_name" FROM {_quote(label_table)}'
    ).fetchall()

    class_names: List[str] = []
    embeddings_parts: List[np.ndarray] = []
    labels_parts: List[np.ndarray] = []

    for label_idx, (label_name, table_name) in enumerate(label_rows):
        cur = conn.execute(f'SELECT "Embedding" FROM {_quote(table_name)}')
        all_embs: List[List[float]] = []
        while True:
            batch = cur.fetchmany(4096)
            if not batch:
                break
            all_embs.extend(r[0] for r in batch)
        if not all_embs:
            continue

        embs = np.array(all_embs, dtype=np.float32)
        embeddings_parts.append(embs)
        labels_parts.append(np.full(len(embs), label_idx, dtype=np.int64))
        class_names.append(label_name)

    conn.close()

    if not embeddings_parts:
        raise RuntimeError(f"Dataset '{dataset_name}': no embeddings found")

    X = np.concatenate(embeddings_parts, axis=0)
    y = np.concatenate(labels_parts, axis=0)
    print(f"  Loaded {dataset_name}: {X.shape[0]} samples, {len(class_names)} classes, dim={X.shape[1]}")
    return X, y, class_names


# ---------------------------------------------------------------------------
# MLP model
# ---------------------------------------------------------------------------

class MLP(nn.Module):
    """Deep MLP with residual connections, batch norm, ReLU activations and dropout."""

    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Residual blocks
        self.block1 = self._make_residual_block(hidden_dim, dropout)
        self.block2 = self._make_residual_block(hidden_dim, dropout)
        self.block3 = self._make_residual_block(hidden_dim, dropout)
        self.block4 = self._make_residual_block(hidden_dim, dropout)
        
        # Output head
        self.output = nn.Linear(hidden_dim, num_classes)
    
    def _make_residual_block(self, hidden_dim: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        
        # Residual connections
        x = x + self.block1(x)
        x = nn.functional.relu(x)
        
        x = x + self.block2(x)
        x = nn.functional.relu(x)
        
        x = x + self.block3(x)
        x = nn.functional.relu(x)
        
        x = x + self.block4(x)
        x = nn.functional.relu(x)
        
        return self.output(x)


# ---------------------------------------------------------------------------
# Training loop (single dataset)
# ---------------------------------------------------------------------------

def train_single_dataset(
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    class_names: List[str],
    output_dir: Path,
    class_subset: List[int] = None,
    *,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
    hidden_dim: int = 512,
    dropout: float = 0.1,
    weight_decay: float = 1e-4,
    test_ratio: float = 0.2,
    device: str = "cpu",
    seed: int = 42,
    early_stop_patience: int = 20,
    samples_per_class: int | None = None,
    train_shots: int | None = None,
    test_shots: int | None = None,
) -> dict:
    """Train a 3-layer MLP on one dataset or subset, return accuracy dict."""

    rng = np.random.RandomState(seed)

    # Filter to class subset if provided
    if class_subset is not None:
        # Create mapping from old class indices to new ones
        mask = np.isin(y, class_subset)
        X_filtered = X[mask].copy()
        y_filtered = y[mask].copy()

        # Remap class labels to 0..len(subset)-1
        class_name_subset = [class_names[i] for i in class_subset]
        for new_idx, old_idx in enumerate(sorted(class_subset)):
            y_filtered[y_filtered == old_idx] = new_idx + 1000  # Temporary offset
        y_filtered = y_filtered - 1000

        X = X_filtered
        y = y_filtered
        class_names = class_name_subset

    num_classes = len(class_names)
    input_dim = X.shape[1]

    # ---- stratified train/test split ----
    train_idx, test_idx = [], []
    for c in range(num_classes):
        c_idx = np.where(y == c)[0]
        rng.shuffle(c_idx)

        if train_shots is not None or test_shots is not None:
            # Few-shot mode: take exactly train_shots train and test_shots test per class.
            n_train = train_shots if train_shots is not None else len(c_idx) - (test_shots or 1)
            n_test = test_shots if test_shots is not None else len(c_idx) - n_train
            if len(c_idx) < n_train + n_test:
                raise RuntimeError(
                    f"Class {c} of '{dataset_name}' has only {len(c_idx)} samples "
                    f"but {n_train} train + {n_test} test = {n_train + n_test} requested."
                )
            train_idx.extend(c_idx[:n_train].tolist())
            test_idx.extend(c_idx[n_train:n_train + n_test].tolist())
        else:
            # Ratio mode (default): optionally cap total samples per class first.
            if samples_per_class is not None:
                c_idx = c_idx[:samples_per_class]
            n_test = max(1, int(len(c_idx) * test_ratio))
            test_idx.extend(c_idx[:n_test].tolist())
            train_idx.extend(c_idx[n_test:].tolist())

    X_train = torch.tensor(X[train_idx], dtype=torch.float32)
    y_train = torch.tensor(y[train_idx], dtype=torch.long)
    X_test = torch.tensor(X[test_idx], dtype=torch.float32)
    y_test = torch.tensor(y[test_idx], dtype=torch.long)

    # Adjust batch size for small datasets (BatchNorm needs at least 2 samples per batch)
    effective_batch_size = min(batch_size, max(2, len(train_idx) // 10))

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=effective_batch_size,
        shuffle=True,
        drop_last=True,  # Important: drop_last=True to avoid single-sample batches with BatchNorm
    )
    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=effective_batch_size,
        shuffle=False,
        drop_last=False,
    )

    # ---- model / optimizer / loss ----
    model = MLP(input_dim, hidden_dim, num_classes, dropout=dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Calculate total batches for linear decay
    total_batches = len(train_loader) * epochs
    
    def lr_lambda(batch_idx: int) -> float:
        if total_batches <= 1:
            return 0.01
        return 1.0 - (batch_idx / (total_batches - 1)) * 0.99
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    loss_fn = nn.CrossEntropyLoss()

    best_test_acc = 0.0
    best_train_acc = 0.0
    epochs_without_improvement = 0
    final_train_acc = 0.0
    final_test_acc = 0.0

    for epoch in range(epochs):
        # --- train ---
        model.train()
        total_loss, total_correct, total_n = 0.0, 0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item() * xb.size(0)
            total_correct += (logits.argmax(1) == yb).sum().item()
            total_n += xb.size(0)

        train_loss = total_loss / total_n
        train_acc = total_correct / total_n
        final_train_acc = train_acc
        if train_acc > best_train_acc:
            best_train_acc = train_acc

        # --- test ---
        model.eval()
        test_loss_sum, test_correct, test_n = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                test_loss_sum += loss_fn(logits, yb).item() * xb.size(0)
                test_correct += (logits.argmax(1) == yb).sum().item()
                test_n += xb.size(0)
        test_loss = test_loss_sum / test_n
        test_acc = test_correct / test_n
        final_test_acc = test_acc

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= early_stop_patience:
            break

    return {
        "num_classes": num_classes,
        "num_samples": len(train_idx) + len(test_idx),
        "train_acc": best_train_acc,
        "final_train_acc": final_train_acc,
        "test_acc": best_test_acc,
        "final_test_acc": final_test_acc,
    }


def train_multiclass_runs(
    dataset_name: str,
    X: np.ndarray,
    y: np.ndarray,
    class_names: List[str],
    device: str = "cpu",
    *,
    epochs: int = 100,
    lr: float = 1e-3,
    batch_size: int = 256,
    hidden_dim: int = 512,
    dropout: float = 0.1,
    weight_decay: float = 1e-4,
    test_ratio: float = 0.2,
    seed: int = 42,
    early_stop_patience: int = 20,
    num_runs: int = 70,
    subset_size: int = 10,
    samples_per_class: int | None = None,
    train_shots: int | None = None,
    test_shots: int | None = None,
) -> dict:
    """Train multiple times on random N-class subsets, return aggregated results."""
    
    num_classes = len(class_names)

    # Fall back to all available classes if the dataset has fewer than subset_size
    effective_subset_size = min(subset_size, num_classes)
    if effective_subset_size < subset_size:
        print(f"  Note: {dataset_name} has only {num_classes} classes; using all {num_classes} instead of {subset_size}")

    # Generate class subsets
    max_combos = calculate_max_combinations(num_classes, effective_subset_size)

    if max_combos == 0:
        print(f"  Warning: Cannot create {effective_subset_size}-class subsets from {num_classes} classes")
        return None

    if max_combos < num_runs:
        print(f"  Note: only {max_combos} unique {effective_subset_size}-class combination(s); "
              f"cycling subsets across {num_runs} runs with different splits")
    print(f"  Creating {num_runs} runs of {effective_subset_size}-class training")
    class_subsets = generate_random_class_subsets(num_classes, effective_subset_size, num_runs, seed)
    
    run_results = []
    
    # Train on each subset
    for run_idx, class_subset in enumerate(tqdm(class_subsets, desc=f"  {dataset_name} runs", leave=False)):
        result = train_single_dataset(
            dataset_name=f"{dataset_name}_run{run_idx+1}",
            X=X,
            y=y,
            class_names=class_names,
            output_dir=Path("/tmp"),  # Temporary, not stored
            class_subset=class_subset,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            hidden_dim=hidden_dim,
            dropout=dropout,
            weight_decay=weight_decay,
            test_ratio=test_ratio,
            device=device,
            seed=seed + run_idx,
            early_stop_patience=early_stop_patience,
            samples_per_class=samples_per_class,
            train_shots=train_shots,
            test_shots=test_shots,
        )

        result["run"] = run_idx + 1
        result["selected_classes"] = class_subset
        run_results.append(result)
    
    # Aggregate results
    train_accs = [r["train_acc"] for r in run_results]
    final_train_accs = [r["final_train_acc"] for r in run_results]
    test_accs = [r["test_acc"] for r in run_results]
    final_accs = [r["final_test_acc"] for r in run_results]
    
    aggregated = {
        "dataset": dataset_name,
        "total_classes": int(num_classes),
        "num_runs": len(run_results),
        "subset_size": effective_subset_size,
        "samples_per_class": int(samples_per_class) if samples_per_class is not None else None,
        "max_possible_combinations": int(max_combos),
        "runs": [
            {
                "run": int(r["run"]),
                "num_classes": int(r["num_classes"]),
                "num_samples": int(r["num_samples"]),
                "train_acc": float(r["train_acc"]),
                "final_train_acc": float(r["final_train_acc"]),
                "test_acc": float(r["test_acc"]),
                "final_test_acc": float(r["final_test_acc"]),
                "selected_classes": [int(x) for x in r["selected_classes"]],
            }
            for r in run_results
        ],
        "average_train_acc": float(np.mean(train_accs)),
        "std_train_acc": float(np.std(train_accs)),
        "min_train_acc": float(np.min(train_accs)),
        "max_train_acc": float(np.max(train_accs)),
        "average_final_train_acc": float(np.mean(final_train_accs)),
        "std_final_train_acc": float(np.std(final_train_accs)),
        "average_test_acc": float(np.mean(test_accs)),
        "std_test_acc": float(np.std(test_accs)),
        "min_test_acc": float(np.min(test_accs)),
        "max_test_acc": float(np.max(test_accs)),
        "average_final_test_acc": float(np.mean(final_accs)),
        "std_final_test_acc": float(np.std(final_accs)),
    }
    
    return aggregated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MLP baseline for DuckDB-stored embedding datasets")
    parser.add_argument("--db", required=True, help="Path to hierarchical DuckDB file")
    parser.add_argument("--output", default="mlp_baseline/results", help="Root output directory")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs per dataset")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--hidden-dim", type=int, default=2048, help="MLP hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout rate")
    parser.add_argument("--weight-decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--early-stop-patience", type=int, default=50, help="Early stopping patience (epochs)")
    parser.add_argument("--test-ratio", type=float, default=0.2, help="Fraction of data for test set")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--datasets", nargs="*", default=None, help="Only train on these datasets (default: all)")
    parser.add_argument("--multiclass-runs", type=int, default=70, help="Number of 10-class subset runs per dataset")
    parser.add_argument("--subset-size", type=int, default=10, help="Number of classes per subset")
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=None,
        help="Maximum number of samples to use per class before train/test split (default: use all)",
    )
    parser.add_argument(
        "--train-shots",
        type=int,
        default=None,
        help="Exact number of training samples per class (few-shot mode). Overrides --samples-per-class and --test-ratio.",
    )
    parser.add_argument(
        "--test-shots",
        type=int,
        default=None,
        help="Exact number of test samples per class (few-shot mode). Defaults to same as --train-shots if only that is set.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to use for training (e.g. cpu, cuda, cuda:0, mps). Defaults to auto-detect.",
    )
    args = parser.parse_args()

    if args.device is not None:
        device = args.device
    else:
        device = (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    print(f"Using device: {device}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- determine which datasets to train on ----
    all_ds_names = list_datasets(args.db)
    if args.datasets:
        ds_names = [n for n in args.datasets if n in all_ds_names]
        missing = set(args.datasets) - set(ds_names)
        for m in missing:
            print(f"  [warn] Requested dataset '{m}' not found in DB, skipping")
    else:
        ds_names = all_ds_names

    if not ds_names:
        print("No datasets to train on. Exiting.")
        sys.exit(1)

    print(f"\nTraining MLP baseline on {len(ds_names)} dataset(s) with {args.multiclass_runs} runs each\n")

    # ---- train each dataset (loaded one at a time) ----
    all_results: Dict[str, dict] = {}
    
    for ds_name in sorted(ds_names):
        print(f"Loading {ds_name}...")
        X, y, class_names = load_single_dataset(args.db, ds_name)
        print(f"── {ds_name} ({X.shape[0]} samples, {len(class_names)} classes) ──")
        
        result = train_multiclass_runs(
            dataset_name=ds_name,
            X=X,
            y=y,
            class_names=class_names,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            weight_decay=args.weight_decay,
            test_ratio=args.test_ratio,
            seed=args.seed,
            early_stop_patience=args.early_stop_patience,
            num_runs=args.multiclass_runs,
            subset_size=args.subset_size,
            samples_per_class=args.samples_per_class,
            train_shots=args.train_shots,
            test_shots=args.test_shots,
        )

        if result:
            all_results[ds_name] = result
            print(f"  => avg_test_acc={result['average_test_acc']:.4f} "
                  f"(std={result['std_test_acc']:.4f}, min={result['min_test_acc']:.4f}, max={result['max_test_acc']:.4f})\n")

    # ---- write summary JSON ----
    json_path = output_dir / "results_multiclass.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults written to {json_path}")


if __name__ == "__main__":
    main()
