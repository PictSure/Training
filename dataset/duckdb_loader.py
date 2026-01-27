from __future__ import annotations
import random
import duckdb
from collections import defaultdict
import json
from pathlib import Path
import argparse
from tqdm import tqdm
import random
import torch
from torch.utils.data import IterableDataset, DataLoader
from typing import Dict, Iterator, List, Optional, Tuple


class DuckDBEmbeddingDataset(IterableDataset):
    """
    Streams embedding groups from a DuckDB database (combined reader + iterable).

    Each yielded item is a 4-tuple of tensors:
        context_images: FloatTensor [C, D]
        context_labels: LongTensor  [C]
        pred_image:     FloatTensor [D]
        pred_label:     LongTensor  [] (scalar)

    Parameters
    ----------
    db_path : str
        Path to the DuckDB database file.
    dataset_name : str
        Which dataset inside the DB to sample from.
    n_samples : int
        Number of context samples PER CLASS.
    n_classes : int
        Number of classes per group.
    groups_per_epoch : Optional[int]
        How many sample groups to produce per iteration over the dataset.
        If None, the iterator is infinite; stop via DataLoader's `num_batches` or an outer loop.
    dtype : torch.dtype
        Tensor dtype for embeddings (default: float32).
    device : Optional[torch.device]
        If provided, tensors are created on this device.
    table_name : str
        DuckDB table name containing columns: "Dataset", "Label", "Embedding".
    read_only : bool
        Open DuckDB connection in read-only mode (default: True).
    verbose : bool
        If True, prints a short connection message per worker.
    """

    def __init__(
        self,
        *,
        db_path: str,
        dataset_name: str,
        n_samples: int,
        n_classes: int,
        groups_per_epoch: Optional[int] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = "cpu",
        table_name: str = "embeddings",
        read_only: bool = True,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.db_path = db_path
        self.dataset_name = dataset_name
        self.n_samples = n_samples
        self.n_classes = n_classes
        self.groups_per_epoch = groups_per_epoch
        self.dtype = dtype
        self.device = device
        self.table_name = table_name
        self.read_only = read_only
        self.verbose = verbose
        self._num_workers = 0

        self.embedding_dim = self._get_embedding_dim()

    def _get_embedding_dim(self) -> int:
        """
        Get embedding dimension by sampling one embedding from the DB.
        """
        conn = self._open_conn()
        try:
            query = f"""
                SELECT "Embedding"
                FROM {self.table_name}
                WHERE "Dataset" = ?
                LIMIT 1
            """
            row = conn.execute(query, [self.dataset_name]).fetchone()
            if row is None:
                raise RuntimeError(
                    f"No embeddings found for dataset '{self.dataset_name}' in table '{self.table_name}'."
                )
            emb = row[0]
            emb_list = self._to_float_list(emb)
            return len(emb_list)
        finally:
            conn.close()

    def _open_conn(self) -> duckdb.DuckDBPyConnection:
        conn = duckdb.connect(database=self.db_path, read_only=self.read_only)
        return conn

    def _get_dataset_labels(self, conn: duckdb.DuckDBPyConnection) -> Dict[str, List[str]]:
        """
        Extract all unique datasets and their labels.

        Returns:
            dict: {dataset_name: [label1, label2, ...]} (labels sorted for determinism).
        """
        query = f"""
            SELECT DISTINCT "Dataset", "Label"
            FROM {self.table_name}
        """
        results = conn.execute(query).fetchall()
        dataset_labels = defaultdict(set)
        for dataset, label in results:
            dataset_labels[dataset].add(label)
        return {ds: sorted(labels) for ds, labels in dataset_labels.items()}

    @staticmethod
    def _to_float_list(e) -> List[float]:
        # duckdb list/array -> list already
        if isinstance(e, (list, tuple)):
            return [float(x) for x in e]
        # stringified "[...]" -> parse
        if isinstance(e, str):
            return [float(x) for x in e.strip("[]").split(",") if x.strip() != ""]
        # fallback (e.g., numpy arrays)
        try:
            return [float(x) for x in e]
        except TypeError:
            raise TypeError(
                f"Unsupported embedding type: {type(e)}. "
                "Expected list-like or stringified list."
            )

    def _sample(
        self,
        conn: duckdb.DuckDBPyConnection,
        dataset_labels: Dict[str, List[str]],
        *,
        dataset_name: str,
        n_samples: int,
        n_classes: int,
    ) -> dict:
        """
        Sample embeddings for a given dataset, returning a record like:
        {
          "Context": [(label_idx:int, embedding:list[float]), ...],  # length = n_samples * n_classes
          "Prediction": (label_idx:int, embedding:list[float]),      # one extra example
          "Dataset": str
        }
        """
        # --- validations ---
        if dataset_name not in dataset_labels:
            raise ValueError(f"Dataset '{dataset_name}' not found.")

        all_labels = dataset_labels[dataset_name]

        if n_classes < 1:
            raise ValueError("n_classes must be >= 1.")
        if n_classes > len(all_labels):
            raise ValueError(
                f"n_classes ({n_classes}) exceeds number of labels in dataset "
                f"({len(all_labels)})."
            )
        if n_samples < 1:
            raise ValueError("n_samples must be >= 1.")

        # --- (1) select n_classes labels ---
        chosen_labels = random.sample(all_labels, n_classes) if n_classes > 1 else [all_labels[0]]

        # --- (2) map to indices 0..n_classes-1 ---
        label_to_index = {lbl: idx for idx, lbl in enumerate(chosen_labels)}

        samples: List[Tuple[int, List[float]]] = []

        # --- (3) sample n_samples per class ---
        query_main = f"""
            SELECT "Embedding"
            FROM {self.table_name}
            WHERE "Dataset" = ? AND "Label" = ?
            ORDER BY random()
            LIMIT {n_samples}
        """
        for label in chosen_labels:
            rows = conn.execute(query_main, [dataset_name, label]).fetchall()
            index = label_to_index[label]
            for (emb,) in rows:
                samples.append((index, self._to_float_list(emb)))

        # --- (4) add one extra sample from a random label among the selected classes ---
        extra_label = random.choice(chosen_labels)
        query_extra = f"""
            SELECT "Embedding"
            FROM {self.table_name}
            WHERE "Dataset" = ? AND "Label" = ?
            ORDER BY random()
            LIMIT 1
        """
        extra = conn.execute(query_extra, [dataset_name, extra_label]).fetchone()
        if extra is None:
            raise RuntimeError(
                f"No embeddings found for dataset '{dataset_name}' and label '{extra_label}'."
            )

        record = {
            "Context": samples,
            "Prediction": (label_to_index[extra_label], self._to_float_list(extra[0])),
            "Dataset": dataset_name,
        }
        return record

    def _to_tensors(
        self, record: dict
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Convert the record dict into tensors with stable shapes.
        record = {
          "Context": [(label_idx:int, embedding:list[float]), ...],
          "Prediction": (label_idx:int, embedding:list[float]),
          "Dataset": str
        }
        """
        context = record["Context"]
        pred_lbl, pred_emb = record["Prediction"]

        # context -> [C, D] and [C]
        context_labels = torch.tensor(
            [int(lbl) for (lbl, _) in context],
            dtype=torch.long,
            device=self.device,
        )  # [C]
        context_images = torch.tensor(
            [list(map(float, emb)) for (_, emb) in context],
            dtype=self.dtype,
            device=self.device,
        )  # [C, D]

        # prediction -> [D] and []
        pred_image = torch.tensor(
            list(map(float, pred_emb)), dtype=self.dtype, device=self.device
        )  # [D]
        pred_label = torch.tensor(int(pred_lbl), dtype=torch.long, device=self.device)  # []

        return context_images, context_labels, pred_image, pred_label

    # ------------------------
    # IterableDataset interface
    # ------------------------
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Each worker creates its own DuckDB connection and samples groups.
        """
        # Make sampling deterministic per-worker if the user sets worker_init_fn or generator.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Seed python's RNG for this worker from torch's worker seed
            base_seed = torch.initial_seed()  # large 64-bit number
            random.seed(base_seed ^ (worker_info.id + 0x9E3779B97F4A7C15))

        self._num_workers = worker_info.num_workers if worker_info is not None else 1

        conn = self._open_conn()
        try:
            # Discover labels per worker (avoids sharing state across processes)
            dataset_labels = self._get_dataset_labels(conn)
            if self.verbose:
                print(
                    f"✅ Connected to DuckDB at {self.db_path}. "
                    f"Found datasets: {list(dataset_labels.keys())}"
                )

            n = 0
            while self.groups_per_epoch is None or n < (self.groups_per_epoch // self._num_workers):
                record = self._sample(
                    conn,
                    dataset_labels,
                    dataset_name=self.dataset_name,
                    n_samples=self.n_samples,
                    n_classes=self.n_classes,
                )
                yield self._to_tensors(record)
                n += 1
        finally:
            conn.close()

    def __len__(self) -> int:
        # Only defined when groups_per_epoch is set.
        if self.groups_per_epoch is None:
            raise TypeError(
                "Length is undefined for infinite DuckDBEmbeddingDataset. "
                "Set groups_per_epoch to make it finite."
            )
        return self.groups_per_epoch


def collate_embedding_batch(batch):
    """
    batch: list of tuples (context_images [C,D], context_labels [C], pred_image [D], pred_label [])
    Returns:
        context_images: [B, C, D]
        context_labels: [B, C]
        pred_image:     [B, D]
        pred_label:     [B]
    """
    c_imgs = torch.stack([b[0] for b in batch], dim=0)       # [B, C, D]
    c_labs = torch.stack([b[1] for b in batch], dim=0)       # [B, C]
    p_imgs = torch.stack([b[2] for b in batch], dim=0)       # [B, D]
    p_labs = torch.stack([b[3] for b in batch], dim=0).view(-1)  # [B]
    return c_imgs, c_labs, p_imgs, p_labs