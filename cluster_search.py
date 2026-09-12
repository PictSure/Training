#!/usr/bin/env python3
"""Launch controlled transformer-size experiments and plot accuracy versus model size."""
import argparse
import copy
import os
from typing import Iterable, List, Mapping, Optional

import matplotlib.pyplot as plt
import torch
import yaml

from trainer_duckdb import Trainer


def detect_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _parse_int_list(value: Optional[str]) -> List[int]:
    if not value:
        return []
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _build_candidates(overrides: List[int], defaults: List[int]) -> List[int]:
    if overrides:
        return sorted({value for value in overrides if value > 0})
    return sorted(defaults)


def compute_model_size(embed_dim: int, nheads: int, nlayers: int) -> int:
    return embed_dim * nheads * nlayers


def _load_config(path: str) -> Mapping:
    with open(path, "r") as handle:
        return yaml.safe_load(handle)


def _ensure_output_directory(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_search(
    config_path: str,
    output_dir: str,
    epochs: int,
    embed_dims: List[int],
    nheads: List[int],
    nlayers: List[int],
    run_prefix: str,
    device: str,
) -> List[Mapping]:
    base_config = copy.deepcopy(_load_config(config_path))
    _ensure_output_directory(output_dir)

    # Build valid combinations (embed_dim must be divisible by nheads)
    valid_combinations = []
    for embed_dim in embed_dims:
        for nhead in nheads:
            for nlayer in nlayers:
                if embed_dim % nhead == 0:
                    valid_combinations.append((embed_dim, nhead, nlayer))
                else:
                    print(f"Skipping invalid config: embed_dim={embed_dim}, nheads={nhead}, nlayers={nlayer} (embed_dim not divisible by nheads)")
    
    print(f"\nRunning {len(valid_combinations)} valid configurations")
    
    trainer_args = argparse.Namespace(new=True)
    results = []

    for embed_dim, nhead, nlayer in valid_combinations:
        variant_name = f"{run_prefix}-e{embed_dim}-h{nhead}-l{nlayer}"
        variant_config = copy.deepcopy(base_config)
        variant_config["model"]["embed_dim"] = embed_dim
        variant_config["model"]["nheads"] = nhead
        variant_config["model"]["nlayers"] = nlayer
        variant_config["optimizer"]["epochs"] = epochs
        variant_config["paths"]["output"] = output_dir
        variant_config["name"] = variant_name

        print(f"\nRunning {variant_name}: embed_dim={embed_dim}, nheads={nhead}, nlayers={nlayer}")
        trainer = Trainer(variant_config, device, trainer_args)
        trainer.train()

        accuracy = None
        if trainer.accuracies:
            accuracy = max(trainer.accuracies)

        results.append(
            {
                "name": variant_name,
                "embed_dim": embed_dim,
                "nheads": nhead,
                "nlayers": nlayer,
                "model_size": compute_model_size(embed_dim, nhead, nlayer),
                "accuracy": accuracy,
            }
        )
    return results


def plot_results(results: List[Mapping], output_dir: str) -> str:
    sorted_results = sorted(results, key=lambda entry: entry["model_size"])
    sizes = [entry["model_size"] for entry in sorted_results if entry["accuracy"] is not None]
    accuracies = [entry["accuracy"] for entry in sorted_results if entry["accuracy"] is not None]

    if not sizes or not accuracies:
        raise RuntimeError("No accuracy data available to plot")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(sizes, accuracies, color="tab:blue")
    for entry in sorted_results:
        if entry["accuracy"] is None:
            continue
        ax.annotate(
            f"{entry['nheads']}h-{entry['nlayers']}l-{entry['embed_dim']}e",
            (entry["model_size"], entry["accuracy"]),
            textcoords="offset points",
            xytext=(0, 6),
            ha="center",
            fontsize=8,
        )
    ax.set_xlabel("Model size (embed_dim × nheads × nlayers)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Cluster search: accuracy vs. model size")
    ax.grid(True, linestyle="--", alpha=0.35)
    plt.tight_layout()

    plot_path = os.path.join(output_dir, "accuracy_vs_model_size.png")
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


def echo_results(results: List[Mapping]) -> None:
    for entry in sorted(results, key=lambda item: item["model_size"]):
        accuracy = entry["accuracy"]
        accuracy_display = f"{accuracy:.4f}" if accuracy is not None else "N/A"
        print(
            f"{entry['name']}: size={entry['model_size']}, embed={entry['embed_dim']}, "
            f"heads={entry['nheads']}, layers={entry['nlayers']}, acc={accuracy_display}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-search transformer sizes with DuckDB data")
    parser.add_argument("--config", "-c", default="configs/local_duckdb_search.yaml")
    parser.add_argument("--output-dir", default="output/cluster_search")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument(
        "--embed-dims",
        help="Comma-separated embed dim candidates (default: 64,128,256,512,1024).",
        default=None,
    )
    parser.add_argument(
        "--nheads",
        help="Comma-separated head counts (default: 4,8,16).",
        default=None,
    )
    parser.add_argument(
        "--nlayers",
        help="Comma-separated layer counts (default: 2,4,6,8,12).",
        default=None,
    )
    parser.add_argument("--run-prefix", default="cluster-search")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    base_config = _load_config(args.config)

    embed_dims = _build_candidates(_parse_int_list(args.embed_dims), [64, 256, 512, 1024])
    nheads = _build_candidates(_parse_int_list(args.nheads), [4, 8])
    nlayers = _build_candidates(_parse_int_list(args.nlayers), [1, 2, 4, 6])

    device = args.device or detect_device()

    results = run_search(
        config_path=args.config,
        output_dir=args.output_dir,
        epochs=args.epochs,
        embed_dims=embed_dims,
        nheads=nheads,
        nlayers=nlayers,
        run_prefix=args.run_prefix,
        device=device,
    )

    echo_results(results)
    plot_path = plot_results(results, args.output_dir)
    print(f"Scatter plot saved to {plot_path}")


if __name__ == "__main__":
    main()
