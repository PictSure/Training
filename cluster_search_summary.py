#!/usr/bin/env python3
"""Summarize cluster-search runs into one line per run with overall accuracy."""

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt


RUN_PATTERN = re.compile(
    r"^(?P<prefix>.+)-e(?P<embed_dim>\d+)-h(?P<nheads>\d+)-l(?P<nlayers>\d+)_(?P<timestamp>\d{8}_\d{6})$"
)


@dataclass
class RunSummary:
    run_dir: str
    run_name: str
    timestamp: Optional[str]
    embed_dim: Optional[int]
    nheads: Optional[int]
    nlayers: Optional[int]
    best_train_acc: Optional[float]
    best_test_acc: Optional[float]
    overall_acc: Optional[float]
    total_params: Optional[int]
    curve_epochs: List[int]
    curve_values: List[float]
    curve_metric: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one-line summaries for cluster-search runs with overall accuracy"
    )
    parser.add_argument(
        "--cluster-dir",
        default="output/cluster_search",
        help="Directory containing run subfolders with epoch_metrics.csv",
    )
    parser.add_argument(
        "--format",
        choices=["text", "csv"],
        default="text",
        help="Output format",
    )
    parser.add_argument(
        "--sort-by",
        choices=["name", "accuracy", "embed", "timestamp"],
        default="accuracy",
        help="Sort key for output lines",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort descending (default for accuracy is descending, others ascending)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Create a matplotlib plot for all runs",
    )
    parser.add_argument(
        "--plot-path",
        default=None,
        help="Path to save the generated plot (default: <cluster-dir>/training_runs_accuracy.png)",
    )
    return parser.parse_args()


def _safe_float(value: str) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _safe_int(value) -> Optional[int]:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number


def _fmt_acc(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def _extract_run_metadata(folder_name: str) -> tuple[str, Optional[int], Optional[int], Optional[int], Optional[str]]:
    match = RUN_PATTERN.match(folder_name)
    if not match:
        return folder_name, None, None, None, None
    run_name = folder_name
    return (
        run_name,
        int(match.group("embed_dim")),
        int(match.group("nheads")),
        int(match.group("nlayers")),
        match.group("timestamp"),
    )


def _compute_total_parameters(
    embedding_input_dim: Optional[int],
    num_classes: Optional[int],
    embed_dim: Optional[int],
    nlayers: Optional[int],
) -> Optional[int]:
    if embedding_input_dim is None or num_classes is None or embed_dim is None or nlayers is None:
        return None

    e = embed_dim
    d_model = 2 * e
    d_ff = 4 * e

    x_projection = embedding_input_dim * e + e
    y_projection = num_classes * e + e
    transformer_per_layer = (
        (3 * d_model * d_model)
        + (3 * d_model)
        + (d_model * d_model)
        + d_model
        + (d_model * d_ff)
        + d_ff
        + (d_ff * d_model)
        + d_model
        + (2 * d_model)
        + (2 * d_model)
    )
    transformer_total = nlayers * transformer_per_layer
    classifier = (2 * e) * num_classes + num_classes
    return x_projection + y_projection + transformer_total + classifier


def _summarize_metrics_file(run_folder: Path) -> Optional[RunSummary]:
    metrics_path = run_folder / "epoch_metrics.csv"
    if not metrics_path.is_file():
        return None

    hyperparams_path = run_folder / "hyperparameters.json"
    hyperparams = {}
    if hyperparams_path.is_file():
        with hyperparams_path.open("r") as handle:
            hyperparams = json.load(handle)

    best_train_acc: Optional[float] = None
    best_test_acc: Optional[float] = None
    train_acc_by_epoch: dict[int, float] = {}
    test_acc_by_epoch: dict[int, float] = {}

    with metrics_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metric_name = row.get("metric_name")
            metric_value = _safe_float(row.get("metric_value", ""))
            epoch_idx_raw = row.get("epoch_idx", "")
            if metric_value is None:
                continue
            try:
                epoch_idx = int(epoch_idx_raw)
            except (TypeError, ValueError):
                continue

            if metric_name == "acc":
                if best_train_acc is None or metric_value > best_train_acc:
                    best_train_acc = metric_value
                train_acc_by_epoch[epoch_idx] = metric_value
            elif metric_name == "test_acc":
                if best_test_acc is None or metric_value > best_test_acc:
                    best_test_acc = metric_value
                test_acc_by_epoch[epoch_idx] = metric_value

    if test_acc_by_epoch:
        selected_curve = sorted(test_acc_by_epoch.items())
        curve_metric = "test_acc"
    else:
        selected_curve = sorted(train_acc_by_epoch.items())
        curve_metric = "acc"

    curve_epochs = [epoch for epoch, _ in selected_curve]
    curve_values = [value for _, value in selected_curve]

    run_name, embed_dim, nheads, nlayers, timestamp = _extract_run_metadata(run_folder.name)
    if embed_dim is None:
        embed_dim = _safe_int(hyperparams.get("model", {}).get("embed_dim"))
    if nheads is None:
        nheads = _safe_int(hyperparams.get("model", {}).get("nheads"))
    if nlayers is None:
        nlayers = _safe_int(hyperparams.get("model", {}).get("nlayers"))

    embedding_input_dim = _safe_int(hyperparams.get("embedding_dim"))
    num_classes = _safe_int(hyperparams.get("dataloader", {}).get("num_classes"))
    total_params = _compute_total_parameters(
        embedding_input_dim=embedding_input_dim,
        num_classes=num_classes,
        embed_dim=embed_dim,
        nlayers=nlayers,
    )

    overall_acc = best_test_acc if best_test_acc is not None else best_train_acc
    return RunSummary(
        run_dir=run_folder.name,
        run_name=run_name,
        timestamp=timestamp,
        embed_dim=embed_dim,
        nheads=nheads,
        nlayers=nlayers,
        best_train_acc=best_train_acc,
        best_test_acc=best_test_acc,
        overall_acc=overall_acc,
        total_params=total_params,
        curve_epochs=curve_epochs,
        curve_values=curve_values,
        curve_metric=curve_metric,
    )


def collect_summaries(cluster_dir: Path) -> List[RunSummary]:
    summaries: List[RunSummary] = []
    for child in sorted(cluster_dir.iterdir()):
        if not child.is_dir():
            continue
        summary = _summarize_metrics_file(child)
        if summary is not None:
            summaries.append(summary)
    return summaries


def sort_summaries(summaries: Iterable[RunSummary], sort_by: str, descending: bool) -> List[RunSummary]:
    runs = list(summaries)

    if sort_by == "accuracy":
        default_desc = True
        key = lambda item: item.overall_acc if item.overall_acc is not None else float("-inf")
    elif sort_by == "embed":
        default_desc = False
        key = lambda item: (
            item.embed_dim if item.embed_dim is not None else -1,
            item.nheads if item.nheads is not None else -1,
            item.nlayers if item.nlayers is not None else -1,
            item.run_name,
        )
    elif sort_by == "timestamp":
        default_desc = False
        key = lambda item: (item.timestamp or "", item.run_name)
    else:
        default_desc = False
        key = lambda item: item.run_name

    return sorted(runs, key=key, reverse=descending or (sort_by == "accuracy" and not descending and default_desc))


def print_text_lines(summaries: Iterable[RunSummary]) -> None:
    for run in summaries:
        print(
            f"{run.run_name}: overall_acc={_fmt_acc(run.overall_acc)}, "
            f"train_best={_fmt_acc(run.best_train_acc)}, "
            f"test_best={_fmt_acc(run.best_test_acc)}, "
            f"params={run.total_params if run.total_params is not None else 'N/A'}, "
            f"embed={run.embed_dim if run.embed_dim is not None else 'N/A'}, "
            f"heads={run.nheads if run.nheads is not None else 'N/A'}, "
            f"layers={run.nlayers if run.nlayers is not None else 'N/A'}"
        )


def print_csv_lines(summaries: Iterable[RunSummary]) -> None:
    writer = csv.writer(os.sys.stdout)
    writer.writerow(
        [
            "run_name",
            "run_dir",
            "timestamp",
            "embed_dim",
            "nheads",
            "nlayers",
            "total_params",
            "overall_acc",
            "best_train_acc",
            "best_test_acc",
        ]
    )
    for run in summaries:
        writer.writerow(
            [
                run.run_name,
                run.run_dir,
                run.timestamp or "",
                run.embed_dim if run.embed_dim is not None else "",
                run.nheads if run.nheads is not None else "",
                run.nlayers if run.nlayers is not None else "",
                run.total_params if run.total_params is not None else "",
                "" if run.overall_acc is None else f"{run.overall_acc:.6f}",
                "" if run.best_train_acc is None else f"{run.best_train_acc:.6f}",
                "" if run.best_test_acc is None else f"{run.best_test_acc:.6f}",
            ]
        )


def _run_label(run: RunSummary) -> str:
    if run.embed_dim is not None and run.nheads is not None and run.nlayers is not None:
        return f"e{run.embed_dim}-h{run.nheads}-l{run.nlayers}"
    return run.run_name


def _build_equal_spread_colors(runs: List[RunSummary]) -> dict[str, tuple]:
    if not runs:
        return {}

    ranked = sorted(
        runs,
        key=lambda run: (
            run.total_params if run.total_params is not None else -1,
            run.run_name,
        ),
    )
    total = len(ranked)
    cmap = plt.get_cmap("RdYlGn")
    color_map: dict[str, tuple] = {}

    for index, run in enumerate(ranked):
        if total == 1:
            color_position = 0.5
        else:
            color_position = 1.0 - (index / (total - 1))
        color_map[run.run_name] = cmap(color_position)
    return color_map


def create_training_plot(summaries: List[RunSummary], plot_path: Path) -> Path:
    runs_with_curves = [run for run in summaries if run.curve_epochs and run.curve_values]
    if not runs_with_curves:
        raise RuntimeError("No runs with epoch accuracy curves available for plotting")

    colors_by_run = _build_equal_spread_colors(runs_with_curves)

    fig, ax = plt.subplots(figsize=(14, 8))

    for run in runs_with_curves:
        run_color = colors_by_run.get(run.run_name, plt.get_cmap("RdYlGn")(0.5))
        (line,) = ax.plot(
            run.curve_epochs,
            run.curve_values,
            linestyle="-",
            linewidth=1.5,
            alpha=0.85,
            label=_run_label(run),
            color=run_color,
        )

        peak_epoch, peak_acc = max(
            zip(run.curve_epochs, run.curve_values),
            key=lambda item: item[1],
        )
        ax.scatter(
            [peak_epoch],
            [peak_acc],
            color=line.get_color(),
            edgecolor="black",
            linewidth=0.6,
            s=55,
            marker="o",
            zorder=4,
        )

    ax.set_title("Training Curves Across Cluster Search Runs (Peak per Run Highlighted)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, frameon=False)

    plt.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path)
    plt.close(fig)
    return plot_path


def main() -> None:
    args = parse_args()
    cluster_dir = Path(args.cluster_dir)

    if not cluster_dir.is_dir():
        raise FileNotFoundError(f"Cluster search directory not found: {cluster_dir}")

    summaries = collect_summaries(cluster_dir)
    if not summaries:
        raise RuntimeError(f"No run folders with epoch_metrics.csv found in {cluster_dir}")

    sorted_summaries = sort_summaries(
        summaries=summaries,
        sort_by=args.sort_by,
        descending=args.descending,
    )

    if args.format == "csv":
        print_csv_lines(sorted_summaries)
    else:
        print_text_lines(sorted_summaries)

    if args.plot:
        output_path = (
            Path(args.plot_path)
            if args.plot_path
            else cluster_dir / "training_runs_accuracy.png"
        )
        saved_path = create_training_plot(sorted_summaries, output_path)
        print(f"Plot saved to {saved_path}")


if __name__ == "__main__":
    main()