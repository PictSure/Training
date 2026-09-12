#!/usr/bin/env python3
"""Generate a scaling-law style plot: max accuracy vs total parameters (log scale)."""

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import matplotlib.pyplot as plt


@dataclass
class ScalingPoint:
    run_name: str
    total_params: int
    max_train_acc: Optional[float]
    max_test_acc: Optional[float]
    max_accuracy: float
    embed_dim: Optional[int]
    nheads: Optional[int]
    nlayers: Optional[int]
    embedding_model: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a scaling-law plot from cluster-search outputs"
    )
    parser.add_argument(
        "--cluster-dir",
        default="output/cluster_search",
        help="Directory with run folders that contain epoch_metrics.csv and hyperparameters.json",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output image path (default: <cluster-dir>/scaling_law_max_accuracy_log.png)",
    )
    parser.add_argument(
        "--top-k-annotate",
        type=int,
        default=5,
        help="Annotate top-k runs by max accuracy",
    )
    parser.add_argument(
        "--metric",
        choices=["overall", "train", "test"],
        default="overall",
        help="Metric to plot: overall=max(test,train), train=max(train), test=max(test)",
    )
    return parser.parse_args()


def _safe_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_float(value) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


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


def _infer_embedding_model_name(hyper: dict) -> str:
    encoder = hyper.get("encoder")
    if isinstance(encoder, str) and encoder.strip():
        return encoder.strip()

    duckdb_path = hyper.get("duckdb-path")
    if isinstance(duckdb_path, str) and duckdb_path.strip():
        filename = Path(duckdb_path).stem.lower()
        for token in ["dinov3", "dinov2", "dino", "clip", "mae", "eva", "vit"]:
            if token in filename:
                return token
        return filename

    return "unknown"


def _format_param_count(total_params: int) -> str:
    if total_params >= 1_000_000_000:
        return f"{total_params / 1_000_000_000:.2f}B"
    if total_params >= 1_000_000:
        return f"{total_params / 1_000_000:.1f}M"
    if total_params >= 1_000:
        return f"{total_params / 1_000:.1f}K"
    return str(total_params)


def _read_max_accuracies(metrics_file: Path) -> tuple[Optional[float], Optional[float]]:
    max_train: Optional[float] = None
    max_test: Optional[float] = None
    with metrics_file.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            metric_name = row.get("metric_name")
            metric_value = _safe_float(row.get("metric_value"))
            if metric_value is None:
                continue
            if metric_name == "acc":
                if max_train is None or metric_value > max_train:
                    max_train = metric_value
            elif metric_name == "test_acc":
                if max_test is None or metric_value > max_test:
                    max_test = metric_value
    return max_train, max_test


def _choose_metric(max_train: Optional[float], max_test: Optional[float], metric: str) -> Optional[float]:
    if metric == "train":
        return max_train
    if metric == "test":
        return max_test
    if max_test is not None:
        return max_test
    return max_train


def collect_scaling_points(cluster_dir: Path, metric: str) -> List[ScalingPoint]:
    points: List[ScalingPoint] = []
    for run_folder in sorted(cluster_dir.iterdir()):
        if not run_folder.is_dir():
            continue

        metrics_file = run_folder / "epoch_metrics.csv"
        hyper_file = run_folder / "hyperparameters.json"
        if not metrics_file.is_file() or not hyper_file.is_file():
            continue

        with hyper_file.open("r") as handle:
            hyper = json.load(handle)

        embed_dim = _safe_int(hyper.get("model", {}).get("embed_dim"))
        nheads = _safe_int(hyper.get("model", {}).get("nheads"))
        nlayers = _safe_int(hyper.get("model", {}).get("nlayers"))
        embedding_input_dim = _safe_int(hyper.get("embedding_dim"))
        num_classes = _safe_int(hyper.get("dataloader", {}).get("num_classes"))

        total_params = _compute_total_parameters(
            embedding_input_dim=embedding_input_dim,
            num_classes=num_classes,
            embed_dim=embed_dim,
            nlayers=nlayers,
        )
        if total_params is None or total_params <= 0:
            continue

        max_train, max_test = _read_max_accuracies(metrics_file)
        chosen = _choose_metric(max_train, max_test, metric=metric)
        if chosen is None:
            continue

        points.append(
            ScalingPoint(
                run_name=run_folder.name,
                total_params=total_params,
                max_train_acc=max_train,
                max_test_acc=max_test,
                max_accuracy=chosen,
                embed_dim=embed_dim,
                nheads=nheads,
                nlayers=nlayers,
                embedding_model=_infer_embedding_model_name(hyper),
            )
        )
    return points


def _linear_fit_on_loglog_error(points: Iterable[ScalingPoint]) -> tuple[float, float]:
    xs = [math.log10(point.total_params) for point in points]
    ys = [math.log10(max(1.0 - point.max_accuracy, 1e-8)) for point in points]
    n = len(xs)
    if n < 2:
        raise RuntimeError("Need at least 2 runs for a trend line")

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    var_x = sum((x - mean_x) ** 2 for x in xs)
    if var_x == 0:
        return 0.0, mean_y
    cov_xy = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    slope = cov_xy / var_x
    intercept = mean_y - slope * mean_x
    return slope, intercept


def make_plot(points: List[ScalingPoint], output_path: Path, top_k_annotate: int, metric_name: str) -> Path:
    if not points:
        raise RuntimeError("No valid runs found for scaling-law plot")

    points = sorted(points, key=lambda point: point.total_params)
    x_vals = [point.total_params for point in points]
    y_vals = [max(1.0 - point.max_accuracy, 1e-8) for point in points]

    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(
        x_vals,
        y_vals,
        c=[math.log10(point.total_params) for point in points],
        cmap="viridis",
        s=60,
        alpha=0.9,
        edgecolors="black",
        linewidths=0.4,
    )

    slope, intercept = _linear_fit_on_loglog_error(points)
    min_x = min(x_vals)
    max_x = max(x_vals)
    x_line = [10 ** (math.log10(min_x) + i * (math.log10(max_x) - math.log10(min_x)) / 200) for i in range(201)]
    y_line = [10 ** (intercept + slope * math.log10(x)) for x in x_line]
    ax.plot(x_line, y_line, linestyle="--", linewidth=2.0, color="tab:orange", label="Log-log fit on error")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_xlabel("Total Parameters (log scale)")
    ax.set_ylabel("1 - Maximum Accuracy (log scale, inverted)")
    ax.set_title(f"Scaling Law Plot ({metric_name} max accuracy, high-end expanded)")
    ax.grid(True, which="both", linestyle="--", alpha=0.35)

    best_point = max(points, key=lambda point: point.max_accuracy)
    best_error = max(1.0 - best_point.max_accuracy, 1e-8)
    ax.scatter([best_point.total_params], [best_error], marker="*", s=180, color="red", zorder=5)
    ax.annotate(
        f"Best: {_format_param_count(best_point.total_params)}\n{best_point.embedding_model}",
        xy=(best_point.total_params, best_error),
        xytext=(12, -16),
        textcoords="offset points",
        va="top",
        fontsize=9,
        color="red",
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": "red", "lw": 1},
    )

    top_points = sorted(points, key=lambda point: point.max_accuracy, reverse=True)[: max(top_k_annotate, 0)]
    for point in top_points:
        point_error = max(1.0 - point.max_accuracy, 1e-8)
        ax.annotate(
            f"{point.max_accuracy:.3f}",
            (point.total_params, point_error),
            textcoords="offset points",
            xytext=(4, -10),
            fontsize=8,
            alpha=0.8,
        )

    min_acc = min(point.max_accuracy for point in points)
    max_acc = max(point.max_accuracy for point in points)
    acc_span = max_acc - min_acc

    if acc_span <= 0.08:
        step = 0.01
    elif acc_span <= 0.2:
        step = 0.02
    else:
        step = 0.05

    start_tick = math.floor(min_acc / step) * step
    end_tick = math.ceil(max_acc / step) * step

    accuracy_ticks = []
    tick = start_tick
    while tick <= end_tick + 1e-12:
        bounded_tick = min(max(tick, 1e-6), 1 - 1e-6)
        accuracy_ticks.append(bounded_tick)
        tick += step

    tick_positions = [max(1.0 - tick_value, 1e-8) for tick_value in accuracy_ticks]
    ax.set_yticks(tick_positions)
    ax.set_yticklabels([f"{tick_value:.2f}" for tick_value in accuracy_ticks])

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("log10(total parameters)")
    ax.legend(loc="lower right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path


def main() -> None:
    args = parse_args()
    cluster_dir = Path(args.cluster_dir)
    if not cluster_dir.is_dir():
        raise FileNotFoundError(f"Cluster directory not found: {cluster_dir}")

    output_path = (
        Path(args.output)
        if args.output
        else cluster_dir / "scaling_law_max_accuracy_log.png"
    )

    points = collect_scaling_points(cluster_dir=cluster_dir, metric=args.metric)
    saved = make_plot(
        points=points,
        output_path=output_path,
        top_k_annotate=args.top_k_annotate,
        metric_name=args.metric,
    )

    print(f"Collected {len(points)} runs")
    print(f"Scaling-law plot saved to {saved}")


if __name__ == "__main__":
    main()
