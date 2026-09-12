"""Plot MLP baseline performance and optional ICL comparison.

Usage:
    python mlp_baseline/plot_mlp_results.py \
        --input mlp_baseline/results/results_multiclass.json

Multiple named inputs:
    python mlp_baseline/plot_mlp_results.py \
        --input mlp_baseline/results/results_multiclass.json \
                mlp_baseline/10-shot-results/results_multiclass.json \
        --input-labels full 10-shot

Include train accuracy:
    python mlp_baseline/plot_mlp_results.py \
        --input mlp_baseline/results/results_multiclass.json \
        --include-train-acc

Compare against ICL final metrics from epoch_metrics.csv:
    python mlp_baseline/plot_mlp_results.py \
        --input mlp_baseline/results/results_multiclass.json \
        --icl-metrics output/dinov2_normal/test_20260220_193516/epoch_metrics.csv
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


TRAIN_KEY_CANDIDATES: Sequence[str] = (
    "train_acc",
    "best_train_acc",
    "final_train_acc",
)

TEST_KEY_CANDIDATES: Sequence[str] = (
    "test_acc",
    "best_test_acc",
    "final_test_acc",
)


def load_results(path: Path) -> Dict[str, Mapping]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected top-level dict in {path}, got {type(payload).__name__}")
    if not payload:
        raise ValueError(f"No dataset entries found in {path}")

    return payload


def collect_numeric_run_keys(results: Mapping[str, Mapping]) -> set[str]:
    keys: set[str] = set()
    for dataset in results.values():
        runs = dataset.get("runs", [])
        if not isinstance(runs, list):
            continue
        for run in runs:
            if not isinstance(run, dict):
                continue
            for key, value in run.items():
                if isinstance(value, (int, float)):
                    keys.add(key)
    return keys


def resolve_metric_key(
    *,
    available_keys: set[str],
    requested_key: str | None,
    candidates: Sequence[str],
    metric_name: str,
) -> str:
    key = requested_key
    if key is None:
        for candidate in candidates:
            if candidate in available_keys:
                key = candidate
                break

    if key is None:
        raise ValueError(
            f"Could not find a {metric_name} key in run entries. "
            f"Use --{metric_name}-key to set it explicitly."
        )
    if key not in available_keys:
        raise ValueError(f"{metric_name} key '{key}' not found in run entries")

    return key


def parse_metric_value(value: str) -> float | None:
    """Parse scalar metrics that may be serialized as strings or single-item lists."""
    try:
        return float(value)
    except (TypeError, ValueError):
        pass

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, (int, float)):
            return float(parsed)
        if isinstance(parsed, (list, tuple)) and len(parsed) == 1 and isinstance(parsed[0], (int, float)):
            return float(parsed[0])
    except Exception:
        return None

    return None


def load_single_icl_from_epoch_csv(path: Path, phase: str, summary_mode: str) -> Dict[str, float]:
    """Load per-dataset ICL accuracy from an epoch_metrics.csv file."""
    phase_lc = phase.lower()
    best_values: Dict[str, float] = {}
    final_values: Dict[str, Tuple[int, float]] = {}
    available_phases: set[str] = set()

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_phase = (row.get("phase") or "").strip().lower()
            if row_phase:
                available_phases.add(row_phase)
            if row_phase != phase_lc:
                continue

            metric_name = (row.get("metric_name") or "").strip()
            if not metric_name.startswith("acc_"):
                continue

            dataset_name = metric_name.removeprefix("acc_")
            metric_value = parse_metric_value((row.get("metric_value") or "").strip())
            if metric_value is None:
                continue

            try:
                epoch_idx = int((row.get("epoch_idx") or "0").strip())
            except ValueError:
                continue

            if summary_mode == "best":
                if dataset_name not in best_values or metric_value > best_values[dataset_name]:
                    best_values[dataset_name] = metric_value
                continue

            previous = final_values.get(dataset_name)
            if previous is None or epoch_idx >= previous[0]:
                final_values[dataset_name] = (epoch_idx, metric_value)

    if phase_lc not in available_phases:
        available = ", ".join(sorted(available_phases)) if available_phases else "none"
        raise ValueError(
            f"Phase '{phase}' not present in '{path}'. Available phases: {available}"
        )

    if summary_mode == "best":
        return best_values
    return {dataset: value for dataset, (_, value) in final_values.items()}


def load_single_icl_from_json(path: Path, summary_mode: str) -> Dict[str, float]:
    """Load per-dataset ICL accuracy from a JSON summary, if compatible."""
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict JSON structure in '{path}'")

    if all(isinstance(v, (int, float)) for v in payload.values()):
        return {str(k): float(v) for k, v in payload.items()}

    summary_candidates = (
        ("best_test_acc", "test_acc", "final_test_acc")
        if summary_mode == "best"
        else ("final_test_acc", "test_acc", "best_test_acc")
    )

    extracted: Dict[str, float] = {}
    for dataset_name, value in payload.items():
        if not isinstance(value, dict):
            continue
        for key in summary_candidates:
            if key in value and isinstance(value[key], (int, float)):
                extracted[str(dataset_name)] = float(value[key])
                break

    if not extracted:
        raise ValueError(
            f"Could not extract dataset accuracy values from JSON '{path}'. "
            "Expected either {dataset: number} or {dataset: {final_test_acc/best_test_acc/...}}"
        )

    return extracted


def load_single_icl_metrics(path: Path, phase: str, summary_mode: str) -> Dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(f"ICL metric file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return load_single_icl_from_epoch_csv(path=path, phase=phase, summary_mode=summary_mode)
    if suffix == ".json":
        return load_single_icl_from_json(path=path, summary_mode=summary_mode)

    raise ValueError(
        f"Unsupported ICL file extension for '{path}'. "
        "Supported formats: .csv (epoch_metrics.csv), .json"
    )


def load_icl_metrics(
    metric_paths: Sequence[Path],
    phase: str,
    summary_mode: str,
) -> Dict[str, np.ndarray]:
    """Load one or more ICL metric files and aggregate per-dataset values."""
    merged: Dict[str, List[float]] = {}

    for path in metric_paths:
        per_file = load_single_icl_metrics(path=path, phase=phase, summary_mode=summary_mode)
        for dataset_name, score in per_file.items():
            merged.setdefault(dataset_name, []).append(float(score))

    if not merged:
        raise ValueError("No ICL accuracy values were extracted from the provided metric files")

    return {dataset_name: np.array(scores, dtype=float) for dataset_name, scores in merged.items()}


def collect_dataset_metrics(
    results: Mapping[str, Mapping],
    test_key: str,
    train_key: str | None = None,
) -> List[Tuple[str, np.ndarray | None, np.ndarray]]:
    collected: List[Tuple[str, np.ndarray | None, np.ndarray]] = []
    for dataset_name in sorted(results.keys()):
        dataset = results[dataset_name]
        runs = dataset.get("runs", [])
        if not isinstance(runs, list):
            continue

        test_values = [float(r[test_key]) for r in runs if isinstance(r, dict) and isinstance(r.get(test_key), (int, float))]

        if not test_values:
            continue

        train_values: np.ndarray | None = None
        if train_key is not None:
            train_values_list = [
                float(r[train_key])
                for r in runs
                if isinstance(r, dict) and isinstance(r.get(train_key), (int, float))
            ]
            if not train_values_list:
                continue
            train_values = np.array(train_values_list)

        collected.append((dataset_name, train_values, np.array(test_values)))

    if not collected:
        if train_key is None:
            raise ValueError(f"No datasets had test metric '{test_key}' available")
        raise ValueError(
            "No datasets had both metrics available. "
            f"Requested keys: train='{train_key}', test='{test_key}'"
        )

    return collected


# Tab-10 palette: test colors and lighter train counterparts
_TAB10_COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
    "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
]
_TAB10_LIGHT = [
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5",
    "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]


def make_grouped_bar_plot(
    named_dataset_metrics: List[Tuple[str, List[Tuple[str, np.ndarray | None, np.ndarray]]]],
    test_key: str,
    output_path: Path,
    train_key: str | None = None,
    icl_metrics: Mapping[str, np.ndarray] | None = None,
    icl_label: str = "ICL",
) -> None:
    include_train = train_key is not None
    include_icl = icl_metrics is not None and len(icl_metrics) > 0
    n_inputs = len(named_dataset_metrics)
    single = n_inputs == 1

    # Collect union of all dataset names; sort by mean test accuracy across all inputs.
    all_names: set[str] = set()
    for _, dm in named_dataset_metrics:
        for name, _, _ in dm:
            all_names.add(name)

    def _sort_key(name: str) -> float:
        vals: List[float] = []
        for _, dm in named_dataset_metrics:
            for n, _, tv in dm:
                if n == name:
                    vals.append(float(np.mean(tv)))
        return float(np.mean(vals)) if vals else 0.0

    names = sorted(all_names, key=_sort_key, reverse=True)
    x = np.arange(len(names), dtype=float)

    # Build series: (label, color, [samples_or_None per dataset position])
    series_definitions: List[Tuple[str, str, List[np.ndarray | None]]] = []
    for idx, (input_label, dm) in enumerate(named_dataset_metrics):
        train_map = {n: tv for n, tv, _ in dm if tv is not None}
        test_map = {n: tv for n, _, tv in dm}
        test_color = "tab:blue" if single else _TAB10_COLORS[idx % len(_TAB10_COLORS)]
        train_color = "tab:orange" if single else _TAB10_LIGHT[idx % len(_TAB10_LIGHT)]
        series_label_test = f"MLP {test_key}" if single else f"{input_label} test"
        series_label_train = f"MLP {train_key}" if single else f"{input_label} train"
        if include_train:
            series_definitions.append((
                series_label_train,
                train_color,
                [train_map.get(name) for name in names],
            ))
        series_definitions.append((
            series_label_test,
            test_color,
            [test_map.get(name) for name in names],
        ))
    if include_icl:
        assert icl_metrics is not None
        icl_color = "tab:green" if single else _TAB10_COLORS[n_inputs % len(_TAB10_COLORS)]
        series_definitions.append((
            icl_label,
            icl_color,
            [icl_metrics.get(name) for name in names],
        ))

    num_series = len(series_definitions)
    total_group_width = 0.8
    width = total_group_width / num_series
    offsets = (np.arange(num_series) - (num_series - 1) / 2.0) * width
    fig_width = max(10.0, 0.9 * len(names))

    fig, ax = plt.subplots(figsize=(fig_width, 6.0))

    jitter_rng = np.random.default_rng(42)
    global_max_mean = 0.0
    for series_idx, (label, color, samples_by_dataset) in enumerate(series_definitions):
        means = np.array([
            float(np.mean(samples)) if samples is not None and len(samples) > 0 else np.nan
            for samples in samples_by_dataset
        ])
        stds = np.array([
            float(np.std(samples)) if samples is not None and len(samples) > 0 else np.nan
            for samples in samples_by_dataset
        ])
        positions = x + offsets[series_idx]
        valid = np.isfinite(means)

        bar_container = ax.bar(
            positions[valid],
            means[valid],
            width * 0.95,
            yerr=stds[valid],
            capsize=2,
            label=label,
            color=color,
            alpha=0.9,
        )
        if np.any(valid):
            global_max_mean = max(global_max_mean, float(np.nanmax(means[valid])))
            value_labels = [
                np.format_float_positional(float(v), precision=6, trim="-")
                for v in means[valid]
            ]
            ax.bar_label(
                bar_container,
                labels=value_labels,
                padding=1,
                fontsize=6,
                rotation=90,
            )

        # Overlay run-level points with slight jitter to show dispersion.
        for i, samples in enumerate(samples_by_dataset):
            if samples is None or len(samples) == 0:
                continue
            jitter = jitter_rng.uniform(-width * 0.25, width * 0.25, size=len(samples))
            ax.scatter(
                np.full_like(samples, positions[i]) + jitter,
                samples,
                s=8,
                alpha=0.25,
                color="black",
            )

    title_parts: List[str] = ["MLP"]
    if single:
        title_parts.append("Train/Test" if include_train else "Test")
    else:
        title_parts.append(f"[{', '.join(lbl for lbl, _ in named_dataset_metrics)}]")
    if include_icl:
        title_parts.append(f"vs {icl_label}")
    ax.set_title("Multiclass Performance (" + ", ".join(title_parts) + ")")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Accuracy")
    y_max = min(1.2, max(1.05, global_max_mean + 0.1))
    ax.set_ylim(0.0, y_max)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def format_mean_std(values: np.ndarray | None) -> str:
    if values is None or len(values) == 0:
        return "n/a"
    return f"{float(np.mean(values)):.4f} +/- {float(np.std(values)):.4f}"


def save_accuracy_csv(
    named_dataset_metrics: List[Tuple[str, List[Tuple[str, np.ndarray | None, np.ndarray]]]],
    csv_path: Path,
    *,
    test_key: str,
    train_key: str | None = None,
    icl_metrics: Mapping[str, np.ndarray] | None = None,
    icl_label: str = "ICL",
) -> None:
    """Write per-dataset accuracy summary (mean +/- std) to a CSV file."""
    include_train = train_key is not None
    include_icl = icl_metrics is not None and len(icl_metrics) > 0
    n_inputs = len(named_dataset_metrics)
    single = n_inputs == 1

    all_names: set[str] = set()
    for _, dm in named_dataset_metrics:
        for name, _, _ in dm:
            all_names.add(name)

    def _sort_key(name: str) -> float:
        vals: List[float] = []
        for _, dm in named_dataset_metrics:
            for n, _, tv in dm:
                if n == name:
                    vals.append(float(np.mean(tv)))
        return float(np.mean(vals)) if vals else 0.0

    names = sorted(all_names, key=_sort_key, reverse=True)

    def _metric_headers(col: str) -> List[str]:
        return [f"{col}_mean", f"{col}_std"]

    def _metric_values(values: np.ndarray | None) -> List[str]:
        if values is None or len(values) == 0:
            return ["", ""]
        return [f"{float(np.mean(values)):.4f}", f"{float(np.std(values)):.4f}"]

    headers: List[str] = ["dataset"]
    for input_label, _ in named_dataset_metrics:
        if include_train:
            col = f"MLP {train_key}" if single else f"{input_label} train"
            headers.extend(_metric_headers(col))
        col = f"MLP {test_key}" if single else f"{input_label} test"
        headers.extend(_metric_headers(col))
    if include_icl:
        headers.extend(_metric_headers(icl_label))

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for name in names:
            row: List[str] = [name]
            for _, dm in named_dataset_metrics:
                train_map = {n: tv for n, tv, _ in dm if tv is not None}
                test_map = {n: tv for n, _, tv in dm}
                if include_train:
                    row.extend(_metric_values(train_map.get(name)))
                row.extend(_metric_values(test_map.get(name)))
            if include_icl:
                assert icl_metrics is not None
                row.extend(_metric_values(icl_metrics.get(name)))
            writer.writerow(row)


def print_accuracy_table(
    named_dataset_metrics: List[Tuple[str, List[Tuple[str, np.ndarray | None, np.ndarray]]]],
    *,
    test_key: str,
    train_key: str | None = None,
    icl_metrics: Mapping[str, np.ndarray] | None = None,
    icl_label: str = "ICL",
) -> None:
    """Print a compact table of per-dataset accuracy numbers (mean +/- std)."""
    include_train = train_key is not None
    include_icl = icl_metrics is not None and len(icl_metrics) > 0
    n_inputs = len(named_dataset_metrics)
    single = n_inputs == 1

    all_names: set[str] = set()
    for _, dm in named_dataset_metrics:
        for name, _, _ in dm:
            all_names.add(name)

    def _sort_key(name: str) -> float:
        vals: List[float] = []
        for _, dm in named_dataset_metrics:
            for n, _, tv in dm:
                if n == name:
                    vals.append(float(np.mean(tv)))
        return float(np.mean(vals)) if vals else 0.0

    names = sorted(all_names, key=_sort_key, reverse=True)

    headers: List[str] = ["dataset"]
    for input_label, _ in named_dataset_metrics:
        if include_train:
            headers.append(f"MLP {train_key}" if single else f"{input_label} train")
        headers.append(f"MLP {test_key}" if single else f"{input_label} test")
    if include_icl:
        headers.append(icl_label)

    rows: List[List[str]] = []
    for name in names:
        row: List[str] = [name]
        for _, dm in named_dataset_metrics:
            train_map = {n: tv for n, tv, _ in dm if tv is not None}
            test_map = {n: tv for n, _, tv in dm}
            if include_train:
                row.append(format_mean_std(train_map.get(name)))
            row.append(format_mean_std(test_map.get(name)))
        if include_icl:
            assert icl_metrics is not None
            row.append(format_mean_std(icl_metrics.get(name)))
        rows.append(row)

    widths = [len(h) for h in headers]
    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    print("\nAccuracy Summary (mean +/- std)")
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * widths[i] for i in range(len(headers))))
    for row in rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot train/test performance from multiclass JSON results")
    parser.add_argument(
        "--input",
        nargs="+",
        default=["mlp_baseline/results/results_multiclass.json"],
        help="One or more paths to results_multiclass.json files",
    )
    parser.add_argument(
        "--input-labels",
        nargs="*",
        default=None,
        help="Labels for each input file (defaults to filename stems)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to output PNG (default: based on first input path)",
    )
    parser.add_argument(
        "--include-train-acc",
        action="store_true",
        help="Include train accuracy in the plot",
    )
    parser.add_argument("--train-key", default=None, help="Run-level JSON key for train accuracy")
    parser.add_argument("--test-key", default=None, help="Run-level JSON key for test accuracy")
    parser.add_argument(
        "--icl-metrics",
        nargs="*",
        default=None,
        help="Optional ICL metric files for comparison (.csv epoch_metrics or .json summaries)",
    )
    parser.add_argument(
        "--icl-phase",
        default="train",
        help="Phase to use when reading ICL epoch_metrics.csv (default: train)",
    )
    parser.add_argument(
        "--icl-summary",
        choices=("final", "best"),
        default="final",
        help="How to summarize ICL epoch metrics per dataset (default: final)",
    )
    parser.add_argument(
        "--icl-label",
        default="ICL",
        help="Legend/column label for ICL comparison results",
    )
    args = parser.parse_args()

    input_paths = [Path(p) for p in args.input]
    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(f"Input file not found: {p}")

    if args.input_labels is not None:
        if len(args.input_labels) != len(input_paths):
            raise ValueError(
                f"--input-labels must have the same number of entries as --input "
                f"({len(args.input_labels)} labels for {len(input_paths)} files)"
            )
        input_labels = args.input_labels
    else:
        input_labels = [p.stem for p in input_paths]

    output_name_parts = ["test"]
    if args.include_train_acc:
        output_name_parts.insert(0, "train")
    if args.icl_metrics:
        output_name_parts.append("icl")
    default_output_name = "_".join(output_name_parts) + "_accuracy.png"
    output_path = Path(args.output) if args.output else input_paths[0].parent / default_output_name

    # Determine metric keys from the first input.
    first_results = load_results(input_paths[0])
    available_keys = collect_numeric_run_keys(first_results)
    test_key = resolve_metric_key(
        available_keys=available_keys,
        requested_key=args.test_key,
        candidates=TEST_KEY_CANDIDATES,
        metric_name="test",
    )

    train_key: str | None = None
    if args.include_train_acc:
        train_key = resolve_metric_key(
            available_keys=available_keys,
            requested_key=args.train_key,
            candidates=TRAIN_KEY_CANDIDATES,
            metric_name="train",
        )
    elif args.train_key is not None:
        print("[warn] --train-key was provided but --include-train-acc is disabled; train metric will be ignored")

    # Load all inputs.
    named_dataset_metrics: List[Tuple[str, List[Tuple[str, np.ndarray | None, np.ndarray]]]] = []
    for i, (path, label) in enumerate(zip(input_paths, input_labels)):
        results = first_results if i == 0 else load_results(path)
        dm = collect_dataset_metrics(results, test_key=test_key, train_key=train_key)
        named_dataset_metrics.append((label, dm))

    icl_metrics: Dict[str, np.ndarray] | None = None
    if args.icl_metrics:
        icl_paths = [Path(p) for p in args.icl_metrics]
        icl_metrics = load_icl_metrics(
            metric_paths=icl_paths,
            phase=args.icl_phase,
            summary_mode=args.icl_summary,
        )
        print(
            f"Loaded ICL metrics from {len(icl_paths)} file(s) "
            f"using phase='{args.icl_phase}' and summary='{args.icl_summary}'"
        )

    print_accuracy_table(
        named_dataset_metrics,
        test_key=test_key,
        train_key=train_key,
        icl_metrics=icl_metrics,
        icl_label=args.icl_label,
    )
    csv_path = output_path.with_suffix(".csv")
    save_accuracy_csv(
        named_dataset_metrics,
        csv_path,
        test_key=test_key,
        train_key=train_key,
        icl_metrics=icl_metrics,
        icl_label=args.icl_label,
    )
    make_grouped_bar_plot(
        named_dataset_metrics,
        test_key=test_key,
        output_path=output_path,
        train_key=train_key,
        icl_metrics=icl_metrics,
        icl_label=args.icl_label,
    )

    print(f"Saved plot to: {output_path}")
    print(f"Saved CSV to:  {csv_path}")
    if train_key is None:
        print(f"Using metric: test='{test_key}'")
    else:
        print(f"Using metrics: train='{train_key}', test='{test_key}'")
    if icl_metrics is not None:
        all_dm_names = {name for _, dm in named_dataset_metrics for name, _, _ in dm}
        overlap = sum(1 for name in all_dm_names if name in icl_metrics)
        print(f"ICL dataset overlap with baseline: {overlap}/{len(all_dm_names)}")


if __name__ == "__main__":
    main()
