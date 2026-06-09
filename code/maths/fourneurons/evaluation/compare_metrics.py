"""Compare aggregate scoring metrics from a list of result directories.

Example:
    python -m fourneurons.evaluation.compare_metrics \
        /scratch/results/math/competitionmath/run_a \
        /scratch/results/math/competitionmath/run_b \
        --split full \
        --output_dir /scratch/results/math/competitionmath/compare_metrics
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional


METRICS = ("pass@1", "pass@8", "box_compliance")


PASTEL_COLORS = (
    "#8fbdd3",
    "#f0b98d",
    "#a8d8b8",
    "#c7b7dd",
    "#efaaa9",
    "#d7cc8f",
)
METRIC_COLORS = {
    "pass@1": "#8fbdd3",
    "pass@8": "#f0b98d",
    "box_compliance": "#a8d8b8",
}


def percent_axis_bounds(values, fallback_ymin: float) -> tuple[float, float]:
    percent_values = [value * 100 for value in values if value is not None]
    if not percent_values:
        return fallback_ymin, 100.0
    ymin = max(0.0, (min(percent_values) // 10) * 10)
    if ymin >= 100.0:
        ymin = 90.0
    return ymin, 100.0


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_box_compliance(scored: dict) -> Optional[float]:
    detailed_results = scored.get("detailed_results")
    if not isinstance(detailed_results, list):
        return None

    boxed = 0
    total = 0
    for result in detailed_results:
        if not isinstance(result, dict):
            continue
        completions = result.get("completions")
        if not isinstance(completions, list):
            continue
        for completion in completions:
            if not isinstance(completion, dict):
                continue
            total += 1
            if "boxed" in completion:
                boxed += int(bool(completion["boxed"]))
            else:
                boxed += int(completion.get("extracted") is not None)

    if total == 0:
        return None
    return boxed / total


def metrics_for_result_dir(
    result_dir: Path,
    split: str,
    label: Optional[str],
    run_index: int,
) -> dict:
    scored_path = result_dir / f"{split}_scored.json"
    if not scored_path.is_file():
        raise FileNotFoundError(f"Missing scored file: {scored_path}")

    scored = read_json(scored_path)
    metrics = scored.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Missing metrics object in {scored_path}")

    box_compliance = metrics.get("box_compliance")
    if box_compliance is None:
        box_compliance = compute_box_compliance(scored)

    return {
        "run": label or result_dir.name,
        "run_index": run_index,
        "result_dir": str(result_dir),
        "scored_path": str(scored_path),
        "n_problems": scored.get("n_problems"),
        "n_completions": scored.get("n_completions"),
        "pass@1": metrics.get("pass@1"),
        "pass@8": metrics.get("pass@8"),
        "box_compliance": box_compliance,
    }


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "run",
        "n_problems",
        "n_completions",
        "pass@1",
        "pass@8",
        "box_compliance",
        "scored_path",
        "result_dir",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: item["run_index"]):
            writer.writerow({column: row.get(column) for column in columns})


def plot_metrics(rows: list[dict], output_path: Path, ymin: float, ymax: float) -> None:
    import matplotlib.pyplot as plt

    ordered = sorted(rows, key=lambda row: row["run_index"])
    labels = [row["run"] for row in ordered]
    x = list(range(len(ordered)))

    _, ax = plt.subplots(figsize=(5.0, 3.2))

    axis_values = []
    for metric, marker in (
        ("pass@1", "o"),
        ("pass@8", "s"),
        ("box_compliance", "^"),
    ):
        values = [row.get(metric) for row in ordered]
        axis_values.extend(values)
        if any(value is not None for value in values):
            ax.plot(
                x,
                [value * 100 if value is not None else None for value in values],
                marker=marker,
                linewidth=2,
                color=METRIC_COLORS.get(metric),
                label=metric,
            )

    ax.set_title("Evaluation Metrics")
    ax.set_xlabel("Result")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    ax.set_ylim(*percent_axis_bounds(axis_values, ymin))
    ax.figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(output_path, dpi=160)
    plt.close(ax.figure)


def plot_metric_bars(rows: list[dict], output_path: Path, ymin: float, ymax: float) -> Path:
    import matplotlib.pyplot as plt

    ordered = sorted(rows, key=lambda row: row["run_index"])
    labels = [row["run"] for row in ordered]
    x = list(range(len(ordered)))
    values = [
        row.get("pass@8") * 100 if row.get("pass@8") is not None else float("nan")
        for row in ordered
    ]

    _, ax = plt.subplots(figsize=(4.8, 3.2))
    ax.bar(
        x,
        values,
        width=0.62,
        color=[PASTEL_COLORS[i % len(PASTEL_COLORS)] for i in range(len(ordered))],
        edgecolor="white",
        linewidth=0.8,
    )

    ax.set_title("pass@8 by Run")
    ax.set_ylabel("pass@8 (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(*percent_axis_bounds([row.get("pass@8") for row in ordered], ymin))
    ax.figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(output_path, dpi=160)
    plt.close(ax.figure)
    return output_path


def parse_labels(labels: Optional[list[str]], result_dirs: list[Path]) -> list[Optional[str]]:
    if labels is None:
        return [None] * len(result_dirs)
    if len(labels) != len(result_dirs):
        raise ValueError(
            f"Expected {len(result_dirs)} labels for {len(result_dirs)} result dirs, "
            f"got {len(labels)}."
        )
    return labels


def default_output_dir(result_dirs: list[Path]) -> Path:
    parents = {path.resolve().parent for path in result_dirs}
    if len(parents) == 1:
        return next(iter(parents)) / "compare_metrics"
    return Path("compare_metrics")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare pass@1, pass@8, and box_compliance from existing scored "
            "result folders. Each folder must contain <split>_scored.json."
        )
    )
    parser.add_argument(
        "result_dirs",
        nargs="+",
        type=Path,
        help="Result directories to compare.",
    )
    parser.add_argument("--split", default="val", help="Dataset split name.")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=None,
        help=(
            "Directory where metrics.csv and metrics.png are written. Defaults "
            "to compare_metrics under the common parent."
        ),
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional display labels, one per result directory.",
    )
    parser.add_argument(
        "--ymin",
        type=float,
        default=20.0,
        help="Minimum y-axis value in percent.",
    )
    parser.add_argument(
        "--ymax",
        type=float,
        default=100.0,
        help="Maximum y-axis value in percent.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result_dirs = [path.resolve() for path in args.result_dirs]
    for result_dir in result_dirs:
        if not result_dir.is_dir():
            raise NotADirectoryError(f"Result directory not found: {result_dir}")

    labels = parse_labels(args.labels, result_dirs)
    rows = [
        metrics_for_result_dir(
            result_dir=result_dir,
            split=args.split,
            label=label,
            run_index=run_index,
        )
        for run_index, (result_dir, label) in enumerate(zip(result_dirs, labels))
    ]

    output_dir = args.output_dir or default_output_dir(result_dirs)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "metrics.csv"
    plot_path = output_dir / "metrics.png"
    bar_plot_path = output_dir / "metrics_bars.png"

    write_csv(rows, csv_path)
    plot_metrics(rows, plot_path, ymin=args.ymin, ymax=args.ymax)
    plot_metric_bars(rows, bar_plot_path, ymin=args.ymin, ymax=args.ymax)

    print(f"Read {len(rows)} scored result files.")
    print(f"Wrote metrics CSV: {csv_path}")
    print(f"Wrote metrics plot: {plot_path}")
    print(f"Wrote metrics bar plot: {bar_plot_path}")


if __name__ == "__main__":
    main()
