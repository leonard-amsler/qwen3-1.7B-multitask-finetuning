"""Compare scored evaluation results by problem category.

Example:
    python -m fourneurons.evaluation.compare_categories \
        /scratch/results/math/competitionmath/run_a \
        /scratch/results/math/competitionmath/run_b \
        --split val \
        --output_dir /scratch/results/math/competitionmath/compare_categories
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

from evaluate.pass_at_k import compute_pass_at_k_for_dataset


DEFAULT_CATEGORY_FIELDS = ("category", "type", "level")


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


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
    return rows


def read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_category_field(rows: Iterable[dict], requested: Optional[str]) -> str:
    if requested is not None:
        return requested

    seen_rows = list(rows)
    for field in DEFAULT_CATEGORY_FIELDS:
        if any(row.get(field) not in (None, "") for row in seen_rows):
            return field

    fields = ", ".join(DEFAULT_CATEGORY_FIELDS)
    raise ValueError(
        f"Could not find a category field in generations rows. "
        f"Tried: {fields}. Pass --category_field explicitly."
    )


def get_boxed_count(detail: dict) -> Optional[int]:
    if isinstance(detail.get("boxed"), int):
        return detail["boxed"]

    completions = detail.get("completions")
    if not isinstance(completions, list):
        return None

    boxed = 0
    total = 0
    for completion in completions:
        if not isinstance(completion, dict):
            continue
        total += 1
        if "boxed" in completion:
            boxed += int(bool(completion["boxed"]))
        else:
            boxed += int(completion.get("extracted") is not None)

    return boxed if total else None


def category_rows_for_result_dir(
    result_dir: Path,
    split: str,
    category_field: Optional[str],
    label: Optional[str],
    run_index: int,
) -> list[dict]:
    gens_path = result_dir / f"{split}_gens.jsonl"
    scored_path = result_dir / f"{split}_scored.json"

    if not gens_path.is_file():
        raise FileNotFoundError(f"Missing generations file: {gens_path}")
    if not scored_path.is_file():
        raise FileNotFoundError(f"Missing scored file: {scored_path}")

    gens_rows = read_jsonl(gens_path)
    scored = read_json(scored_path)
    details = scored.get("detailed_results")
    if not isinstance(details, list):
        raise ValueError(f"Missing detailed_results list in {scored_path}")
    if len(gens_rows) != len(details):
        raise ValueError(
            f"Mismatched row counts for {result_dir}: "
            f"{len(gens_rows)} generation rows, {len(details)} scored rows."
        )

    category_field = detect_category_field(gens_rows, category_field)
    run_label = label or result_dir.name
    by_category: dict[str, list[tuple[dict, dict]]] = defaultdict(list)

    for gen_row, detail in zip(gens_rows, details):
        category = gen_row.get(category_field)
        if category in (None, ""):
            category = "uncategorized"
        by_category[str(category)].append((gen_row, detail))

    rows = []
    for category, pairs in sorted(by_category.items()):
        n_values = {int(detail["n"]) for _, detail in pairs if "n" in detail}
        if not n_values:
            raise ValueError(f"No per-problem completion count found in {scored_path}")
        if len(n_values) != 1:
            raise ValueError(
                f"Category {category!r} in {scored_path} has mixed completion counts: "
                f"{sorted(n_values)}"
            )
        n_completions = n_values.pop()
        correct_counts = [int(detail.get("c", 0)) for _, detail in pairs]
        k_values = [k for k in (1, 8) if k <= n_completions]
        metrics = compute_pass_at_k_for_dataset(correct_counts, n_completions, k_values)

        boxed_counts = [get_boxed_count(detail) for _, detail in pairs]
        if all(boxed is not None for boxed in boxed_counts):
            metrics["box_compliance"] = sum(boxed_counts) / (len(boxed_counts) * n_completions)

        rows.append(
            {
                "run": run_label,
                "run_index": run_index,
                "result_dir": str(result_dir),
                "category_field": category_field,
                "category": category,
                "n_problems": len(pairs),
                "n_completions": n_completions,
                "pass@1": metrics.get("pass@1"),
                "pass@8": metrics.get("pass@8"),
                "box_compliance": metrics.get("box_compliance"),
            }
        )

    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "run",
        "category",
        "category_field",
        "n_problems",
        "n_completions",
        "pass@1",
        "pass@8",
        "box_compliance",
        "result_dir",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def safe_filename(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9._-]+", "_", text)
    return text.strip("_") or "category"


def plot_category(rows: list[dict], category: str, output_path: Path) -> None:
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

    ax.set_title(category)
    ax.set_xlabel("Result")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    ax.set_ylim(*percent_axis_bounds(axis_values, 20.0))
    ax.figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(output_path, dpi=160)
    plt.close(ax.figure)


def plot_all_categories(rows: list[dict], output_dir: Path) -> list[Path]:
    by_category: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)

    paths = []
    for category, category_rows in sorted(by_category.items()):
        path = output_dir / f"{safe_filename(category)}.png"
        plot_category(category_rows, category, path)
        paths.append(path)
    return paths


def category_display_name(category: str) -> str:
    labels = {
        "Counting & Probability": "Count. & Prob.",
        "Intermediate Algebra": "Interm. Alg.",
        "Number Theory": "Number Theory",
    }
    return labels.get(category, category)


def plot_category_metric_bars(rows: list[dict], output_path: Path) -> Path:
    import matplotlib.pyplot as plt

    runs = sorted({(row["run_index"], row["run"]) for row in rows})
    categories = sorted({row["category"] for row in rows})
    lookup = {(row["category"], row["run_index"]): row for row in rows}

    group_width = 0.82
    bar_width = group_width / max(len(runs), 1) * 0.92
    centers = list(range(len(categories)))
    _, ax = plt.subplots(figsize=(7.2, 3.4))

    for run_pos, (run_index, run_label) in enumerate(runs):
        xs = [
            center - group_width / 2 + bar_width / 2 + run_pos * bar_width
            for center in centers
        ]
        values = []
        for category in categories:
            value = lookup.get((category, run_index), {}).get("pass@8")
            values.append(value * 100 if value is not None else float("nan"))
        ax.bar(
            xs,
            values,
            width=bar_width,
            color=PASTEL_COLORS[run_pos % len(PASTEL_COLORS)],
            edgecolor="white",
            linewidth=0.8,
            label=run_label,
        )

    ax.set_title("pass@8 by Category")
    ax.set_ylabel("pass@8 (%)")
    ax.set_xticks(centers)
    ax.set_xticklabels(
        [category_display_name(category) for category in categories],
        rotation=30,
        ha="right",
    )
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=min(3, max(len(runs), 1)), fontsize=8)
    ax.set_ylim(*percent_axis_bounds([lookup.get((category, run_index), {}).get("pass@8") for category in categories for run_index, _ in runs], 20.0))
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare existing scored result folders by problem category. Each result "
            "folder must contain <split>_gens.jsonl and <split>_scored.json."
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
            "Directory where category_metrics.csv and category plots are written. "
            "Defaults to compare_categories under the common parent."
        ),
    )
    parser.add_argument(
        "--category_field",
        default=None,
        help="Generation-row field to group by. Defaults to auto-detecting category, type, or level.",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional display labels, one per result directory.",
    )
    return parser.parse_args()


def default_output_dir(result_dirs: list[Path]) -> Path:
    parents = {path.resolve().parent for path in result_dirs}
    if len(parents) == 1:
        return next(iter(parents)) / "compare_categories"
    return Path("compare_categories")


def main() -> None:
    args = parse_args()
    result_dirs = [path.resolve() for path in args.result_dirs]
    for result_dir in result_dirs:
        if not result_dir.is_dir():
            raise NotADirectoryError(f"Result directory not found: {result_dir}")

    labels = parse_labels(args.labels, result_dirs)
    rows = []
    for run_index, (result_dir, label) in enumerate(zip(result_dirs, labels)):
        rows.extend(
            category_rows_for_result_dir(
                result_dir=result_dir,
                split=args.split,
                category_field=args.category_field,
                label=label,
                run_index=run_index,
            )
        )

    output_dir = args.output_dir or default_output_dir(result_dirs)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "category_metrics.csv"
    write_csv(rows, csv_path)
    plot_paths = plot_all_categories(rows, output_dir)
    summary_plot_path = output_dir / "all_categories_metrics.png"
    plot_category_metric_bars(rows, summary_plot_path)

    print(f"Read {len(result_dirs)} result directories.")
    print(f"Wrote category metrics CSV: {csv_path}")
    print(f"Wrote {len(plot_paths)} category plots to: {output_dir}")
    print(f"Wrote category summary plot: {summary_plot_path}")


if __name__ == "__main__":
    main()
