"""Plot scored checkpoint evaluation results.

Example:
    python fourneurons/evaluation/score_all_checkpoints.py \
        --results_dir /scratch/results/math/competitionmath \
        --run_id 20260526-161659 \
        --split full
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import re
from pathlib import Path
from typing import Optional, Tuple, Union


CHECKPOINT_RE = re.compile(r"^(?P<run_id>.+)_(?P<checkpoint>checkpoint-(?P<step>\d+)|final)$")


def checkpoint_sort_key(row: dict) -> Tuple[int, Union[int, str]]:
    checkpoint = row["checkpoint"]
    if checkpoint == "final":
        return (1, 0)

    match = re.fullmatch(r"checkpoint-(\d+)", checkpoint)
    if match:
        return (0, int(match.group(1)))

    return (2, checkpoint)


def parse_result_dir(path: Path) -> Optional[dict]:
    """Return checkpoint metadata from a result directory name.

    Expected names are like:
        20260526-161659_checkpoint-500
        20260526-161659_final
    """
    match = CHECKPOINT_RE.fullmatch(path.name)
    if not match:
        return None

    step = match.group("step")
    return {
        "run_id": match.group("run_id"),
        "checkpoint": match.group("checkpoint"),
        "step": int(step) if step is not None else None,
        "path": path,
    }


def find_scored_results(results_dir: Path, run_id: Optional[str], split: str) -> list[dict]:
    rows = []
    scored_name = f"{split}_scored.json"

    for scored_path in sorted(results_dir.glob(f"*/{scored_name}")):
        parsed = parse_result_dir(scored_path.parent)
        if parsed is None:
            continue
        if run_id is not None and parsed["run_id"] != run_id:
            continue

        rows.append(read_scored_result(parsed, scored_path))

    return sorted(rows, key=checkpoint_sort_key)


def score_missing_results(
    results_dir: Path,
    run_id: Optional[str],
    split: str,
    benchmark: str,
    force_rescore: bool = False,
) -> int:
    scored = 0
    generations_name = f"{split}_gens.jsonl"

    for generations_path in sorted(results_dir.glob(f"*/{generations_name}")):
        parsed = parse_result_dir(generations_path.parent)
        if parsed is None:
            continue
        if run_id is not None and parsed["run_id"] != run_id:
            continue

        scored_path = generations_path.parent / f"{split}_scored.json"
        if scored_path.is_file() and not force_rescore:
            continue

        command = [
            sys.executable,
            "-m",
            "evaluate.score",
            "--generations",
            str(generations_path),
            "--benchmark",
            benchmark,
            "--output",
            str(scored_path),
        ]
        action = "Rescoring" if scored_path.is_file() else "Scoring missing"
        print(f"{action} result: {generations_path}", flush=True)
        print(" ".join(command), flush=True)
        subprocess.run(command, check=True)
        scored += 1

    return scored


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
            boxed += int(completion.get("extracted") is not None)

    if total == 0:
        return None
    return boxed / total


def read_scored_result(row: dict, scored_path: Path) -> dict:
    with open(scored_path, "r", encoding="utf-8") as f:
        scored = json.load(f)

    metrics = scored.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError(f"Missing metrics object in {scored_path}")

    box_compliance = metrics.get("box_compliance")
    if box_compliance is None:
        box_compliance = compute_box_compliance(scored)

    return {
        **row,
        "scored_path": scored_path,
        "n_problems": scored.get("n_problems"),
        "n_completions": scored.get("n_completions"),
        "pass@1": metrics.get("pass@1"),
        "pass@8": metrics.get("pass@8"),
        "box_compliance": box_compliance,
    }


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "run_id",
        "checkpoint",
        "step",
        "n_problems",
        "n_completions",
        "pass@1",
        "pass@8",
        "box_compliance",
        "scored_path",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def plot_metrics(rows: list[dict], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    x = list(range(len(rows)))
    labels = [row["checkpoint"].replace("checkpoint-", "ckpt-") for row in rows]

    fig_width = max(8, min(18, len(rows) * 0.8))
    _, ax = plt.subplots(figsize=(fig_width, 5))

    for metric, marker in (
        ("pass@1", "o"),
        ("pass@8", "s"),
        ("box_compliance", "^"),
    ):
        points = [row.get(metric) for row in rows]
        if any(point is not None for point in points):
            ax.plot(
                x,
                [point * 100 if point is not None else None for point in points],
                marker=marker,
                linewidth=2,
                label=metric,
            )

    ax.set_title("Checkpoint Evaluation")
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(output_path, dpi=160)
    plt.close(ax.figure)


def default_output_prefix(results_dir: Path, run_id: Optional[str], split: str) -> Path:
    name = f"{run_id}_{split}_checkpoint_metrics" if run_id else f"{split}_checkpoint_metrics"
    return results_dir / name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read existing scored JSON files and plot pass@1/pass@8/box_compliance "
            "across checkpoints. "
            "Result directories should be named like <run_id>_checkpoint-500 or <run_id>_final."
        )
    )
    parser.add_argument(
        "results_dir",
        nargs="?",
        type=Path,
        help="Directory containing checkpoint result directories with <split>_scored.json files.",
    )
    parser.add_argument(
        "--results_dir",
        dest="results_dir_option",
        type=Path,
        default=None,
        help="Directory containing checkpoint result directories with <split>_scored.json files.",
    )
    parser.add_argument("--run_id", default=None, help="Only include this training run id.")
    parser.add_argument("--split", default="val")
    parser.add_argument("--benchmark", default="math", help="Benchmark to pass to evaluate.score.")
    parser.add_argument(
        "--score_missing",
        action="store_true",
        help="Create missing <split>_scored.json files from existing <split>_gens.jsonl files.",
    )
    parser.add_argument(
        "--force_rescore",
        action="store_true",
        help="Recreate <split>_scored.json files from existing <split>_gens.jsonl files, overwriting existing scores.",
    )
    parser.add_argument(
        "--output_prefix",
        type=Path,
        default=None,
        help="Output prefix for .csv and .png files.",
    )
    args = parser.parse_args()
    if args.results_dir is None and args.results_dir_option is None:
        parser.error("results_dir is required, either positionally or with --results_dir.")
    if args.results_dir is not None and args.results_dir_option is not None:
        parser.error("pass results_dir either positionally or with --results_dir, not both.")
    if args.results_dir_option is not None:
        args.results_dir = args.results_dir_option
    del args.results_dir_option
    return args


def main() -> None:
    args = parse_args()
    if not args.results_dir.is_dir():
        raise NotADirectoryError(f"Results directory not found: {args.results_dir}")

    if args.score_missing or args.force_rescore:
        scored_count = score_missing_results(
            args.results_dir,
            args.run_id,
            args.split,
            args.benchmark,
            force_rescore=args.force_rescore,
        )
        action = "Recreated" if args.force_rescore else "Created"
        print(f"{action} {scored_count} scored file(s).")

    rows = find_scored_results(args.results_dir, args.run_id, args.split)
    if not rows:
        run_msg = f" for run_id={args.run_id}" if args.run_id else ""
        raise FileNotFoundError(
            f"No {args.split}_scored.json files found in checkpoint result directories "
            f"under {args.results_dir}{run_msg}"
        )

    output_prefix = args.output_prefix or default_output_prefix(
        args.results_dir, args.run_id, args.split
    )
    csv_path = output_prefix.with_suffix(".csv")
    plot_path = output_prefix.with_suffix(".png")

    write_csv(rows, csv_path)
    plot_metrics(rows, plot_path)

    print(f"Read {len(rows)} scored checkpoint result files.")
    print(f"Wrote metrics CSV: {csv_path}")
    print(f"Wrote comparison plot: {plot_path}")


if __name__ == "__main__":
    main()
