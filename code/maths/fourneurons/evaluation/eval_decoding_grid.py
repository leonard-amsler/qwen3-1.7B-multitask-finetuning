"""Run a small decoding grid for one checkpoint.

Example:
    python -m fourneurons.evaluation.eval_decoding_grid \
        /scratch/checkpoints/math/20260526-161659/checkpoint-12500 \
        --benchmark math \
        --dataset competitionmath \
        --split full \
        --max_num_samples 1000
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_TEMPERATURES = (0.5, 0.6, 0.7)
DEFAULT_TOP_PS = (0.8, 0.9, 0.95)


def float_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def checkpoint_tag(checkpoint: Path) -> str:
    if checkpoint.parent.name:
        return f"{checkpoint.parent.name}_{checkpoint.name}"
    return checkpoint.name


def result_root(args: argparse.Namespace) -> Path:
    return Path("/scratch/results") / args.benchmark / args.dataset


def run_name_for(args: argparse.Namespace, temperature: float, top_p: float) -> str:
    return (
        f"{args.output_prefix}{checkpoint_tag(args.checkpoint)}"
        f"_temp{float_tag(temperature)}"
        f"_top_p{float_tag(top_p)}"
        f"_top_k{args.top_k}"
    )


def generations_path(args: argparse.Namespace, run_name: str) -> Path:
    return result_root(args) / run_name / f"{args.split}_gens.jsonl"


def scored_path(args: argparse.Namespace, run_name: str) -> Path:
    return result_root(args) / run_name / f"{args.split}_scored.json"


def merged_model_dir(args: argparse.Namespace) -> Path:
    return result_root(args) / f"{args.output_prefix}{checkpoint_tag(args.checkpoint)}_merged"


def build_eval_command(
    args: argparse.Namespace,
    run_name: str,
    temperature: float,
    top_p: float,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "fourneurons.evaluation.eval",
        args.benchmark,
        args.dataset,
        args.split,
        run_name,
        "--checkpoint",
        str(args.checkpoint),
        "--num_generations",
        str(args.num_generations),
        "--max_tokens",
        str(args.max_tokens),
        "--temperature",
        str(temperature),
        "--top_p",
        str(top_p),
        "--top_k",
        str(args.top_k),
        "--merged_model_dir",
        str(merged_model_dir(args)),
    ]
    if args.prompt_file_path:
        command.extend(["--prompt_file_path", args.prompt_file_path])
    if args.max_num_samples is not None:
        command.extend(["--max_num_samples", str(args.max_num_samples)])
    return command


def build_score_command(args: argparse.Namespace, run_name: str) -> list[str]:
    return [
        sys.executable,
        "-m",
        "evaluate.score",
        "--generations",
        str(generations_path(args, run_name)),
        "--benchmark",
        args.benchmark,
        "--output",
        str(scored_path(args, run_name)),
    ]


def read_metrics(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        scored = json.load(f)
    metrics = scored.get("metrics", {})
    return {
        "n_problems": scored.get("n_problems"),
        "n_completions": scored.get("n_completions"),
        "pass@1": metrics.get("pass@1"),
        "pass@8": metrics.get("pass@8"),
        "box_compliance": metrics.get("box_compliance"),
    }


def write_summary(args: argparse.Namespace, rows: list[dict]) -> Path:
    output_path = result_root(args) / f"{args.output_prefix}{checkpoint_tag(args.checkpoint)}_decoding_grid.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "temperature",
        "top_p",
        "top_k",
        "n_problems",
        "n_completions",
        "pass@1",
        "pass@8",
        "box_compliance",
        "run_name",
        "scored_path",
    ]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})
    return output_path


def parse_float_list(raw: str) -> list[float]:
    values = [float(item.strip()) for item in raw.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("list must contain at least one value")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run temperature/top_p decoding sweeps for one LoRA checkpoint, "
            "using top_k=20 and n=8 by default."
        )
    )
    parser.add_argument("checkpoint", type=Path, help="LoRA checkpoint directory to evaluate.")
    parser.add_argument("--benchmark", default="math")
    parser.add_argument("--dataset", default="competitionmath")
    parser.add_argument("--split", default="full")
    parser.add_argument("--prompt_file_path", default="fourneurons/prompts/math.txt")
    parser.add_argument("--temperatures", type=parse_float_list, default=list(DEFAULT_TEMPERATURES))
    parser.add_argument("--top_ps", type=parse_float_list, default=list(DEFAULT_TOP_PS))
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument(
        "--max_num_samples",
        type=int,
        default=None,
        help="Maximum number of samples per grid point. Omit to evaluate the whole split.",
    )
    parser.add_argument(
        "--output_prefix",
        default="decodegrid_",
        help="Prefix for result directories and the summary CSV.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip generation for grid points that already have <split>_gens.jsonl.",
    )
    parser.add_argument(
        "--skip_scored",
        action="store_true",
        help="Skip scoring for grid points that already have <split>_scored.json.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running them.")
    args = parser.parse_args()

    if not args.checkpoint.is_dir():
        parser.error(f"checkpoint directory not found: {args.checkpoint}")
    if args.output_prefix and "/" in args.output_prefix:
        parser.error("--output_prefix must not contain path separators")
    if args.top_k <= 0:
        parser.error("--top_k must be positive")
    if args.num_generations <= 0:
        parser.error("--num_generations must be positive")
    if args.max_tokens <= 0:
        parser.error("--max_tokens must be positive")
    if args.max_num_samples is not None and args.max_num_samples <= 0:
        parser.error("--max_num_samples must be positive when provided")
    for temperature in args.temperatures:
        if temperature <= 0:
            parser.error("--temperatures values must be positive")
    for top_p in args.top_ps:
        if not (0 < top_p <= 1):
            parser.error("--top_ps values must be in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    rows = []
    print(f"Checkpoint: {args.checkpoint}", flush=True)
    print(f"Shared merged model dir: {merged_model_dir(args)}", flush=True)

    for temperature in args.temperatures:
        for top_p in args.top_ps:
            run_name = run_name_for(args, temperature, top_p)
            gen_path = generations_path(args, run_name)
            score_path = scored_path(args, run_name)

            if args.skip_existing and gen_path.is_file():
                print(f"Skipping existing generations: {gen_path}", flush=True)
            else:
                eval_command = build_eval_command(args, run_name, temperature, top_p)
                print(f"Evaluating {run_name}", flush=True)
                print(" ".join(eval_command), flush=True)
                if not args.dry_run:
                    subprocess.run(eval_command, check=True)

            if args.skip_scored and score_path.is_file():
                print(f"Skipping existing score: {score_path}", flush=True)
            else:
                if not args.dry_run and not gen_path.is_file():
                    raise FileNotFoundError(f"Cannot score missing generations file: {gen_path}")
                score_command = build_score_command(args, run_name)
                print(f"Scoring {run_name}", flush=True)
                print(" ".join(score_command), flush=True)
                if not args.dry_run:
                    subprocess.run(score_command, check=True)

            row = {
                "temperature": temperature,
                "top_p": top_p,
                "top_k": args.top_k,
                "run_name": run_name,
                "scored_path": score_path,
            }
            if score_path.is_file():
                row.update(read_metrics(score_path))
            rows.append(row)

    if args.dry_run:
        summary_path = result_root(args) / f"{args.output_prefix}{checkpoint_tag(args.checkpoint)}_decoding_grid.csv"
        print(f"Dry run complete. Grid summary would be: {summary_path}", flush=True)
        return

    summary_path = write_summary(args, rows)
    print(f"Wrote grid summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
