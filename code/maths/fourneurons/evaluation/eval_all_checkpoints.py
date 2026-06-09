"""Run generation evaluation for every checkpoint in a training run.

Example:
    python -m fourneurons.evaluation.eval_all_checkpoints \
        /scratch/checkpoints/math/20260526-161659 \
        --benchmark math \
        --dataset competitionmath \
        --split full \
        --max_tokens 4096 \
        --max_num_samples 1000
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple, Union


CHECKPOINT_RE = re.compile(r"checkpoint-(\d+)")


def checkpoint_sort_key(path: Path) -> Tuple[int, Union[int, str]]:
    if path.name == "final":
        return (1, 0)

    match = CHECKPOINT_RE.fullmatch(path.name)
    if match:
        return (0, int(match.group(1)))

    return (2, path.name)


def find_checkpoints(run_dir: Path, include_final: bool = True) -> list[Path]:
    checkpoints = [
        path
        for path in run_dir.iterdir()
        if path.is_dir() and CHECKPOINT_RE.fullmatch(path.name)
    ]

    final_dir = run_dir / "final"
    if include_final and final_dir.is_dir():
        checkpoints.append(final_dir)

    return sorted(checkpoints, key=checkpoint_sort_key)


def output_dir_for(args: argparse.Namespace, checkpoint: Path) -> Path:
    run_id = args.run_dir.name
    run_name = f"{args.output_prefix}{run_id}_{checkpoint.name}"
    return Path("/scratch/results") / args.benchmark / args.dataset / run_name


def generations_path_for(args: argparse.Namespace, checkpoint: Path) -> Path:
    return output_dir_for(args, checkpoint) / f"{args.split}_gens.jsonl"


def scored_path_for(args: argparse.Namespace, checkpoint: Path) -> Path:
    return output_dir_for(args, checkpoint) / f"{args.split}_scored.json"


def build_eval_command(args: argparse.Namespace, checkpoint: Path) -> list[str]:
    output_dir = output_dir_for(args, checkpoint)
    run_name = output_dir.name

    command = [
        sys.executable,
        "-m",
        "fourneurons.evaluation.eval",
        args.benchmark,
        args.dataset,
        args.split,
        run_name,
        "--checkpoint",
        str(checkpoint),
        "--num_generations",
        str(args.num_generations),
        "--max_tokens",
        str(args.max_tokens),
    ]

    if args.prompt_file_path:
        command.extend(["--prompt_file_path", args.prompt_file_path])
    if args.max_num_samples is not None:
        command.extend(["--max_num_samples", str(args.max_num_samples)])

    return command


def build_score_command(args: argparse.Namespace, checkpoint: Path) -> list[str]:
    return [
        sys.executable,
        "-m",
        "evaluate.score",
        "--generations",
        str(generations_path_for(args, checkpoint)),
        "--benchmark",
        args.benchmark,
        "--output",
        str(scored_path_for(args, checkpoint)),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fourneurons.evaluation.eval on every checkpoint directory in a run."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Training run directory, e.g. /scratch/checkpoints/math/20260526-161659.",
    )
    parser.add_argument("--benchmark", default="math")
    parser.add_argument("--dataset", default="competitionmath")
    parser.add_argument("--split", default="full")
    parser.add_argument("--prompt_file_path", default="fourneurons/prompts/math.txt")
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument(
        "--max_num_samples",
        type=int,
        default=None,
        help="Maximum number of samples per checkpoint. Omit to evaluate the whole split.",
    )
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument(
        "--output_prefix",
        default="",
        help=(
            "Optional prefix prepended to each result directory name, e.g. "
            "'gen16_tok8192_'."
        ),
    )
    parser.add_argument(
        "--no_final",
        action="store_true",
        help="Skip the final adapter directory if it exists.",
    )
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip generation for checkpoints that already have <split>_gens.jsonl.",
    )
    parser.add_argument(
        "--skip_scored",
        action="store_true",
        help="Skip scoring for checkpoints that already have <split>_scored.json.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print the commands without running them.",
    )
    args = parser.parse_args()

    if args.max_tokens <= 0:
        parser.error("--max_tokens must be positive")
    if args.num_generations <= 0:
        parser.error("--num_generations must be positive")
    if args.max_num_samples is not None and args.max_num_samples <= 0:
        parser.error("--max_num_samples must be positive when provided")
    if args.output_prefix and "/" in args.output_prefix:
        parser.error("--output_prefix must not contain path separators")

    return args


def main() -> None:
    args = parse_args()
    if not args.run_dir.is_dir():
        raise NotADirectoryError(f"Run directory not found: {args.run_dir}")

    checkpoints = find_checkpoints(args.run_dir, include_final=not args.no_final)
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint-* directories found in {args.run_dir}"
            + (" and no final directory found" if not args.no_final else "")
        )

    print(f"Found {len(checkpoints)} checkpoint(s) in {args.run_dir}", flush=True)
    for checkpoint in checkpoints:
        generations_path = generations_path_for(args, checkpoint)
        scored_path = scored_path_for(args, checkpoint)

        should_run_generation = not (args.skip_existing and generations_path.is_file())
        if should_run_generation:
            command = build_eval_command(args, checkpoint)
            print(f"Evaluating checkpoint: {checkpoint}", flush=True)
            print(" ".join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, check=True)
        else:
            print(f"Skipping existing generations: {generations_path}", flush=True)

        should_run_scoring = not (args.skip_scored and scored_path.is_file())
        if should_run_scoring:
            if not args.dry_run and not generations_path.is_file():
                raise FileNotFoundError(
                    f"Cannot score missing generations file: {generations_path}"
                )

            score_command = build_score_command(args, checkpoint)
            print(f"Scoring checkpoint: {checkpoint}", flush=True)
            print(" ".join(score_command), flush=True)
            if not args.dry_run:
                subprocess.run(score_command, check=True)
        else:
            print(f"Skipping existing scored output: {scored_path}", flush=True)


if __name__ == "__main__":
    main()
