"""Generate 8 completions for the RL prompt pool and score them.

This is phase 2 of docs/math_rl.md: use the best current math SFT
checkpoint to pre-score candidate RL prompts with the same boxed-answer scorer as
the benchmark.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


BEST_MATH500_16K_CHECKPOINT = Path(
    "/scratch/checkpoints/math/qwen3-1.7b-lora-math-mixed_20260526-220009/checkpoint-4458"
)
DEFAULT_RUN_NAME = "rl_pool_prescore_mixed_ckpt4458_tok16k_n8"
DEFAULT_PROMPT_FILE = Path("fourneurons/prompts/math.txt")


def generations_path(benchmark: str, dataset: str, run_name: str, split: str) -> Path:
    return Path("/scratch/results") / benchmark / dataset / run_name / f"{split}_gens.jsonl"


def scored_path(benchmark: str, dataset: str, run_name: str, split: str) -> Path:
    return Path("/scratch/results") / benchmark / dataset / run_name / f"{split}_scored.json"


def build_eval_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "fourneurons.evaluation.eval",
        args.benchmark,
        args.dataset,
        args.split,
        args.run_name,
        "--checkpoint",
        str(args.checkpoint),
        "--num_generations",
        str(args.num_generations),
        "--max_tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--top_p",
        str(args.top_p),
        "--prompt_file_path",
        str(args.prompt_file_path),
        "--generation_batch_size",
        str(args.generation_batch_size),
    ]
    if not args.no_resume_generation:
        command.append("--resume_generation")
    if args.max_num_samples is not None:
        command.extend(["--max_num_samples", str(args.max_num_samples)])
    if args.top_k is not None:
        command.extend(["--top_k", str(args.top_k)])
    if args.merged_model_dir is not None:
        command.extend(["--merged_model_dir", str(args.merged_model_dir)])
    return command


def build_score_command(args: argparse.Namespace) -> list[str]:
    return [
        sys.executable,
        "-m",
        "evaluate.score",
        "--generations",
        str(generations_path(args.benchmark, args.dataset, args.run_name, args.split)),
        "--benchmark",
        args.benchmark,
        "--output",
        str(scored_path(args.benchmark, args.dataset, args.run_name, args.split)),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", default="math")
    parser.add_argument("--dataset", default="rl_prompt_pool")
    parser.add_argument("--split", default="train")
    parser.add_argument("--run_name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--checkpoint", type=Path, default=BEST_MATH500_16K_CHECKPOINT)
    parser.add_argument("--prompt_file_path", type=Path, default=DEFAULT_PROMPT_FILE)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--generation_batch_size", type=int, default=16)
    parser.add_argument("--max_num_samples", type=int, default=None)
    parser.add_argument("--merged_model_dir", type=Path, default=None)
    parser.add_argument("--no_resume_generation", action="store_true")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_scoring", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if args.num_generations != 8:
        parser.error("Phase 2 should use --num_generations 8.")
    if args.max_tokens <= 0:
        parser.error("--max_tokens must be positive.")
    if args.max_num_samples is not None and args.max_num_samples <= 0:
        parser.error("--max_num_samples must be positive when provided.")
    if args.generation_batch_size <= 0:
        parser.error("--generation_batch_size must be positive.")
    if args.temperature <= 0:
        parser.error("--temperature must be positive.")
    if not (0 < args.top_p <= 1):
        parser.error("--top_p must be in (0, 1].")
    if not args.checkpoint.is_dir():
        parser.error(f"Checkpoint directory does not exist: {args.checkpoint}")
    if not args.prompt_file_path.is_file():
        parser.error(f"Prompt file does not exist: {args.prompt_file_path}")

    pool_path = Path("/scratch/data") / args.benchmark / args.dataset / "splits" / f"{args.dataset}_{args.split}.jsonl"
    if not pool_path.is_file():
        parser.error(
            f"Prompt pool does not exist: {pool_path}. Run fourneurons.data.math_rl_prompt_pool first."
        )

    return args


def run(command: list[str], *, dry_run: bool) -> None:
    print(" ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    gens = generations_path(args.benchmark, args.dataset, args.run_name, args.split)
    scored = scored_path(args.benchmark, args.dataset, args.run_name, args.split)

    print(f"Checkpoint: {args.checkpoint}", flush=True)
    print(f"Generations: {gens}", flush=True)
    print(f"Scored output: {scored}", flush=True)

    if not args.skip_generation:
        run(build_eval_command(args), dry_run=args.dry_run)
    if not args.skip_scoring:
        if not args.dry_run and not gens.is_file():
            raise FileNotFoundError(f"Cannot score missing generations file: {gens}")
        run(build_score_command(args), dry_run=args.dry_run)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
