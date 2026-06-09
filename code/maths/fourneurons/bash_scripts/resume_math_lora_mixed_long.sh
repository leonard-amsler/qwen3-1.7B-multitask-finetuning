#!/usr/bin/env bash
# Submit a non-interactive Run:AI job that resumes the interrupted mixed-long math SFT run.

set -euo pipefail

RUN_ID="${RUN_ID:-qwen3-1.7b-lora-math-mixed-long_20260528-174630}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/checkpoints/math/${RUN_ID}}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-latest}"

export FOURNEURONS_RUN_ID="${FOURNEURONS_RUN_ID:-${RUN_ID}}"
export WANDB_NAME="${WANDB_NAME:-${RUN_ID}}"
export OUTPUT_DIR
export RESUME_FROM_CHECKPOINT
export WANDB_RUN_ID="${WANDB_RUN_ID:-23yc0y4m}"
export WANDB_RESUME="${WANDB_RESUME:-must}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/train_math_lora_mixed.sh" "${1:-math-lora-mixed-resume}"