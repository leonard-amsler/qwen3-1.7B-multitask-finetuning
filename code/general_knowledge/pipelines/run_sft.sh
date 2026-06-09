#!/usr/bin/env bash
# =============================================================================
# run_sft.sh — SFT model (on-policy self-distillation)
# -----------------------------------------------------------------------------
# Trains the SFT model reported in the paper: LoRA fine-tune (r=64, alpha=128)
# of Qwen3-1.7B on the on-policy CoT corpus (the 1.7B's own cleaned correct
# reasoning), then merges the adapter (bakes thinking=ON) and scores it on the
# GK benchmark.
# Self-contained: builds the on-policy corpus (self-distillation) if missing.
# Output: $CKPT/gk_sft/vllm
#
# Usage:
#   chmod +x pipelines/run_sft.sh
#   ./pipelines/run_sft.sh
#   N_SAMPLES=1 ./pipelines/run_sft.sh   # quick smoke
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

ensure_model_sft   # builds on-policy corpus + SFT + merge if missing

run_test sft "$CKPT/gk_sft/vllm"
echo "=== run_sft done. Model: $CKPT/gk_sft/vllm ==="
