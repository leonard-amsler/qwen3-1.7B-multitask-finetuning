#!/usr/bin/env bash
# =============================================================================
# run_distillation.sh — compare the three CoT distillation strategies
# -----------------------------------------------------------------------------
# Reproduces the distillation comparison in the report (figure "GK benchmark").
# Trains one SFT model per CoT corpus, with the SAME SFT recipe, then scores
# each on the GK benchmark:
#   - off-policy distillation   (14B teacher CoTs)            -> $CKPT/gk_offpolicy
#   - contrastive distillation  (14B + refute best distractor) -> $CKPT/gk_contrastive
#   - on-policy self-distillation (1.7B own cleaned CoTs = SFT) -> $CKPT/gk_sft
# Self-contained: each corpus and model is built automatically if missing.
#
# Usage:
#   chmod +x pipelines/run_distillation.sh
#   ./pipelines/run_distillation.sh                 # all three, n=8
#   N_SAMPLES=1 ./pipelines/run_distillation.sh     # quick smoke
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

ensure_model_offpolicy
ensure_model_contrastive
ensure_model_sft

run_test offpolicy   "$CKPT/gk_offpolicy/vllm"
run_test contrastive "$CKPT/gk_contrastive/vllm"
run_test sft         "$CKPT/gk_sft/vllm"

echo "=== run_distillation done. Reports under $EVAL/gk_{offpolicy,contrastive,sft}/ ==="
echo "    on-policy self-distillation is the model reported as SFT."
