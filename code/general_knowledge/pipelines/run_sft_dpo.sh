#!/usr/bin/env bash
# =============================================================================
# run_sft_dpo.sh — SFT+DPO model (FINAL)
# -----------------------------------------------------------------------------
# On-policy preference tuning on top of SFT: sample correct-vs-incorrect pairs
# from the SFT model itself, then DPO (beta=0.05, lr=1e-5, 3 epochs) with the
# SFT model as the frozen reference policy. Merges and scores on the GK benchmark.
# Self-contained: builds the SFT model (full chain) + DPO pairs if missing.
# Output: $CKPT/gk_sft_dpo/vllm
#
# Usage:
#   chmod +x pipelines/run_sft_dpo.sh
#   ./pipelines/run_sft_dpo.sh
#   N_SAMPLES=1 ./pipelines/run_sft_dpo.sh   # quick smoke
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

ensure_dpo_pairs   # builds SFT model (full chain) + dpo_pairs_sft.jsonl if missing

echo "=== [SFT+DPO] DPO LoRA (beta=0.05, lr=1e-5, 3 epochs) ==="
python -m fourneurons.scripts.train_dpo \
  --base_model      "$CKPT/gk_sft/vllm" \
  --pairs           "$DATA/dpo_pairs_sft.jsonl" \
  --output_dir      "$CKPT/gk_sft_dpo" \
  --final_model_dir "$CKPT/gk_sft_dpo/adapter" \
  --num_epochs 3 --learning_rate 1e-5 --beta 0.05 --bf16

echo "=== [SFT+DPO] merge LoRA (DPO adapter on SFT policy) ==="
python -m fourneurons.scripts.merge_lora \
  --adapter_dir "$CKPT/gk_sft_dpo/adapter" \
  --output_dir  "$CKPT/gk_sft_dpo/vllm" \
  --base_model  "$CKPT/gk_sft/vllm"

run_test sft_dpo "$CKPT/gk_sft_dpo/vllm"
echo "=== run_sft_dpo done. FINAL MODEL: $CKPT/gk_sft_dpo/vllm ==="
