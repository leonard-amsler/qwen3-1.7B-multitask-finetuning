#!/usr/bin/env bash
# =============================================================================
# V11b — STRONG on-policy DPO on top of V9b  (FINAL MODEL)
# -----------------------------------------------------------------------------
# Same DPO pairs as V11, stronger signal (beta=0.05, lr=1e-5, 3 epochs).
# load_best_model_at_end keeps the least overfit checkpoint.
# WINS: dev_full n=8 pass@1 0.600 (vs v9b 0.580, +2.0 on 4/4 sources),
#       pass@8 0.883 (~ v9b, intact). DPO sharpens the distribution.
# Self-contained: ensure_dpo_pairs builds gk_v9b + pairs if needed.
# Output: $CKPT/gk_v11b/vllm
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

ensure_dpo_pairs   # auto: gk_v9b (full chain) + dpo_pairs_v9b.jsonl

echo "=== [V11b] strong DPO LoRA (beta=0.05, lr=1e-5, 3 epochs) ==="
python -m fourneurons.scripts.train_dpo \
  --base_model "$CKPT/gk_v9b/vllm" \
  --pairs "$DATA/dpo_pairs_v9b.jsonl" \
  --output_dir "$CKPT/gk_v11b" \
  --final_model_dir "$CKPT/gk_v11b/adapter" \
  --num_epochs 3 --learning_rate 1e-5 --beta 0.05 --bf16

echo "=== [V11b] merge LoRA ==="
python -m fourneurons.scripts.merge_lora \
  --adapter_dir "$CKPT/gk_v11b/adapter" \
  --output_dir  "$CKPT/gk_v11b/vllm" \
  --base_model  "$CKPT/gk_v9b/vllm"

run_test v11b "$CKPT/gk_v11b/vllm"
echo "=== [V11b] done. FINAL MODEL: $CKPT/gk_v11b/vllm ==="
