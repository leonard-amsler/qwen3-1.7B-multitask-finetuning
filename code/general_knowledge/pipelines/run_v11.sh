#!/usr/bin/env bash
# =============================================================================
# V11 — Timid on-policy DPO on top of V9b
# -----------------------------------------------------------------------------
# Sample (correct vs incorrect) pairs from v9b, then timid DPO
# (beta=0.1, lr=5e-6, 1 epoch) -> NO-OP (0.610 = v9b). Documented negative result.
# Self-contained: ensure_dpo_pairs builds gk_v9b (full chain) + pairs if needed.
# Output: $CKPT/gk_v11/vllm
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

ensure_dpo_pairs   # auto: gk_v9b (full chain) + dpo_pairs_v9b.jsonl

echo "=== [V11] timid DPO LoRA (beta=0.1, lr=5e-6, 1 epoch) ==="
python -m fourneurons.scripts.train_dpo \
  --base_model "$CKPT/gk_v9b/vllm" \
  --pairs "$DATA/dpo_pairs_v9b.jsonl" \
  --output_dir "$CKPT/gk_v11" \
  --final_model_dir "$CKPT/gk_v11/adapter" \
  --num_epochs 1 --beta 0.1 --bf16

echo "=== [V11] merge LoRA (DPO adapter on v9b policy) ==="
python -m fourneurons.scripts.merge_lora \
  --adapter_dir "$CKPT/gk_v11/adapter" \
  --output_dir  "$CKPT/gk_v11/vllm" \
  --base_model  "$CKPT/gk_v9b/vllm"

run_test v11 "$CKPT/gk_v11/vllm"
echo "=== [V11] done. Model: $CKPT/gk_v11/vllm ==="
