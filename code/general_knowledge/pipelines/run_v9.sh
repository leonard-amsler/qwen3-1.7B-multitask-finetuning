#!/usr/bin/env bash
# =============================================================================
# V9 — Re-SFT on V8 data with STRONG LoRA (format fix)
# -----------------------------------------------------------------------------
# V9 = SAME data as V8 (1.7B self-distill), but LoRA r=64/alpha=128 instead of
# r=8. \boxed{} coverage goes from 51% to 86%; dev_full from 0.39 to 0.60.
# Self-contained: if train_v8 is missing, ensure_v8_data builds it
# (self-distill generate + assemble from train_v6, also auto-built).
# Output: $CKPT/gk_v9/vllm   | dev_full (n=1): pass@1 ~0.600
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

ensure_v8_data   # build $DATA/train_v8 (+ cache) if absent

echo "=== [V9] SFT strong LoRA (r=64, alpha=128) on train_v8 ==="
python -m fourneurons.scripts.train \
    --dataset_dir       "$DATA/train_v8" \
    --output_dir        "$CKPT/gk_v9" \
    --final_model_dir   "$CKPT/gk_v9/adapter" \
    --num_epochs 1 --learning_rate 2e-4 \
    --lora_r 64 --lora_alpha 128 --max_seq_length 4096 --bf16

echo "=== [V9] merge LoRA ==="
python -m fourneurons.scripts.merge_lora \
    --adapter_dir "$CKPT/gk_v9/adapter" \
    --output_dir  "$CKPT/gk_v9/vllm" \
    --base_model  "$BASE" --device cpu

run_test v9 "$CKPT/gk_v9/vllm"
echo "=== [V9] done. Model: $CKPT/gk_v9/vllm ==="
