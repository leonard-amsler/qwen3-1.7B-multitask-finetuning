#!/usr/bin/env bash
# =============================================================================
# V6 — Long 14B distillation + surgical re-distillation
# -----------------------------------------------------------------------------
# SFT on CoTs distilled from Qwen3-14B-AWQ: v5 cache (short) + v6_long cache
# (mmlu/mmlu_world/mmlu_pro_cot, max_tokens=2048), merged last-wins.
# Self-contained: builds distill caches + train_v6 if missing.
# Output: $CKPT/gk_v6/vllm   | dev_full (n=1): pass@1 ~0.541
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

ensure_train_v6   # build v5 + v6_long caches + train_v6 dataset if absent

echo "=== [V6] SFT LoRA (r=64, alpha=128) ==="
python -m fourneurons.scripts.train \
    --dataset_dir       "$DATA/train_v6" \
    --output_dir        "$CKPT/gk_v6" \
    --final_model_dir   "$CKPT/gk_v6/adapter" \
    --num_epochs 1 --learning_rate 2e-4 \
    --per_device_batch_size 2 --grad_accum 8 \
    --lora_r 64 --lora_alpha 128 --max_seq_length 4096 \
    --eval_steps 200 --save_steps 200 --logging_steps 20 --bf16

echo "=== [V6] merge LoRA (+ thinking baked) ==="
python -m fourneurons.scripts.merge_lora \
    --adapter_dir "$CKPT/gk_v6/adapter" \
    --output_dir  "$CKPT/gk_v6/vllm" \
    --base_model  "$BASE" --device cpu

run_test v6 "$CKPT/gk_v6/vllm"
echo "=== [V6] done. Model: $CKPT/gk_v6/vllm ==="
