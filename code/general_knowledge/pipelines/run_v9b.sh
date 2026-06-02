#!/usr/bin/env bash
# =============================================================================
# V9b — V9 + anti-loop cleanup (select_best) at assembly
# -----------------------------------------------------------------------------
# 59% of V8 self-distilled traces looped (-> truncations without \boxed{}).
# V9b re-assembles the V8 cache keeping, per question, the shortest clean
# correct sample; falls back to v6 CoT if all loop. \boxed{} -> 94.5%.
# Self-contained: ensure_train_v9b builds V8 data + train_v6 + train_v9b as needed.
# Output: $CKPT/gk_v9b/vllm  | dev_full n=8: pass@1 0.580, pass@8 0.892
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

ensure_train_v9b   # auto: V8 data (cache) + train_v6 (fallback) + select_best assemble

echo "=== [V9b] SFT strong LoRA (r=64, alpha=128) ==="
python -m fourneurons.scripts.train \
    --dataset_dir       "$DATA/train_v9b" \
    --output_dir        "$CKPT/gk_v9b" \
    --final_model_dir   "$CKPT/gk_v9b/adapter" \
    --num_epochs 1 --learning_rate 2e-4 \
    --lora_r 64 --lora_alpha 128 --max_seq_length 4096 --bf16

echo "=== [V9b] merge LoRA ==="
python -m fourneurons.scripts.merge_lora \
    --adapter_dir "$CKPT/gk_v9b/adapter" \
    --output_dir  "$CKPT/gk_v9b/vllm" \
    --base_model  "$BASE" --device cpu

run_test v9b "$CKPT/gk_v9b/vllm"
echo "=== [V9b] done. Model: $CKPT/gk_v9b/vllm ==="
