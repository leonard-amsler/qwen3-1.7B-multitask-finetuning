#!/usr/bin/env bash
# =============================================================================
# V10 — Contrastive 14B distillation (off-policy) on STEM
# -----------------------------------------------------------------------------
# On top of v6's clean data architecture, add CoTs where 14B justifies the correct
# answer AND refutes the most tempting wrong line (contrastive prompt, option-agnostic).
# Result: off-policy < on-policy -> V10 < V9b.
# Self-contained: builds v5 + v6_long caches if missing.
# Output: $CKPT/gk_v10/vllm  | dev_full (n=1): pass@1 ~0.554
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

ensure_distill_v5
ensure_distill_v6_long

echo "=== [V10] contrastive distillation (14B) ==="
python -m fourneurons.distill.distill \
    --teacher Qwen/Qwen3-14B-AWQ --quantization awq_marlin \
    --output "$DATA/distilled_cot_v10_contrastive.jsonl" \
    --sources mmlu mmlu_pro_cot --reasoning_style contrastive \
    --max_tokens 2048 --max_model_len 4096

echo "=== [V10] build_train (v5 + v6_long + contrastive, last-wins) ==="
python -m fourneurons.data.build_train \
    --output_dir          "$DATA/train_v10" \
    --total               30000 --max_variants 1 \
    --distilled_cot_cache "$DATA/distilled_cot_v5.jsonl" \
                          "$DATA/distilled_cot_v6_long.jsonl" \
                          "$DATA/distilled_cot_v10_contrastive.jsonl" \
    --seed                42

echo "=== [V10] SFT LoRA + merge ==="
python -m fourneurons.scripts.train \
    --dataset_dir       "$DATA/train_v10" \
    --output_dir        "$CKPT/gk_v10" \
    --final_model_dir   "$CKPT/gk_v10/adapter" \
    --num_epochs 1 --learning_rate 2e-4 \
    --lora_r 64 --lora_alpha 128 --max_seq_length 4096 --bf16
python -m fourneurons.scripts.merge_lora \
    --adapter_dir "$CKPT/gk_v10/adapter" \
    --output_dir  "$CKPT/gk_v10/vllm" \
    --base_model  "$BASE" --device cpu

run_test v10 "$CKPT/gk_v10/vllm"
echo "=== [V10] done. Model: $CKPT/gk_v10/vllm ==="
