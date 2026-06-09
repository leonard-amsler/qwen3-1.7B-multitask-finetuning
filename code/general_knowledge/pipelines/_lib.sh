#!/usr/bin/env bash
# =============================================================================
# _lib.sh — shared helpers for the General Knowledge pipelines (sourced by run_*.sh)
# -----------------------------------------------------------------------------
# Each ensure_* function builds an artifact ONLY if it is missing, and the
# functions chain together, so every run_*.sh is self-contained and can be run
# in any order. Naming follows the report:
#   - Baseline                  (raw Qwen3-1.7B, with / without a format prompt)
#   - Distillation comparison   (off-policy / contrastive / on-policy CoT corpora)
#   - SFT                       (on-policy self-distillation)
#   - SFT+DPO                   (preference tuning on top of SFT, final model)
# All models are scored on the GK benchmark (validation_samples/...dev_full.jsonl).
# =============================================================================

# --- Shared config (override via environment variables if needed) ------------
_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$_LIB_DIR/.." && pwd)}"
DATA="${DATA:-/scratch/data}"
CKPT="${CKPT:-/scratch/checkpoints}"
EVAL="${EVAL:-/scratch/eval}"
BASE="${BASE:-Qwen/Qwen3-1.7B}"
GK_BENCH="${GK_BENCH:-validation_samples/general_knowledge_dev_full.jsonl}"
N_SAMPLES="${N_SAMPLES:-8}"   # 8 = reliable metric (like CI); 1 = quick smoke test

# Format-encouraging system prompt for the "Base (sp)" baseline (override via env).
FORMAT_PROMPT="${FORMAT_PROMPT:-Answer the multiple-choice question. Reason step by step, then give your final answer as a single letter inside \\boxed{}, e.g. \\boxed{A}. Do not write anything after the box.}"

# Group model eval — trained with sp_group_think.txt (unlike GK SFT/DPO models).
GROUP_MODEL="${GROUP_MODEL:-/scratch/checkpoints/group_model/learnable_cat_5e3}"
GROUP_SP_FILE="${GROUP_SP_FILE:-$ROOT/fourneurons/prompts/sp_group_think.txt}"

# =============================================================================
# Teacher (14B) distillation caches — used by the off-policy & contrastive corpora
# =============================================================================
ensure_distill_14b_short() {
  [[ -f "$DATA/distilled_cot_14b_short.jsonl" ]] && return 0
  echo "  [ensure] 14B distillation (short CoTs)"
  python -m fourneurons.distill.distill \
      --teacher Qwen/Qwen3-14B-AWQ \
      --output "$DATA/distilled_cot_14b_short.jsonl" \
      --max_tokens 1024 --gpu_memory_utilization 0.85
}

ensure_distill_14b_long() {
  [[ -f "$DATA/distilled_cot_14b_long.jsonl" ]] && return 0
  echo "  [ensure] 14B distillation (long step-by-step CoTs on STEM)"
  python -m fourneurons.distill.distill \
      --teacher Qwen/Qwen3-14B-AWQ \
      --output "$DATA/distilled_cot_14b_long.jsonl" \
      --sources mmlu mmlu_world mmlu_pro_cot --max_tokens 2048
}

# =============================================================================
# CoT corpora (SFT datasets) — one per distillation strategy in the report
# =============================================================================

# Off-policy distillation: questions + 14B CoTs (short + long, last-wins).
# Also serves as the question source for on-policy self-distillation.
ensure_data_offpolicy() {
  [[ -d "$DATA/gk_offpolicy_data" ]] && return 0
  echo "  [ensure] build off-policy CoT corpus"
  ensure_distill_14b_short
  ensure_distill_14b_long
  python -m fourneurons.data.build_train \
      --output_dir          "$DATA/gk_offpolicy_data" \
      --total               30000 --max_variants 1 \
      --distilled_cot_cache "$DATA/distilled_cot_14b_short.jsonl" \
                            "$DATA/distilled_cot_14b_long.jsonl" \
      --seed                42
}

# Contrastive distillation: off-policy CoTs + a 14B trace that also refutes the
# most tempting wrong answer (contrastive, option-agnostic).
ensure_data_contrastive() {
  [[ -d "$DATA/gk_contrastive_data" ]] && return 0
  echo "  [ensure] build contrastive CoT corpus"
  ensure_distill_14b_short
  ensure_distill_14b_long
  if [[ ! -f "$DATA/distilled_cot_contrastive.jsonl" ]]; then
    echo "  [ensure] 14B contrastive distillation"
    python -m fourneurons.distill.distill \
        --teacher Qwen/Qwen3-14B-AWQ --quantization awq_marlin \
        --output "$DATA/distilled_cot_contrastive.jsonl" \
        --sources mmlu mmlu_pro_cot --reasoning_style contrastive \
        --max_tokens 2048 --max_model_len 4096
  fi
  python -m fourneurons.data.build_train \
      --output_dir          "$DATA/gk_contrastive_data" \
      --total               30000 --max_variants 1 \
      --distilled_cot_cache "$DATA/distilled_cot_14b_short.jsonl" \
                            "$DATA/distilled_cot_14b_long.jsonl" \
                            "$DATA/distilled_cot_contrastive.jsonl" \
      --seed                42
}

# On-policy self-distillation cache: the 1.7B base draws its own CoTs (4 samples
# per question, thinking on) on the off-policy question set.
ensure_selfdistill_cache() {
  ensure_data_offpolicy
  if [[ ! -f "$DATA/gk_selfdistill/self_distill_cache.jsonl" ]]; then
    echo "  [ensure] on-policy self-distillation — generate"
    python -m fourneurons.distill.self_distill \
        --teacher Qwen/Qwen3-1.7B --quantization "" --enable_thinking \
        --source_dataset "$DATA/gk_offpolicy_data" \
        --output_dir "$DATA/gk_selfdistill" \
        --cot_source_tag self_distill_baseline \
        --n_samples 4 --max_tokens 3000 --max_model_len 4096 \
        --temperature 0.6 --top_p 0.95 --top_k 20 \
        --gpu_memory_utilization 0.92 --chunk_size 2000 \
        --no_fallback_to_source --phase generate
  fi
}

# On-policy CoT corpus: re-assemble the self-distill cache keeping, per question,
# the shortest clean correct trace (anti-loop select_best). This is the SFT data.
ensure_data_onpolicy() {
  [[ -d "$DATA/gk_onpolicy_data" ]] && return 0
  echo "  [ensure] build on-policy CoT corpus (select_best, anti-loop)"
  ensure_selfdistill_cache
  python -m fourneurons.distill.self_distill \
      --teacher Qwen/Qwen3-1.7B --quantization "" \
      --source_dataset "$DATA/gk_offpolicy_data" \
      --output_dir "$DATA/gk_onpolicy_data" \
      --cache_path "$DATA/gk_selfdistill/self_distill_cache.jsonl" \
      --cot_source_tag self_distill_clean \
      --n_samples 4 --select_best --max_thinking_chars 4000 \
      --phase assemble
}

# =============================================================================
# Models — same SFT recipe (LoRA r=64, alpha=128) for all three corpora
# =============================================================================
# usage: _sft_and_merge <corpus_dir> <ckpt_dir>
_sft_and_merge() {
  local data_dir="$1" ckpt_dir="$2"
  python -m fourneurons.scripts.train \
      --dataset_dir     "$data_dir" \
      --output_dir      "$ckpt_dir" \
      --final_model_dir "$ckpt_dir/adapter" \
      --num_epochs 1 --learning_rate 2e-4 \
      --lora_r 64 --lora_alpha 128 --max_seq_length 4096 --bf16
  python -m fourneurons.scripts.merge_lora \
      --adapter_dir "$ckpt_dir/adapter" \
      --output_dir  "$ckpt_dir/vllm" \
      --base_model  "$BASE" --device cpu
}

ensure_model_offpolicy() {
  [[ -d "$CKPT/gk_offpolicy/vllm" ]] && return 0
  echo "  [ensure] SFT + merge (off-policy distillation)"
  ensure_data_offpolicy
  _sft_and_merge "$DATA/gk_offpolicy_data" "$CKPT/gk_offpolicy"
}

ensure_model_contrastive() {
  [[ -d "$CKPT/gk_contrastive/vllm" ]] && return 0
  echo "  [ensure] SFT + merge (contrastive distillation)"
  ensure_data_contrastive
  _sft_and_merge "$DATA/gk_contrastive_data" "$CKPT/gk_contrastive"
}

# On-policy self-distillation SFT = the SFT model reported in the paper.
ensure_model_sft() {
  [[ -d "$CKPT/gk_sft/vllm" ]] && return 0
  echo "  [ensure] SFT + merge (on-policy self-distillation = SFT)"
  ensure_data_onpolicy
  _sft_and_merge "$DATA/gk_onpolicy_data" "$CKPT/gk_sft"
}

# Preference pairs sampled from the SFT model (correct vs incorrect, 8 draws).
ensure_dpo_pairs() {
  [[ -f "$DATA/dpo_pairs_sft.jsonl" ]] && return 0
  echo "  [ensure] build DPO pairs (8 draws from SFT)"
  ensure_model_sft
  python -m fourneurons.scripts.build_dpo_pairs \
      --model "$CKPT/gk_sft/vllm" \
      --dataset_dir "$DATA/gk_onpolicy_data" \
      --output "$DATA/dpo_pairs_sft.jsonl" \
      --n_examples 4000 --n_per_prompt 8 --max_tokens 2048
}

# =============================================================================
# GK benchmark evaluation
# =============================================================================
# usage: run_test <tag> <model_dir> [system_prompt]
run_test() {
  local tag="$1" model="$2" sys="${3:-}"
  echo "=== [$tag] GK benchmark (n=$N_SAMPLES) ==="
  if [[ -n "$sys" ]]; then
    python -m fourneurons.eval.run_inference \
        --model "$model" --input "$GK_BENCH" \
        --output "$EVAL/gk_$tag/gkbench_${tag}_generations.jsonl" \
        --system_prompt "$sys" \
        --n "$N_SAMPLES" --max_tokens 4096 --max_model_len 4096 --gpu_memory_utilization 0.90
  else
    python -m fourneurons.eval.run_inference \
        --model "$model" --input "$GK_BENCH" \
        --output "$EVAL/gk_$tag/gkbench_${tag}_generations.jsonl" \
        --n "$N_SAMPLES" --max_tokens 4096 --max_model_len 4096 --gpu_memory_utilization 0.90
  fi
  python -m fourneurons.eval.report_by_bucket \
      --generations "$EVAL/gk_$tag/gkbench_${tag}_generations.jsonl" \
      --output      "$EVAL/gk_$tag/gkbench_${tag}_report.json"
}
