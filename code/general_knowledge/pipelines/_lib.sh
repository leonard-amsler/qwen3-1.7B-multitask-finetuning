#!/usr/bin/env bash
# =============================================================================
# _lib.sh — shared helpers for GK pipelines (sourced by run_*.sh)
# -----------------------------------------------------------------------------
# Each ensure_* function builds an artifact ONLY if it is missing.
# They chain together (e.g. ensure_v8_data calls ensure_train_v6), making every
# run_*.sh self-contained and runnable in any order.
# All "train_v8 data", train_v6, etc. commands live here once.
# =============================================================================

# --- Shared config (override via environment variables if needed) ------------
DATA="${DATA:-/scratch/data}"
CKPT="${CKPT:-/scratch/checkpoints}"
EVAL="${EVAL:-/scratch/eval}"
BASE="${BASE:-Qwen/Qwen3-1.7B}"
DEV="${DEV:-validation_samples/general_knowledge_dev_full.jsonl}"
N_SAMPLES="${N_SAMPLES:-8}"   # 8 = reliable metric (like CI); 1 = quick smoke test

# --- Shared 14B distillations (short v5 + long v6) ---------------------------
ensure_distill_v5() {
  [[ -f "$DATA/distilled_cot_v5.jsonl" ]] && return 0
  echo "  [ensure] distill v5 (14B, short)"
  python -m fourneurons.distill.distill \
      --teacher Qwen/Qwen3-14B-AWQ \
      --output "$DATA/distilled_cot_v5.jsonl" \
      --max_tokens 1024 --gpu_memory_utilization 0.85
}

ensure_distill_v6_long() {
  [[ -f "$DATA/distilled_cot_v6_long.jsonl" ]] && return 0
  echo "  [ensure] distill v6_long (14B, STEM)"
  python -m fourneurons.distill.distill \
      --teacher Qwen/Qwen3-14B-AWQ \
      --output "$DATA/distilled_cot_v6_long.jsonl" \
      --sources mmlu mmlu_world mmlu_pro_cot --max_tokens 2048
}

# --- SFT dataset train_v6 (questions + 14B CoTs) -----------------------------
ensure_train_v6() {
  [[ -d "$DATA/train_v6" ]] && return 0
  echo "  [ensure] build train_v6"
  ensure_distill_v5
  ensure_distill_v6_long
  python -m fourneurons.data.build_train \
      --output_dir          "$DATA/train_v6" \
      --total               30000 --max_variants 1 \
      --distilled_cot_cache "$DATA/distilled_cot_v5.jsonl" "$DATA/distilled_cot_v6_long.jsonl" \
      --seed                42
}

# --- V8 data: self-distillation from the 1.7B baseline -----------------------
# Produces train_v8 (SFT dataset) AND train_v8/self_distill_cache.jsonl (V9b cache)
ensure_v8_data() {
  ensure_train_v6
  if [[ ! -f "$DATA/train_v8/self_distill_cache.jsonl" ]]; then
    echo "  [ensure] self-distill 1.7B — generate (V8 cache)"
    python -m fourneurons.distill.self_distill \
        --teacher Qwen/Qwen3-1.7B --quantization "" --enable_thinking \
        --source_dataset "$DATA/train_v6" \
        --output_dir "$DATA/train_v8" \
        --cot_source_tag self_distill_baseline \
        --n_samples 4 --max_tokens 3000 --max_model_len 4096 \
        --temperature 0.6 --top_p 0.95 --top_k 20 \
        --gpu_memory_utilization 0.92 --chunk_size 2000 \
        --no_fallback_to_source --phase generate
  fi
  if [[ ! -d "$DATA/train_v8" || -z "$(ls -A "$DATA/train_v8" 2>/dev/null | grep -v self_distill_cache)" ]]; then
    echo "  [ensure] self-distill 1.7B — assemble (train_v8 dataset)"
    python -m fourneurons.distill.self_distill \
        --teacher Qwen/Qwen3-1.7B \
        --source_dataset "$DATA/train_v6" \
        --output_dir "$DATA/train_v8" \
        --cot_source_tag self_distill_baseline \
        --n_samples 4 --no_fallback_to_source --phase assemble
  fi
}

# --- V9b dataset: clean re-assembly (select_best) from V8 cache --------------
ensure_train_v9b() {
  [[ -d "$DATA/train_v9b" ]] && return 0
  echo "  [ensure] assemble train_v9b (select_best, anti-loop)"
  ensure_v8_data
  python -m fourneurons.distill.self_distill \
      --teacher Qwen/Qwen3-1.7B --quantization "" \
      --source_dataset "$DATA/train_v6" \
      --output_dir "$DATA/train_v9b" \
      --cache_path "$DATA/train_v8/self_distill_cache.jsonl" \
      --cot_source_tag self_distill_clean \
      --n_samples 4 --select_best --max_thinking_chars 4000 \
      --phase assemble
}

# --- V9b model (base policy for V11/V11b DPO) --------------------------------
ensure_model_v9b() {
  [[ -d "$CKPT/gk_v9b/vllm" ]] && return 0
  echo "  [ensure] SFT + merge gk_v9b"
  ensure_train_v9b
  python -m fourneurons.scripts.train \
      --dataset_dir "$DATA/train_v9b" \
      --output_dir "$CKPT/gk_v9b" \
      --final_model_dir "$CKPT/gk_v9b/adapter" \
      --num_epochs 1 --learning_rate 2e-4 \
      --lora_r 64 --lora_alpha 128 --max_seq_length 4096 --bf16
  python -m fourneurons.scripts.merge_lora \
      --adapter_dir "$CKPT/gk_v9b/adapter" \
      --output_dir "$CKPT/gk_v9b/vllm" \
      --base_model "$BASE" --device cpu
}

# --- DPO pairs sampled from v9b ------------------------------------------------
ensure_dpo_pairs() {
  [[ -f "$DATA/dpo_pairs_v9b.jsonl" ]] && return 0
  echo "  [ensure] build_dpo_pairs (8 draws from v9b)"
  ensure_model_v9b
  ensure_train_v9b
  python -m fourneurons.scripts.build_dpo_pairs \
      --model "$CKPT/gk_v9b/vllm" \
      --dataset_dir "$DATA/train_v9b" \
      --output "$DATA/dpo_pairs_v9b.jsonl" \
      --n_examples 4000 --n_per_prompt 8 --max_tokens 2048
}

# --- dev_full benchmark eval ---------------------------------------------------
# usage: run_test <version> <model_dir>
run_test() {
  local v="$1" model="$2"
  echo "=== [$v] TEST dev_full (n=$N_SAMPLES) ==="
  python -m fourneurons.eval.run_inference \
      --model "$model" --input "$DEV" \
      --output "$EVAL/gk_$v/devfull_${v}_generations.jsonl" \
      --n "$N_SAMPLES" --max_tokens 4096 --max_model_len 4096 --gpu_memory_utilization 0.90
  python -m fourneurons.eval.report_by_bucket \
      --generations "$EVAL/gk_$v/devfull_${v}_generations.jsonl" \
      --output      "$EVAL/gk_$v/devfull_${v}_report.json"
}
