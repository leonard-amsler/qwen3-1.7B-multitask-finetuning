#!/usr/bin/env bash
# =============================================================================
# eval_all.sh — re-evaluate every KEPT GK model on all local benchmarks
# -----------------------------------------------------------------------------
# No retraining: assumes the merged vLLM checkpoints already exist under
#   $CKPT/gk_<version>/vllm   (built by the run_*.sh pipelines).
# For each version it runs inference + the per-bucket report on:
#   - dev_full  (n=1000, reliable, has meta -> by_source / macro_cat / n_options)
#   - dev_small (n=220,  noisy, kept for continuity with earlier plans)
# Reports land in $EVAL/gk_<version>/.
#
# Usage:
#   chmod +x pipelines/eval_all.sh
#   ./pipelines/eval_all.sh                 # all kept versions, n=8 (like CI)
#   N_SAMPLES=1 ./pipelines/eval_all.sh      # quick smoke (n=1)
#   VERSIONS="v9b v11b" ./pipelines/eval_all.sh   # subset
#   WITH_BASELINE=1 ./pipelines/eval_all.sh  # also score raw Qwen3-1.7B
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

# Kept versions = those that have a run_*.sh pipeline.
VERSIONS="${VERSIONS:-v6 v9 v9b v10 v11 v11b}"
DEV_FULL="${DEV_FULL:-validation_samples/general_knowledge_dev_full.jsonl}"
DEV_SMALL="${DEV_SMALL:-validation_samples/general_knowledge_dev_small.jsonl}"

eval_one() {  # eval_one <tag> <model_dir>
  local tag="$1" model="$2"
  if [[ ! -d "$model" ]]; then
    echo "!! [$tag] missing checkpoint: $model — run pipelines/run_${tag}.sh first; skipping."
    return 0
  fi
  for bench in full small; do
    local dev out rep
    if [[ "$bench" == full ]]; then dev="$DEV_FULL"; else dev="$DEV_SMALL"; fi
    out="$EVAL/gk_$tag/dev${bench}_${tag}_generations.jsonl"
    rep="$EVAL/gk_$tag/dev${bench}_${tag}_report_n${N_SAMPLES}.json"
    echo "=== [$tag] dev_${bench} (n=$N_SAMPLES) ==="
    python -m fourneurons.eval.run_inference \
        --model "$model" --input "$dev" --output "$out" \
        --n "$N_SAMPLES" --max_tokens 4096 --max_model_len 4096 \
        --gpu_memory_utilization 0.90
    python -m fourneurons.eval.report_by_bucket \
        --generations "$out" --output "$rep"
  done
}

if [[ "${WITH_BASELINE:-0}" == "1" ]]; then
  eval_one baseline "$BASE"
fi

for V in $VERSIONS; do
  eval_one "$V" "$CKPT/gk_$V/vllm"
done

echo "=== eval_all done. Reports under $EVAL/gk_*/dev*_report_n${N_SAMPLES}.json ==="
