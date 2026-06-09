#!/usr/bin/env bash
# =============================================================================
# run_baseline.sh — score the raw Qwen3-1.7B baseline (no training)
# -----------------------------------------------------------------------------
# Two reference points on the GK benchmark, at n=8 like the CI:
#   - "baseline"        : Base (no sp) — the bare MCQ question through the chat
#                         template (no prompt engineering, thinking on)
#   - "baseline_prompt" : Base (sp)    — same model + a format-encouraging system
#                         prompt (isolates the gain from prompting alone)
# Reports land in $EVAL/gk_baseline{,_prompt}/. No checkpoint needed.
#
# Usage:
#   chmod +x pipelines/run_baseline.sh
#   ./pipelines/run_baseline.sh                 # both baselines, n=8
#   N_SAMPLES=1 ./pipelines/run_baseline.sh     # quick smoke
#   ONLY=noprompt ./pipelines/run_baseline.sh   # or ONLY=prompt
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

run_baseline() {  # run_baseline <tag> [system_prompt]
  local tag="$1" sys="${2:-}"
  local out="$EVAL/gk_$tag/gkbench_${tag}_generations.jsonl"
  local rep="$EVAL/gk_$tag/gkbench_${tag}_report.json"
  echo "=== [$tag] GK benchmark (n=$N_SAMPLES) ==="
  if [[ -n "$sys" ]]; then
    python -m fourneurons.eval.run_inference \
        --model "$BASE" --input "$GK_BENCH" --output "$out" \
        --system_prompt "$sys" \
        --n "$N_SAMPLES" --max_tokens 4096 --max_model_len 4096 \
        --gpu_memory_utilization 0.90
  else
    python -m fourneurons.eval.run_inference \
        --model "$BASE" --input "$GK_BENCH" --output "$out" \
        --n "$N_SAMPLES" --max_tokens 4096 --max_model_len 4096 \
        --gpu_memory_utilization 0.90
  fi
  python -m fourneurons.eval.report_by_bucket --generations "$out" --output "$rep"
}

case "${ONLY:-both}" in
  noprompt) run_baseline baseline ;;
  prompt)   run_baseline baseline_prompt "$FORMAT_PROMPT" ;;
  both)     run_baseline baseline
            run_baseline baseline_prompt "$FORMAT_PROMPT" ;;
  *) echo "ONLY must be one of: noprompt | prompt | both"; exit 1 ;;
esac

echo "=== run_baseline done. Reports under $EVAL/gk_baseline{,_prompt}/ ==="
