#!/usr/bin/env bash
# =============================================================================
# run_group.sh — score the merged group model on the GK benchmark
# -----------------------------------------------------------------------------
# Unlike GK task-specific models (SFT / SFT+DPO), the group model was trained
# and evaluated with sp_group_think.txt. This script loads that prompt and
# runs n=8 inference + bucket report.
#
# Usage:
#   chmod +x pipelines/run_group.sh
#   ./pipelines/run_group.sh
#   N_SAMPLES=1 ./pipelines/run_group.sh
#   GROUP_MODEL=/scratch/checkpoints/group_model/learnable_cat_5e3 ./pipelines/run_group.sh
#   GROUP_SP_FILE=/path/to/sp_group_think.txt ./pipelines/run_group.sh
# =============================================================================
set -euo pipefail
source "$(dirname "$0")/_lib.sh"

[[ -f "$GROUP_SP_FILE" ]] || {
  echo "ERROR: group system prompt not found: $GROUP_SP_FILE" >&2
  exit 1
}
[[ -e "$GROUP_MODEL" ]] || {
  echo "ERROR: group model not found: $GROUP_MODEL" >&2
  exit 1
}

SP="$(cat "$GROUP_SP_FILE")"
mkdir -p "$EVAL/gk_group"

run_test group "$GROUP_MODEL" "$SP"

echo "=== run_group done. Report: $EVAL/gk_group/gkbench_group_report.json ==="
