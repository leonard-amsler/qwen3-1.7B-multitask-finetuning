#!/usr/bin/env bash
# CS-552 - submit a non-interactive math checkpoint evaluator job on Run:AI.
#
# Usage:
#   GASPAR=<gaspar> GROUP=<gXX> \
#     ./fourneurons/bash_scripts/eval_math_all_checkpoints.sh \
#     /scratch/checkpoints/math/20260526-161659 4096 1000
#
# Equivalent env overrides:
#   RUN_DIR=/scratch/checkpoints/math/20260526-161659
#   MAX_TOKENS=4096
#   MAX_NUM_SAMPLES=1000
#   OUTPUT_PREFIX=gen16_tok8192_
#
# Optional overrides:
#   REPO_DIR=/scratch/leo/merged_repo
#   IMAGE=ayushkumartarun/course-cs-552-standard:v1
#   BENCHMARK=math
#   DATASET=competitionmath
#   SPLIT=full
#   NUM_GENERATIONS=8
#   PROMPT_FILE_PATH=fourneurons/prompts/math.txt
#   EXTRA_EVAL_ARGS="--skip_existing"
#   EXTRA_EVAL_CMD="python -m fourneurons.evaluation.eval_all_checkpoints ..."

set -euo pipefail

GASPAR="${GASPAR:-lamsler}"
GROUP="${GROUP:-g17}"

if [[ -z "${GASPAR}" || "${GASPAR}" == "gaspar" ]]; then
  echo "ERROR: set GASPAR, e.g. GASPAR=lamsler GROUP=g17 $0" >&2
  exit 1
fi

if [[ -z "${GROUP}" || "${GROUP}" == "gxx" || "${GROUP}" == "gXX" ]]; then
  echo "ERROR: set GROUP, e.g. GASPAR=lamsler GROUP=g17 $0" >&2
  exit 1
fi

if ! command -v runai >/dev/null 2>&1; then
  echo "ERROR: runai CLI was not found in this shell." >&2
  echo "Run this from the RCP/login environment, not from inside an already running pod." >&2
  exit 127
fi

RUN_DIR="${RUN_DIR:-${1:-}}"
MAX_TOKENS="${MAX_TOKENS:-${2:-4096}}"
MAX_NUM_SAMPLES="${MAX_NUM_SAMPLES:-${3:-1000}}"

if [[ -z "${RUN_DIR}" ]]; then
  echo "ERROR: provide RUN_DIR or pass it as the first argument." >&2
  echo "Example: $0 /scratch/checkpoints/math/20260526-161659 4096 1000" >&2
  exit 2
fi

GPUS="${GPUS:-1}"
NODE="${NODE:-a100-40g}"
SUFFIX="${SUFFIX:-math-all-ckpt-eval}"
JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"

IMAGE="${IMAGE:-ayushkumartarun/course-cs-552-standard:v1}"
REPO_DIR="${REPO_DIR:-/scratch/leo/merged_repo}"
BENCHMARK="${BENCHMARK:-math}"
DATASET="${DATASET:-competitionmath}"
SPLIT="${SPLIT:-full}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
PROMPT_FILE_PATH="${PROMPT_FILE_PATH:-fourneurons/prompts/math.txt}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"

if [[ -n "${EXTRA_EVAL_CMD:-}" ]]; then
  EVAL_CMD="${EXTRA_EVAL_CMD}"
else
  EVAL_CMD="python -m fourneurons.evaluation.eval_all_checkpoints ${RUN_DIR} --benchmark ${BENCHMARK} --dataset ${DATASET} --split ${SPLIT} --num_generations ${NUM_GENERATIONS} --max_tokens ${MAX_TOKENS} --max_num_samples ${MAX_NUM_SAMPLES} --prompt_file_path ${PROMPT_FILE_PATH}"
  if [[ -n "${OUTPUT_PREFIX}" ]]; then
    EVAL_CMD="${EVAL_CMD} --output_prefix ${OUTPUT_PREFIX}"
  fi
  EVAL_CMD="${EVAL_CMD} ${EXTRA_EVAL_ARGS}"
fi

SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

printf '>>> Submitting %s (non-interactive, %s GPU)\n' "${JOB_NAME}" "${GPUS}"
printf '>>> Repo: %s\n' "${REPO_DIR}"
printf '>>> Run dir: %s\n' "${RUN_DIR}"
printf '>>> Command: %s\n' "${EVAL_CMD}"

runai submit \
  --name "${JOB_NAME}" \
  -p "${PROJECT}" \
  --image "${IMAGE}" \
  --gpu "${GPUS}" \
  --large-shm \
  --node-pools "${NODE}" \
  --working-dir /scratch \
  --environment HF_HOME=/scratch/hf_cache \
  --environment HF_HUB_ENABLE_HF_TRANSFER=1 \
  --environment HF_TOKEN="${HF_TOKEN:-}" \
  --environment HUGGING_FACE_HUB_TOKEN="${HF_TOKEN:-}" \
  --environment WANDB_KEY="${WANDB_KEY:-}" \
  --environment WANDB_DIR=/scratch/wandb \
  --environment PYTORCH_ALLOC_CONF=expandable_segments:True \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro" \
  --existing-pvc "claimname=${SHARED_RW_PVC},path=/shared-rw" \
  --command -- /bin/bash -lc "\
    set -euo pipefail && \
    mkdir -p /scratch/hf_cache /scratch/wandb /scratch/checkpoints/math && \
    ln -sf \"\$(command -v python3)\" /usr/local/bin/python && \
    cd \"${REPO_DIR}\" && \
    ${EVAL_CMD}"

cat <<EOM

>>> Job submitted: ${JOB_NAME}

Watch it start:  runai describe job ${JOB_NAME} -p ${PROJECT}
Stream logs:     runai logs -f ${JOB_NAME} -p ${PROJECT}
Shell in pod:    runai bash ${JOB_NAME} -p ${PROJECT}
Stop the job:    runai delete job ${JOB_NAME} -p ${PROJECT}

This job is non-interactive: it evaluates each checkpoint and exits when the
last checkpoint finishes or a command fails.
EOM
