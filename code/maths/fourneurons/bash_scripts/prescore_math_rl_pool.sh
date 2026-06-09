#!/usr/bin/env bash
# CS-552 - submit a non-interactive RL prompt-pool prescore job on Run:AI.
#
# Usage:
#   GASPAR=<gaspar> GROUP=<gXX> ./fourneurons/bash_scripts/prescore_math_rl_pool.sh
#
# Defaults to:
#   python -m fourneurons.scripts.prescore_math_rl_pool --dataset rl_prompt_pool_40k --run_name rl_pool_40k_prescore_mixed_ckpt4458_tok16k_n8
#
# Optional overrides:
#   REPO_DIR=/scratch/leo/merged_repo
#   IMAGE=ayushkumartarun/course-cs-552-standard:v1
#   DATASET=rl_prompt_pool_40k
#   RUN_NAME=rl_pool_40k_prescore_mixed_ckpt4458_tok16k_n8
#   GENERATION_BATCH_SIZE=16
#   EXTRA_PRESCORE_ARGS="--no_resume_generation"
#   EXTRA_PRESCORE_ARGS="--skip_generation"  # score existing generations only
#   EXTRA_PRESCORE_CMD="python -m fourneurons.scripts.prescore_math_rl_pool ..."

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

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

GPUS="${GPUS:-1}"
NODE="${NODE:-a100-40g}"
SUFFIX="${SUFFIX:-${1:-math-rl-prescore}}"
JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"

IMAGE="${IMAGE:-ayushkumartarun/course-cs-552-standard:v1}"
REPO_DIR="${REPO_DIR:-/scratch/leo/merged_repo}"
DATASET="${DATASET:-rl_prompt_pool_40k}"
RUN_NAME="${RUN_NAME:-rl_pool_40k_prescore_mixed_ckpt4458_tok16k_n8}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-16}"
EXTRA_PRESCORE_ARGS="${EXTRA_PRESCORE_ARGS:-}"

if [[ -n "${EXTRA_PRESCORE_CMD:-}" ]]; then
  PRESCORE_CMD="${EXTRA_PRESCORE_CMD}"
else
  PRESCORE_CMD="python -m fourneurons.scripts.prescore_math_rl_pool --dataset ${DATASET} --run_name ${RUN_NAME} --generation_batch_size ${GENERATION_BATCH_SIZE}"
  if [[ -n "${EXTRA_PRESCORE_ARGS}" ]]; then
    PRESCORE_CMD="${PRESCORE_CMD} ${EXTRA_PRESCORE_ARGS}"
  fi
fi

SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

HF_TOKEN="${HF_TOKEN:-}"
WANDB_KEY="${WANDB_KEY:-${WANDB_API_KEY:-}}"
WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_KEY}}"

printf '>>> Submitting %s (non-interactive, %s GPU)\n' "${JOB_NAME}" "${GPUS}"
printf '>>> Repo: %s\n' "${REPO_DIR}"
printf '>>> Dataset: %s\n' "${DATASET}"
printf '>>> Run name: %s\n' "${RUN_NAME}"
printf '>>> Generation batch size: %s\n' "${GENERATION_BATCH_SIZE}"
printf '>>> Command: %s\n' "${PRESCORE_CMD}"

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
  --environment HF_TOKEN="${HF_TOKEN}" \
  --environment HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}" \
  --environment WANDB_KEY="${WANDB_KEY}" \
  --environment WANDB_API_KEY="${WANDB_API_KEY}" \
  --environment WANDB_DIR=/scratch/wandb \
  --environment PYTORCH_ALLOC_CONF=expandable_segments:True \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro" \
  --existing-pvc "claimname=${SHARED_RW_PVC},path=/shared-rw" \
  --command -- /bin/bash -lc "\
    set -euo pipefail && \
    mkdir -p /scratch/hf_cache /scratch/wandb /scratch/results && \
    ln -sf \"\$(command -v python3)\" /usr/local/bin/python && \
    cd \"${REPO_DIR}\" && \
    ${PRESCORE_CMD}"

cat <<EOM

>>> Job submitted: ${JOB_NAME}

Watch it start:  runai describe job ${JOB_NAME} -p ${PROJECT}
Stream logs:     runai logs -f ${JOB_NAME} -p ${PROJECT}
Shell in pod:    runai bash ${JOB_NAME} -p ${PROJECT}
Stop the job:    runai delete job ${JOB_NAME} -p ${PROJECT}

This job is non-interactive: it generates the RL prompt pool in batches, resumes
from the existing generations prefix on restart, scores the full file, then
exits when prescoring finishes or fails.
EOM
