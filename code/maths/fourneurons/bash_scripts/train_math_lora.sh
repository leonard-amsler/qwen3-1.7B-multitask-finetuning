#!/usr/bin/env bash
# CS-552 — submit a non-interactive math LoRA SFT job on Run:AI.
#
# Usage:
#   GASPAR=<gaspar> GROUP=<gXX> ./fourneurons/bash_scripts/train_math_lora.sh
#
# Optional overrides:
#   REPO_DIR=/scratch/leo/merged_repo
#   IMAGE=ayushkumartarun/course-cs-552-standard:v1
#   HF_TOKEN=...
#   WANDB_KEY=...              # train_math.py currently reads WANDB_KEY
#   WANDB_API_KEY=...          # also accepted; copied into WANDB_KEY if needed
#   EXTRA_TRAIN_CMD='python -m fourneurons.scripts.train_math'

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

set -euo pipefail

GASPAR="lamsler"              # <-- YOUR GASPAR EPFL username.
GROUP="g17"                  # <-- YOUR TEAM, e.g. g07.

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

GPUS=1
NODE="${NODE:-a100-40g}"
SUFFIX="${1:-math-lora-train}"
JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"

IMAGE="${IMAGE:-ayushkumartarun/course-cs-552-standard:v1}"
REPO_DIR="${REPO_DIR:-/scratch/leo/merged_repo}"
TRAIN_CMD="${EXTRA_TRAIN_CMD:-python -m fourneurons.scripts.train_math}"

SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

HF_TOKEN="${HF_TOKEN:-}"
WANDB_KEY="${WANDB_KEY:-}"

if [[ -n "${WANDB_KEY}" ]]; then
  WANDB_AUTH_SOURCE="WANDB_KEY environment variable"
else
  WANDB_AUTH_SOURCE="not provided"
fi

printf '>>> Submitting %s (non-interactive, 1 GPU)\n' "${JOB_NAME}"
printf '>>> Repo: %s\n' "${REPO_DIR}"
printf '>>> Command: %s\n' "${TRAIN_CMD}"
printf '>>> W&B auth: %s\n' "${WANDB_AUTH_SOURCE}"

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
    ${TRAIN_CMD}"

cat <<EOM

>>> Job submitted: ${JOB_NAME}

Watch it start:  runai describe job ${JOB_NAME} -p ${PROJECT}
Stream logs:     runai logs -f ${JOB_NAME} -p ${PROJECT}
Shell in pod:    runai bash ${JOB_NAME} -p ${PROJECT}
Stop the job:    runai delete job ${JOB_NAME} -p ${PROJECT}

This job is non-interactive: it runs the training command directly and exits
when training finishes or fails.
EOM
