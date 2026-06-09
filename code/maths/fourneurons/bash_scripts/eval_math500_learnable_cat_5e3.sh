#!/usr/bin/env bash
# CS-552 - submit a non-interactive Math500 eval job on Run:AI.
#
# Usage:
#   GASPAR=<gaspar> GROUP=<gXX> ./fourneurons/bash_scripts/eval_math500_learnable_cat_5e3.sh
#
# Defaults to evaluating:
#   /scratch/checkpoints/group_model/learnable_cat_5e3
#
# on:
#   /scratch/data/math/math500/splits/math500_full.jsonl
#
# Optional overrides:
#   REPO_DIR=/scratch/leo/merged_repo
#   IMAGE=ayushkumartarun/course-cs-552-standard:v1
#   MODEL_DIR=/scratch/checkpoints/group_model/learnable_cat_5e3
#   PROMPT_FILE_PATH=/scratch/nico/standard-project-m2-4neurons/fourneurons/prompts/sp_group_think.txt
#   RUN_NAME=learnable_cat_5e3_math500_full_tok16k_n8_sp_group_think
#   GENERATION_BATCH_SIZE=16

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
SUFFIX="${SUFFIX:-${1:-math500-lcat5e3}}"
JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"

IMAGE="${IMAGE:-ayushkumartarun/course-cs-552-standard:v1}"
REPO_DIR="${REPO_DIR:-/scratch/leo/merged_repo}"
BENCHMARK="${BENCHMARK:-math}"
DATASET="${DATASET:-math500}"
SPLIT="${SPLIT:-full}"
MODEL_DIR="${MODEL_DIR:-/scratch/checkpoints/group_model/learnable_cat_5e3}"
PROMPT_FILE_PATH="${PROMPT_FILE_PATH:-/scratch/nico/standard-project-m2-4neurons/fourneurons/prompts/sp_group_think.txt}"
RUN_NAME="${RUN_NAME:-learnable_cat_5e3_math500_full_tok16k_n8_sp_group_think}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
MAX_TOKENS="${MAX_TOKENS:-16384}"
GENERATION_BATCH_SIZE="${GENERATION_BATCH_SIZE:-16}"

OUTPUT_DIR="/scratch/results/${BENCHMARK}/${DATASET}/${RUN_NAME}"
GENERATIONS_FILE="${OUTPUT_DIR}/${SPLIT}_gens.jsonl"
SCORED_FILE="${OUTPUT_DIR}/${SPLIT}_scored.json"

EVAL_CMD="python -m fourneurons.evaluation.eval ${BENCHMARK} ${DATASET} ${SPLIT} ${RUN_NAME} --checkpoint ${MODEL_DIR} --merged_model_dir ${MODEL_DIR} --prompt_file_path ${PROMPT_FILE_PATH} --num_generations ${NUM_GENERATIONS} --max_tokens ${MAX_TOKENS} --generation_batch_size ${GENERATION_BATCH_SIZE} --resume_generation"
SCORE_CMD="python -m evaluate.score --generations ${GENERATIONS_FILE} --benchmark ${BENCHMARK} --output ${SCORED_FILE}"
JOB_CMD="${EVAL_CMD} && ${SCORE_CMD}"

SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

HF_TOKEN="${HF_TOKEN:-}"
WANDB_KEY="${WANDB_KEY:-${WANDB_API_KEY:-}}"
WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_KEY}}"

if ((${#JOB_NAME} > 55)); then
  echo "ERROR: Run:AI workload name is too long (${#JOB_NAME} chars, max 55): ${JOB_NAME}" >&2
  echo "Use a shorter suffix, e.g. $0 math500" >&2
  exit 2
fi

printf '>>> Submitting %s (non-interactive, %s GPU)\n' "${JOB_NAME}" "${GPUS}"
printf '>>> Repo: %s\n' "${REPO_DIR}"
printf '>>> Model dir: %s\n' "${MODEL_DIR}"
printf '>>> Prompt: %s\n' "${PROMPT_FILE_PATH}"
printf '>>> Dataset: %s/%s/%s\n' "${BENCHMARK}" "${DATASET}" "${SPLIT}"
printf '>>> Run name: %s\n' "${RUN_NAME}"
printf '>>> Max tokens: %s\n' "${MAX_TOKENS}"
printf '>>> Num generations: %s\n' "${NUM_GENERATIONS}"
printf '>>> Generation batch size: %s\n' "${GENERATION_BATCH_SIZE}"
printf '>>> Command: %s\n' "${JOB_CMD}"

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
    ${JOB_CMD}"

cat <<EOM

>>> Job submitted: ${JOB_NAME}

Watch it start:  runai describe job ${JOB_NAME} -p ${PROJECT}
Stream logs:     runai logs -f ${JOB_NAME} -p ${PROJECT}
Shell in pod:    runai bash ${JOB_NAME} -p ${PROJECT}
Stop the job:    runai delete job ${JOB_NAME} -p ${PROJECT}

This job is non-interactive: it generates Math500 completions with the
evaluator defaults for temperature/top-p, scores the generations, then exits.

Generations: ${GENERATIONS_FILE}
Scores:      ${SCORED_FILE}
EOM
