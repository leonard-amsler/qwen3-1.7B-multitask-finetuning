#!/usr/bin/env bash
# CS-552 - submit non-interactive baseline math evaluations on Run:AI.
#
# Submits three independent jobs so Run:AI can run them in parallel:
#   1. competitionmath/full with fourneurons/prompts/math.txt
#   2. competitionmath/full with fourneurons/prompts/math_simple.txt
#   3. competitionmath/full without a system prompt
#
# Usage:
#   GASPAR=<gaspar> GROUP=<gXX> ./fourneurons/bash_scripts/eval_baseline.sh
#
# Optional overrides:
#   REPO_DIR=/scratch/leo/merged_repo
#   IMAGE=ayushkumartarun/course-cs-552-standard:v1
#   BENCHMARK=math
#   DATASET=competitionmath
#   SPLIT=full
#   NUM_GENERATIONS=8
#   MAX_NUM_SAMPLES=1000
#   PROMPT_FILE_PATH=fourneurons/prompts/math.txt
#   PROMPT_RUN_NAME=baseline_evaluation_with_prompt
#   SIMPLE_PROMPT_FILE_PATH=fourneurons/prompts/math_simple.txt
#   SIMPLE_PROMPT_RUN_NAME=baseline_evaluation_simple_prompt
#   NO_PROMPT_RUN_NAME=baseline_evaluation_no_prompt
#   EXTRA_EVAL_ARGS="--max_tokens 4096"

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
SUFFIX="${1:-baseeval}"
TIMESTAMP="$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"

IMAGE="${IMAGE:-ayushkumartarun/course-cs-552-standard:v1}"
REPO_DIR="${REPO_DIR:-/scratch/leo/merged_repo}"
BENCHMARK="${BENCHMARK:-math}"
DATASET="${DATASET:-competitionmath}"
SPLIT="${SPLIT:-full}"
NUM_GENERATIONS="${NUM_GENERATIONS:-8}"
MAX_NUM_SAMPLES="${MAX_NUM_SAMPLES:-1000}"
PROMPT_FILE_PATH="${PROMPT_FILE_PATH:-fourneurons/prompts/math.txt}"
PROMPT_RUN_NAME="${PROMPT_RUN_NAME:-baseline_evaluation_with_prompt}"
SIMPLE_PROMPT_FILE_PATH="${SIMPLE_PROMPT_FILE_PATH:-fourneurons/prompts/math_simple.txt}"
SIMPLE_PROMPT_RUN_NAME="${SIMPLE_PROMPT_RUN_NAME:-baseline_evaluation_simple_prompt}"
NO_PROMPT_RUN_NAME="${NO_PROMPT_RUN_NAME:-baseline_evaluation_no_prompt}"
EXTRA_EVAL_ARGS="${EXTRA_EVAL_ARGS:-}"

EVAL_WITH_PROMPT_CMD="python fourneurons/evaluation/eval.py ${BENCHMARK} ${DATASET} ${SPLIT} ${PROMPT_RUN_NAME} --base --prompt_file_path ${PROMPT_FILE_PATH} --num_generations ${NUM_GENERATIONS} --max_num_samples ${MAX_NUM_SAMPLES} ${EXTRA_EVAL_ARGS}"
EVAL_SIMPLE_PROMPT_CMD="python fourneurons/evaluation/eval.py ${BENCHMARK} ${DATASET} ${SPLIT} ${SIMPLE_PROMPT_RUN_NAME} --base --prompt_file_path ${SIMPLE_PROMPT_FILE_PATH} --num_generations ${NUM_GENERATIONS} --max_num_samples ${MAX_NUM_SAMPLES} ${EXTRA_EVAL_ARGS}"
EVAL_NO_PROMPT_CMD="python fourneurons/evaluation/eval.py ${BENCHMARK} ${DATASET} ${SPLIT} ${NO_PROMPT_RUN_NAME} --base --num_generations ${NUM_GENERATIONS} --max_num_samples ${MAX_NUM_SAMPLES} ${EXTRA_EVAL_ARGS}"

SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

HF_TOKEN="${HF_TOKEN:-}"
WANDB_KEY="${WANDB_KEY:-}"

submit_eval_job() {
  local job_name="$1"
  local eval_cmd="$2"

  printf '>>> Submitting %s (non-interactive, %s GPU)\n' "${job_name}" "${GPUS}"
  printf '>>> Repo: %s\n' "${REPO_DIR}"
  printf '>>> Command: %s\n' "${eval_cmd}"

  runai submit \
    --name "${job_name}" \
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
      mkdir -p /scratch/hf_cache /scratch/wandb /scratch/results/math && \
      ln -sf \"\$(command -v python3)\" /usr/local/bin/python && \
      cd \"${REPO_DIR}\" && \
      ${eval_cmd}"
}

PROMPT_JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-p-${TIMESTAMP}"
SIMPLE_PROMPT_JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-sp-${TIMESTAMP}"
NO_PROMPT_JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-np-${TIMESTAMP}"

for job_name in "${PROMPT_JOB_NAME}" "${SIMPLE_PROMPT_JOB_NAME}" "${NO_PROMPT_JOB_NAME}"; do
  if ((${#job_name} > 55)); then
    echo "ERROR: Run:AI workload name is too long (${#job_name} chars, max 55): ${job_name}" >&2
    echo "Use a shorter suffix, e.g. $0 baseeval" >&2
    exit 2
  fi
done

submit_eval_job "${PROMPT_JOB_NAME}" "${EVAL_WITH_PROMPT_CMD}"
submit_eval_job "${SIMPLE_PROMPT_JOB_NAME}" "${EVAL_SIMPLE_PROMPT_CMD}"
submit_eval_job "${NO_PROMPT_JOB_NAME}" "${EVAL_NO_PROMPT_CMD}"

cat <<EOM

>>> Jobs submitted:
    ${PROMPT_JOB_NAME}
    ${SIMPLE_PROMPT_JOB_NAME}
    ${NO_PROMPT_JOB_NAME}

Watch them start:
  runai describe job ${PROMPT_JOB_NAME} -p ${PROJECT}
  runai describe job ${SIMPLE_PROMPT_JOB_NAME} -p ${PROJECT}
  runai describe job ${NO_PROMPT_JOB_NAME} -p ${PROJECT}

Stream logs:
  runai logs -f ${PROMPT_JOB_NAME} -p ${PROJECT}
  runai logs -f ${SIMPLE_PROMPT_JOB_NAME} -p ${PROJECT}
  runai logs -f ${NO_PROMPT_JOB_NAME} -p ${PROJECT}

Stop jobs:
  runai delete job ${PROMPT_JOB_NAME} -p ${PROJECT}
  runai delete job ${SIMPLE_PROMPT_JOB_NAME} -p ${PROJECT}
  runai delete job ${NO_PROMPT_JOB_NAME} -p ${PROJECT}

These jobs are independent and non-interactive, so Run:AI can schedule them in
parallel when resources are available.
EOM
