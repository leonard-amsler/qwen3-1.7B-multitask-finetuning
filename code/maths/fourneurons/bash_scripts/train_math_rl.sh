#!/usr/bin/env bash
# CS-552 - submit a non-interactive math GRPO/RL job on Run:AI.
#
# Usage:
#   GASPAR=<gaspar> GROUP=<gXX> ./fourneurons/bash_scripts/train_math_rl.sh
#
# Defaults to a first-1k smoke run:
#   1. Slice the first 1k existing RL-pool generations and prompts.
#   2. Score them with evaluate.score.
#   3. Select frontier rows with 1-7 correct out of 8.
#   4. Run fourneurons.scripts.train_math_rl with vLLM enabled.
#
# Optional overrides:
#   REPO_DIR=/scratch/leo/merged_repo
#   IMAGE=ayushkumartarun/course-cs-552-standard:v1
#   FIRST_N=1000
#   INCLUDE_CORRECT=1-7
#   BUILD_FIRST1K_FRONTIER=0  # skip setup and use MATH_RL_TRAIN_FILE as-is
#   MATH_RL_TRAIN_FILE=/scratch/data/math/rl_frontier_40k_first1k/splits/rl_frontier_40k_first1k_train.jsonl
#   MATH_RL_MAX_TRAINING_SAMPLES=380
#   MATH_RL_MAX_VALIDATION_SAMPLES=0
#   MATH_RL_MAX_STEPS=20
#   MATH_RL_MAX_COMPLETION_LENGTH=16384
#   MATH_RL_MAX_CONTEXT_LENGTH=20000
#   MATH_RL_NUM_GENERATIONS_EVAL=2
#   MATH_RL_PER_DEVICE_EVAL_BATCH_SIZE=2
#   MATH_RL_VLLM_GPU_MEMORY_UTILIZATION=0.3
#   MATH_RL_VLLM_MAX_MODEL_LENGTH=20000
#   MATH_RL_KL_BETA=0
#   MATH_RL_TRAINER_BF16=0
#   MATH_RL_VLLM_IMPORTANCE_SAMPLING_CORRECTION=0
#   MATH_RL_VLLM_ENABLE_SLEEP_MODE=1
#   OUTPUT_DIR=/scratch/checkpoints/math/<run>
#   RESUME_FROM_CHECKPOINT=/scratch/checkpoints/math/<run>/checkpoint-<step>

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
SUFFIX="${SUFFIX:-${1:-math-rl-smoke}}"
JOB_NAME="cs552-${GASPAR}-${GROUP}-${SUFFIX}-$(date +%H%M%S)"
PROJECT="course-cs-552-${GASPAR}"

IMAGE="${IMAGE:-ayushkumartarun/course-cs-552-standard:v1}"
REPO_DIR="${REPO_DIR:-/scratch/leo/merged_repo}"

SOURCE_GENS_FILE="${SOURCE_GENS_FILE:-/scratch/results/math/rl_prompt_pool_40k/rl_pool_40k_prescore_mixed_ckpt4458_tok16k_n8/train_gens.jsonl}"
SOURCE_POOL_FILE="${SOURCE_POOL_FILE:-/scratch/data/math/rl_prompt_pool_40k/splits/rl_prompt_pool_40k_train.jsonl}"
FIRST_N="${FIRST_N:-1000}"
INCLUDE_CORRECT="${INCLUDE_CORRECT:-1-7}"
BUILD_FIRST1K_FRONTIER="${BUILD_FIRST1K_FRONTIER:-1}"

FIRST1K_DIR="${FIRST1K_DIR:-/scratch/data/math/rl_frontier_40k_first1k}"
FIRST1K_SPLITS_DIR="${FIRST1K_DIR}/splits"
FIRST1K_POOL_FILE="${FIRST1K_POOL_FILE:-${FIRST1K_SPLITS_DIR}/rl_pool_40k_first1k_train.jsonl}"
FIRST1K_GENS_FILE="${FIRST1K_GENS_FILE:-${FIRST1K_SPLITS_DIR}/rl_pool_40k_first1k_gens.jsonl}"
FIRST1K_SCORED_FILE="${FIRST1K_SCORED_FILE:-${FIRST1K_SPLITS_DIR}/rl_pool_40k_first1k_scored.json}"
MATH_RL_TRAIN_FILE="${MATH_RL_TRAIN_FILE:-${FIRST1K_SPLITS_DIR}/rl_frontier_40k_first1k_train.jsonl}"

MATH_RL_MAX_TRAINING_SAMPLES="${MATH_RL_MAX_TRAINING_SAMPLES:-380}"
MATH_RL_MAX_VALIDATION_SAMPLES="${MATH_RL_MAX_VALIDATION_SAMPLES:-0}"
MATH_RL_MAX_STEPS="${MATH_RL_MAX_STEPS:-20}"
MATH_RL_MAX_COMPLETION_LENGTH="${MATH_RL_MAX_COMPLETION_LENGTH:-16384}"
MATH_RL_MAX_CONTEXT_LENGTH="${MATH_RL_MAX_CONTEXT_LENGTH:-20000}"
MATH_RL_NUM_GENERATIONS_EVAL="${MATH_RL_NUM_GENERATIONS_EVAL:-2}"
MATH_RL_PER_DEVICE_EVAL_BATCH_SIZE="${MATH_RL_PER_DEVICE_EVAL_BATCH_SIZE:-2}"
MATH_RL_USE_VLLM="${MATH_RL_USE_VLLM:-1}"
MATH_RL_VLLM_GPU_MEMORY_UTILIZATION="${MATH_RL_VLLM_GPU_MEMORY_UTILIZATION:-0.3}"
MATH_RL_VLLM_MODE="${MATH_RL_VLLM_MODE:-colocate}"
MATH_RL_VLLM_MAX_MODEL_LENGTH="${MATH_RL_VLLM_MAX_MODEL_LENGTH:-${MATH_RL_MAX_CONTEXT_LENGTH}}"
MATH_RL_KL_BETA="${MATH_RL_KL_BETA:-0}"
MATH_RL_TRAINER_BF16="${MATH_RL_TRAINER_BF16:-0}"
MATH_RL_VLLM_IMPORTANCE_SAMPLING_CORRECTION="${MATH_RL_VLLM_IMPORTANCE_SAMPLING_CORRECTION:-0}"
MATH_RL_VLLM_ENABLE_SLEEP_MODE="${MATH_RL_VLLM_ENABLE_SLEEP_MODE:-1}"
MATH_RL_MASK_TRUNCATED_COMPLETIONS="${MATH_RL_MASK_TRUNCATED_COMPLETIONS:-0}"

SCRATCH_PVC="course-cs-552-scratch-${GROUP}"
SHARED_RO_PVC="course-cs-552-shared-ro"
SHARED_RW_PVC="course-cs-552-shared-rw"

HF_TOKEN="${HF_TOKEN:-}"
WANDB_KEY="${WANDB_KEY:-${WANDB_API_KEY:-}}"
WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_KEY}}"
WANDB_PROJECT="${WANDB_PROJECT:-math-rl}"
FOURNEURONS_RUN_NAME="${FOURNEURONS_RUN_NAME:-qwen3-1.7b-lora-math-rl-smoke}"
FOURNEURONS_RUN_ID="${FOURNEURONS_RUN_ID:-}"
WANDB_NAME="${WANDB_NAME:-}"
WANDB_RUN_ID="${WANDB_RUN_ID:-}"
WANDB_RESUME="${WANDB_RESUME:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
RESUME_FROM_CHECKPOINT="${RESUME_FROM_CHECKPOINT:-}"

if [[ -n "${WANDB_KEY}" ]]; then
  WANDB_AUTH_SOURCE="WANDB_KEY environment variable"
else
  WANDB_AUTH_SOURCE="not provided"
fi

printf '>>> Submitting %s (non-interactive, %s GPU)\n' "${JOB_NAME}" "${GPUS}"
printf '>>> Repo: %s\n' "${REPO_DIR}"
printf '>>> Train file: %s\n' "${MATH_RL_TRAIN_FILE}"
printf '>>> Build first-1k frontier: %s\n' "${BUILD_FIRST1K_FRONTIER}"
printf '>>> RL steps: %s\n' "${MATH_RL_MAX_STEPS}"
printf '>>> vLLM: %s (%s, gpu_memory_utilization=%s)\n' \
  "${MATH_RL_USE_VLLM}" "${MATH_RL_VLLM_MODE}" "${MATH_RL_VLLM_GPU_MEMORY_UTILIZATION}"
printf '>>> W&B auth: %s\n' "${WANDB_AUTH_SOURCE}"
if [[ -n "${OUTPUT_DIR}" ]]; then
  printf '>>> Output directory: %s\n' "${OUTPUT_DIR}"
fi
if [[ -n "${RESUME_FROM_CHECKPOINT}" ]]; then
  printf '>>> Resume checkpoint: %s\n' "${RESUME_FROM_CHECKPOINT}"
fi
if [[ -n "${WANDB_RUN_ID}" ]]; then
  printf '>>> W&B resume id: %s (%s)\n' "${WANDB_RUN_ID}" "${WANDB_RESUME:-allow}"
fi

RUNAI_OPTIONAL_ENV=()
[[ -n "${FOURNEURONS_RUN_NAME}" ]] && RUNAI_OPTIONAL_ENV+=(--environment "FOURNEURONS_RUN_NAME=${FOURNEURONS_RUN_NAME}")
[[ -n "${FOURNEURONS_RUN_ID}" ]] && RUNAI_OPTIONAL_ENV+=(--environment "FOURNEURONS_RUN_ID=${FOURNEURONS_RUN_ID}")
[[ -n "${WANDB_NAME}" ]] && RUNAI_OPTIONAL_ENV+=(--environment "WANDB_NAME=${WANDB_NAME}")
[[ -n "${WANDB_RUN_ID}" ]] && RUNAI_OPTIONAL_ENV+=(--environment "WANDB_RUN_ID=${WANDB_RUN_ID}")
[[ -n "${WANDB_RESUME}" ]] && RUNAI_OPTIONAL_ENV+=(--environment "WANDB_RESUME=${WANDB_RESUME}")
[[ -n "${OUTPUT_DIR}" ]] && RUNAI_OPTIONAL_ENV+=(--environment "OUTPUT_DIR=${OUTPUT_DIR}")
[[ -n "${RESUME_FROM_CHECKPOINT}" ]] && RUNAI_OPTIONAL_ENV+=(--environment "RESUME_FROM_CHECKPOINT=${RESUME_FROM_CHECKPOINT}")

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
  --environment WANDB_PROJECT="${WANDB_PROJECT}" \
  --environment MATH_RL_TRAIN_FILE="${MATH_RL_TRAIN_FILE}" \
  --environment MATH_RL_MAX_TRAINING_SAMPLES="${MATH_RL_MAX_TRAINING_SAMPLES}" \
  --environment MATH_RL_MAX_VALIDATION_SAMPLES="${MATH_RL_MAX_VALIDATION_SAMPLES}" \
  --environment MATH_RL_MAX_STEPS="${MATH_RL_MAX_STEPS}" \
  --environment MATH_RL_MAX_COMPLETION_LENGTH="${MATH_RL_MAX_COMPLETION_LENGTH}" \
  --environment MATH_RL_MAX_CONTEXT_LENGTH="${MATH_RL_MAX_CONTEXT_LENGTH}" \
  --environment MATH_RL_NUM_GENERATIONS_EVAL="${MATH_RL_NUM_GENERATIONS_EVAL}" \
  --environment MATH_RL_PER_DEVICE_EVAL_BATCH_SIZE="${MATH_RL_PER_DEVICE_EVAL_BATCH_SIZE}" \
  --environment MATH_RL_USE_VLLM="${MATH_RL_USE_VLLM}" \
  --environment MATH_RL_KL_BETA="${MATH_RL_KL_BETA}" \
  --environment MATH_RL_TRAINER_BF16="${MATH_RL_TRAINER_BF16}" \
  --environment MATH_RL_VLLM_MODE="${MATH_RL_VLLM_MODE}" \
  --environment MATH_RL_VLLM_MAX_MODEL_LENGTH="${MATH_RL_VLLM_MAX_MODEL_LENGTH}" \
  --environment MATH_RL_VLLM_IMPORTANCE_SAMPLING_CORRECTION="${MATH_RL_VLLM_IMPORTANCE_SAMPLING_CORRECTION}" \
  --environment MATH_RL_VLLM_ENABLE_SLEEP_MODE="${MATH_RL_VLLM_ENABLE_SLEEP_MODE}" \
  --environment MATH_RL_VLLM_GPU_MEMORY_UTILIZATION="${MATH_RL_VLLM_GPU_MEMORY_UTILIZATION}" \
  --environment MATH_RL_MASK_TRUNCATED_COMPLETIONS="${MATH_RL_MASK_TRUNCATED_COMPLETIONS}" \
  "${RUNAI_OPTIONAL_ENV[@]}" \
  --environment PYTORCH_ALLOC_CONF=expandable_segments:True \
  --existing-pvc "claimname=${SCRATCH_PVC},path=/scratch" \
  --existing-pvc "claimname=${SHARED_RO_PVC},path=/shared-ro" \
  --existing-pvc "claimname=${SHARED_RW_PVC},path=/shared-rw" \
  --command -- /bin/bash -lc "\
    set -euo pipefail && \
    mkdir -p /scratch/hf_cache /scratch/wandb /scratch/checkpoints/math '${FIRST1K_SPLITS_DIR}' && \
    ln -sf \"\$(command -v python3)\" /usr/local/bin/python && \
    cd '${REPO_DIR}' && \
    if [[ '${BUILD_FIRST1K_FRONTIER}' == '1' ]]; then \
      sed -n '1,${FIRST_N}p' '${SOURCE_POOL_FILE}' > '${FIRST1K_POOL_FILE}' && \
      sed -n '1,${FIRST_N}p' '${SOURCE_GENS_FILE}' > '${FIRST1K_GENS_FILE}' && \
      python -m evaluate.score --generations '${FIRST1K_GENS_FILE}' --benchmark math --output '${FIRST1K_SCORED_FILE}' && \
      python -m fourneurons.data.select_math_rl_frontier \
        --pool '${FIRST1K_POOL_FILE}' \
        --scored '${FIRST1K_SCORED_FILE}' \
        --output '${MATH_RL_TRAIN_FILE}' \
        --include-correct '${INCLUDE_CORRECT}'; \
    fi && \
    python -m fourneurons.scripts.train_math_rl"

cat <<EOM

>>> Job submitted: ${JOB_NAME}

Watch it start:  runai describe job ${JOB_NAME} -p ${PROJECT}
Stream logs:     runai logs -f ${JOB_NAME} -p ${PROJECT}
Shell in pod:    runai bash ${JOB_NAME} -p ${PROJECT}
Stop the job:    runai delete job ${JOB_NAME} -p ${PROJECT}

This job is non-interactive: it prepares the first-1k frontier split under
/scratch, starts math GRPO training with vLLM, and exits when training finishes
or fails.
EOM
