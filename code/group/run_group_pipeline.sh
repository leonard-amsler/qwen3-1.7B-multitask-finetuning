#!/bin/bash

################################################################################
# Group Model Merging & SFT Pipeline
################################################################################
# This script executes the complete group training pipeline:
# 1. Build mixed dataset (combining all domain validation splits)
# 2. Download base model
# 3. Evaluate baseline model
# 4. Merge adapters using TIES, DARE-TIES, and Learnable CAT methods
# 5. Evaluate each merged model
# 6. Train SFT model on mixed data
# 7. Evaluate SFT model
#
# PREREQUISITES:
# This pipeline expects individual domain pipelines to have been run first.
# It requires trained LoRA adapters from:
#   - Safety pipeline (SFT)
#   - Math pipeline
#   - Multilingual pipeline
#   - General Knowledge pipeline
#
# All results organized in: /scratch/results/group/mixed/
# Checkpoints in: /scratch/checkpoints/group_model/
# Run from: code/group/ directory
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

# Paths
CODE_DIR="/scratch/nico/standard-project-m3-4neurons/code/group"
RESULTS_DIR="${CODE_DIR}/results"
SYSTEM_PROMPT="${CODE_DIR}/fourneurons/prompts/sp_group_think.txt"

# ADAPTER PATHS - Update these to match your individual pipeline outputs!
# These are the default paths from the report; adjust if your adapters are elsewhere.
ADAPTER_SAFETY="/scratch/checkpoints/safety/20260518-215854/final"
ADAPTER_MATH="/scratch/checkpoints/math/qwen3-1.7b-lora-math-rl_20260602-064411/checkpoint-200/"
ADAPTER_MULTILINGUAL="/scratch/checkpoints/multilingual/mmmlu_sft3_long2/checkpoint-6875"
ADAPTER_GK="/scratch/checkpoints/gk_v11b/adapter"

# Ensure we're in the right directory
cd "$CODE_DIR" || exit 1

echo -e "${GREEN}===============================================${NC}"
echo -e "${GREEN}Group Model Merging & SFT Pipeline${NC}"
echo -e "${GREEN}===============================================${NC}"
echo "Working directory: $(pwd)"
echo "Results directory: ${RESULTS_DIR}"
echo ""

################################################################################
# Prerequisite Check: Verify adapter paths exist
################################################################################
echo -e "${YELLOW}Checking adapter prerequisites...${NC}"

check_adapter() {
  local name=$1
  local path=$2
  if [ ! -d "$path" ]; then
    echo -e "${RED}✗ Error: $name adapter not found at $path${NC}"
    echo "  Please verify that the individual pipelines have been run."
    echo "  Update ADAPTER_* paths in this script if they differ."
    exit 1
  fi
}

check_adapter "Safety" "$ADAPTER_SAFETY"
check_adapter "Math" "$ADAPTER_MATH"
check_adapter "Multilingual" "$ADAPTER_MULTILINGUAL"
check_adapter "General Knowledge" "$ADAPTER_GK"

echo -e "${GREEN}✓ All adapters found${NC}\n"

################################################################################
# Step 1: Build mixed dataset
################################################################################
echo -e "${YELLOW}[1/7] Building mixed validation/training datasets...${NC}"
python -c "from fourneurons.data.group import build_eval_dataset; build_eval_dataset(n_total=1500, split_name='val')"
echo -e "${GREEN}✓ Mixed datasets created${NC}\n"

################################################################################
# Step 2: Download base model
################################################################################
echo -e "${YELLOW}[2/7] Downloading base model (Qwen3-1.7B)...${NC}"
python -m fourneurons.scripts.download_base_model
echo -e "${GREEN}✓ Base model downloaded${NC}\n"

################################################################################
# Step 3: Evaluate baseline model
################################################################################
echo -e "${YELLOW}[3/7] Evaluating baseline model on mixed dataset...${NC}"

echo "  3a. Generate predictions (no system prompt)"
python -m fourneurons.evaluation.eval \
  group mixed val baseline_nosp \
  --base

echo "  3b. Generate predictions (with system prompt)"
python -m fourneurons.evaluation.eval \
  group mixed val baseline_sp \
  --base \
  --prompt_file_path "$SYSTEM_PROMPT"

echo "  3c. Score baseline results"
python -m evaluate.score_wandb \
  --generations /scratch/results/group/mixed/baseline_nosp/val_gens.jsonl \
  --benchmark group \
  --output /scratch/results/group/mixed/baseline_nosp/val_scored.json \
  --run_name baseline_nosp_scoring

python -m evaluate.score_wandb \
  --generations /scratch/results/group/mixed/baseline_sp/val_gens.jsonl \
  --benchmark group \
  --output /scratch/results/group/mixed/baseline_sp/val_scored.json \
  --run_name baseline_sp_scoring

echo -e "${GREEN}✓ Baseline evaluation complete${NC}\n"

################################################################################
# Step 4: Merge adapters using TIES, DARE-TIES, and Learnable CAT
################################################################################
echo -e "${YELLOW}[4/7] Merging adapters using three methods...${NC}"

echo "  4a. TIES merging (equal weights, density=0.7)"
python -m fourneurons.scripts.merge_ties \
  --adapters "$ADAPTER_SAFETY" "$ADAPTER_MATH" "$ADAPTER_MULTILINGUAL" "$ADAPTER_GK"

echo "  4b. DARE-TIES merging (density=0.7)"
python -m fourneurons.scripts.merge_dare \
  0.7 \
  --adapters "$ADAPTER_SAFETY" "$ADAPTER_MATH" "$ADAPTER_MULTILINGUAL" "$ADAPTER_GK"

echo "  4c. Learnable CAT merging (learns per-adapter weights)"
python -m fourneurons.scripts.merge_cat_learnable \
  --adapters "$ADAPTER_SAFETY" "$ADAPTER_MATH" "$ADAPTER_MULTILINGUAL" "$ADAPTER_GK"

echo -e "${GREEN}✓ All merging methods complete${NC}\n"

################################################################################
# Step 5: Evaluate merged models
################################################################################
echo -e "${YELLOW}[5/7] Evaluating merged models...${NC}"

for method in ties dare_merged_density0.7 learnable_cat_5e3; do
  echo "  Evaluating: $method"
  
  MERGED_PATH="/scratch/checkpoints/group_model/$method"
  
  python -m fourneurons.evaluation.eval \
    group mixed val "merged_$method" \
    --checkpoint "$MERGED_PATH" \
    --merged
  
  python -m evaluate.score_wandb \
    --generations /scratch/results/group/mixed/merged_${method}/val_gens.jsonl \
    --benchmark group \
    --output /scratch/results/group/mixed/merged_${method}/val_scored.json \
    --run_name "merged_${method}_scoring"
done

echo -e "${GREEN}✓ Merged model evaluation complete${NC}\n"

################################################################################
# Step 6: Train SFT model on mixed data
################################################################################
echo -e "${YELLOW}[6/7] Training SFT model on mixed dataset...${NC}"
echo "  Using all domain data, 6 epochs, LoRA fine-tuning."
python -m fourneurons.scripts.train_group \
  --run_name group_sft_mixed \
  --epochs 6 \
  --single_dataset_batches

echo -e "${GREEN}✓ SFT training complete${NC}\n"

################################################################################
# Step 7: Evaluate SFT model
################################################################################
echo -e "${YELLOW}[7/7] Evaluating SFT model on mixed dataset...${NC}"

# Find the latest SFT checkpoint (final epoch)
SFT_CHECKPOINT=$(find /scratch/checkpoints/group -maxdepth 1 -type d -name "group_sft_mixed" | head -1)
if [ ! -d "$SFT_CHECKPOINT/checkpoint-"* ]; then
  SFT_CHECKPOINT=$(find /scratch/checkpoints/group -maxdepth 1 -type d -printf '%T@ %p\n' | grep "group_sft" | sort -rn | head -1 | cut -d' ' -f2-)
fi

if [ -d "$SFT_CHECKPOINT" ]; then
  # Use the final checkpoint (last epoch)
  SFT_FINAL_CHECKPOINT=$(find "$SFT_CHECKPOINT" -name "checkpoint-*" -type d | sort -V | tail -1)
  if [ -z "$SFT_FINAL_CHECKPOINT" ]; then
    SFT_FINAL_CHECKPOINT="$SFT_CHECKPOINT/final"
  fi
  
  echo "  Using SFT checkpoint: $SFT_FINAL_CHECKPOINT"
  
  python -m fourneurons.evaluation.eval \
    group mixed val sft_final \
    --checkpoint "$SFT_FINAL_CHECKPOINT"
  
  python -m evaluate.score_wandb \
    --generations /scratch/results/group/mixed/sft_final/val_gens.jsonl \
    --benchmark group \
    --output /scratch/results/group/mixed/sft_final/val_scored.json \
    --run_name sft_final_scoring
  
  echo -e "${GREEN}✓ SFT evaluation complete${NC}\n"
else
  echo -e "${RED}✗ Error: SFT checkpoint not found${NC}"
  exit 1
fi

################################################################################
# Pipeline Complete
################################################################################
echo -e "${GREEN}===============================================${NC}"
echo -e "${GREEN}✓ PIPELINE COMPLETE!${NC}"
echo -e "${GREEN}===============================================${NC}"
echo ""
echo "All results stored in: /scratch/results/group/mixed/"
echo ""
echo "Results by approach:"
echo "├── baseline_nosp/             # Base model, no system prompt"
echo "├── baseline_sp/               # Base model, with system prompt"
echo "├── merged_ties/               # TIES merged model"
echo "├── merged_dare_merged_density0.7/  # DARE-TIES merged model"
echo "├── merged_learnable_cat_5e3/  # Learnable CAT merged model"
echo "└── sft_final/                 # SFT fine-tuned model (final epoch)"
echo ""
echo "Each result directory contains:"
echo "  - val_gens.jsonl      : Generated completions"
echo "  - val_scored.json     : Scored results"
echo ""
echo ""
