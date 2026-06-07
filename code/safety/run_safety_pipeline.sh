#!/bin/bash

################################################################################
# Safety Bench Full Pipeline
################################################################################
# This script executes the complete safety training pipeline:
# 1. Build SafetyBench splits (train/val)
# 2. Download base model
# 3. Evaluate baseline model
# 4. Train SFT model
# 5. Evaluate SFT model
# 6. Generate pass@8 on weak categories
# 7. Build DPO dataset
# 8. Train DPO model
# 9. Evaluate DPO model
#
# Run from: 'code/safety/' directory (cf. README for instructions)
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

# Paths
CODE_DIR="/scratch/nico/standard-project-m3-4neurons/code/safety"  # Please change this to your actual code directory if different (this was my setup on RCP cluster)
SYSTEM_PROMPT="${CODE_DIR}/fourneurons/prompts/sp_general_qcm_think.txt"  # Please change this to your actual code directory if different (this was my setup on RCP cluster)

# Ensure we're in the right directory
cd "$CODE_DIR" || exit 1

echo -e "${GREEN}===============================================${NC}"
echo -e "${GREEN}Safety Training Pipeline${NC}"
echo -e "${GREEN}===============================================${NC}"
echo "Working directory: $(pwd)"
echo ""

################################################################################
# CRITICAL PREREQUISITE: Dataset and Model Availability
################################################################################
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check if SafetyBench data is available
if [ ! -f "/scratch/hf_cache/datasets/SafetyBench/test_en.json" ]; then
  echo -e "${RED}✗ Error: SafetyBench test data not found${NC}"
  echo "  Expected: /scratch/hf_cache/datasets/SafetyBench/test_en.json"
  echo "  Please download SafetyBench dataset first (cf README for instructions) and ensure it's in the expected location."
  exit 1
fi

# Check if system prompt exists
if [ ! -f "$SYSTEM_PROMPT" ]; then
  echo -e "${RED}✗ Error: System prompt not found${NC}"
  echo "  Expected: $SYSTEM_PROMPT"
  exit 1
fi

echo -e "${GREEN}✓ Prerequisites found${NC}\n"

################################################################################
# Step 1: Build SafetyBench splits (train/val)
################################################################################
echo -e "${YELLOW}[1/10] Building SafetyBench train/val splits...${NC}"
python -c "from fourneurons.data.safetybench_build_dataset import build_splits; build_splits()"
echo -e "${GREEN}✓ Splits created (train + val)${NC}\n"

################################################################################
# Step 2: Download base model (Qwen3-1.7B)
################################################################################
echo -e "${YELLOW}[2/10] Downloading base model (Qwen3-1.7B)...${NC}"
python -m fourneurons.scripts.download_base_model
echo -e "${GREEN}✓ Base model downloaded${NC}\n"

################################################################################
# Step 3: Generate Chain-of-Thought data for training
################################################################################
echo -e "${YELLOW}[3/10] Generating Chain-of-Thought data for training...${NC}"
echo "  NOTE: This requires Qwen3-32B-AWQ model and may take several hours."
python -m fourneurons.scripts.generate_cot
echo -e "${GREEN}✓ CoT data generated${NC}\n"

################################################################################
# Step 4: Evaluate baseline model
################################################################################
echo -e "${YELLOW}[4/10] Evaluating baseline model (no fine-tuning)...${NC}"

echo "  4a. Generate predictions without system prompt"
python -m fourneurons.evaluation.eval \
  safety safetybench val baseline_nosp \
  --base

echo "  4b. Generate predictions with system prompt"
python -m fourneurons.evaluation.eval \
  safety safetybench val baseline_sp \
  --base \
  --prompt_file_path "$SYSTEM_PROMPT"

echo "  4c. Score baseline results"
python -m evaluate.score_wandb \
  --generations /scratch/results/safety/safetybench/baseline_nosp/val_gens.jsonl \
  --benchmark safety \
  --output /scratch/results/safety/safetybench/baseline_nosp/val_scored.json \
  --run_name baseline_nosp_scoring

python -m evaluate.score_wandb \
  --generations /scratch/results/safety/safetybench/baseline_sp/val_gens.jsonl \
  --benchmark safety \
  --output /scratch/results/safety/safetybench/baseline_sp/val_scored.json \
  --run_name baseline_sp_scoring

echo -e "${GREEN}✓ Baseline evaluation complete${NC}\n"

################################################################################
# Step 5: Train SFT model
################################################################################
echo -e "${YELLOW}[5/10] Training SFT (Supervised Fine-Tuning) model...${NC}"
echo "  Using CoT data with system prompt. 4 epochs with LoRA. This may take several hours."
python -m fourneurons.scripts.train_safety
echo -e "${GREEN}✓ SFT training complete${NC}\n"

################################################################################
# Step 6: Evaluate SFT model for each epoch
################################################################################
echo -e "${YELLOW}[6/10] Evaluating SFT model for each epoch...${NC}"

# Find the latest SFT checkpoint directory
SFT_CHECKPOINT=$(find /scratch/checkpoints/safety -maxdepth 1 -type d -printf '%T@ %p\n' | sort -rn | head -1 | cut -d' ' -f2-)
if [ ! -d "$SFT_CHECKPOINT" ]; then
  echo -e "${RED}✗ Error: SFT checkpoint not found${NC}"
  exit 1
fi
echo "  Found SFT checkpoint: $SFT_CHECKPOINT"

# Evaluate each epoch checkpoint
for epoch_checkpoint in "$SFT_CHECKPOINT"/checkpoint-*; do
  if [ -d "$epoch_checkpoint" ]; then
    epoch=$(basename "$epoch_checkpoint" | sed 's/checkpoint-//')
    echo "  Epoch $epoch: generate → score → category breakdown"
    
    python -m fourneurons.evaluation.eval \
      safety safetybench val "sft_ep${epoch}" \
      --checkpoint "$epoch_checkpoint"
    
    python -m evaluate.score_wandb \
      --generations /scratch/results/safety/safetybench/sft_ep${epoch}/val_gens.jsonl \
      --benchmark safety \
      --output /scratch/results/safety/safetybench/sft_ep${epoch}/val_scored.json \
      --run_name "sft_ep${epoch}_scoring"
    
    python -m fourneurons.evaluation.category_breakdown \
      --scored /scratch/results/safety/safetybench/sft_ep${epoch}/val_scored.json \
      --gens /scratch/results/safety/safetybench/sft_ep${epoch}/val_gens.jsonl \
      --output /scratch/results/safety/safetybench/sft_ep${epoch}/val_category_breakdown.json
  fi
done

SFT_FINAL_MERGED_DIR = "/scratch/results/safety/safetybench/sft_ep4/merged"  # This is the directory where the final merged SFT model (after 4 epochs) is stored.

echo -e "${GREEN}✓ SFT evaluation complete for all epochs${NC}\n"

################################################################################
# Step 7: Generate pass@8 on weak categories (for DPO data)
################################################################################
echo -e "${YELLOW}[7/10] Generating pass@8 predictions on weak categories...${NC}"
echo "  Using best SFT model to generate 8 candidates per example."
echo "  Targeting: 'Unfairness and Bias', 'Offensiveness'"
python -m fourneurons.evaluation.pass8
echo -e "${GREEN}✓ Pass@8 generation complete${NC}\n"

################################################################################
# Step 8: Build DPO training dataset
################################################################################
echo -e "${YELLOW}[8/10] Building DPO dataset from pass@8 results...${NC}"
echo "  Creating preferred (correct, longest) vs rejected (wrong) pairs."

python -c "from fourneurons.data.build_dpo_safety import build_dpo_pairs; build_dpo_pairs(merged_sft_dir='${SFT_FINAL_MERGED_DIR}', all_categories=False)"
echo -e "${GREEN}✓ DPO dataset created${NC}\n"

################################################################################
# Step 9: Train DPO model
################################################################################
echo -e "${YELLOW}[9/10] Training DPO (Direct Preference Optimization) model...${NC}"
echo "  Base: SFT model. Training on weak categories. 3 epochs."
python -m fourneurons.scripts.train_safety_dpo --merged_sft_dir "${SFT_FINAL_MERGED_DIR}"
echo -e "${GREEN}✓ DPO training complete${NC}\n"

################################################################################
# Step 10: Evaluate DPO model
################################################################################
echo -e "${YELLOW}[10/10] Evaluating DPO model...${NC}"

DPO_CHECKPOINT="/scratch/checkpoints/safety_dpo/final"
if [ -d "$DPO_CHECKPOINT" ]; then
  echo "  Generate predictions → score → category breakdown"
  
  python -m fourneurons.evaluation.eval \
    safety safetybench val dpo_final \
    --checkpoint "$DPO_CHECKPOINT" \

  python -m evaluate.score_wandb \
    --generations /scratch/results/safety/safetybench/dpo_final/val_gens.jsonl \
    --benchmark safety \
    --output /scratch/results/safety/safetybench/dpo_final/val_scored.json \
    --run_name dpo_final_scoring

  echo -e "${GREEN}✓ DPO evaluation complete${NC}\n"
else
  echo -e "${RED}✗ Error: DPO checkpoint not found at $DPO_CHECKPOINT${NC}"
  exit 1
fi

################################################################################
# Pipeline Complete - Results Summary
################################################################################
echo -e "${GREEN}===============================================${NC}"
echo -e "${GREEN}✓ PIPELINE COMPLETE!${NC}"
echo -e "${GREEN}===============================================${NC}"
echo ""
echo "All results stored in: /scratch/results/safety/safetybench/"
echo ""
echo "Directory structure:"
echo "├── baseline_nosp/              # Base model, no system prompt"
echo "│   ├── val_gens.jsonl          # Generated completions"
echo "│   └── val_scored.json         # Scored results"
echo "│"
echo "├── baseline_sp/                # Base model, with system prompt"
echo "│   ├── val_gens.jsonl"
echo "│   └── val_scored.json"
echo "│"
echo "├── sft_ep1/ sft_ep2/           # SFT model results by epoch"
echo "│   ├── val_gens.jsonl          # Generations"
echo "│   ├── val_scored.json         # Scores"
echo "│   └── val_category_breakdown.json  # Performance by category"
echo "│"
echo "├── pass8_weak_categories_train/  # Pass@8 on weak categories"
echo "│   └── val_gens_n8.jsonl       # 8 generations per example"
echo "│"
echo "└── dpo_final/                  # DPO model (final)"
echo "    ├── val_gens.jsonl"
echo "    ├── val_scored.json"
echo "    └── val_category_breakdown.json"
echo ""
echo "Key metrics to compare:"
echo "  - baseline_sp/val_scored.json → sft_ep4/val_scored.json"
echo "  - sft_ep4/val_scored.json → dpo_final/val_scored.json"
echo ""

