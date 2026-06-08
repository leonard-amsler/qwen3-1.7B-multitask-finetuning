# Multilinguality - Running instructions

## Overview

This branch focuses on improving multilingual performance, particularly on the MMLU benchmark, through a combination of data augmentation, distillation, and fine-tuning techniques.

### Pipeline Steps

The main training pipeline is executed via the following commands:

1. **Data Preparation and Augmentation** → Format all datasets, Generate distractors and CoT explanations for MMLU questions in multiple languages using the base model.

```bash
# Set up environment
cd /scratch/nathan/repo/
source ../.venv/bin/activate

# Get and format datasets
python fourneurons/data/multilingual.py

# Generate MMLU distractors using Qwen3-32B-AWQ, output to mmmlu_more_qcms
python fourneurons/scripts/augment_mcq_choices.py multilingual mmmlu train Qwen/Qwen3-32B-AWQ mmmlu_more_qcms

# Generate CoT explanations for MMLU questions using Qwen3-32B-AWQ, output to mmmlu_more_qcms
python fourneurons/scripts/distilled_reasoning_traces.py multilingual mmmlu_more_qcms train Qwen/Qwen3-32B-AWQ mmmlu_more_qcms --n 4
```

1. **Baseline Evaluation** → Evaluate the base model on the datasets.

```bash
# MMMLU
# 1. Generate
python fourneurons/evaluation/eval.py multilingual mmmlu quicktest base_model_mmmlu --checkpoint /scratch/checkpoints/multilingual/base_patched --n 8 --no_lora
# 2. Score with the official script
python -m evaluate.score_wandb --generations /scratch/results/multilingual/mmmlu/base_model_mmmlu/quicktest_gens.jsonl --benchmark multilingual --output /scratch/results/multilingual/mmmlu/base_model_mmmlu/quicktest_scored.jsonl --run_name base_model_mmmlu_scoring

# XCOPA
# 1. Generate
python fourneurons/evaluation/eval.py multilingual xcopa test base_model_xcopa --checkpoint /scratch/checkpoints/multilingual/base_patched --n 1 --no_lora
# 2. Score with the official script
python -m evaluate.score_wandb --generations /scratch/results/multilingual/mmmlu/base_model_xcopa/test_gens.jsonl --benchmark multilingual --output /scratch/results/multilingual/xcopa/base_model_xcopa/test_scored.jsonl --run_name base_model_xcopa_scoring

# MMLU ProX
# 1. Generate
python fourneurons/evaluation/eval.py multilingual mmmlu_prox quicktest base_model_mmmlu_prox --checkpoint /scratch/checkpoints/multilingual/base_patched --n 1 --no_lora
# 2. Score with the official script
python -m evaluate.score_wandb --generations /scratch/results/multilingual/mmmlu_prox/base_model_mmmlu_prox/quicktest_gens.jsonl --benchmark multilingual --output /scratch/results/multilingual/mmmlu_prox/base_model_mmmlu_prox/quicktest_scored.jsonl --run_name base_model_mmmlu_prox_scoring

```

3. **SFT Training** → Supervised Fine-Tuning with LoRA on the augmented MMLU data (12 epochs).

```bash
python fourneurons/scripts/train_multilingual.py sft --run_name mmmlu_sft --epochs 12
```

4. **SFT Evaluation** → Score each epoch on MMLU (all languages) + category breakdown.

```bash
# Patch the chat template to use our multilingual prompt
python fourneurons/scripts/patch_chat_template.py /scratch/checkpoints/multilingual/mmmlu_sft/checkpoint-6875 /scratch/nathan/repo/fourneurons/prompts/multilingual_cot_teacher.txt

# Evaluate the model on MMLU or any other dataset (available: xcopa, mmmlu_prox, mmmlu) to get pass@1 and pass@8
python fourneurons/evaluation/eval.py multilingual mmmlu quicktest mmmlu_sft_6875 --checkpoint /scratch/checkpoints/multilingual/mmmlu_sft/checkpoint-6875 --n 8

# Use the course's official scoring script to get detailed results and log to Weights & Biases
python -m evaluate.score_wandb --generations /scratch/results/multilingual/mmmlu/mmmlu_sft_6875/quicktest_gens.jsonl --benchmark multilingual --output /scratch/results/multilingual/mmmlu/mmmlu_sft_6875/quicktest_scored.jsonl --run_name mmmlu_sft_6875_scoring
```

Here, `checkpoint-6875` corresponds to the best epoch based on validation performance, but you can evaluate any checkpoint to see how performance evolves across epochs, and use any dataset among the available ones for evaluation.

5. **GRPO Training** → Grouped Reward Policy Optimization on the same data, using the model's own generations as preferences.

```bash
# Use the best SFT checkpoint as the starting point for GRPO training, and specify the number of generations to use for preference generation.
python fourneurons/scripts/train_multilingual.py grpo --run_name mmmlu_grpo --model_path /scratch/results/multilingual/mmmlu/mmmlu_sft_6875/merged --num_generations 8
```

Again, you can specify any SFT checkpoint as the starting point for GRPO training, and adjust the number of generations used for preference generation to see how it impacts final performance.

6. **GRPO Evaluation** → Final model scoring on MMLU (all languages) + category breakdown.

```bash
# Patch the chat template to use our multilingual prompt
python fourneurons/scripts/patch_chat_template.py /scratch/checkpoints/multilingual/mmmlu_grpo/checkpoint-50 /scratch/nathan/repo/fourneurons/prompts/multilingual_cot_teacher.txt

# Evaluate the model on MMLU or any other dataset (available: xcopa, mmmlu_prox, mmmlu) to get pass@1 and pass@8
python fourneurons/evaluation/eval.py multilingual mmmlu quicktest mmmlu_mmmlu_grpo_50 --checkpoint /scratch/checkpoints/multilingual/mmmlu_grpo/checkpoint-50 --n 8 --no_lora
python -m evaluate.score_wandb --generations /scratch/results/multilingual/mmmlu/mmmlu_mmmlu_grpo_50/quicktest_gens.jsonl --benchmark multilingual --output /scratch/results/multilingual/mmmlu/mmmlu_mmmlu_grpo_50/quicktest_scored.jsonl --run_name mmmlu_mmmlu_grpo_50_scoring
```

Again, `checkpoint-50` is interchangeable with any checkpoint from GRPO training, and you can evaluate on any dataset among the available ones to see how performance evolves across epochs and datasets.

### Results

Once everything has run, you can go in `/scratch/results/multilingual/` to find the generations and scores for each evaluation run, and compare how the base model, SFT checkpoints, and GRPO checkpoints perform across datasets and languages. You can also check Weights & Biases for detailed logs and visualizations of the training and evaluation runs.
