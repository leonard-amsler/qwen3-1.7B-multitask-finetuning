# Safety Benchmark Pipeline

Complete pipeline for training and evaluating safety models on SafetyBench dataset as described in the project report.

## Quick Start

```bash
cd /scratch/nico/standard-project-m3-4neurons/code/safety
bash run_safety_pipeline.sh
```

## What the Pipeline Does

The script executes a complete 10-step training pipeline:

1. **Build splits** → SafetyBench train/val dataset splits
2. **Download model** → Qwen3-1.7B base model  
3. **Generate CoT** → Chain-of-Thought training data (requires 32B model, ~hours)
4. **Baseline eval** → Evaluate base model (±system prompt)
5. **Train SFT** → Supervised Fine-Tuning with LoRA (4 epochs)
6. **Eval SFT epochs** → Score each epoch + category breakdown
7. **Pass@8** → Generate 8 candidates on weak categories
8. **Build DPO data** → Create preference pairs from pass@8 results
9. **Train DPO** → Direct Preference Optimization (3 epochs)
10. **Eval DPO** → Final model scoring + analysis

## Prerequisites

- SafetyBench dataset downloaded: `/scratch/hf_cache/datasets/SafetyBench/`
  - Required files: `test_en.json`, `test_answers_en.json`
  - Download from Hugging Face if not present: `datasets.load_dataset("SafetyBench", split="test")`
- System prompt file exists: `fourneurons/prompts/sp_general_qcm_think.txt`
- All dependencies installed (see `requirements.txt`)

## Output Locations

All results organized under `/scratch/results/safety/safetybench/`:

```
results/
├── baseline_nosp/              # Base model (no system prompt)
├── baseline_sp/                # Base model (with system prompt)
├── sft_ep1/ sft_ep2/ ...      # SFT by epoch (4 epochs)
├── pass8_weak_categories_train/  # Pass@8 generations
└── dpo_final/                  # DPO final model
```

Each directory contains:
- `val_gens.jsonl` → Generated completions
- `val_scored.json` → Evaluation scores
- `val_category_breakdown.json` → Performance by category (SFT only)

Model checkpoints saved to: `/scratch/checkpoints/`.
Merged models (for DPO training) saved to: `/scratch/results/safety/safetybench/{run_name}/merged/`.

## Customizing Paths

**Hardcoded paths adapted to RCP cluster setup.** Update these in `run_safety_pipeline.sh` if different:

| Path | Variable | Use |
|------|----------|-----|
| `/scratch/nico/standard-project-m3-4neurons/code/safety` | `CODE_DIR` | Script location |
| `/scratch/hf_cache/` | Fixed | Model/dataset cache |
| `/scratch/data/` | Fixed | Data splits |
| `/scratch/checkpoints/` | Fixed | Training checkpoints |
| `/scratch/results/` | Fixed | Evaluation results |

If any path doesn't match your setup, edit the script before running.

Some paths might also need to be updated in individual training scripts (e.g., `PROMPT_FILE` in `train_safety.py`) if they reference specific directories that doesn't match your setup.

## Key Hyperparameters

Directly from project report (in scripts, not configurable):

- **SFT**: 4 epochs, batch=16 (eff), lr=2e-4, LoRA r=16
- **DPO**: 3 epochs, batch=16 (eff), lr=5e-6, β=0.1
- **Generation**: temperature=0.7, top_p=0.9, max_tokens=16384

## Monitoring

The script prints progress with color-coded output:
- 🟢 GREEN = Step completed
- 🟡 YELLOW = Step starting
- 🔴 RED = Error

Logs can be captured:
```bash
bash run_safety_pipeline.sh | tee pipeline.log
```

## Troubleshooting

**Module import errors:** Script must be run from `code/safety/` directory to ensure correct imports. If you see `ModuleNotFoundError`, check your current directory and adjust paths in the script if necessary.

**Path not found errors:** Update hardcoded paths in `run_safety_pipeline.sh` for your setup, or in individual scripts if necessary.

**GPU OOM:** Reduce batch size or max_length in individual training scripts (results would be affected)

**Step X takes too long:** CoT generation (step 3) requires 32B model - estimate 4+ hours. Other steps should complete in minutes to hours depending on hardware.
