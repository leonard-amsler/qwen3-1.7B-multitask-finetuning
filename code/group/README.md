# Group Benchmark Pipeline

Complete pipeline for training and evaluating group models using adapter merging and SFT, as described in the project report.

## Quick Start

```bash
cd /scratch/nico/standard-project-m3-4neurons/code/group
bash run_group_pipeline.sh
```

## What the Pipeline Does

The script executes a complete 7-step training pipeline:

1. **Build dataset** → Mixed validation/training dataset from all domains
2. **Download model** → Qwen3-1.7B base model  
3. **Baseline eval** → Evaluate base model on mixed data (±system prompt)
4. **Merge adapters** → Three methods: TIES, DARE-TIES (ρ=0.7), Learnable CAT
5. **Eval merged** → Score all merged models on mixed data
6. **Train SFT** → Supervised Fine-Tuning with LoRA (6 epochs) on mixed data
7. **Eval SFT** → Final model scoring on mixed data

## Prerequisites

**CRITICAL:** This pipeline requires all individual domain pipelines to be executed first.

You must have trained LoRA adapters from:
- **Safety** pipeline → `/scratch/checkpoints/safety/TIMESTAMP/final`
- **Math** pipeline → `/scratch/checkpoints/math/TIMESTAMP/checkpoint-*`
- **Multilingual** pipeline → `/scratch/checkpoints/multilingual/TIMESTAMP/checkpoint-*`
- **Knowledge (GK)** pipeline → `/scratch/checkpoints/gk_vXX/adapter`

### Before Running This Script

1. Run the individual domain pipelines (safety, math, multilingual, knowledge)
2. Note the checkpoint paths where adapters are saved
3. Update the `ADAPTER_*` variables in `run_group_pipeline.sh` to point to your adapter locations

### Adapter Path Configuration

Edit the top of `run_group_pipeline.sh` and update these variables:

```bash
# Default paths from report (these may differ in your setup):
ADAPTER_SAFETY="/scratch/checkpoints/safety/20260518-215854/final"
ADAPTER_MATH="/scratch/checkpoints/math/qwen3-1.7b-lora-math-rl_20260602-064411/checkpoint-200/"
ADAPTER_MULTILINGUAL="/scratch/checkpoints/multilingual/mmmlu_sft3_long2/checkpoint-6875"
ADAPTER_GK="/scratch/checkpoints/gk_v11b/adapter"
```

**If you see errors about missing adapters:**
- Check that all domain pipelines have been executed
- Verify the checkpoint paths exist: `ls -la /scratch/checkpoints/safety/*/final` etc.
- Update the paths in the script to match your setup

### Other Prerequisites

- System prompt file exists: `fourneurons/prompts/sp_group_think.txt`
- All dependencies installed (see `requirements.txt`)
- Sufficient disk space for results (~several GB)

## Output Locations

All results organized under `/scratch/results/group/mixed/`:

```
results/
├── baseline_nosp/                      # Base model (no system prompt)
├── baseline_sp/                        # Base model (with system prompt)
├── merged_ties/                        # TIES adapter merge results
├── merged_dare_merged_density0.7/      # DARE-TIES merge (ρ=0.7) results
├── merged_learnable_cat_5e3/           # Learnable CAT merge results
└── sft_final/                          # SFT fine-tuned model (6 epochs)
```

Each directory contains:
- `val_gens.jsonl` → Generated completions (one per example)
- `val_scored.json` → Evaluation scores and pass@1 metrics

Model checkpoints saved to:
- SFT model: `/scratch/checkpoints/group/`
- Merged models (temporary): `/scratch/checkpoints/group_model/{method_name}/`

## Customizing Adapter Paths

**Hardcoded adapter paths adapted to RCP cluster setup.** If your adapters are in different locations:

1. Run individual pipelines and note their checkpoint output times
2. Find the checkpoint directories:
   ```bash
   ls -la /scratch/checkpoints/safety/
   ls -la /scratch/checkpoints/math/
   ls -la /scratch/checkpoints/multilingual/
   ls -la /scratch/checkpoints/gk_*
   ```
3. Update the `ADAPTER_*` variables in `run_group_pipeline.sh` accordingly

Example: If your safety checkpoint is at `/scratch/checkpoints/safety/20260520-120000/final`:
```bash
ADAPTER_SAFETY="/scratch/checkpoints/safety/20260520-120000/final"
```

Some paths might also need to be updated in individual scripts (e.g., in merging or training scripts) if they reference hardcoded directories that don't match your setup.

## Key Hyperparameters

Directly from project report (in scripts, not configurable):

- **Adapter Merging**: 
  - TIES: equal weights, density threshold=0.7
  - DARE-TIES: random pruning (ρ=0.7), then TIES
  - Learnable CAT: learns per-adapter weights (lr=5e-3, 100 steps)
- **SFT**: 6 epochs, batch=16 (eff), lr=2e-4, LoRA r=16
- **Generation**: temperature=0.7, top_p=0.9, max_tokens=16384

## Monitoring

The script prints progress with color-coded output:
- 🟢 GREEN = Step completed
- 🟡 YELLOW = Step starting
- 🔴 RED = Error (check your adapter paths!)

Logs can be captured:
```bash
bash run_group_pipeline.sh | tee group_pipeline.log
```

## Comparing Results

After the pipeline completes, compare model performance:

```bash
# View baseline performance
cat /scratch/results/group/mixed/baseline_sp/val_scored.json | jq '.metrics'

# View all merged methods
for method in ties dare_merged_density0.7 learnable_cat_5e3; do
  echo "=== $method ==="
  cat /scratch/results/group/mixed/merged_${method}/val_scored.json | jq '.metrics'
done

# View SFT final
cat /scratch/results/group/mixed/sft_final/val_scored.json | jq '.metrics'
```

Key metrics to check:
- `pass@1` → Pass at one (main metric)

## Troubleshooting

**Error: "Adapter not found"**
- The script exits early if adapters are missing
- Run individual domain pipelines first
- Update `ADAPTER_*` paths in the script to match your checkpoint locations
- Verify paths exist: `ls -la /path/to/adapter`

**Module import errors**
- Script must be run from `code/group/` directory to ensure correct imports
- If you see `ModuleNotFoundError`, check your current directory

**"Path not found" errors**
- Update hardcoded paths in `run_group_pipeline.sh` for your setup
- Some individual scripts may also have hardcoded paths that need updating

**GPU OOM during SFT training**
- Reduce batch size in `train_group.py` (would require modifying the script and results would be affected)

**Step X takes unusually long**
- Merged model evaluation (step 5) can take 1-2 hours per method
- SFT training (step 6) takes 4-6 hours for 6 epochs