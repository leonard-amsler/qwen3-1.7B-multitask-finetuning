# General Knowledge — reproduction pipelines (team 4neurons)

Required base model: **`Qwen/Qwen3-1.7B`**. Each script reproduces one part of the
report end-to-end (data → SFT/DPO → merge → **GK benchmark eval**) and writes a
vLLM-ready checkpoint to `/scratch/checkpoints/gk_<name>/vllm`.

> Scripts **do not push to HuggingFace** — they stop after local evaluation.
> To submit, manually upload the corresponding `vllm/` directory.

## Scripts

These map one-to-one to what is described in the report.

| Script | What it produces | Output checkpoint |
|---|---|---|
| `run_baseline.sh` | Raw Qwen3-1.7B reference: **Base (no sp)** and **Base (sp)** (format prompt) | — (no training) |
| `run_distillation.sh` | The **distillation comparison**: trains one SFT model per CoT corpus (off-policy / contrastive / on-policy) with the same recipe and scores all three on the GK benchmark | `gk_offpolicy`, `gk_contrastive`, `gk_sft` |
| `run_sft.sh` | The **SFT** model (on-policy self-distillation) | `gk_sft/vllm` |
| `run_sft_dpo.sh` | The **SFT+DPO** model (final, preference tuning on top of SFT) | `gk_sft_dpo/vllm` |
| `run_group.sh` | The **group model** (CAT merge) on the GK benchmark, with `sp_group_think.txt` | — (eval only) |

The on-policy point of the distillation comparison **is** the SFT model
(`gk_sft`): the report keeps on-policy self-distillation as SFT.

## How to run

```bash
chmod +x pipelines/*.sh

# Each script is self-contained and can be run in any order: missing data,
# caches, or upstream checkpoints are built automatically (ensure_* in _lib.sh).

./pipelines/run_baseline.sh        # Base (no sp) + Base (sp)
./pipelines/run_distillation.sh    # off-policy vs contrastive vs on-policy (SFT)
./pipelines/run_sft.sh             # SFT only
./pipelines/run_sft_dpo.sh         # SFT+DPO (final model)
./pipelines/run_group.sh            # group model on GK benchmark (with group prompt)
```

Baseline variants and a quick smoke test:

```bash
ONLY=prompt   ./pipelines/run_baseline.sh   # only Base (sp)
ONLY=noprompt ./pipelines/run_baseline.sh   # only Base (no sp)
N_SAMPLES=1   ./pipelines/run_sft.sh         # n=1 smoke test (final metric is n=8)
```

Evaluation writes a per-source / macro-category / option-count JSON report under
`$EVAL/gk_<name>/`. Compare close models **only at n=8** (n=1 is noisy ±~3 pts).

## Configuration (`_lib.sh`)

Shared settings, override via environment variables (defaults shown):

| Variable | Default | Purpose |
|---|---|---|
| `DATA` | `/scratch/data` | Distill caches and CoT corpora |
| `CKPT` | `/scratch/checkpoints` | LoRA adapters and merged models |
| `EVAL` | `/scratch/eval` | Generations and JSON reports |
| `BASE` | `Qwen/Qwen3-1.7B` | Base model |
| `GK_BENCH` | `validation_samples/general_knowledge_dev_full.jsonl` | GK benchmark |
| `N_SAMPLES` | `8` | Completions per question at eval (`1` = quick smoke test) |
| `FORMAT_PROMPT` | (format instruction) | System prompt for **Base (sp)** |
| `GROUP_MODEL` | `/scratch/checkpoints/group_model/learnable_cat_5e3` | Merged group model path |
| `GROUP_SP_FILE` | `fourneurons/prompts/sp_group_think.txt` | Group system prompt file |

All build steps are idempotent `ensure_*` helpers (skip if the artifact exists)
that chain together:

```
14B distill (short/long) ─► off-policy corpus ─► self-distill cache ─► on-policy corpus ─► SFT ─► DPO pairs ─► SFT+DPO
                          └► contrastive corpus ─► contrastive model
```

## Code files used

- `fourneurons/distill/distill.py` — CoT distillation from the 14B teacher
  (`--reasoning_style contrastive` for the contrastive corpus). Prompts in
  `fourneurons/distill/prompts.py`.
- `fourneurons/distill/self_distill.py` — on-policy self-distillation from the
  1.7B base (`--select_best` / `--max_thinking_chars` = anti-loop filtering).
- `fourneurons/data/build_train.py` — SFT corpus assembly (layered caches, last-wins).
- `fourneurons/scripts/train.py` — LoRA SFT.
- `fourneurons/scripts/build_dpo_pairs.py` + `train_dpo.py` — DPO pipeline.
- `fourneurons/scripts/merge_lora.py` — merge adapter + **bake `enable_thinking=ON`**
  into the chat template + write `generation_config.json`.
- `fourneurons/eval/run_inference.py` + `report_by_bucket.py` — GK benchmark evaluation.
