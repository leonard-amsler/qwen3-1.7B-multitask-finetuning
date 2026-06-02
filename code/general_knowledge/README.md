# General Knowledge — Team 4neurons

Fine-tuning **Qwen/Qwen3-1.7B** for multiple-choice general-knowledge questions (2–20 options). The model must answer with reasoning and a final letter in `\boxed{LETTER}`. Thinking mode is baked into the chat template at merge time (CI does not pass `enable_thinking` as a kwarg).

All project code lives in the **`fourneurons/`** Python package. End-to-end reproduction scripts for each model version are in **`pipelines/`**.

For the full research narrative, diagnostics, and numbers, see [`PLAN_FINAL.md`](PLAN_FINAL.md). Earlier iteration plans: [`PLAN_V5.md`](PLAN_V5.md) through [`PLAN_V10.md`](PLAN_V10.md).

---

## Project overview

We optimize pass@1 on knowledge MCQs while avoiding three traps identified across iterations:

- **Goodhart's law** — overfitting an in-distribution dev set that does not generalize to CI.
- **Format collapse** — short, shallow chain-of-thought that scores well locally but fails on hard questions.
- **Off-policy distillation** — teaching the 1.7B student from a 14B teacher it cannot faithfully imitate.

### Version lineage (v5 → v11b)

| Version | Idea | Key result |
|---|---|---|
| **v5** | Rigorous dataset rebuild: strict CoT, 14B distillation, typed distractors, OOD dev | First model to beat baseline on public CI (+0.12); format collapse persisted (~150 tok CoTs) |
| **v6** | Long 14B CoTs on STEM sources (10–20 sentences, given-answer prompts) | **dev_full 0.541**; stable reference for 14B off-policy distillation |
| **v7** | 14B teacher in thinking mode (blind solve) | Regression — surface imitation without real reasoning |
| **v8** | Self-distill from 1.7B baseline, weak LoRA (r=8) | Format broken (51% `\boxed{}`); true score ~0.52 vs 0.39 measured |
| **v9** | Same v8 data, strong LoRA (r=64) | Confirmed format drives +21 pts; 86% `\boxed{}` |
| **v9b** | v9 + `select_best` anti-loop filtering | **dev_full 0.609** (n=1); pass@1 **0.580** (n=8); on-policy beats v6 and v10 |
| **v10** | Contrastive 14B distillation on STEM (refute common wrong reasoning) | dev_full 0.554 — off-policy underperforms v9b |
| **v11** | Timid DPO on v9b draws (β=0.1, 1 epoch) | No-op (~0.610 vs 0.609) |
| **v11b** | Strong on-policy DPO on v9b (β=0.05, 3 epochs) | **Final model** — pass@1 **0.600** (n=8), +2.0 vs v9b on all four sources |

**Final model: v11b** — self-distill → SFT (v9b) → preference tuning on the model's own correct vs incorrect draws.

---

## Repository layout

```
general_knowledge/
├── fourneurons/          # All source code (see below)
├── pipelines/            # One script per model version (start here to run)
├── validation_samples/   # Dev sets (dev_small, dev_full, OOD)
├── evaluate/             # CI-compatible scoring (benchmarks, extract_answer)
├── configs/              # Training configs
├── docker/               # Cluster / Docker setup
├── PLAN_*.md             # Iteration plans and diagnostics
└── requirements.txt
```

### `fourneurons/` package

| Module | Role |
|---|---|
| `fourneurons/data/` | Dataset loaders (MMLU, TriviaQA, BoolQ, …), augmentation, `build_train.py` |
| `fourneurons/distill/` | 14B CoT distillation (`distill.py`) and 1.7B self-distillation (`self_distill.py`) |
| `fourneurons/scripts/` | LoRA SFT (`train.py`), DPO (`train_dpo.py`, `build_dpo_pairs.py`), merge (`merge_lora.py`) |
| `fourneurons/eval/` | Inference (`run_inference.py`), bucket reports (`report_by_bucket.py`), dev set builders |
| `fourneurons/utils/` | Config and misc helpers |

Run any module with `python -m fourneurons.<module>` from this directory (after install).

---

## Setup

```bash
cd code/general_knowledge
pip install -r requirements.txt   # installs fourneurons in editable mode (-e .)
```

Requires Python ≥ 3.9, CUDA GPU(s) for training and distillation, and HuggingFace access for `Qwen/Qwen3-1.7B` and `Qwen/Qwen3-14B-AWQ`.

Default paths assume a cluster layout (`/scratch/data`, `/scratch/checkpoints`, `/scratch/eval`). Override with environment variables (see below).

---

## Running model versions

**Go to [`pipelines/`](pipelines/)** to reproduce any version end-to-end (data → SFT/DPO → merge → eval).

Each script is **self-contained** and can be run in any order. Missing prerequisites (distill caches, datasets, upstream checkpoints) are built automatically via `ensure_*` helpers in [`pipelines/_lib.sh`](pipelines/_lib.sh).

```bash
chmod +x pipelines/*.sh

# Final model (builds full chain as needed)
./pipelines/run_v11b.sh

# Other versions
./pipelines/run_v6.sh
./pipelines/run_v9.sh
./pipelines/run_v9b.sh
./pipelines/run_v10.sh
./pipelines/run_v11.sh
```

### Which script maps to which version

| Script | Version | Output checkpoint | Notes |
|---|---|---|---|
| `run_v6.sh` | **v6** | `gk_v6/vllm` | Long 14B distill, given-answer prompts |
| `run_v9.sh` | **v9** | `gk_v9/vllm` | Re-SFT on v8 data, strong LoRA; auto-builds `train_v8` |
| `run_v9b.sh` | **v9b** | `gk_v9b/vllm` | v9 + anti-loop `select_best` |
| `run_v10.sh` | **v10** | `gk_v10/vllm` | Contrastive 14B distill on STEM |
| `run_v11.sh` | **v11** | `gk_v11/vllm` | Timid DPO on v9b |
| `run_v11b.sh` | **v11b** | `gk_v11b/vllm` | **Final model** — strong DPO on v9b |

There is no dedicated v8 script: v8 differs from v9 only in LoRA strength (r=8 vs r=64). The v8 self-distillation dataset is built inside `ensure_v8_data()` when running v9 or v9b.

Scripts **do not push to HuggingFace**. After a run, upload the corresponding `vllm/` directory manually for CI submission.

### Environment variables

Set these before running a pipeline (defaults shown):

| Variable | Default | Purpose |
|---|---|---|
| `DATA` | `/scratch/data` | Distill caches and HF datasets |
| `CKPT` | `/scratch/checkpoints` | LoRA adapters and merged models |
| `EVAL` | `/scratch/eval` | Generation outputs and JSON reports |
| `BASE` | `Qwen/Qwen3-1.7B` | Base model |
| `DEV` | `validation_samples/general_knowledge_dev_full.jsonl` | Eval set |
| `N_SAMPLES` | `8` | Completions per question at eval (`1` = quick smoke test) |

Quick smoke test:

```bash
N_SAMPLES=1 ./pipelines/run_v9.sh
```

More detail: [`pipelines/README.md`](pipelines/README.md).




