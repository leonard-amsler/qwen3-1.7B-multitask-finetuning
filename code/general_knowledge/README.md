# General Knowledge — Team 4neurons

Fine-tuning **Qwen/Qwen3-1.7B** for multiple-choice general-knowledge questions (2–20 options). The model must answer with reasoning and a final letter in `\boxed{LETTER}`. Thinking mode is baked into the chat template at merge time (CI does not pass `enable_thinking` as a kwarg).

All project code lives in the **`fourneurons/`** Python package. End-to-end reproduction scripts are in **`pipelines/`** (see [`pipelines/README.md`](pipelines/README.md)).

---

## Project overview

We optimize pass@1 on knowledge MCQs while avoiding three traps:

- **Goodhart's law** — overfitting an in-distribution dev set that does not generalize to CI.
- **Format collapse** — short, shallow chain-of-thought that scores well locally but fails on hard questions.
- **Off-policy distillation** — teaching the 1.7B student from a 14B teacher it cannot faithfully imitate.

The reported pipeline has three parts, all evaluated on the **GK benchmark**
(`validation_samples/general_knowledge_dev_full.jsonl`):

| Stage | Idea | Script |
|---|---|---|
| **Baseline** | Raw Qwen3-1.7B, with / without a format system prompt | `pipelines/run_baseline.sh` |
| **Distillation comparison** | Same SFT recipe on three CoT corpora — off-policy (14B), contrastive (14B), on-policy self-distillation (1.7B) | `pipelines/run_distillation.sh` |
| **SFT** | On-policy self-distillation, the best CoT corpus | `pipelines/run_sft.sh` |
| **SFT+DPO** | Preference tuning on the SFT model's own correct vs incorrect draws (**final model**) | `pipelines/run_sft_dpo.sh` |

On-policy self-distillation wins the comparison and is the model reported as
**SFT**; **SFT+DPO** is the final submission.

---

## Repository layout

```
general_knowledge/
├── fourneurons/          # All source code (see below)
├── pipelines/            # Reproduction scripts (start here to run)
├── validation_samples/   # GK benchmark dev sets
├── evaluate/             # CI-compatible scoring (benchmarks, extract_answer)
├── docker/               # Cluster / Docker setup
└── requirements.txt
```

### `fourneurons/` package

| Module | Role |
|---|---|
| `fourneurons/data/` | Dataset loaders (MMLU, TriviaQA, BoolQ, …), augmentation, `build_train.py` |
| `fourneurons/distill/` | 14B CoT distillation (`distill.py`) and 1.7B self-distillation (`self_distill.py`) |
| `fourneurons/scripts/` | LoRA SFT (`train.py`), DPO (`train_dpo.py`, `build_dpo_pairs.py`), merge (`merge_lora.py`) |
| `fourneurons/eval/` | Inference (`run_inference.py`), bucket reports (`report_by_bucket.py`) |
| `fourneurons/utils/` | Config and misc helpers |

Run any module with `python -m fourneurons.<module>` from this directory (after install).

---

## Setup

```bash
cd code/general_knowledge
pip install -r requirements.txt   # installs fourneurons in editable mode (-e .)
```

Requires Python ≥ 3.9, a CUDA GPU for training and distillation, and HuggingFace access for `Qwen/Qwen3-1.7B` and `Qwen/Qwen3-14B-AWQ`.

Default paths assume a cluster layout (`/scratch/data`, `/scratch/checkpoints`, `/scratch/eval`). Override with environment variables (see [`pipelines/README.md`](pipelines/README.md)).

---

## Running

Each script is **self-contained** and can be run in any order. Missing prerequisites (distill caches, CoT corpora, upstream checkpoints) are built automatically via `ensure_*` helpers in [`pipelines/_lib.sh`](pipelines/_lib.sh).

```bash
chmod +x pipelines/*.sh

./pipelines/run_baseline.sh        # Base (no sp) + Base (sp)
./pipelines/run_distillation.sh    # off-policy vs contrastive vs on-policy (SFT)
./pipelines/run_sft.sh             # SFT
./pipelines/run_sft_dpo.sh         # SFT+DPO (final model)
```

Quick smoke test (n=1 instead of the reliable n=8):

```bash
N_SAMPLES=1 ./pipelines/run_sft.sh
```

Scripts **do not push to HuggingFace**. After a run, upload the corresponding `vllm/` directory manually for CI submission.

More detail: [`pipelines/README.md`](pipelines/README.md).
