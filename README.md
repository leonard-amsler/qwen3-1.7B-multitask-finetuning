# Post-training Qwen3-1.7B for 5 Reasoning Tasks

**Team 4neurons** — EPFL **CS-552 Modern Natural Language Processing**, Spring 2026 · Final project (Milestone 3)

Léonard Amsler · Noé Boulud · Nathan Gromb · Nicolas Teissier

---

Adapting a small language model to several specialised domains at once is hard: task-specific
fine-tuning tends to erode general capability. This project post-trains **Qwen3-1.7B** on four
unrelated tasks — mathematical reasoning, general knowledge, safety, and multilinguality — using
LoRA SFT on chain-of-thought-augmented data followed by task-specific RL, then studies how to fold
those four specialists back into a **single group model** via LoRA adapter merging and mixed-dataset SFT.

## 📄 The paper

**[`final_report/4neurons_report.pdf`](final_report/4neurons_report.pdf)** is the authoritative write-up —
method, datasets, ablations, per-category and per-language breakdowns, ethical discussion, and full
hyper-parameters in the appendix. Read it first; everything in this repository exists to reproduce it.

Three findings, in short:

- **CoT distillation is the main lever.** Supervised fine-tuning on distilled reasoning traces drives
  most of the task-specific gain, well beyond what prompting alone recovers.
- **RL is complementary but task-dependent.** GRPO and DPO help on some tasks and not others; a
  reward signal that does not penalise malformed output separately from wrong answers buys little.
- **Merging beats retraining.** **CAT** adapter merging composes the four specialists in weight space
  and edges out mixed-dataset SFT — at negligible additional compute.

Headline numbers on the validation benchmarks (see Table 1 of the report for the full grid, including
baselines, format-compliance rates and every method tried):

| Task | Benchmark | Best method | Score |
|---|---|---|---|
| Mathematics | MATH500 | SFT + GRPO | **85.2%** Pass@8 |
| Multilinguality | MMMLU / XCOPA / MMLU-ProX | SFT | **65.0%** Pass@1 |
| General knowledge | MMLU / MMLU-Pro & others | SFT + DPO | **60.4%** Pass@1 |
| Safety | SafetyBench | SFT | **83.5%** Pass@1 |
| **Group** | all four | **CAT merge** | **69.9%** |

## 🗂 Repository architecture

The four task codebases started unified and diverged quickly. Rather than merge them back together
artificially, they are kept as **five independent codespaces** under [`code/`](code/) — one per task,
plus the group model. Each is self-contained: its own dependencies, configs, entry points, and its own
README describing how to run it.

```
.
├── final_report/
│   └── 4neurons_report.pdf     ← the paper
├── code/
│   ├── README.md               ← index of the five codespaces
│   ├── general_knowledge/
│   ├── maths/
│   ├── multilinguality/
│   ├── safety/
│   └── group/                  ← LoRA merging (CAT / TIES / DARE-TIES) + mixed-dataset SFT
└── M2-archive/                 ← Milestone 2 template & validation splits, kept for reference
```

Within a codespace the layout is broadly consistent:

| Path | Contents |
|---|---|
| `fourneurons/` | The project Python package — `data/` (dataset building, CoT & distractor augmentation), `evaluation/`, `prompts/`, `model/`, `scripts/`, `utils/` |
| `evaluate/` | Standalone scorer mirroring the course CI's answer-extraction and `pass@k` logic, for gating checkpoints locally |
| `configs/` | Training and generation configurations |
| `pipelines/` · `bash_scripts/` · `run_*_pipeline.sh` | End-to-end reproduction entry points, mapping one-to-one onto the report's experiments |
| `docker/` | Image build and Run:AI job submission helpers for the course cluster |
| `validation_samples/` · `results/` | Held-out validation splits and scored evaluation output |

All training runs target a **single A100** at a time.

## 🚀 Where to start

Per-task instructions live with the code — these READMEs are the run instructions, and this page
deliberately does not duplicate them:

| Codespace | Run instructions |
|---|---|
| Index of all five | [`code/README.md`](code/README.md) |
| Mathematics | [`code/maths/README.md`](code/maths/README.md) — the most detailed runbook: data building, SFT & GRPO, decoding-parameter grids, result plotting |
| General knowledge | [`code/general_knowledge/README.md`](code/general_knowledge/README.md) + [`pipelines/README.md`](code/general_knowledge/pipelines/README.md) |
| Multilinguality | [`code/multilinguality/README.md`](code/multilinguality/README.md) |
| Safety | [`code/safety/README.md`](code/safety/README.md) |
| Group model | [`code/group/README.md`](code/group/README.md) |

Each codespace also ships an `evaluate/README.md` for the standalone scorer and a
`fourneurons/evaluation/README.md` for the batched generation harness.

## ℹ️ About this public mirror

This is a public copy of the team's coursework repository, published after the course concluded.
The commit history is preserved in full; it was rewritten only to remove a credential that had been
committed in the original, all `.DS_Store` files, and the raw generation dumps
(`code/safety/results/*/val_gens*.jsonl`, ~224 MB). The **scored** outputs those dumps were derived
from — `val_scored.json`, `summary.json`, `val_category_breakdown.json` — are all still here.

Pipelines expect `HF_TOKEN` and `WANDB_API_KEY` to be supplied through the environment.
