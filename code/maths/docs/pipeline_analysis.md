# Pipeline Analysis And Run Commands

This document describes the current `merged_repo` pipeline and how it relates
to the math-SFT implementation in `standard-project-m2-4neurons`. It is meant
as a first orientation pass before porting the math reasoning code into
`merged_repo`.

## Repository Roles

- `merged_repo` is currently the main branch workspace. It contains the course
  starter code, shared evaluation code, Run:AI submission helpers, and scripts
  for safety and multilingual SFT experiments.
- `standard-project-m2-4neurons` is the math branch. It contains the developed
  math LoRA/SFT pipeline, including OpenMathInstruct-2 training, optional
  OpenR1 mixing, simple competition-math SFT, LoRA merge, chat-template
  injection, and Hugging Face upload helpers.

The immediate adaptation goal is to copy or reconcile the math-specific
training pieces from the math branch into `merged_repo`, while keeping the
merged repo's safety/multilingual work.

## Environment And Cluster Setup

The cluster workflow is documented in `RCP_GUIDE.md` and implemented by
`docker/submit.sh`.

One-time setup:

```bash
runai config cluster rcp-caas-prod
runai login
runai config project course-cs-552-<gaspar>
```

Before launching jobs, edit `docker/submit.sh`:

```bash
GASPAR="<your-gaspar>"
GROUP="<your-group>"  # for example g07
```

Launch an interactive A100 pod:

```bash
cd /scratch/leo/merged_repo/docker
./submit.sh
```

Then connect with one of:

```bash
runai describe job <job-name> -p course-cs-552-<gaspar>
runai port-forward <job-name> --port 8888:8888 -p course-cs-552-<gaspar>
runai bash <job-name> -p course-cs-552-<gaspar>
```

Stop the pod when finished:

```bash
runai delete job <job-name> -p course-cs-552-<gaspar>
```

Large artifacts should live under `/scratch`: datasets in `/scratch/data`,
checkpoints in `/scratch/checkpoints`, HF cache in `/scratch/hf_cache`, and
W&B logs in `/scratch/wandb`.

## Current `merged_repo` Pipeline

### 1. Install Dependencies

From the repository root inside the pod:

```bash
cd /scratch/leo/merged_repo
pip install -r requirements.txt
```

`requirements.txt` includes PyTorch, Transformers, datasets, PEFT, TRL,
vLLM, Accelerate, W&B, and installs this package editable with `-e .`.

### 2. Cache The Base Model

`fourneurons/scripts/download_base_model.py` downloads the required base model:

```bash
python -m fourneurons.scripts.download_base_model
```

It caches `Qwen/Qwen3-1.7B` through Hugging Face. Other scripts assume the
snapshot is available under:

```text
/scratch/hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/
```

### 3. Build SafetyBench Splits

`fourneurons/data/safetybench_data_loader.py` reads local SafetyBench JSON
files:

```text
/scratch/hf_cache/datasets/SafetyBench/test_en.json
/scratch/hf_cache/datasets/SafetyBench/test_answers_en.json
```

It formats each row as:

```json
{"prompt": "question\n\nA) ...", "answer": "A", "category": "..."}
```

`fourneurons/data/safetybench_build_dataset.py` shuffles and splits this data:

```bash
python -m fourneurons.data.safetybench_build_dataset
```

Output:

```text
/scratch/data/safety/safetybench/splits/safetybench_train.jsonl
/scratch/data/safety/safetybench/splits/safetybench_val.jsonl
```

### 4. Generate Safety CoT Traces

`fourneurons/scripts/generate_cot.py` uses a teacher model
`Qwen/Qwen3-32B-AWQ` with vLLM to generate chain-of-thought completions for
the SafetyBench train split. It keeps only completions whose boxed answer
matches the label.

Expected command:

```bash
python -m fourneurons.scripts.generate_cot
```

Input:

```text
/scratch/data/safety/safetybench/splits/safetybench_train.jsonl
```

Output:

```text
/scratch/data/safety/safetybench/cot/safetybench_train_cot.jsonl
```

Important: this script currently has hard-coded paths to one user's checkout,
for example `/scratch/nico/standard-project-m2-4neurons/prompts/...`. Those
paths should be changed to repo-relative prompt paths before relying on it.

### 5. Format Examples For SFT

`fourneurons/data/format_for_sft.py` converts raw or CoT examples into text
ready for TRL `SFTTrainer`.

Behavior:

- If the sample has `completion`, it uses that as the assistant content.
- Otherwise it builds a short answer: `The correct answer is \boxed{X}`.
- If a prompt file is supplied, it adds a system message before the user turn.
- It serializes the messages with `tokenizer.apply_chat_template(...,
  add_generation_prompt=False)`.

This formatter is used by the current safety and multilingual training
scripts.

### 6. Train Safety LoRA

`fourneurons/scripts/train_safety.py` trains a LoRA adapter with TRL
`SFTTrainer`.

Expected command:

```bash
python -m fourneurons.scripts.train_safety
```

Main assumptions:

- Base model is the cached Qwen3-1.7B snapshot.
- Training data is
  `/scratch/data/safety/safetybench/cot/safetybench_train_cot.jsonl`.
- Output goes to `/scratch/checkpoints/safety/<timestamp>`.
- W&B project is `safety-sft`.
- LoRA uses `r=16`, `lora_alpha=32`, `target_modules="all-linear"`.
- Training uses 4 epochs, batch size 4, grad accumulation 4, bf16, max length
  2048.

Important: `PROMPT_FILE` is hard-coded to `/scratch/nico/...`; change it to a
path in `merged_repo/prompts/` before running.

### 7. Prepare Multilingual Data

`fourneurons/data/multilingual.py` prepares multilingual MCQ datasets:

- `prepare_mmmlu(config)` loads MMMLU for Italian, Spanish, Chinese, Hindi,
  plus a Russian MMLU dataset.
- It formats each row into the course prompt/answer schema.
- It stratifies train/validation/test splits by language.
- It saves locally under `/scratch/data/multilingual/mmmlu`.
- It pushes the dataset to `cs-552-2026-4neurons/mmmlu`.
- `prepare_mmmlu_jsonl(config)` writes JSONL splits under
  `/scratch/data/multilingual/mmmlu/splits`.

Command:

```bash
python -m fourneurons.data.multilingual
```

This reads `configs/basic_config.yml` for owner, languages, and data paths.

### 8. Augment Multilingual MCQ Choice Counts

`fourneurons/data/augment_mcq_choices.py` is intended to create additional
wrong choices so multilingual training covers variable option counts from 2 to
20, matching the course CI requirements.

The script contains three stages:

- `augment(...)`: generate additional distractors with a teacher model.
- `balance_augmented(...)`: rebalance the number of choices per language.
- `format_augmented(...)`: shuffle choices and write back to prompt/answer
  format.

Current CLI behavior only runs balancing and formatting because `augment(...)`
is commented out at the bottom.

Example command shape:

```bash
python -m fourneurons.data.augment_mcq_choices \
  multilingual mmmlu train Qwen/Qwen3-32B-AWQ mmmlu_more_qcms --cont
```

Before use, decide whether to uncomment the augmentation stage or run the
functions separately.

### 9. Generate Distilled Reasoning Traces

`fourneurons/data/distilled_reasoning_traces.py` uses a teacher model to
generate CoT completions for a prepared dataset. It keeps examples where the
boxed answer matches the gold label, then pushes the resulting dataset to the
Hugging Face Hub.

Example from the script:

```bash
python -m fourneurons.data.distilled_reasoning_traces \
  multilingual mmmlu_more_qcms train Qwen/Qwen3-32B-AWQ mmmlu_more_qcms
```

It expects formatted inputs under:

```text
/scratch/data/<benchmark>/<dataset>/splits/temp/<dataset>_<split>_formatted.jsonl
```

and writes:

```text
/scratch/data/<benchmark>/<run_name>/splits/<run_name>_<split>.jsonl
```

### 10. Train Multilingual LoRA

`fourneurons/scripts/train_multilingual.py` trains a LoRA adapter on a prepared
multilingual JSONL split.

Command:

```bash
python -m fourneurons.scripts.train_multilingual
```

Main assumptions:

- Input is
  `/scratch/data/multilingual/mmmlu_more_qcms/splits/mmmlu_more_qcms_train.jsonl`.
- Output is `/scratch/checkpoints/multilingual`.
- LoRA uses `r=16`, `lora_alpha=32`, `target_modules="all-linear"`.
- Training uses 2 epochs, batch size 4, grad accumulation 4, bf16, max length
  2048.

### 11. Evaluate Locally

There are two evaluation layers.

The generic generator is `fourneurons/evaluation/eval.py`:

```bash
python -m fourneurons.evaluation.eval \
  safety safetybench val safety_lora_eval \
  --checkpoint /scratch/checkpoints/safety/<run_id>/final
```

For a base-model evaluation:

```bash
python -m fourneurons.evaluation.eval \
  safety safetybench val base_safety_eval --base
```

It writes generations to:

```text
/scratch/results/<benchmark>/<dataset>/<run_name>/<split>_gens.jsonl
```

Then score with the course-style scorer:

```bash
python -m evaluate.score \
  --generations /scratch/results/safety/safetybench/safety_lora_eval/val_gens.jsonl \
  --benchmark safety \
  --output /scratch/results/safety/safetybench/safety_lora_eval/val_scored.json
```

There are also two one-off safety validation scripts:

```bash
python -m fourneurons.scripts.baseline_eval
python -m fourneurons.scripts.sp_baseline_eval
```

These generate safety validation completions and print the matching
`evaluate.score` command.

### 12. Patch Chat Templates

The course CI calls:

```python
tokenizer.apply_chat_template(messages, add_generation_prompt=True)
```

so any default system prompt or thinking-mode choice must be encoded in the
tokenizer's chat template.

`fourneurons/scripts/patch_chat_template.py` is intended to:

- load a tokenizer from a checkpoint,
- inject a default system prompt if the caller does not provide one,
- force Qwen3 thinking on or off,
- save the patched tokenizer.

Intended command shape:

```bash
python -m fourneurons.scripts.patch_chat_template \
  /scratch/checkpoints/safety/<run_id>/final \
  prompts/sp_general_qcm_think.txt \
  --thinking
```

Known issues in the current file:

- It imports `fourneurons.prompts.prompt_loader`, but `prompt_loader.py` is in
  top-level `prompts/`.
- The final call uses `args.prompt_file_pathname` instead of
  `args.prompt_file_path`.
- The prompt text is interpolated directly into Jinja and may need JSON
  escaping for quotes/backslashes/newlines.

### 13. Push To Hugging Face

`fourneurons/scripts/push_to_hf.py` is intended to push a full checkpoint and
tokenizer to a Hugging Face model repo, then push a `generation_config.json`.

Intended command shape:

```bash
python -m fourneurons.scripts.push_to_hf \
  cs-552-2026-4neurons/safety_model \
  /scratch/checkpoints/safety/<merged_or_full_checkpoint>
```

Known issue: the script uses `argparse` but does not import it.

Also note that LoRA adapters should be merged into full model weights before
course submission. The course CI expects a vLLM-loadable checkpoint with
weights, config, tokenizer files, chat template, and `generation_config.json`
at the repo root.

## Math Pipeline In The Math Branch

The math branch contains two related math SFT pipelines.

### A. Config-Driven OpenMathInstruct/OpenR1 LoRA

Files to port:

```text
configs/math_lora_sft.yml
configs/math_lora_sft_mixed.yml
fourneurons/data/math_sft.py
fourneurons/data/openr1_math.py
fourneurons/scripts/train_math_lora.py
fourneurons/scripts/merge_math_lora.py
fourneurons/scripts/modify_chat_template.py
docs/math_lora_finetuning.md
```

Training data:

- Primary dataset: `nvidia/OpenMathInstruct-2`, split `train_1M`.
- Optional mixed dataset: `open-r1/OpenR1-Math-220k`.

Formatting:

- User message is the math problem.
- Assistant target is a reasoning trace plus a final boxed answer.
- OpenMathInstruct examples wrap `generated_solution` in `<think>...</think>`
  and append `Therefore, the final answer is \boxed{...}.`
- OpenR1 examples choose the shortest correct and complete trace that fits the
  token budget, and use it verbatim.
- Labels mask the prompt tokens with `-100`, so loss is computed only on the
  assistant response.

Smoke test:

```bash
python -m fourneurons.scripts.train_math_lora \
  --config configs/math_lora_sft.yml \
  --dry-run \
  --max-train-samples 4
```

Train OpenMathInstruct-only:

```bash
python -m fourneurons.scripts.train_math_lora \
  --config configs/math_lora_sft.yml
```

Train mixed OpenMathInstruct + OpenR1:

```bash
python -m fourneurons.scripts.train_math_lora \
  --config configs/math_lora_sft_mixed.yml
```

Debug run:

```bash
python -m fourneurons.scripts.train_math_lora \
  --config configs/math_lora_sft.yml \
  --max-steps 100 \
  --max-train-samples 2000 \
  --output-dir /scratch/checkpoints/math_lora_sft_debug
```

Merge adapter:

```bash
python -m fourneurons.scripts.merge_math_lora \
  --config configs/math_lora_sft.yml \
  --adapter-dir /scratch/checkpoints/math_lora_sft_full \
  --output-dir /scratch/checkpoints/math_lora_sft_merged
```

For the mixed config:

```bash
python -m fourneurons.scripts.merge_math_lora \
  --config configs/math_lora_sft_mixed.yml \
  --adapter-dir /scratch/checkpoints/math_lora_sft_mixed \
  --output-dir /scratch/checkpoints/math_lora_sft_mixed_merged
```

Inject the default math system prompt into the merged checkpoint:

```bash
python -m fourneurons.scripts.modify_chat_template \
  --model-path /scratch/checkpoints/math_lora_sft_mixed_merged \
  --smoke-test
```

Score generated math validation completions:

```bash
python -m evaluate.score \
  --generations generations/math_lora_valid.jsonl \
  --benchmark math
```

Upload once the checkpoint is merged and validated:

```bash
hf upload cs-552-2026-4neurons/math_model \
  /scratch/checkpoints/math_lora_sft_mixed_merged \
  --commit-message "Upload merged math checkpoint"
```

### B. Simpler Competition-Math SFT

Files to port if this path is still useful:

```text
fourneurons/data/converters/convert_competition_math.py
fourneurons/scripts/simpler/simple_SFT.py
fourneurons/scripts/simpler/merge_simple_sft_lora.py
fourneurons/scripts/simpler/update_simple_sft_chat_template.py
fourneurons/scripts/simpler/upload_model_to_hf.py
docs/simple_sft_run_commands.md
```

Convert a saved competition math dataset:

```bash
python -m fourneurons.data.converters.convert_competition_math \
  /scratch/hf_cache/competition_math_train_saved \
  /scratch/hf_cache/competition_math_train_thinking_sft
```

Dry run:

```bash
python -m fourneurons.scripts.simpler.simple_SFT \
  --data-file /scratch/hf_cache/competition_math_train_thinking_sft \
  --dry-run \
  --max-train-samples 1
```

Train:

```bash
python -m fourneurons.scripts.simpler.simple_SFT \
  --model-name-or-path /scratch/hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e \
  --cache-dir /scratch/hf_cache \
  --data-file /scratch/hf_cache/competition_math_train_thinking_sft \
  --output-dir /scratch/checkpoints/qwen3-1.7b-competition-simple-sft-lora \
  --report-to wandb \
  --wandb-project 4neurons-math \
  --wandb-name simple-sft-run2
```

Merge:

```bash
python -m fourneurons.scripts.simpler.merge_simple_sft_lora \
  --adapter-dir /scratch/checkpoints/qwen3-1.7b-competition-simple-sft-lora \
  --output-dir /scratch/checkpoints/qwen3-1.7b-competition-simple-sft-merged \
  --overwrite
```

Inject/update the default system prompt:

```bash
python -m fourneurons.scripts.simpler.update_simple_sft_chat_template \
  --model-path /scratch/checkpoints/qwen3-1.7b-competition-simple-sft-merged \
  --smoke-test
```

Upload:

```bash
python -m fourneurons.scripts.simpler.upload_model_to_hf \
  --model-dir /scratch/checkpoints/qwen3-1.7b-competition-simple-sft-merged \
  --repo-id cs-552-2026-4neurons/math_model \
  --replace-existing \
  --commit-message "Upload merged simple-SFT math checkpoint"
```

## Recommended Command Order After Math Port

Once the math files are added to `merged_repo`, the clean math workflow should
be:

```bash
cd /scratch/leo/merged_repo
pip install -r requirements.txt
python -m fourneurons.scripts.download_base_model
python -m fourneurons.scripts.train_math_lora --config configs/math_lora_sft_mixed.yml --dry-run --max-train-samples 4
python -m fourneurons.scripts.train_math_lora --config configs/math_lora_sft_mixed.yml
python -m fourneurons.scripts.merge_math_lora --config configs/math_lora_sft_mixed.yml
python -m fourneurons.scripts.modify_chat_template --model-path /scratch/checkpoints/math_lora_sft_mixed_merged --smoke-test
```

Then generate validation completions and score:

```bash
python -m evaluate.score --generations <math_generations.jsonl> --benchmark math --output <math_scored.json>
```

Finally upload the merged checkpoint to:

```text
cs-552-2026-4neurons/math_model
```

## Porting Checklist

- Copy math configs into `merged_repo/configs/`.
- Copy `fourneurons/data/math_sft.py` and `fourneurons/data/openr1_math.py`.
- Copy `fourneurons/scripts/train_math_lora.py`,
  `fourneurons/scripts/merge_math_lora.py`, and
  `fourneurons/scripts/modify_chat_template.py`.
- Optionally copy the simpler competition-math path under
  `fourneurons/scripts/simpler/` and
  `fourneurons/data/converters/`.
- Reconcile imports with the merged repo layout, especially `prompts`.
- Avoid hard-coded user paths such as `/scratch/nico/...`; use repo-relative
  prompt files or CLI arguments.
- Keep existing merged-repo safety/multilingual scripts.
- After porting, run at least the math dry run and a Python compile check before
  launching a full training job.

## Known Current Issues To Fix Before Heavy Runs

- `fourneurons/scripts/patch_chat_template.py` has a bad import path and an
  `args.prompt_file_pathname` typo.
- `fourneurons/scripts/push_to_hf.py` is missing `import argparse`.
- Several scripts choose the first snapshot from `os.listdir(SNAPSHOT_DIR)`;
  this is convenient but not deterministic if multiple snapshots exist.
- Some paths are hard-coded to `/scratch/nico/...` or a specific snapshot hash.
- `fourneurons/evaluation/eval.py` accepts `--prompt_file_path` but does not
  currently apply that prompt when building prompts.
- `augment_mcq_choices.py` currently comments out the actual augmentation
  function in its CLI path.
