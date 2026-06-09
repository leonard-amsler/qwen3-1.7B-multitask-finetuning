# Math SFT Pipeline

This is the current math reasoning workflow in `merged_repo`: build the JSONL datasets, train either OpenMathInstruct-only or mixed OpenMathInstruct + OpenR1Math LoRA SFT, evaluate pass@8, patch the merged tokenizer chat template, and push the merged model to Hugging Face.

Run commands from the repo root:

```bash
cd /scratch/leo/merged_repo
```

## 1. Build The Datasets

### OpenMathInstruct

Builder:

```text
fourneurons/data/openMathInstruct_build_jsonl.py
```

It loads:

```text
nvidia/OpenMathInstruct-2, split=train_1M
```

and writes:

```text
/scratch/data/math/openmathinstruct/splits/openmathinstruct_train.jsonl
/scratch/data/math/openmathinstruct/splits/openmathinstruct_val.jsonl
/scratch/data/math/openmathinstruct/splits/openmathinstruct_test.jsonl
```

Build it:

```bash
python -m fourneurons.data.openMathInstruct_build_jsonl
```

### OpenR1Math

Builder:

```text
fourneurons/data/openR1math_build_jsonl.py
```

It loads:

```text
open-r1/OpenR1-Math-220k, config=default, split=train
```

and writes:

```text
/scratch/data/math/openR1math/splits/openR1math_train.jsonl
/scratch/data/math/openR1math/splits/openR1math_val.jsonl
/scratch/data/math/openR1math/splits/openR1math_test.jsonl
```

Build it:

```bash
python -m fourneurons.data.openR1math_build_jsonl
```

The OpenR1 builder selects the shortest generation that is both reasoning-complete and math-verified correct, then appends a canonical final answer box. This is intentional because the evaluator extracts the last `\boxed{...}`.

### Sanity Checks

```bash
ls -lh /scratch/data/math/openmathinstruct/splits/
ls -lh /scratch/data/math/openR1math/splits/
head -n 1 /scratch/data/math/openmathinstruct/splits/openmathinstruct_val.jsonl
head -n 1 /scratch/data/math/openR1math/splits/openR1math_val.jsonl
```

Every row should contain at least:

```json
{
  "prompt": "problem statement",
  "answer": "final answer",
  "completion": "assistant training target ending with \\boxed{answer}"
}
```

## 2. Train The Model

There are two training entrypoints.

OpenMathInstruct-only:

```text
fourneurons/scripts/train_math.py
```

Mixed OpenMathInstruct + OpenR1Math:

```text
fourneurons/scripts/train_math_mixed.py
```

Both use:

```text
fourneurons/prompts/math.txt
```

as the system prompt during SFT formatting.

### Non-Interactive Run:AI Submitter

Submitter:

```text
fourneurons/bash_scripts/train_math_lora.sh
```

It loads `.env` from the repo root if present. Example `.env`:

```bash
HF_TOKEN=hf_xxx
WANDB_KEY=xxx
```

It also sets:

```bash
PYTORCH_ALLOC_CONF=expandable_segments:True
```

to reduce CUDA memory fragmentation.

By default, the submitter runs OpenMathInstruct-only training:

```bash
./fourneurons/bash_scripts/train_math_lora.sh
```

To run mixed training, override the command:

```bash
EXTRA_TRAIN_CMD='python -m fourneurons.scripts.train_math_mixed' \
  ./fourneurons/bash_scripts/train_math_lora.sh
```

Useful Run:AI commands are printed after submission:

```bash
runai describe job <job-name> -p course-cs-552-<gaspar>
runai logs -f <job-name> -p course-cs-552-<gaspar>
runai bash <job-name> -p course-cs-552-<gaspar>
runai delete job <job-name> -p course-cs-552-<gaspar>
```

### Current Training Settings

The current memory-safe settings are:

```text
per_device_train_batch_size = 1
gradient_accumulation_steps = 16
per_device_eval_batch_size = 1
max_length = 4096
eval_steps = 500
```

The mixed trainer token-filters before applying sample caps: it formats each JSONL row, drops rows whose formatted token count exceeds `4096`, and keeps collecting until it reaches the target count. With current defaults this means `50k` valid train examples total, split as `25k` OpenMathInstruct and `25k` OpenR1Math when enough valid rows exist.

Checkpoints are written under:

```text
/scratch/checkpoints/math/<run_id>/
```

The final adapter is:

```text
/scratch/checkpoints/math/<run_id>/final
```

Epoch/checkpoint directories such as `checkpoint-3125` are also valid LoRA adapter checkpoints for evaluation.

## 3. Evaluate The Adapter

Evaluator:

```text
fourneurons/evaluation/eval.py
```

It does three things:

- merges the LoRA adapter into the base Qwen3-1.7B model,
- runs vLLM generation,
- writes a JSONL file with a `completions` list per row.

It now supports:

- `--num_generations 8` for pass@8 generation,
- `--prompt_file_path fourneurons/prompts/math.txt` to use the same system prompt as training,
- `--max_num_samples N` to evaluate on a subset,
- PEFT bitsandbytes dispatch disabling during merge.

Evaluate OpenMathInstruct validation with pass@8:

```bash
python fourneurons/evaluation/eval.py \
  math openmathinstruct val math_adapter_pass8_<run_id> \
  --checkpoint /scratch/checkpoints/math/<run_id>/final \
  --num_generations 8 \
  --prompt_file_path fourneurons/prompts/math.txt
```

For a specific intermediate checkpoint:

```bash
python fourneurons/evaluation/eval.py \
  math openmathinstruct val math_adapter_pass8_<run_id>_ckpt3125 \
  --checkpoint /scratch/checkpoints/math/<run_id>/checkpoint-3125 \
  --num_generations 8 \
  --prompt_file_path fourneurons/prompts/math.txt
```

For a quick smoke evaluation:

```bash
python fourneurons/evaluation/eval.py \
  math openmathinstruct val smoke_<run_id> \
  --checkpoint /scratch/checkpoints/math/<run_id>/checkpoint-3125 \
  --num_generations 2 \
  --max_num_samples 20 \
  --prompt_file_path fourneurons/prompts/math.txt
```

The evaluator writes:

```text
/scratch/results/math/openmathinstruct/<eval_run_name>/val_gens.jsonl
/scratch/results/math/openmathinstruct/<eval_run_name>/merged/
```

Important: use a fresh `<eval_run_name>` for every new checkpoint. If the `merged/` directory already exists, the evaluator reuses it and will not re-merge the adapter.

Run generation evaluation and scoring for every checkpoint in one training run:

```bash
python -m fourneurons.evaluation.eval_all_checkpoints \
  /scratch/checkpoints/math/<run_id> \
  --benchmark math \
  --dataset openmathinstruct \
  --split val \
  --num_generations 8 \
  --max_tokens 4096 \
  --max_num_samples 1000 \
  --prompt_file_path fourneurons/prompts/math.txt
```

This creates result directories named like `<run_id>_checkpoint-500` and `<run_id>_final`, each with `<split>_gens.jsonl` and `<split>_scored.json`.

## 4. Score pass@1 And pass@8

Score with:

```bash
python -m evaluate.score \
  --generations /scratch/results/math/openmathinstruct/<eval_run_name>/val_gens.jsonl \
  --benchmark math \
  --output /scratch/results/math/openmathinstruct/<eval_run_name>/val_scored.json
```

With 8 completions per row, the scorer automatically reports both:

```text
pass@1=...
pass@8=...
```

Inspect details:

```bash
less /scratch/results/math/openmathinstruct/<eval_run_name>/val_scored.json
```

Plot all scored checkpoint result directories:

```bash
python -m fourneurons.evaluation.score_all_checkpoints \
  /scratch/results/math/openmathinstruct \
  --run_id <run_id> \
  --split val
```

This writes:

```text
/scratch/results/math/openmathinstruct/<run_id>_val_checkpoint_metrics.csv
/scratch/results/math/openmathinstruct/<run_id>_val_checkpoint_metrics.png
```

Quick check that the generations file has 8 completions:

```bash
python - <<'PY'
import json
path = "/scratch/results/math/openmathinstruct/<eval_run_name>/val_gens.jsonl"
with open(path) as f:
    row = json.loads(next(f))
print(row.keys())
print(len(row["completions"]))
print(row["completions"][0][:500])
PY
```

## 5. Patch The Merged Chat Template

The course CI calls `tokenizer.apply_chat_template(messages, add_generation_prompt=True)` without passing your custom system prompt. Therefore, before upload, patch the merged model tokenizer so it injects the math system prompt by default.

Patch the merged evaluator output:

```bash
python fourneurons/scripts/patch_chat_template.py \
  /scratch/results/math/openmathinstruct/<eval_run_name>/merged \
  fourneurons/prompts/math.txt \
  --thinking
```

This writes into the merged directory in place unless `--output_dir` is provided. The script also prints a rendered prompt for verification.

Use `--thinking` if you want Qwen thinking mode forced on in the template. Omit it to force thinking off.

## 6. Push To Hugging Face

Push only a merged full model directory, not a LoRA adapter directory. Before pushing, verify the merged directory has full model files:

```bash
ls /scratch/results/math/openmathinstruct/<eval_run_name>/merged
```

Expected files include:

```text
config.json
generation_config.json
model*.safetensors
tokenizer_config.json
chat_template.jinja
```

Do not push directories that only contain:

```text
adapter_config.json
adapter_model.safetensors
```

Current push script:

```text
fourneurons/scripts/push_to_hf.py
```

Before using it, make sure it imports `argparse` and loads with `torch_dtype="bfloat16"` rather than deprecated `dtype="bfloat16"`.

Push command:

```bash
export HF_TOKEN=hf_xxx

python -m fourneurons.scripts.push_to_hf \
  cs-552-2026-4neurons/math_model \
  /scratch/results/math/openmathinstruct/<eval_run_name>/merged
```

## 7. Expected End-To-End Commands

OpenMathInstruct-only:

```bash
cd /scratch/leo/merged_repo

python -m fourneurons.data.openMathInstruct_build_jsonl

./fourneurons/bash_scripts/train_math_lora.sh

python fourneurons/evaluation/eval.py \
  math openmathinstruct val math_adapter_pass8_<run_id> \
  --checkpoint /scratch/checkpoints/math/<run_id>/final \
  --num_generations 8 \
  --prompt_file_path fourneurons/prompts/math.txt

python -m evaluate.score \
  --generations /scratch/results/math/openmathinstruct/math_adapter_pass8_<run_id>/val_gens.jsonl \
  --benchmark math \
  --output /scratch/results/math/openmathinstruct/math_adapter_pass8_<run_id>/val_scored.json

python fourneurons/scripts/patch_chat_template.py \
  /scratch/results/math/openmathinstruct/math_adapter_pass8_<run_id>/merged \
  fourneurons/prompts/math.txt \
  --thinking
```

Mixed OpenMathInstruct + OpenR1Math:

```bash
cd /scratch/leo/merged_repo

python -m fourneurons.data.openMathInstruct_build_jsonl
python -m fourneurons.data.openR1math_build_jsonl

EXTRA_TRAIN_CMD='python -m fourneurons.scripts.train_math_mixed' \
  ./fourneurons/bash_scripts/train_math_lora.sh

python fourneurons/evaluation/eval.py \
  math openmathinstruct val mixed_pass8_<run_id> \
  --checkpoint /scratch/checkpoints/math/<run_id>/final \
  --num_generations 8 \
  --prompt_file_path fourneurons/prompts/math.txt

python -m evaluate.score \
  --generations /scratch/results/math/openmathinstruct/mixed_pass8_<run_id>/val_gens.jsonl \
  --benchmark math \
  --output /scratch/results/math/openmathinstruct/mixed_pass8_<run_id>/val_scored.json

python fourneurons/scripts/patch_chat_template.py \
  /scratch/results/math/openmathinstruct/mixed_pass8_<run_id>/merged \
  fourneurons/prompts/math.txt \
  --thinking
```
