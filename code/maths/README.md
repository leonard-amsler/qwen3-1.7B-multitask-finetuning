# Maths Codespace Runbook

This directory contains the math reasoning pipeline used by the `fourneurons`
package: dataset builders, LoRA SFT training, GRPO/RL training, generation,
scoring, and result visualization.

Run Python commands from this directory unless a command says otherwise:

```bash
cd /scratch/leo/m3-repo/code/maths
```

The package is not installed by default. Running from `code/maths` puts both
`fourneurons` and the standalone `evaluate` scorer on `PYTHONPATH`.

## What Is Here

```text
fourneurons/data/          Dataset builders and SFT/RL formatting utilities.
fourneurons/scripts/       Training, prescoring, push, and helper entrypoints.
fourneurons/evaluation/    vLLM generation, checkpoint sweeps, and plots.
fourneurons/bash_scripts/  Run:AI non-interactive job submitters.
fourneurons/prompts/       Math system prompts.
evaluate/                 Local scorer mirroring the CI boxed-answer logic.
tools/                    Analysis utilities such as MATH-500 overlap checks.
docs/                     Older design notes and pipeline notes.
```

Large artifacts are expected under `/scratch`:

```text
/scratch/hf_cache                         Hugging Face model and dataset cache
/scratch/data/math/<dataset>/splits       JSONL datasets
/scratch/checkpoints/math/<run_id>        training checkpoints
/scratch/results/math/<dataset>/<run>     generations, merged models, scores, plots
/scratch/wandb                            W&B local files
```

## Environment

### Preferred: course/Run:AI image

Most training and evaluation code needs CUDA, vLLM, Transformers, PEFT, TRL,
datasets, and W&B. The least painful environment is the course image used by
the Run:AI wrappers:

```bash
IMAGE=ayushkumartarun/course-cs-552-standard:v1
```

On the cluster, configure Run:AI once:

```bash
runai config cluster rcp-caas-prod
runai login
runai config project course-cs-552-<gaspar>
```

Then use the scripts in `fourneurons/bash_scripts/`. Some wrappers still
default to an older repo path (`/scratch/leo/merged_repo`), so pass:

```bash
REPO_DIR=/scratch/leo/m3-repo/code/maths
```

The SFT wrappers `train_math_lora.sh` and `train_math_lora_mixed.sh` currently
assign `GASPAR` and `GROUP` inside the files. Edit those two variables before
submitting, or update the scripts to read them from the environment.

### Local or interactive pod setup

There is no maths-local `requirements.txt`. If you need to build an environment
manually, use Python 3.12 and install the packages imported by this directory:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu128 \
  numpy pandas requests datasets huggingface_hub transformers torch pyarrow \
  tokenizers sentencepiece orjson peft trl accelerate vllm wandb tqdm \
  matplotlib seaborn scikit-learn scipy pydantic pyyaml
```

Set common runtime environment variables:

```bash
export HF_HOME=/scratch/hf_cache
export HF_HUB_ENABLE_HF_TRANSFER=1
export WANDB_DIR=/scratch/wandb
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_TOKEN=hf_xxx
export WANDB_KEY=xxx
export WANDB_API_KEY="$WANDB_KEY"
```

Cache the base model. The scripts expect a local Qwen3-1.7B snapshot under
`/scratch/hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/`.

```bash
python -m fourneurons.scripts.download_base_model
```

## Important Data Format

All math datasets are converted to JSONL. Training rows normally contain:

```json
{"prompt": "...", "answer": "...", "completion": "...\\boxed{answer}"}
```

Evaluation rows need at least:

```json
{"prompt": "...", "answer": "..."}
```

Generation outputs append:

```json
{"completions": ["...\\boxed{...}", "..."]}
```

The scorer extracts the last `\boxed{...}` and compares it to `answer`, so
training completions and model outputs should end with a boxed final answer.

## Build SFT and Evaluation Data

Build OpenMathInstruct SFT data:

```bash
python -m fourneurons.data.openMathInstruct_build_jsonl
```

Outputs:

```text
/scratch/data/math/openmathinstruct/splits/openmathinstruct_train.jsonl
/scratch/data/math/openmathinstruct/splits/openmathinstruct_val.jsonl
/scratch/data/math/openmathinstruct/splits/openmathinstruct_test.jsonl
```

Build OpenR1Math SFT data:

```bash
python -m fourneurons.data.openR1math_build_jsonl
```

Outputs:

```text
/scratch/data/math/openR1math/splits/openR1math_train.jsonl
/scratch/data/math/openR1math/splits/openR1math_val.jsonl
/scratch/data/math/openR1math/splits/openR1math_test.jsonl
```

The OpenR1 builder keeps the shortest generation that is both reasoning-complete
and math-verified correct, then appends a canonical boxed answer.

Build optional evaluation datasets:

```bash
python -m fourneurons.data.competitionmath_build_jsonl
python -m fourneurons.data.math500_build_jsonl
```

Outputs:

```text
/scratch/data/math/competitionmath/splits/competitionmath_{train,val,test,full}.jsonl
/scratch/data/math/math500/splits/math500_full.jsonl
```

Quick checks:

```bash
head -n 1 /scratch/data/math/openmathinstruct/splits/openmathinstruct_train.jsonl
head -n 1 /scratch/data/math/openR1math/splits/openR1math_train.jsonl
head -n 1 /scratch/data/math/math500/splits/math500_full.jsonl
```

## Build GRPO/RL Data

The RL pipeline uses prompt-only data. It samples completions from the current
SFT checkpoint, scores them, and keeps "frontier" examples where the model is
not always right.

Build the prompt pool:

```bash
python -m fourneurons.data.math_rl_prompt_pool \
  --source-limit openmathinstruct=50000 \
  --source-limit openR1math=50000 \
  --source-limit numinamath_1_5=50000 \
  --source-limit nemotron_math_v2=50000
```

Default outputs:

```text
/scratch/data/math/rl_prompt_pool/splits/rl_prompt_pool_train.jsonl
/scratch/data/math/rl_prompt_pool/splits/rl_prompt_pool_train.summary.json
```

For a named 40k pool compatible with the existing Run:AI wrapper:

```bash
python -m fourneurons.data.math_rl_prompt_pool \
  --dataset_name rl_prompt_pool_40k \
  --out_dir /scratch/data/math/rl_prompt_pool_40k/splits \
  --source-limit openmathinstruct=10000 \
  --source-limit openR1math=10000 \
  --source-limit numinamath_1_5=10000 \
  --source-limit nemotron_math_v2=10000
```

Pre-score the pool with 8 generations from an SFT checkpoint:

```bash
python -m fourneurons.scripts.prescore_math_rl_pool \
  --dataset rl_prompt_pool \
  --run_name rl_pool_prescore_mixed_ckpt4458_tok16k_n8 \
  --checkpoint /scratch/checkpoints/math/qwen3-1.7b-lora-math-mixed_20260526-220009/checkpoint-4458 \
  --generation_batch_size 16
```

This runs generation and scoring. The important outputs are:

```text
/scratch/results/math/rl_prompt_pool/<run_name>/train_gens.jsonl
/scratch/results/math/rl_prompt_pool/<run_name>/train_scored.json
```

For a smoke pre-score:

```bash
python -m fourneurons.scripts.prescore_math_rl_pool \
  --dataset rl_prompt_pool \
  --run_name rl_pool_prescore_smoke20 \
  --max_num_samples 20 \
  --generation_batch_size 4
```

Select the frontier split:

```bash
python -m fourneurons.data.select_math_rl_frontier \
  --pool /scratch/data/math/rl_prompt_pool/splits/rl_prompt_pool_train.jsonl \
  --scored /scratch/results/math/rl_prompt_pool/rl_pool_prescore_mixed_ckpt4458_tok16k_n8/train_scored.json \
  --output /scratch/data/math/rl_frontier/splits/rl_frontier_train.jsonl \
  --include-correct 1-7
```

Use `--include-correct 1-3` for a narrower hard-frontier run, or `--max_rows N`
to cap the selected training set.

Run:AI prescore wrapper:

```bash
GASPAR=<gaspar> GROUP=<gXX> REPO_DIR=/scratch/leo/m3-repo/code/maths \
  ./fourneurons/bash_scripts/prescore_math_rl_pool.sh
```

## Train Models

### OpenMathInstruct-only LoRA SFT

Local or inside an interactive GPU pod:

```bash
python -m fourneurons.scripts.train_math
```

Defaults:

```text
data:      /scratch/data/math/openmathinstruct/splits
prompt:    fourneurons/prompts/math.txt
train cap: 250000 examples
val cap:   256 examples
epochs:    4
context:   4096 tokens
output:    /scratch/checkpoints/math/<timestamp>
final:     /scratch/checkpoints/math/<timestamp>/final
```

Run:AI, after editing `GASPAR` and `GROUP` at the top of the script:

```bash
REPO_DIR=/scratch/leo/m3-repo/code/maths \
  ./fourneurons/bash_scripts/train_math_lora.sh
```

### Mixed OpenMathInstruct + OpenR1Math LoRA SFT

Local or inside an interactive GPU pod:

```bash
python -m fourneurons.scripts.train_math_mixed
```

Useful overrides:

```bash
FOURNEURONS_RUN_NAME=qwen3-1.7b-lora-math-mixed-16k \
WANDB_PROJECT=math-sft \
python -m fourneurons.scripts.train_math_mixed
```

Resume:

```bash
OUTPUT_DIR=/scratch/checkpoints/math/<run_id> \
RESUME_FROM_CHECKPOINT=latest \
FOURNEURONS_RUN_ID=<run_id> \
WANDB_NAME=<run_id> \
WANDB_RUN_ID=<wandb-run-id> \
WANDB_RESUME=must \
python -m fourneurons.scripts.train_math_mixed
```

Defaults:

```text
data:      50k total valid train examples, half OpenMathInstruct and half OpenR1Math
val cap:   256 total valid validation examples
epochs:    4
context:   16384 tokens
LoRA:      r=16, alpha=32, dropout=0.05, target_modules=all-linear
output:    /scratch/checkpoints/math/<run_id>
```

Run:AI, after editing `GASPAR` and `GROUP` at the top of the script:

```bash
REPO_DIR=/scratch/leo/m3-repo/code/maths \
  ./fourneurons/bash_scripts/train_math_lora_mixed.sh
```

Resume the named long run via wrapper, with the same `GASPAR`/`GROUP` note:

```bash
REPO_DIR=/scratch/leo/m3-repo/code/maths \
  ./fourneurons/bash_scripts/resume_math_lora_mixed_long.sh
```

### GRPO/RL math model

Local or inside an interactive GPU pod:

```bash
MATH_RL_INIT_CHECKPOINT=/scratch/checkpoints/math/<sft_run>/checkpoint-4458 \
MATH_RL_TRAIN_FILE=/scratch/data/math/rl_frontier/splits/rl_frontier_train.jsonl \
FOURNEURONS_RUN_NAME=qwen3-1.7b-lora-math-rl \
python -m fourneurons.scripts.train_math_rl
```

Defaults are conservative for a first run:

```text
init checkpoint: /scratch/checkpoints/math/qwen3-1.7b-lora-math-mixed_20260526-220009/checkpoint-4458
train file:      /scratch/data/math/rl_frontier/splits/rl_frontier_train.jsonl
train cap:       4096 prompts
epochs:          1
group size:      8 generations
completion cap:  16384 tokens
context cap:     20000 tokens
learning rate:   1e-6
KL beta:         0.001
loss type:       dapo
reward:          +1 correct boxed, -0.25 wrong boxed, -0.75 missing box
output:          /scratch/checkpoints/math/<run_id>
```

Common smoke settings:

```bash
MATH_RL_MAX_TRAINING_SAMPLES=380 \
MATH_RL_MAX_VALIDATION_SAMPLES=0 \
MATH_RL_MAX_STEPS=20 \
MATH_RL_KL_BETA=0 \
MATH_RL_TRAINER_BF16=0 \
python -m fourneurons.scripts.train_math_rl
```

Run:AI smoke wrapper:

```bash
GASPAR=<gaspar> GROUP=<gXX> REPO_DIR=/scratch/leo/m3-repo/code/maths \
  ./fourneurons/bash_scripts/train_math_rl.sh
```

The wrapper defaults to slicing the first 1000 existing pool generations,
rescoring them, selecting frontier rows with `1-7` correct out of 8, then
training for a short smoke run. Set `BUILD_FIRST1K_FRONTIER=0` and
`MATH_RL_TRAIN_FILE=...` to use an already prepared frontier split.

## Generate and Evaluate

The main evaluator is:

```bash
python -m fourneurons.evaluation.eval \
  <benchmark> <dataset> <split> <run_name> \
  --checkpoint <lora_or_full_checkpoint> \
  --num_generations 8 \
  --max_tokens 4096 \
  --prompt_file_path fourneurons/prompts/math.txt
```

It expects input data at:

```text
/scratch/data/<benchmark>/<dataset>/splits/<dataset>_<split>.jsonl
```

It writes:

```text
/scratch/results/<benchmark>/<dataset>/<run_name>/<split>_gens.jsonl
/scratch/results/<benchmark>/<dataset>/<run_name>/merged
```

Use a fresh `run_name` for each checkpoint. If the `merged` directory already
exists, the evaluator reuses it and will not re-merge a different adapter.

### Single checkpoint

Evaluate one checkpoint on MATH-500 with 8 generations:

```bash
python -m fourneurons.evaluation.eval \
  math math500 full <eval_run_name> \
  --checkpoint /scratch/checkpoints/math/<run_id>/checkpoint-4458 \
  --num_generations 8 \
  --max_tokens 16384 \
  --prompt_file_path fourneurons/prompts/math.txt \
  --generation_batch_size 16 \
  --resume_generation
```

Evaluate the final adapter:

```bash
python -m fourneurons.evaluation.eval \
  math competitionmath full <run_id>_final \
  --checkpoint /scratch/checkpoints/math/<run_id>/final \
  --num_generations 8 \
  --max_tokens 4096 \
  --prompt_file_path fourneurons/prompts/math.txt
```

Evaluate the base model:

```bash
python -m fourneurons.evaluation.eval \
  math math500 full qwen3_base_math500 \
  --base \
  --num_generations 8 \
  --max_tokens 4096 \
  --prompt_file_path fourneurons/prompts/math.txt
```

### All checkpoints in a run

```bash
python -m fourneurons.evaluation.eval_all_checkpoints \
  /scratch/checkpoints/math/<run_id> \
  --benchmark math \
  --dataset math500 \
  --split full \
  --num_generations 8 \
  --max_tokens 16384 \
  --prompt_file_path fourneurons/prompts/math.txt \
  --max_num_samples 500 \
  --skip_existing \
  --skip_scored
```

This evaluates each `checkpoint-*` directory and `final` if present, then
scores each generation file. Result directories are named:

```text
/scratch/results/math/math500/<run_id>_checkpoint-<step>
/scratch/results/math/math500/<run_id>_final
```

Run:AI wrapper:

```bash
GASPAR=<gaspar> GROUP=<gXX> REPO_DIR=/scratch/leo/m3-repo/code/maths \
  ./fourneurons/bash_scripts/eval_math_all_checkpoints.sh \
  /scratch/checkpoints/math/<run_id> 16384 500
```

### Decoding grid for one checkpoint

```bash
python -m fourneurons.evaluation.eval_decoding_grid \
  /scratch/checkpoints/math/<run_id>/checkpoint-4458 \
  --benchmark math \
  --dataset math500 \
  --split full \
  --num_generations 8 \
  --max_tokens 16384 \
  --max_num_samples 500 \
  --temperatures 0.5,0.6,0.7 \
  --top_ps 0.8,0.9,0.95 \
  --top_k 20 \
  --prompt_file_path fourneurons/prompts/math.txt
```

This reuses one merged model directory and writes a CSV summary such as:

```text
/scratch/results/math/math500/decodegrid_<run>_<checkpoint>_decoding_grid.csv
```

Run:AI wrapper:

```bash
GASPAR=<gaspar> GROUP=<gXX> REPO_DIR=/scratch/leo/m3-repo/code/maths \
  ./fourneurons/bash_scripts/eval_math_decoding_grid.sh \
  /scratch/checkpoints/math/<run_id>/checkpoint-4458 16384 500
```

## Score Generations

Score any generations JSONL:

```bash
python -m evaluate.score \
  --generations /scratch/results/math/math500/<eval_run_name>/full_gens.jsonl \
  --benchmark math \
  --output /scratch/results/math/math500/<eval_run_name>/full_scored.json
```

The output JSON contains:

```text
benchmark_method
n_problems
n_completions
metrics.pass@1
metrics.pass@8
metrics.box_compliance
detailed_results
```

With fewer than 8 completions per problem, the scorer reports only the pass@k
values that are valid for the number of completions. `box_compliance` is the
fraction of completions with an extractable boxed answer.

Optional W&B logging:

```bash
python -m evaluate.score_wandb \
  --generations /scratch/results/math/math500/<eval_run_name>/full_gens.jsonl \
  --benchmark math \
  --output /scratch/results/math/math500/<eval_run_name>/full_scored.json \
  --run_name <eval_run_name>_scoring
```

## Visualize Results

All visualization commands consume existing scored result directories. Each
result directory must contain `<split>_scored.json`; per-category and per-level
plots also need `<split>_gens.jsonl`.

### Compare a list of models

```bash
python -m fourneurons.evaluation.compare_metrics \
  /scratch/results/math/math500/run_a \
  /scratch/results/math/math500/run_b \
  /scratch/results/math/math500/run_c \
  --split full \
  --labels base mixed_sft grpo \
  --output_dir /scratch/results/math/math500/compare_metrics
```

Outputs:

```text
metrics.csv
metrics.png
metrics_bars.png
```

### Compare by problem category

For CompetitionMath or MATH-500, use the `type` field:

```bash
python -m fourneurons.evaluation.compare_categories \
  /scratch/results/math/math500/run_a \
  /scratch/results/math/math500/run_b \
  --split full \
  --category_field type \
  --labels mixed_sft grpo \
  --output_dir /scratch/results/math/math500/compare_categories
```

Outputs:

```text
category_metrics.csv
all_categories_metrics.png
<one png per category>
```

If `--category_field` is omitted, the script tries `category`, `type`, then
`level`.

### Compare by difficulty level

```bash
python -m fourneurons.evaluation.compare_levels \
  /scratch/results/math/math500/run_a \
  /scratch/results/math/math500/run_b \
  --split full \
  --level_field level \
  --labels mixed_sft grpo \
  --output_dir /scratch/results/math/math500/compare_levels
```

Outputs:

```text
level_metrics.csv
all_levels_metrics.png
<one png per level>
```

If `--level_field` is omitted, the script tries `level`, `difficulty`, then
`math500_level`.

### Plot metrics across checkpoints

After `eval_all_checkpoints`:

```bash
python -m fourneurons.evaluation.score_all_checkpoints \
  /scratch/results/math/math500 \
  --run_id <run_id> \
  --split full \
  --benchmark math \
  --score_missing
```

Outputs:

```text
<run_id>_full_checkpoint_metrics.csv
<run_id>_full_checkpoint_metrics.png
```

## Submission Helpers

Patch a tokenizer chat template to inject the default system prompt:

```bash
python -m fourneurons.scripts.patch_chat_template \
  /scratch/checkpoints/math/<run_id>/final \
  fourneurons/prompts/math.txt \
  --thinking
```

Push a full model directory to Hugging Face:

```bash
python -m fourneurons.scripts.push_to_hf \
  cs-552-2026-4neurons/math_model \
  /scratch/checkpoints/math/<merged_or_full_checkpoint>
```

The push helper expects a full vLLM-loadable model directory. A raw LoRA adapter
directory is not enough for CI submission unless it has already been merged
with the base model.

## Useful Analysis Utility

Check overlap between actual SFT training rows and MATH-500:

```bash
python tools/math500_overlap_check.py \
  --target-per-dataset 125000 \
  --max-length 4096 \
  --output /scratch/results/math500_overlap_report.json
```

This samples the same length-filtered OpenMathInstruct/OpenR1 rows used by SFT
and reports exact, TF-IDF character n-gram, and MinHash-style overlaps.

## Known Caveats

- `fourneurons.data.format_for_sft` imports `prompts.prompt_loader`, but this
  checkout does not include a source `code/maths/prompts/prompt_loader.py`.
  The SFT scripts need that helper on `PYTHONPATH`, or the import should be
  changed to a local file read / `fourneurons.prompts.prompt_loader`.
- Several scripts choose the first local Qwen snapshot with `os.listdir`.
  Keep only one Qwen3-1.7B snapshot under `/scratch/hf_cache/.../snapshots`, or
  update the script to use a deterministic path.
- The Run:AI wrappers default to `/scratch/leo/merged_repo`; pass
  `REPO_DIR=/scratch/leo/m3-repo/code/maths` for this checkout.
- Use a new evaluation `run_name` for every different checkpoint unless you
  intentionally reuse `--merged_model_dir`.
- `score_wandb.py` logs to a hard-coded W&B project name (`safety-eval`) even
  when scoring math generations.
- The safety and multilingual scripts in this tree are inherited from the
  broader project. The math pipeline above is the reproducible path for this
  directory.

## Minimal End-to-End Reproduction

```bash
cd /scratch/leo/m3-repo/code/maths
export HF_HOME=/scratch/hf_cache
export WANDB_DIR=/scratch/wandb

python -m fourneurons.scripts.download_base_model
python -m fourneurons.data.openMathInstruct_build_jsonl
python -m fourneurons.data.openR1math_build_jsonl
python -m fourneurons.data.math500_build_jsonl

python -m fourneurons.scripts.train_math_mixed

python -m fourneurons.evaluation.eval_all_checkpoints \
  /scratch/checkpoints/math/<run_id> \
  --benchmark math \
  --dataset math500 \
  --split full \
  --num_generations 8 \
  --max_tokens 16384 \
  --prompt_file_path fourneurons/prompts/math.txt \
  --max_num_samples 500

python -m fourneurons.evaluation.score_all_checkpoints \
  /scratch/results/math/math500 \
  --run_id <run_id> \
  --split full \
  --benchmark math
```
