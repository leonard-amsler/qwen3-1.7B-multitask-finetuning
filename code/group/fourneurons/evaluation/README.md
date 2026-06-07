# eval.py

Run batched LLM evaluations with optional LoRA merging and scoring.

## Usage

```bash
python eval.py <benchmark> <dataset> <split> <run_name> \
  [--checkpoint PATH] [--base] [--prompt_file_path PATH]
```

## Arguments

- `benchmark`: One of `safety`, `multilingual`, `knowledge`, `math`
- `dataset`: Dataset name (e.g. `safetybench`, `mmlu`)
- `split`: Dataset split (e.g. `val`, `test`)
- `run_name`: Name for this run (used for output directory)

## Options

- `--checkpoint`: Path to LoRA checkpoint (required unless `--base`)
- `--base`: Evaluate base model only (no LoRA)
- `--prompt_file_path`: Path to file containing system prompt for evaluation (if not provided, tokenizer's chat template will not be modified)

## What it does

- Optionally merges LoRA weights into the base model
- Patches tokenizer to always include a system prompt
- Generates completions with vLLM
- Saves outputs to:
  ```
  /scratch/results/<benchmark>/<dataset>/<run_name>/<split>_gens.jsonl
  ```

## Scoring

After generation:

```bash
python -m evaluate.score \
  --generations <output_file> \
  --benchmark <benchmark> \
  --output <scored_file>
```

## Data format

Input must be a `.jsonl` file at:

```
/scratch/data/<benchmark>/<dataset>/splits/<dataset>_<split>.jsonl
```

Each row must contain:

```json
{"prompt": "..."}
```

Outputs will append:

```json
{"completions": ["..."]}
```