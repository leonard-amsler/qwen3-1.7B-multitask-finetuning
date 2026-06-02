# General Knowledge — reproduction pipelines (team 4neurons)

Required base model: **`Qwen/Qwen3-1.7B`**. Each script reproduces one version end-to-end
(data → SFT/DPO → merge → **dev_full eval**) and writes a vLLM-ready checkpoint to
`/scratch/checkpoints/gk_<version>/vllm`.

> Scripts **do not push to HuggingFace** — they stop after local evaluation.
> To submit, manually upload the corresponding `vllm/` directory.

## What we did (summary)

We start from CoTs distilled from a larger teacher (14B) and search for the best
format / reasoning / consistency trade-off for a small 1.7B student. The through-line:

1. **Format** — a weak LoRA (V8) does not enforce `\boxed{}`; strengthening it (V9)
   gains +21 pts. Format is driven by LoRA strength.
2. **Loops** — the 1.7B self-distills into repetitive loops (59% of traces). Filtering
   repetitions and keeping the shortest clean correct trace (V9b) fixes truncations.
3. **On-policy > off-policy** — the 1.7B learns better from its own cleaned reasoning
   (V9b) than from an overly complex 14B teacher (V10, contrastive distillation).
4. **Preference** — timid DPO does nothing (V11); strong on-policy DPO on v9b's own
   draws (V11b) gains +2.0 pass@1 points without breaking pass@8.

Full numbers and diagnostics: see `../PLAN_FINAL.md`.

## Which script runs which version

**Every script is self-contained and can be run in ANY order.** If a prerequisite is
missing (data, cache, or even the v9b model for DPO), it is built automatically via
the `ensure_*` helpers in `_lib.sh`. You can run `run_v9.sh` without ever building V8
data — the script generates it.

| Script | Version | Output | Auto-builds if missing |
|---|---|---|---|
| `run_v6.sh` | **V6** — long 14B distill | `gk_v6/vllm` | distill caches + `train_v6` |
| `run_v9.sh` | **V9** — re-SFT on V8 data, strong LoRA | `gk_v9/vllm` | **`train_v8` data** (+ `train_v6`) |
| `run_v9b.sh` | **V9b** — V9 + anti-loop `select_best` | `gk_v9b/vllm` | V8 data + `train_v6` + `train_v9b` |
| `run_v10.sh` | **V10** — contrastive 14B distill | `gk_v10/vllm` | v5 + v6_long caches |
| `run_v11.sh` | **V11** — timid DPO on v9b | `gk_v11/vllm` | `gk_v9b` (full chain) + DPO pairs |
| `run_v11b.sh` | **V11b** — strong DPO (FINAL) | `gk_v11b/vllm` | `gk_v9b` (full chain) + DPO pairs |

> There is no dedicated "V8" script. **train_v8** (baseline self-distillation) is built
> inside `ensure_v8_data()` in `_lib.sh` and is invoked by V9 and V9b only when
> `train_v8` does not exist yet.

### Shared helpers (`_lib.sh`)

All build steps are idempotent `ensure_*` functions (skip if the artifact already
exists) that chain together:

```
ensure_distill_v5 ─┐
ensure_distill_v6_long ─┴─► ensure_train_v6 ─► ensure_v8_data ─► ensure_train_v9b ─► ensure_model_v9b ─► ensure_dpo_pairs
```

Nothing is recomputed unnecessarily: if you already ran V9b, running V11b will reuse
V8 data and the v9b SFT checkpoint from disk.

## Code files used

- `fourneurons/distill/distill.py` — CoT distillation from 14B
  (`--reasoning_style contrastive` for V10). Prompts in `fourneurons/distill/prompts.py`.
- `fourneurons/distill/self_distill.py` — self-distillation from 1.7B
  (`--select_best` / `--max_thinking_chars` = V9b anti-loop filtering).
- `fourneurons/data/build_train.py` — SFT dataset assembly, layered caches (last-wins).
- `fourneurons/scripts/train.py` — LoRA SFT.
- `fourneurons/scripts/build_dpo_pairs.py` + `train_dpo.py` — DPO pipeline (V11/V11b).
- `fourneurons/scripts/merge_lora.py` — merge adapter + **bake `enable_thinking=ON`**
  into the chat template + write `generation_config.json`.
- `fourneurons/eval/run_inference.py` + `report_by_bucket.py` — dev_full evaluation.

## How to run

```bash
chmod +x pipelines/*.sh

# Any script, any order. Examples:
./pipelines/run_v11b.sh    # FINAL MODEL (builds the full chain as needed)
./pipelines/run_v9.sh      # builds train_v8 data on its own if missing
./pipelines/run_v6.sh
./pipelines/run_v10.sh
```

Shared settings in `_lib.sh` (override via environment variables):
`DATA`, `CKPT`, `EVAL`, `BASE`, `DEV`, and `N_SAMPLES` (completions at eval time;
**8** = reliable like CI, **1** = quick smoke test). Quick test example:

```bash
N_SAMPLES=1 ./pipelines/run_v9.sh
```

Eval writes a per-source/bucket JSON report under `$EVAL/gk_<version>/`.

## Results (dev_full, n=1000)

| Version | pass@1 | note |
|---|---:|---|
| V6 | 0.541 (n=1) | clean 14B distill |
| V9 | 0.600 (n=1) | format fixed |
| V9b | 0.580 (n=8) | + anti-loop, pass@8 0.892 |
| V10 | 0.554 (n=1) | off-policy < on-policy |
| V11 | 0.610 (n=1) | timid DPO = no-op |
| **V11b** | **0.600 (n=8)** | **final**, +2.0 vs v9b, pass@8 0.883 |

> Metric caveat: compare close models **only at n=8** (n=1 is noisy ±~3 pts — a single
> n=1 draw for V11b once showed a misleading 0.640).

> Note: there is intentionally no "V8" script. V8 differs from V9 only in LoRA strength
> (r=8 vs r=64); we keep V9/V9b as the endpoints, and **train_v8 build commands** live
> in `ensure_v8_data()` (`_lib.sh`), triggered automatically by V9 and V9b.
