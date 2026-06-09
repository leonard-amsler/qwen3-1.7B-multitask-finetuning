# Math RL Plan

The next phase should start from the best mixed SFT checkpoint and optimize the model with verifiable rewards. The benchmark evaluates math with 8 sampled completions per problem, so the goal is not only to make the most likely answer correct, but to increase the chance that at least one of 8 trajectories reaches a correct boxed answer.

The RL data should use the training splits as the main prompt source. Existing SFT datasets are still useful for RL if we use only their `prompt` and `answer`, not their supervised completions. This keeps the distribution close to the current checkpoint while letting the model learn from its own sampled solutions.

New hard math datasets should be added, but only through a filtering step. They can diversify reasoning patterns and better match the hidden competition-style benchmark, but noisy labels or unverifiable answers are more damaging in RL than in SFT. The safest design is to build a mixed prompt pool from OpenMathInstruct, OpenR1Math, and any additional hard math sources, then score the current checkpoint on that pool before deciding what to train on.

The central selection rule should be frontier sampling. Generate 8 completions per candidate problem with the current best checkpoint, score them with the same boxed-answer evaluator used by the benchmark, and keep the number correct out of 8. Problems with `1/8` to `3/8` correct are the highest-value RL examples because the model can sometimes solve them but does not solve them reliably. Problems with `4/8` to `7/8` are also useful. Problems with `8/8` are mostly saturated, and problems with `0/8` should be used sparingly because they may be too hard, mislabeled, or outside the model's current reasoning reach.

Implementation should be incremental:

1. Add an RL prompt-pool builder that writes JSONL rows with `prompt`, `answer`, and `source`.
2. Reuse `fourneurons/evaluation/eval.py` with `--num_generations 8` and `evaluate.score` to pre-score candidate prompts.
3. Create a frontier training split biased toward `1 <= correct_count <= 7`, especially `1 <= correct_count <= 3`.
4. Add a new training entrypoint such as `fourneurons/scripts/train_math_rl.py`.
5. Initialize from the best SFT checkpoint and use verifier rewards based on boxed-answer correctness.
6. Start with a conservative GRPO/DAPO-lite setup: group size 8, low learning rate, KL to the SFT checkpoint, short runs, frequent eval.
7. Track pass@8-specific diagnostics after each checkpoint: pass@1, pass@8, correct-count histogram from `0/8` to `8/8`, box compliance, average length, and unique boxed answers per problem.
8. Only after the baseline RL loop works, add diversity-preserving improvements such as difficulty-aware sampling, negative reward for wrong boxed answers, stronger penalty for malformed/no-box outputs, overlong reward shaping, and SFT replay.

The reward should be simple at first: `+1` for a correct boxed answer, a negative reward for an incorrect boxed answer, and a stronger negative reward for missing or malformed boxes. This matches the benchmark contract and directly trains the behavior that matters. Overlong generations should receive a small penalty because the CI has a token limit and stops once the boxed answer appears.

Plain GRPO may improve pass@1 while narrowing the output distribution, which can leave pass@8 flat. For this project, any RL run should be judged by pass@8 and diversity metrics, not by pass@1 alone. If pass@1 rises but unique answers collapse or the `1/8` to `3/8` buckets do not improve, the training is over-sharpening and should use more KL, entropy, negative-reinforcement weighting, or frontier sampling.

Generation config is part of the final system because CI samples using the pushed model configuration. After each promising checkpoint, sweep temperature and top-p locally with 8 completions. A slightly more exploratory decoding setup may improve pass@8 even if it slightly lowers pass@1.

The recommended first experiment is therefore: take the best mixed SFT checkpoint, pre-score a large pool of existing train prompts, train RL on frontier problems with verifier rewards, and evaluate on untouched validation sets. Once that loop is stable, add additional hard math datasets through the same frontier filter.

## Implemented Data-Pool Step

The first implementation step is now split into two small data utilities. `fourneurons.data.math_rl_prompt_pool` builds a verifier-oriented prompt pool containing only `prompt`, `answer`, and `source`, plus lightweight metadata. It intentionally drops supervised completions so RL samples from the current checkpoint. The default output is compatible with the existing evaluator path convention:

```bash
python -m fourneurons.data.math_rl_prompt_pool \
  --source-limit openmathinstruct=50000 \
  --source-limit openR1math=50000 \
  --source-limit numinamath_1_5=50000 \
  --source-limit nemotron_math_v2=50000
```

This writes:

```text
/scratch/data/math/rl_prompt_pool/splits/rl_prompt_pool_train.jsonl
/scratch/data/math/rl_prompt_pool/splits/rl_prompt_pool_train.summary.json
```

The selected Hugging Face additions are:

- `AI-MO/NuminaMath-1.5`: official NuminaMath 1.5 release with strong Hugging Face adoption and direct `problem`/`answer` fields plus validity metadata. Proof rows and invalid rows are filtered out.
- `nvidia/Nemotron-Math-v2`: NVIDIA reasoning dataset with direct `problem`/`expected_answer`, answer-validation metadata, and pass-rate metadata. The builder uses the `medium` split in streaming mode, drops rows whose answer was changed to majority vote, drops tool-use rows, and deduplicates prompts.

Datasets inspected but intentionally excluded: `AIMO-Corpus/PolyMath` has a clean schema but low adoption, `gravermistakes/NuminaMath-1.5-RL-Verifiable` is a lower-signal fork of the official NuminaMath release, `FFHow/OlympiadBench` is multimodal/schema-fragile for this text-only verifier, `KbsdJames/Omni-MATH` is best kept as a hard benchmark/evaluation set rather than training data, `meta-math/MetaMathQA` is popular but SFT-style and requires answer extraction from generated responses, and `SynthLabsAI/Big-Math-RL-Verified` has strong metrics but is gated, so it is not usable as a frictionless default source.

After building the pool, pre-score it with eight generations using the current best SFT checkpoint. The verified best 16k-token Math500 SFT checkpoint is:

```text
/scratch/checkpoints/math/qwen3-1.7b-lora-math-mixed_20260526-220009/checkpoint-4458
```

Its 16k Math500 metrics are `pass@1=0.534`, `pass@8=0.840`, and `box_compliance=0.9405`, which makes it the best SFT checkpoint by pass@8 among the local Math500 16k runs. Run phase 2 with:

```bash
python -m fourneurons.scripts.prescore_math_rl_pool
```

This wraps the existing evaluator and scorer. It runs the equivalent of:

```bash
python -m fourneurons.evaluation.eval \
  math rl_prompt_pool train rl_pool_prescore_mixed_ckpt4458_tok16k_n8 \
  --checkpoint /scratch/checkpoints/math/qwen3-1.7b-lora-math-mixed_20260526-220009/checkpoint-4458 \
  --num_generations 8 \
  --max_tokens 16384 \
  --temperature 0.7 \
  --top_p 0.9 \
  --prompt_file_path fourneurons/prompts/math.txt

python -m evaluate.score \
  --generations /scratch/results/math/rl_prompt_pool/rl_pool_prescore_mixed_ckpt4458_tok16k_n8/train_gens.jsonl \
  --benchmark math \
  --output /scratch/results/math/rl_prompt_pool/rl_pool_prescore_mixed_ckpt4458_tok16k_n8/train_scored.json
```

For a smoke run before launching the full 200k-prompt pre-score, pass `--max_num_samples 20 --run_name rl_pool_prescore_smoke20`.

Then build the frontier split:

```bash
python -m fourneurons.data.select_math_rl_frontier \
  --pool /scratch/data/math/rl_prompt_pool/splits/rl_prompt_pool_train.jsonl \
  --scored /scratch/results/math/rl_prompt_pool/rl_pool_prescore_<run_id>/train_scored.json \
  --output /scratch/data/math/rl_frontier/splits/rl_frontier_train.jsonl \
  --include-correct 1-7
```

Use `--include-correct 1-3` for the highest-value narrow frontier, or keep `1-7` for a larger first RL run.
