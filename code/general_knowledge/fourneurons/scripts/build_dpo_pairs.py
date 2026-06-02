from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer


def _k_bucket(n: int) -> str:
    if n <= 2:
        return "2"
    if n == 3:
        return "3"
    if n == 4:
        return "4"
    if n == 5:
        return "5"
    if n <= 10:
        return "6-10"
    return "11-20"


_BOXED_RE = re.compile(r"\\boxed\{\s*([A-Za-z])\s*\}")


def _extract_letter(text: str) -> str | None:
    matches = _BOXED_RE.findall(text)
    if not matches:
        return None
    return matches[-1].upper()


def _stratified_subset(
    rows: list[dict],
    n_target: int,
    rng: random.Random,
) -> list[dict]:
    if len(rows) <= n_target:
        rng.shuffle(rows)
        return rows
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        key = (r["macro_cat"], _k_bucket(int(r["n_options"])))
        buckets[key].append(r)
    for v in buckets.values():
        rng.shuffle(v)
    out: list[dict] = []
    keys = list(buckets.keys())
    rng.shuffle(keys)
    while len(out) < n_target:
        progressed = False
        for k in keys:
            if buckets[k]:
                out.append(buckets[k].pop())
                progressed = True
                if len(out) >= n_target:
                    break
        if not progressed:
            break
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True, type=Path,
                   help="Path to the vLLM-ready v2 checkpoint (merged).")
    p.add_argument("--dataset_dir", required=True, type=Path,
                   help="HF DatasetDict produced by build_train.py "
                   "(we read its `train` split).")
    p.add_argument("--output", required=True, type=Path,
                   help="JSONL path to write the preference pairs to.")
    p.add_argument("--n_examples", type=int, default=4000,
                   help="How many train rows to sample completions from.")
    p.add_argument("--n_per_prompt", type=int, default=8,
                   help="Number of completions to draw per prompt.")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_pair_length_tokens", type=int, default=2048,
                   help="Skip pairs whose chosen or rejected exceeds this "
                   "many tokens (rough char/4 heuristic). Keeps DPO mem in check.")
    args = p.parse_args(argv)

    rng = random.Random(args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # ----- load dataset
    print(f"[load] dataset from {args.dataset_dir}")
    dsd = load_from_disk(str(args.dataset_dir))
    train = dsd["train"]
    print(f"[load]   train rows: {len(train)}")
    all_rows = [dict(r) for r in train]  # materialise; small enough (~19k)
    rows = _stratified_subset(all_rows, args.n_examples, rng)
    print(f"[load]   sampled {len(rows)} rows (stratified by macro_cat × k_bucket)")

    # ----- render prompts (vLLM expects raw text, chat template applied)
    print(f"[tok] loading tokenizer from {args.model}")
    tok = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True)

    prompts: list[str] = []
    for r in rows:
        # messages[0] is the user turn from format_chat.to_chat_messages
        user_msg = r["messages"][0]
        prompts.append(
            tok.apply_chat_template(
                [user_msg], tokenize=False, add_generation_prompt=True
            )
        )

    from vllm import LLM, SamplingParams

    print(f"[vllm] loading model {args.model}")
    llm = LLM(
        model=str(args.model),
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    sp = SamplingParams(
        n=args.n_per_prompt,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    print(f"[vllm] generating {len(prompts)} prompts × n={args.n_per_prompt} = "
          f"{len(prompts) * args.n_per_prompt} completions...")
    outs = llm.generate(prompts, sp)
    print("[vllm] done.")

    # ----- score + emit pairs
    char_cap = args.max_pair_length_tokens * 4  # rough chars/token

    n_pairs = 0
    n_skipped_no_correct = 0
    n_skipped_no_incorrect = 0
    n_skipped_no_boxed_at_all = 0
    n_skipped_too_long = 0
    by_macro: Counter[str] = Counter()
    by_bucket: Counter[str] = Counter()
    correct_fracs: list[float] = []

    with open(args.output, "w", encoding="utf-8") as fout:
        for row, out in zip(rows, outs):
            gold = row["gold_letter"].upper()
            completions = [o.text for o in out.outputs]

            correct: list[str] = []
            incorrect: list[str] = []
            for c in completions:
                letter = _extract_letter(c)
                if letter == gold:
                    correct.append(c)
                elif letter is not None:
                    incorrect.append(c)

            n_total = len(completions)
            n_correct = len(correct)
            n_incorrect_clean = len(incorrect)
            correct_fracs.append(n_correct / max(1, n_total))

            if n_correct == 0:
                n_skipped_no_correct += 1
                continue
            if n_incorrect_clean == 0:

                if not any(_extract_letter(c) for c in completions):
                    n_skipped_no_boxed_at_all += 1
                else:
                    n_skipped_no_incorrect += 1
                continue

            chosen = rng.choice(correct)
            rejected = rng.choice(incorrect)

            if len(chosen) > char_cap or len(rejected) > char_cap:
                n_skipped_too_long += 1
                continue

            user_msg = row["messages"][0]
            pair = {
                "prompt": [user_msg],
                "chosen": [{"role": "assistant", "content": chosen}],
                "rejected": [{"role": "assistant", "content": rejected}],
                "meta": {
                    "uid": row["uid"],
                    "macro_cat": row["macro_cat"],
                    "n_options": int(row["n_options"]),
                    "gold_letter": gold,
                    "n_correct": n_correct,
                    "n_total": n_total,
                },
            }
            fout.write(json.dumps(pair, ensure_ascii=False) + "\n")
            n_pairs += 1
            by_macro[row["macro_cat"]] += 1
            by_bucket[_k_bucket(int(row["n_options"]))] += 1

    print()
    print("=" * 60)
    print(f"pairs written       : {n_pairs}")
    print(f"skipped (no correct): {n_skipped_no_correct}")
    print(f"skipped (no incorr.): {n_skipped_no_incorrect}")
    print(f"skipped (no boxed)  : {n_skipped_no_boxed_at_all}")
    print(f"skipped (too long)  : {n_skipped_too_long}")
    print(f"mean p_correct      : {sum(correct_fracs)/max(1,len(correct_fracs)):.3f}")
    print(f"-- pairs by macro_cat --")
    for k, v in by_macro.most_common():
        print(f"  {k:20s}: {v}")
    print(f"-- pairs by k_bucket --")
    for k in ("2", "3", "4", "5", "6-10", "11-20"):
        if k in by_bucket:
            print(f"  {k:20s}: {by_bucket[k]}")
    print(f"-> wrote {n_pairs} pairs to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
