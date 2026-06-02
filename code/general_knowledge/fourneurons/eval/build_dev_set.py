from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fourneurons.data.augment import DistractorPool, expand_options
from fourneurons.data.schema import McqExample, stable_uid, write_jsonl
from fourneurons.data.loaders import (
    load_boolq,
    load_commonsenseqa,
    load_mmlu,
    load_mmlu_pro,
    load_mmlu_pro_cot,
)


SMALL_RECIPE = [
    ("mmlu", "test", 80),
    ("mmlu_pro", "test", 60),
    ("commonsenseqa", "validation", 40),
    ("boolq", "validation", 40),
]

FULL_RECIPE = [
    ("mmlu", "test", 400),
    ("mmlu_pro", "test", 300),
    ("commonsenseqa", "validation", 150),
    ("boolq", "validation", 150),
]


SMALL_AUG_11_20 = 20
FULL_AUG_11_20 = 100

AUG_K_RANGE = (11, 20)


_LOADERS = {
    "mmlu": load_mmlu,
    "mmlu_pro": load_mmlu_pro,
    "commonsenseqa": load_commonsenseqa,
    "boolq": load_boolq,
}


def _build_train_blocklist(verbose: bool = True) -> set[str]:
    """Hashes of every question stem in MMLU-Pro-CoT-Train-Labeled."""
    if verbose:
        print("[dev] building training blocklist from MMLU-Pro-CoT...")
    blocklist: set[str] = set()
    for ex in load_mmlu_pro_cot(split="train"):
        blocklist.add(stable_uid(ex.question))
    if verbose:
        print(f"[dev] blocklist size: {len(blocklist)} question hashes.")
    return blocklist


def _stratified_sample(
    examples: list[McqExample],
    n: int,
    rng: random.Random,
) -> list[McqExample]:
    """Approximate uniform sampling across macro categories.

    Buckets examples by `macro_cat`, then round-robins one example per
    bucket (shuffled) until we hit `n` or run out.
    """
    if len(examples) <= n:
        rng.shuffle(examples)
        return examples

    buckets: dict[str, list[McqExample]] = defaultdict(list)
    for ex in examples:
        buckets[ex.macro_cat].append(ex)
    for v in buckets.values():
        rng.shuffle(v)

    out: list[McqExample] = []
    order = list(buckets.keys())
    rng.shuffle(order)
    while len(out) < n:
        progressed = False
        for cat in order:
            if buckets[cat]:
                out.append(buckets[cat].pop())
                progressed = True
                if len(out) >= n:
                    break
        if not progressed:
            break
    return out


def _collect(
    recipe: Sequence[tuple[str, str, int]],
    blocklist: set[str],
    rng: random.Random,
) -> list[McqExample]:
    selected: list[McqExample] = []
    seen_uids: set[str] = set(blocklist)

    for source, split, target in recipe:
        loader = _LOADERS[source]
        print(f"[dev] loading {source}/{split} (target={target})...")
        pool: list[McqExample] = []
        for ex in loader(split=split):
            if ex.uid in seen_uids:
                continue
            seen_uids.add(ex.uid)
            pool.append(ex)
        print(f"[dev]   pool size after dedup: {len(pool)}")
        picked = _stratified_sample(pool, target, rng)
        print(f"[dev]   picked: {len(picked)}")
        selected.extend(picked)

    rng.shuffle(selected)
    return selected


def _k_bucket_label(n_options: int) -> str:
    if n_options <= 2:
        return "2"
    if n_options == 3:
        return "3"
    if n_options == 4:
        return "4"
    if n_options == 5:
        return "5"
    if n_options <= 10:
        return "6-10"
    return "11-20"


def _build_aug_11_20(
    candidates: list[McqExample],
    n_target: int,
    rng: random.Random,
) -> list[McqExample]:

    if n_target <= 0:
        return []

    base_pool = [ex for ex in candidates if ex.n_options <= 10]
    if not base_pool:
        print("[dev] aug-11-20: no candidates with n_options <= 10, skipping.")
        return []

    pool = DistractorPool()
    pool.ingest(candidates)
    print("[dev] aug-11-20: distractor pool sizes per macro_cat: "
          + ", ".join(f"{m}={s}" for m, s in pool.size_by_macro().items()))

    by_macro: dict[str, list[McqExample]] = defaultdict(list)
    for ex in base_pool:
        by_macro[ex.macro_cat].append(ex)
    for v in by_macro.values():
        rng.shuffle(v)

    bases: list[McqExample] = []
    order = list(by_macro.keys())
    rng.shuffle(order)
    while len(bases) < n_target:
        progressed = False
        for cat in order:
            if by_macro[cat]:
                bases.append(by_macro[cat].pop())
                progressed = True
                if len(bases) >= n_target:
                    break
        if not progressed:
            break

    out: list[McqExample] = []
    k_lo, k_hi = AUG_K_RANGE
    for ex in bases:
        k = rng.randint(k_lo, k_hi)
        v = expand_options(ex, k, pool, rng)
        if v is not None:
            out.append(v)
    print(f"[dev] aug-11-20: emitted {len(out)}/{n_target} expansions "
          f"(some bases may have lacked plausible distractors).")
    return out


def _summarize(examples: list[McqExample]) -> str:
    by_source = Counter(ex.source for ex in examples)
    by_macro = Counter(ex.macro_cat for ex in examples)
    by_k = Counter(ex.n_options for ex in examples)
    by_bucket = Counter(_k_bucket_label(ex.n_options) for ex in examples)
    lines = ["=== dev-set summary ===", f"total: {len(examples)}"]
    lines.append("by source:")
    for k, v in by_source.most_common():
        lines.append(f"  {k:30s}: {v:4d}")
    lines.append("by macro_cat:")
    for k, v in by_macro.most_common():
        lines.append(f"  {k:30s}: {v:4d}")
    lines.append("by n_options bucket:")
    for k in ("2", "3", "4", "5", "6-10", "11-20"):
        if k in by_bucket:
            lines.append(f"  {k:30s}: {by_bucket[k]:4d}")
    lines.append("by raw n_options:")
    for k in sorted(by_k):
        lines.append(f"  k={k:<2d}                          : {by_k[k]:4d}")
    return "\n".join(lines)



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output_dir",
        default="validation_samples",
        help="Directory in which to write the two dev-set JSONL files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for stratified sampling (reproducibility).",
    )
    parser.add_argument(
        "--no_blocklist",
        action="store_true",
        help="Skip MMLU-Pro-CoT training-set deduplication (much faster, but "
        "risks dev/train leakage).",
    )
    parser.add_argument(
        "--only",
        choices=("small", "full"),
        default=None,
        help="Only build one of the two artifacts.",
    )
    parser.add_argument(
        "--aug_11_20_small",
        type=int,
        default=SMALL_AUG_11_20,
        help="How many `small` dev examples to additionally expand to "
        f"k ∈ [{AUG_K_RANGE[0]}, {AUG_K_RANGE[1]}]. Set 0 to disable.",
    )
    parser.add_argument(
        "--aug_11_20_full",
        type=int,
        default=FULL_AUG_11_20,
        help="How many `full` dev examples to additionally expand to "
        f"k ∈ [{AUG_K_RANGE[0]}, {AUG_K_RANGE[1]}]. Set 0 to disable.",
    )
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    blocklist: set[str] = set() if args.no_blocklist else _build_train_blocklist()

    targets: list[tuple[str, list, Path, int]] = []
    if args.only in (None, "small"):
        targets.append(
            ("small", SMALL_RECIPE,
             out_dir / "general_knowledge_dev_small.jsonl",
             args.aug_11_20_small)
        )
    if args.only in (None, "full"):
        targets.append(
            ("full", FULL_RECIPE,
             out_dir / "general_knowledge_dev_full.jsonl",
             args.aug_11_20_full)
        )

    for name, recipe, path, n_aug in targets:
        print(f"\n[dev] === building {name} dev set ===")
        local_rng = random.Random(args.seed + (0 if name == "small" else 1))
        examples = _collect(recipe, blocklist, local_rng)
        if n_aug > 0:
            print(f"[dev] adding {n_aug} augmented 11-20-option examples...")
            aug_rng = random.Random(args.seed + (100 if name == "small" else 101))
            aug = _build_aug_11_20(examples, n_aug, aug_rng)
            examples = examples + aug
            local_rng.shuffle(examples)
        n = write_jsonl(path, examples, with_meta=True)
        print(f"[dev] wrote {n} examples to {path}")
        print(_summarize(examples))

    return 0


if __name__ == "__main__":
    sys.exit(main())
