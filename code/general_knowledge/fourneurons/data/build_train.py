from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fourneurons.data.augment import DistractorPool, generate_variants
from fourneurons.data.format_chat import to_training_row
from fourneurons.data.loaders import (
    load_boolq,
    load_commonsenseqa,
    load_ecqa,
    load_mmlu,
    load_mmlu_pro_cot,
    load_mmlu_world,
    load_socialiqa,
    load_triviaqa,
)
from fourneurons.data.schema import McqExample, stable_uid


SOURCE_RECIPE: list[tuple[str, Callable[..., Iterator[McqExample]], dict]] = [
    # --- has real CoT ---
    ("mmlu_pro_cot",  load_mmlu_pro_cot,  {"split": "train"}),
    ("ecqa",          load_ecqa,          {"split": "train"}),
    # --- needs distilled CoT ---
    ("mmlu",          load_mmlu,          {"split": "validation"}),
    ("mmlu_world",    load_mmlu_world,    {}),  # per-subject test+dev (see mmlu_world.py)
    ("triviaqa",      load_triviaqa,      {"split": "train"}),
    ("boolq",         load_boolq,         {"split": "train"}),
    ("socialiqa",     load_socialiqa,     {"split": "train"}),
    ("commonsenseqa", load_commonsenseqa, {"split": "train"}),
]


DEFAULT_MACRO_QUOTAS: dict[str, float] = {
    "stem": 0.25,
    "humanities": 0.20,
    "social_sciences": 0.20,
    "history_geo": 0.20,
    "commonsense": 0.15,
}


def _k_bucket(k: int) -> str:
    if k <= 2:
        return "2"
    if k == 3:
        return "3"
    if k == 4:
        return "4"
    if k == 5:
        return "5"
    if k <= 10:
        return "6-10"
    return "11-20"



def _load_dev_blocklist(paths: Sequence[Path]) -> set[str]:
    """Hash every dev-set question stem to prevent train/dev leakage."""
    blocked: set[str] = set()
    for p in paths:
        if not p.exists():
            print(f"[blocklist] {p}: missing, skipping")
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prompt = row.get("prompt") or ""
                stem = prompt.split("\n\nChoices:")[0].split("\nChoices:")[0]
                blocked.add(stable_uid(stem))
    print(f"[blocklist] {len(blocked)} dev question hashes loaded.")
    return blocked



def _collect_originals(
    blocklist: set[str],
    per_source_cap: Optional[int] = None,
) -> list[McqExample]:
    seen = set(blocklist)
    out: list[McqExample] = []
    for name, loader, kw in SOURCE_RECIPE:
        print(f"\n[load] {name} ({kw})")
        kept = 0
        try:
            for ex in loader(**kw):
                if ex.uid in seen:
                    continue
                seen.add(ex.uid)
                out.append(ex)
                kept += 1
                if per_source_cap and kept >= per_source_cap:
                    break
        except Exception as e:  # network / dataset hiccup → skip source, do not crash run.
            print(f"[load]   {name}: ERROR {type(e).__name__}: {e}")
        print(f"[load]   kept {kept} originals from {name}")
    return out


def _emit_with_augment(
    originals: Sequence[McqExample],
    pool: DistractorPool,
    max_variants: int,
    expand_only: bool,
    rng: random.Random,
) -> Iterator[tuple[McqExample, bool]]:
    for ex in originals:
        yield ex, False
        if max_variants <= 0:
            continue
        for v in generate_variants(
            ex,
            pool,
            rng,
            max_variants=max_variants,
            expand_only=expand_only,
        ):
            yield v, True


def _load_distilled_cache(
    paths: Optional[Path | list[Path]],
) -> dict[str, str]:
    if paths is None:
        return {}
    if isinstance(paths, Path):
        paths = [paths]
    cache: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            print(f"[distill_cache] {path}: missing, skipping (no distilled CoTs)")
            continue
        n_added = 0
        n_overridden = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                uid = row.get("uid")
                cot = row.get("cot")
                if uid and isinstance(cot, str) and cot.strip():
                    if uid in cache:
                        n_overridden += 1
                    cache[uid] = cot.strip()
                    n_added += 1
        print(
            f"[distill_cache] loaded {n_added} CoTs from {path} "
            f"({n_overridden} overrode earlier caches)."
        )
    print(f"[distill_cache] total {len(cache)} distilled CoTs in memory.")
    return cache



def _parse_quotas(raw: Optional[str]) -> dict[str, float]:
    """Parse `--macro_quotas "stem=0.25,..."` into a dict; defaults otherwise."""
    if not raw:
        return dict(DEFAULT_MACRO_QUOTAS)
    out: dict[str, float] = {}
    for piece in raw.split(","):
        if not piece.strip():
            continue
        k, v = piece.split("=")
        out[k.strip()] = float(v.strip())
    s = sum(out.values())
    if abs(s - 1.0) > 1e-3:
        print(f"[quotas] WARNING: sum={s:.3f} (renormalising to 1.0)")
        out = {k: v / s for k, v in out.items()}
    return out


def _quota_sample(
    rows: list[dict],
    total: int,
    macro_quotas: dict[str, float],
    aug_cap_frac: float,
    rng: random.Random,
) -> list[dict]:

    rng.shuffle(rows)

    by_macro_aug: dict[tuple[str, bool], list[dict]] = defaultdict(list)
    for r in rows:
        by_macro_aug[(r["macro_cat"], bool(r["is_augmented"]))].append(r)
    for v in by_macro_aug.values():
        rng.shuffle(v)

    plan: dict[str, int] = {m: int(round(total * q)) for m, q in macro_quotas.items()}
    drift = total - sum(plan.values())
    if drift:
        biggest = max(macro_quotas, key=lambda k: macro_quotas[k])
        plan[biggest] += drift

    out: list[dict] = []
    leftover: int = 0

    for macro, target in plan.items():
        n_aug_cap = int(round(aug_cap_frac * target))
        orig_bucket = by_macro_aug.get((macro, False), [])
        aug_bucket = by_macro_aug.get((macro, True), [])

        take_aug = min(n_aug_cap, len(aug_bucket))
        take_orig = min(target - take_aug, len(orig_bucket))
        # If originals are scarce, allow extra aug rows to fill the gap.
        if take_orig + take_aug < target:
            extra_aug = min(target - take_orig - take_aug, len(aug_bucket) - take_aug)
            take_aug += extra_aug

        for _ in range(take_aug):
            out.append(aug_bucket.pop())
        for _ in range(take_orig):
            out.append(orig_bucket.pop())

        deficit = target - (take_orig + take_aug)
        if deficit > 0:
            leftover += deficit
            print(
                f"[quota] {macro}: target={target}, took only "
                f"{take_orig + take_aug} (deficit={deficit}); "
                f"will redistribute."
            )

    if leftover > 0:
        spare: list[dict] = []
        for bucket in by_macro_aug.values():
            spare.extend(bucket)  # remaining items not yet popped
        rng.shuffle(spare)
        out.extend(spare[:leftover])

    rng.shuffle(out)
    return out[:total]


def _summarize(rows: list[dict]) -> dict:
    by_source = Counter(r["source"] for r in rows)
    by_macro = Counter(r["macro_cat"] for r in rows)
    by_k = Counter(r["k_bucket"] for r in rows)
    by_aug = Counter(r["is_augmented"] for r in rows)
    by_cot = Counter(r.get("cot_source", "?") for r in rows)
    return {
        "total": len(rows),
        "by_source": dict(by_source),
        "by_macro_cat": dict(by_macro),
        "by_k_bucket": {k: by_k[k] for k in ("2", "3", "4", "5", "6-10", "11-20") if k in by_k},
        "by_is_augmented": {str(k): v for k, v in by_aug.items()},
        "by_cot_source": dict(by_cot),
    }


def _print_summary(summary: dict, prefix: str = "") -> None:
    print(f"{prefix}total: {summary['total']}")
    for section in (
        "by_source",
        "by_macro_cat",
        "by_k_bucket",
        "by_is_augmented",
        "by_cot_source",
    ):
        if section not in summary:
            continue
        print(f"{prefix}{section}:")
        for k, v in summary[section].items():
            print(f"{prefix}  {k:24s}: {v}")



def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output_dir", required=True, type=Path)
    parser.add_argument("--total", type=int, default=40000)
    parser.add_argument(
        "--max_variants",
        type=int,
        default=1,
        help="Max augmented variants per original. 0 disables augmentation.",
    )
    parser.add_argument("--per_source_cap", type=int, default=None)
    parser.add_argument("--test_size", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--strict_cot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Drop any row that has neither a real nor a distilled CoT.",
    )
    parser.add_argument(
        "--expand_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate variants only by expanding to k ∈ [6, 20].",
    )
    parser.add_argument(
        "--aug_cap_frac",
        type=float,
        default=0.20,
        help="Max fraction of the final mix that may be augmented rows.",
    )
    parser.add_argument(
        "--macro_quotas",
        type=str,
        default=None,
        help='Override default macro quotas, e.g. '
             '"stem=0.25,humanities=0.20,social_sciences=0.20,'
             'history_geo=0.20,commonsense=0.15".',
    )
    parser.add_argument(
        "--dev_blocklist",
        nargs="+",
        type=Path,
        default=[
            Path("validation_samples/general_knowledge.jsonl"),
            Path("validation_samples/general_knowledge_dev_small.jsonl"),
            Path("validation_samples/general_knowledge_dev_full.jsonl"),
            Path("validation_samples/ood_dev.jsonl"),  # v5 OOD set
        ],
    )
    parser.add_argument(
        "--distilled_cot_cache",
        type=Path,
        nargs="*",
        default=None,
        help="One or more distilled-CoT cache JSONLs to layer in order "
        "(last wins for duplicate uids). v6 typically passes both the "
        "v5 cache and the new v6_long cache.",
    )
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    macro_quotas = _parse_quotas(args.macro_quotas)
    print(f"[build] macro_quotas: {macro_quotas}")
    print(
        f"[build] strict_cot={args.strict_cot}  "
        f"expand_only={args.expand_only}  aug_cap_frac={args.aug_cap_frac}"
    )

    blocklist = _load_dev_blocklist(args.dev_blocklist)
    originals = _collect_originals(blocklist, per_source_cap=args.per_source_cap)
    print(f"\n[build] {len(originals)} originals collected (before strict_cot drop).")

    print("[build] building typed distractor pool...")
    pool = DistractorPool()
    pool.ingest(originals)
    for (mc, t), n in sorted(pool.size_by_macro_type().items()):
        print(f"[build]   pool[{mc:18s}, {t:13s}]: {n}")

    distilled = _load_distilled_cache(args.distilled_cot_cache)

    print(
        f"[build] augmenting (max_variants={args.max_variants}, "
        f"expand_only={args.expand_only})..."
    )
    pre_rows: list[dict] = []
    n_dropped_no_cot = 0
    n_processed = 0
    log_every = max(5000, len(originals) // 20)
    for ex, is_aug in _emit_with_augment(
        originals, pool, args.max_variants, args.expand_only, rng
    ):
        row = to_training_row(
            ex,
            is_augmented=is_aug,
            distilled_cot=distilled.get(ex.uid),
            cot_if_missing=not args.strict_cot,
        )
        if row is None:
            n_dropped_no_cot += 1
        else:
            row["k_bucket"] = _k_bucket(ex.n_options)
            pre_rows.append(row)
        n_processed += 1
        if n_processed % log_every == 0:
            print(
                f"[build]   progress: {n_processed} rows processed "
                f"({len(pre_rows)} kept, {n_dropped_no_cot} dropped no-CoT)..."
            )
    print(
        f"[build] {len(pre_rows)} rows after augmentation+strict_cot "
        f"(dropped {n_dropped_no_cot} rows for missing CoT)."
    )

    print(f"[build] quota-sampling -> {args.total} rows...")
    sampled = _quota_sample(
        pre_rows,
        args.total,
        macro_quotas,
        args.aug_cap_frac,
        rng,
    )

    rng.shuffle(sampled)
    n_test = max(1, int(round(args.test_size * len(sampled))))
    test_rows = sampled[:n_test]
    train_rows = sampled[n_test:]
    print(f"[build] split: train={len(train_rows)}  test={len(test_rows)}")

    print("\n=== TRAIN summary ===")
    train_summary = _summarize(train_rows)
    _print_summary(train_summary)

    print("\n=== TEST summary ===")
    test_summary = _summarize(test_rows)
    _print_summary(test_summary)

    # Save with `datasets` so train.py can `load_from_disk` it.
    from datasets import Dataset, DatasetDict

    def _clean(rows: list[dict]) -> list[dict]:
        return [{k: v for k, v in r.items() if k != "k_bucket"} for r in rows]

    dsd = DatasetDict(
        train=Dataset.from_list(_clean(train_rows)),
        test=Dataset.from_list(_clean(test_rows)),
    )

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    dsd.save_to_disk(str(out_dir))
    with open(out_dir / "build_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "train": train_summary,
                "test": test_summary,
                "macro_quotas": macro_quotas,
                "args": vars(args).copy(),
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    print(f"\n[build] saved to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
