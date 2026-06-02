from __future__ import annotations

import argparse
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fourneurons.data.schema import McqExample, write_jsonl
from fourneurons.data.loaders import load_arc_challenge, load_openbookqa


RECIPE: tuple[tuple[str, callable, dict], ...] = (
    ("arc_challenge", load_arc_challenge, {"split": "test"}),
    ("openbookqa",    load_openbookqa,    {"split": "test"}),
)


def _collect(rng: random.Random) -> list[McqExample]:
    """Load every OOD source and dedupe on the question-stem hash."""
    selected: list[McqExample] = []
    seen_uids: set[str] = set()

    for source, loader, kwargs in RECIPE:
        print(f"[ood] loading {source} {kwargs}...")
        n_kept = 0
        n_dup = 0
        for ex in loader(**kwargs):
            if ex.uid in seen_uids:
                n_dup += 1
                continue
            seen_uids.add(ex.uid)
            selected.append(ex)
            n_kept += 1
        print(f"[ood]   kept {n_kept} from {source} ({n_dup} intra/inter-source dups)")

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


def _summarize(examples: list[McqExample]) -> str:
    by_source = Counter(ex.source for ex in examples)
    by_macro = Counter(ex.macro_cat for ex in examples)
    by_k = Counter(ex.n_options for ex in examples)
    by_bucket = Counter(_k_bucket_label(ex.n_options) for ex in examples)

    lines = ["=== OOD dev-set summary ===", f"total: {len(examples)}"]
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
        help="Directory in which to write `ood_dev.jsonl`.",
    )
    parser.add_argument(
        "--filename",
        default="ood_dev.jsonl",
        help="Output filename (under --output_dir).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for the post-collection shuffle (reproducibility).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on the total number of examples written "
        "(post-shuffle, for smoke tests).",
    )
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / args.filename

    examples = _collect(rng)
    if args.limit is not None:
        examples = examples[: args.limit]
        print(f"[ood] capped to {len(examples)} examples (--limit).")

    n = write_jsonl(out_path, examples, with_meta=True)
    print(f"[ood] wrote {n} examples to {out_path}")
    print(_summarize(examples))
    return 0


if __name__ == "__main__":
    sys.exit(main())
