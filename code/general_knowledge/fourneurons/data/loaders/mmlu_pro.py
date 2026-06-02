"""Loader for `TIGER-Lab/MMLU-Pro` (the original, no CoT).

Used to build the dev set with variable option counts (4..10) that are
*not* present in the CoT-labeled training split. We discard `cot_content`
on purpose to keep dev-set rationales out of training.
"""

from __future__ import annotations

from typing import Iterator, Optional

from ..schema import McqExample
from ._macro_cat import mmlu_pro_category_to_macro

DATASET_ID = "TIGER-Lab/MMLU-Pro"


def load_mmlu_pro(
    split: str = "test",
    limit: Optional[int] = None,
) -> Iterator[McqExample]:
    """Yield `McqExample`s from MMLU-Pro. Available splits: `test`, `validation`."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split=split)
    n_skipped = 0

    for i, row in enumerate(ds):
        if limit is not None and i - n_skipped >= limit:
            break
        question = (row.get("question") or "").strip()
        options = row.get("options") or []
        answer_index = row.get("answer_index")
        if answer_index is None:
            letter = (row.get("answer") or "").strip().upper()
            answer_index = ord(letter) - ord("A") if len(letter) == 1 else None
        category = row.get("category") or ""

        options = [str(o).strip() for o in options if o is not None]
        if not question or len(options) < 2 or answer_index is None:
            n_skipped += 1
            continue
        if not (0 <= int(answer_index) < len(options)):
            n_skipped += 1
            continue

        yield McqExample(
            question=question,
            options=options,
            gold_idx=int(answer_index),
            source="mmlu_pro",
            macro_cat=mmlu_pro_category_to_macro(category),
            cot=None,
            subject=category,
        )

    if n_skipped:
        print(f"[mmlu_pro] skipped {n_skipped} unparsable rows.")
