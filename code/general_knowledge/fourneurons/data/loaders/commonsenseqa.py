"""Loader for CommonsenseQA (5-option commonsense reasoning).

We use `tau/commonsense_qa`. Each row has:
  - `question` : the stem
  - `choices`  : {"label": ["A","B","C","D","E"], "text": [...]}
  - `answerKey`: a letter "A".."E" (empty string in the test split)

We skip rows with an empty answer key (test split items have no labels).
"""

from __future__ import annotations

from typing import Iterator, Optional

from ..schema import McqExample

DATASET_ID = "tau/commonsense_qa"


def load_commonsenseqa(
    split: str = "train",
    limit: Optional[int] = None,
) -> Iterator[McqExample]:
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split=split)
    n_skipped = 0

    for i, row in enumerate(ds):
        if limit is not None and i - n_skipped >= limit:
            break
        question = (row.get("question") or "").strip()
        choices = row.get("choices") or {}
        labels = list(choices.get("label", []) or [])
        texts = list(choices.get("text", []) or [])
        answer_key = (row.get("answerKey") or "").strip().upper()

        if not question or not labels or not texts or len(labels) != len(texts):
            n_skipped += 1
            continue
        if not answer_key or answer_key not in labels:
            n_skipped += 1
            continue

        # Re-order options by canonical label order (A,B,C,...) just in case.
        pairs = sorted(zip(labels, texts), key=lambda p: p[0])
        options = [str(t).strip() for _, t in pairs]
        gold_idx = [lbl for lbl, _ in pairs].index(answer_key)

        yield McqExample(
            question=question,
            options=options,
            gold_idx=gold_idx,
            source="commonsenseqa",
            macro_cat="commonsense",
            cot=None,
            subject="commonsense",
        )

    if n_skipped:
        print(f"[commonsenseqa] skipped {n_skipped} unparsable rows.")
