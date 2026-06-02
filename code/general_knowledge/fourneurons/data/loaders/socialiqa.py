"""Loader for SocialIQA (3-option social commonsense reasoning).

Hugging Face dataset id: `baber/social_i_qa` (parquet mirror).

The original `social_i_qa` / `allenai/social_i_qa` repos ship a loading
script that recent versions of `datasets` no longer support. `baber/social_i_qa`
is a community parquet mirror with identical fields.

Each example has a short context, a question, three options (answerA/B/C)
and a label in {"1","2","3"}. We treat (context + question) as the stem.

Used to fill the **3-option** bucket — none of our other primary sources
ships native 3-option items.
"""

from __future__ import annotations

from typing import Iterator, Optional

from ..schema import McqExample

DATASET_ID = "baber/social_i_qa"


def load_socialiqa(
    split: str = "train",
    limit: Optional[int] = None,
) -> Iterator[McqExample]:
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split=split)
    n_skipped = 0

    for i, row in enumerate(ds):
        if limit is not None and i - n_skipped >= limit:
            break
        context = (row.get("context") or "").strip()
        question = (row.get("question") or "").strip()
        options = [
            (row.get("answerA") or "").strip(),
            (row.get("answerB") or "").strip(),
            (row.get("answerC") or "").strip(),
        ]
        label = (row.get("label") or "").strip()

        if not question or not all(options) or label not in {"1", "2", "3"}:
            n_skipped += 1
            continue
        gold_idx = int(label) - 1
        stem = f"{context}\n\n{question}".strip() if context else question

        yield McqExample(
            question=stem,
            options=options,
            gold_idx=gold_idx,
            source="socialiqa",
            macro_cat="commonsense",
            cot=None,
            subject="social_commonsense",
        )

    if n_skipped:
        print(f"[socialiqa] skipped {n_skipped} unparsable rows.")
