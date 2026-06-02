"""Loader for `google/boolq` reframed as a 2-option MCQ.

BoolQ is originally a reading-comprehension yes/no task with a passage.
Our test-time prompts are closed-book, so we *drop the passage* and frame
the question as a pure 2-option MC:

    {question}?

    Choices:
    A. Yes
    B. No

This makes BoolQ harder than its original setting (many items are
under-determined without the passage). That's acceptable for two reasons:
  1. We mostly want exposure to the 2-option *format*, not knowledge tests.
  2. Closed-book accuracy still beats random for trivia-style items.
"""

from __future__ import annotations

from typing import Iterator, Optional

from ..schema import McqExample

DATASET_ID = "google/boolq"


def load_boolq(
    split: str = "train",
    limit: Optional[int] = None,
) -> Iterator[McqExample]:
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split=split)
    n_skipped = 0

    for i, row in enumerate(ds):
        if limit is not None and i - n_skipped >= limit:
            break
        q = (row.get("question") or "").strip()
        answer = row.get("answer")
        if not q or answer is None:
            n_skipped += 1
            continue

        question = q if q.endswith("?") else q + "?"
        gold_idx = 0 if bool(answer) else 1
        yield McqExample(
            question=question,
            options=["Yes", "No"],
            gold_idx=gold_idx,
            source="boolq",
            macro_cat="commonsense",
            cot=None,
            subject="boolq",
        )

    if n_skipped:
        print(f"[boolq] skipped {n_skipped} unparsable rows.")
