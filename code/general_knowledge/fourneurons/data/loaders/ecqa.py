"""Loader for `tasksource/ecqa` (ExplainCommonsenseQA, 5 options + rationale).

ECQA = CommonsenseQA augmented with human-written free-text rationales.
Fields used:
  - q_text  : question stem
  - q_op1..q_op5 : the 5 option texts
  - q_ans   : the option text that is the correct answer
  - taskB   : the free-text explanation used as our CoT

CommonsenseQA and ECQA share the same question stems, so for the dev set
we use CSQA validation and drop ECQA to avoid leakage. For training, ECQA
is the better source because we get the rationale for free.
"""

from __future__ import annotations

from typing import Iterator, Optional

from ..schema import McqExample

DATASET_ID = "tasksource/ecqa"


def load_ecqa(
    split: str = "train",
    limit: Optional[int] = None,
) -> Iterator[McqExample]:
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split=split)
    n_skipped = 0

    for i, row in enumerate(ds):
        if limit is not None and i - n_skipped >= limit:
            break
        question = (row.get("q_text") or "").strip()
        options = [
            (row.get(f"q_op{k}") or "").strip() for k in range(1, 6)
        ]
        gold_text = (row.get("q_ans") or "").strip()
        cot = (row.get("taskB") or "").strip() or None

        if not question or not all(options) or not gold_text:
            n_skipped += 1
            continue
        # Robust case-insensitive match for the gold option text.
        norm_options = [o.lower() for o in options]
        try:
            gold_idx = norm_options.index(gold_text.lower())
        except ValueError:
            n_skipped += 1
            continue

        yield McqExample(
            question=question,
            options=options,
            gold_idx=gold_idx,
            source="ecqa",
            macro_cat="commonsense",
            cot=cot,
            subject="commonsense",
        )

    if n_skipped:
        print(f"[ecqa] skipped {n_skipped} unparsable rows.")
