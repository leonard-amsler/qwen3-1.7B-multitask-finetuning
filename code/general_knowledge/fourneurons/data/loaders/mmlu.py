"""Loader for the original MMLU (4-option multiple-choice).

We use `cais/mmlu` with the `all` config which concatenates the 57 subjects.
Each row already comes with a clean stem, a 4-element `choices` list, an
integer `answer` (0-3) and the `subject` name.

No CoT in this dataset. Used as:
  - extra training mass (4-option items, breadth across subjects);
  - main source for the dev set held-out from training.
"""

from __future__ import annotations

from typing import Iterator, Optional

from ..schema import McqExample
from ._macro_cat import mmlu_subject_to_macro

DATASET_ID = "cais/mmlu"


def load_mmlu(
    split: str = "test",
    limit: Optional[int] = None,
    config: str = "all",
) -> Iterator[McqExample]:
    """Yield `McqExample`s from MMLU.

    Parameters
    ----------
    split : str
        One of "test", "validation", "dev", "auxiliary_train".
    config : str
        HF config name. "all" concatenates the 57 subjects.
    limit : int | None
        Cap on number of yielded examples (post-skip).
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, config, split=split)
    n_skipped = 0

    for i, row in enumerate(ds):
        if limit is not None and i - n_skipped >= limit:
            break
        question = (row.get("question") or "").strip()
        choices = row.get("choices") or []
        answer = row.get("answer")
        subject = row.get("subject") or ""

        if not question or not choices or answer is None:
            n_skipped += 1
            continue
        choices = [str(c).strip() for c in choices if c is not None]
        if len(choices) < 2 or not (0 <= int(answer) < len(choices)):
            n_skipped += 1
            continue

        yield McqExample(
            question=question,
            options=choices,
            gold_idx=int(answer),
            source="mmlu",
            macro_cat=mmlu_subject_to_macro(subject),
            cot=None,
            subject=subject,
        )

    if n_skipped:
        print(f"[mmlu] skipped {n_skipped} unparsable rows.")
