"""Loader for ARC-Challenge (AI2 Reasoning Challenge, grade-school science).

Hugging Face dataset id: `allenai/ai2_arc`, config `ARC-Challenge`.

Schema (per row):
  - id          : string
  - question    : string (question stem)
  - choices     : {"label": ["A","B","C","D",...], "text": [...]}
  - answerKey   : string, one of the labels in `choices.label`

ARC-Challenge is **never used in training**: it is loaded here only to build
our held-out OOD dev set (`validation_samples/ood_dev.jsonl`). It complements
MMLU/MMLU-Pro because:
  - it is a genuine OOD distribution (different source, different annotators);
  - it is the canonical science-MCQ generalization benchmark in the LM lit.

A small fraction of ARC-Challenge items use **numeric labels** ("1","2","3"...)
instead of letters; we normalise both layouts to the canonical
`{labels sorted}` order so that `gold_idx` always matches `choices.label`.

Most items have exactly 4 options, a handful have 3 or 5; we keep them all
(no forced 4-option filter) — `n_options` is recorded per example anyway and
the downstream bucket report breaks them out cleanly.
"""

from __future__ import annotations

from typing import Iterator, Optional

from ..schema import McqExample

DATASET_ID = "allenai/ai2_arc"
CONFIG_CHALLENGE = "ARC-Challenge"


def load_arc_challenge(
    split: str = "test",
    limit: Optional[int] = None,
) -> Iterator[McqExample]:
    """Yield `McqExample`s from ARC-Challenge.

    Parameters
    ----------
    split : str
        One of "train", "validation", "test". For our OOD dev set we use
        "test" (~1172 items).
    limit : int | None
        Cap on yielded examples (post-skip).
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, CONFIG_CHALLENGE, split=split)
    n_skipped = 0

    for i, row in enumerate(ds):
        if limit is not None and i - n_skipped >= limit:
            break
        question = (row.get("question") or "").strip()
        choices = row.get("choices") or {}
        labels = list(choices.get("label", []) or [])
        texts = list(choices.get("text", []) or [])
        answer_key = (row.get("answerKey") or "").strip()

        if not question or not labels or not texts or len(labels) != len(texts):
            n_skipped += 1
            continue
        if not answer_key or answer_key not in labels:
            n_skipped += 1
            continue

        # Canonicalise option order by ascending label so the gold index
        # always matches the displayed letter ordering (A,B,C,...).
        pairs = sorted(zip(labels, texts), key=lambda p: p[0])
        options = [str(t).strip() for _, t in pairs]
        gold_idx = [lbl for lbl, _ in pairs].index(answer_key)
        if any(not o for o in options):
            n_skipped += 1
            continue

        yield McqExample(
            question=question,
            options=options,
            gold_idx=gold_idx,
            source="arc_challenge",
            macro_cat="stem",
            cot=None,
            subject="arc_science",
        )

    if n_skipped:
        print(f"[arc_challenge] skipped {n_skipped} unparsable rows.")
