"""Loader for OpenBookQA (elementary-science 4-option MCQ).

Hugging Face dataset id: `allenai/openbookqa`, config `main` (~500 test items).

Schema (per row):
  - id              : string
  - question_stem   : string (the actual question)
  - choices         : {"label": ["A","B","C","D"], "text": [...]}
  - answerKey       : string, one of "A".."D"

The `additional` config bundles per-question elementary-science "facts"
(open-book hints). We deliberately use the `main` config: our evaluation is
closed-book, so the model must answer from parametric knowledge alone.

OpenBookQA is **never used in training**: it is loaded here only to build
our held-out OOD dev set (`validation_samples/ood_dev.jsonl`). Pairs nicely
with ARC-Challenge to cover science breadth and depth on OOD items.
"""

from __future__ import annotations

from typing import Iterator, Optional

from ..schema import McqExample

DATASET_ID = "allenai/openbookqa"
CONFIG_MAIN = "main"


def load_openbookqa(
    split: str = "test",
    limit: Optional[int] = None,
) -> Iterator[McqExample]:
    """Yield `McqExample`s from OpenBookQA (main).

    Parameters
    ----------
    split : str
        One of "train", "validation", "test". For our OOD dev set we use
        "test" (~500 items).
    limit : int | None
        Cap on yielded examples (post-skip).
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, CONFIG_MAIN, split=split)
    n_skipped = 0

    for i, row in enumerate(ds):
        if limit is not None and i - n_skipped >= limit:
            break
        question = (row.get("question_stem") or "").strip()
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

        # Canonicalise option order by ascending label (A,B,C,D).
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
            source="openbookqa",
            macro_cat="stem",
            cot=None,
            subject="openbookqa_science",
        )

    if n_skipped:
        print(f"[openbookqa] skipped {n_skipped} unparsable rows.")
