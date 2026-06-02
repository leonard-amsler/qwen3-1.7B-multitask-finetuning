"""Loader for `UW-Madison-Lee-Lab/MMLU-Pro-CoT-Train-Labeled`.

MMLU-Pro with model-generated chain-of-thought. Only a `train` split exists.
Each example has the options inlined inside the `question` text (lines that
start with "A. ", "B. ", ..., up to "J. ") plus a separate `answer` letter
and `chain_of_thoughts` (a list of strings).

This is our main source of CoT-bearing training data; we strip the inlined
options out so the rest of the pipeline can resample option counts uniformly.
"""

from __future__ import annotations

import re
from typing import Iterator, Optional

from ..schema import McqExample
from ._macro_cat import mmlu_pro_category_to_macro

DATASET_ID = "UW-Madison-Lee-Lab/MMLU-Pro-CoT-Train-Labeled"

_OPTION_RE = re.compile(r"^\s*([A-J])[.)]\s+(.*)$")


def _parse_inline_options(question_with_options: str) -> tuple[str, list[str]]:
    """Split the raw `question` field into (stem, options).

    The MMLU-Pro-CoT field looks like:

        What is X?
        A. foo
        B. bar
        C. baz
        ...

    We collect everything before the first "A. " line as the stem, then
    parse each subsequent labeled line as an option. Lines that wrap onto
    the next line (no label) are appended to the previous option.
    """
    stem_lines: list[str] = []
    options: list[str] = []
    in_options = False

    for raw_line in question_with_options.splitlines():
        line = raw_line.rstrip()
        m = _OPTION_RE.match(line)
        if m:
            in_options = True
            options.append(m.group(2).strip())
            continue
        if in_options:
            if line.strip() and options:
                options[-1] = (options[-1] + " " + line.strip()).strip()
        else:
            stem_lines.append(line)

    stem = "\n".join(stem_lines).strip()
    return stem, options


def _parse_cot(cot_field) -> Optional[str]:
    if cot_field is None:
        return None
    if isinstance(cot_field, list):
        text = "\n".join(str(x) for x in cot_field if x is not None)
    else:
        text = str(cot_field)
    text = text.strip()
    return text or None


def load_mmlu_pro_cot(
    split: str = "train",
    limit: Optional[int] = None,
) -> Iterator[McqExample]:
    """Yield `McqExample`s from MMLU-Pro-CoT.

    Drops rows where parsing fails (no options found, gold letter out of
    range, etc.). These are rare but we surface a single warning at the end.
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split=split)
    n_skipped = 0

    for i, row in enumerate(ds):
        if limit is not None and i - n_skipped >= limit:
            break
        raw_q = row.get("question") or ""
        gold_letter = (row.get("answer") or "").strip().upper()
        category = row.get("category") or ""
        cot = _parse_cot(row.get("chain_of_thoughts"))

        stem, options = _parse_inline_options(raw_q)
        if not stem or not options or len(gold_letter) != 1:
            n_skipped += 1
            continue
        gold_idx = ord(gold_letter) - ord("A")
        if not (0 <= gold_idx < len(options)):
            n_skipped += 1
            continue

        yield McqExample(
            question=stem,
            options=options,
            gold_idx=gold_idx,
            source="mmlu_pro_cot",
            macro_cat=mmlu_pro_category_to_macro(category),
            cot=cot,
            subject=category,
        )

    if n_skipped:
        print(f"[mmlu_pro_cot] skipped {n_skipped} unparsable rows.")
