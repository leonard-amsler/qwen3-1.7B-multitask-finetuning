from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from typing import Iterable, Optional

MACRO_CATS = (
    "stem",
    "humanities",
    "social_sciences",
    "history_geo",
    "commonsense",
)

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class McqExample:
    question: str
    options: list[str]
    gold_idx: int
    source: str
    macro_cat: str
    cot: Optional[str] = None
    subject: Optional[str] = None
    uid: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.options:
            raise ValueError("McqExample.options must be non-empty.")
        if not (0 <= self.gold_idx < len(self.options)):
            raise ValueError(
                f"gold_idx={self.gold_idx} out of range for "
                f"{len(self.options)} options."
            )
        if self.macro_cat not in MACRO_CATS:
            raise ValueError(
                f"macro_cat={self.macro_cat!r} not in {MACRO_CATS}."
            )
        if len(self.options) > len(_LETTERS):
            raise ValueError(
                f"Too many options ({len(self.options)}); max supported is "
                f"{len(_LETTERS)}."
            )
        if self.uid is None:
            self.uid = stable_uid(self.question)

    @property
    def gold_letter(self) -> str:
        return _LETTERS[self.gold_idx]

    @property
    def n_options(self) -> int:
        return len(self.options)


def stable_uid(text: str) -> str:
    """Short stable hash over a normalized question stem."""
    norm = re.sub(r"\s+", " ", text.strip().lower())
    return hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


def format_prompt(ex: McqExample) -> str:
    lines = [ex.question.strip(), "", "Choices:"]
    for i, opt in enumerate(ex.options):
        lines.append(f"{_LETTERS[i]}. {opt}")
    return "\n".join(lines)


def to_eval_row(ex: McqExample, with_meta: bool = True) -> dict:
    row = {"prompt": format_prompt(ex), "answer": ex.gold_letter}
    if with_meta:
        row["meta"] = {
            "source": ex.source,
            "macro_cat": ex.macro_cat,
            "n_options": ex.n_options,
            "subject": ex.subject,
            "uid": ex.uid,
        }
    return row


def write_jsonl(path, examples: Iterable[McqExample], with_meta: bool = True) -> int:
    import json
    import os

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for ex in examples:
            row = to_eval_row(ex, with_meta=with_meta)
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n
