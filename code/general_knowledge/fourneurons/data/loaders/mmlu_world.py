"""Loader for the world-knowledge subset of MMLU (4-option MCQ).

Why this loader exists
----------------------
v1 was extremely thin on world knowledge:
  * `history_geo` macro_cat was only 3 % of the training mix;
  * the bulk of MMLU rows we ingested came from `validation` split with no
    subject filter, so STEM and social_sciences drowned the rest.

`mmlu_world` loads **per-subject HF configs** (not `config="all"`).
Important: MMLU's monolithic `auxiliary_train` split (~99k rows) is a
mixed auxiliary corpus that does **not** contain the 57 exam subjects — so
filtering `load_mmlu(split="auxiliary_train", config="all")` always yields
0 world-knowledge rows (we hit this in the first dry-run).

For volume we default to `test` + `dev` per subject (~100–200 + ~5–10
rows each). Overlap with our shipped dev sets is removed downstream by
the question-stem blocklist in `build_train.py`. We deliberately skip
`validation` here because `load_mmlu(validation)` is already in the
global SOURCE_RECIPE.
"""

from __future__ import annotations

from typing import Iterable, Iterator, Optional, Sequence

from ..schema import McqExample
from .mmlu import load_mmlu


DEFAULT_WORLD_SUBJECTS: tuple[str, ...] = (
    # history_geo (7)
    "high_school_world_history",
    "high_school_european_history",
    "high_school_us_history",
    "prehistory",
    "high_school_geography",
    "global_facts",
    "us_foreign_policy",
    # humanities (8) -- moral_scenarios deliberately excluded
    # (notoriously low-accuracy on all MMLU systems, format is awkward).
    "jurisprudence",
    "international_law",
    "world_religions",
    "formal_logic",
    "logical_fallacies",
    "moral_disputes",
    "philosophy",
    "professional_law",
    # social_sciences (14)
    "business_ethics",
    "econometrics",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_microeconomics",
    "high_school_psychology",
    "human_sexuality",
    "management",
    "marketing",
    "professional_accounting",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
)

# Per-subject configs only expose test/dev/validation — not auxiliary_train.
DEFAULT_WORLD_SPLITS: tuple[str, ...] = ("test", "dev")


def load_mmlu_world(
    splits: Optional[Sequence[str]] = None,
    limit: Optional[int] = None,
    subjects: Optional[Iterable[str]] = None,
    # Legacy alias: if someone passes split=..., map it to splits=(split,).
    split: Optional[str] = None,
) -> Iterator[McqExample]:
    """Yield `McqExample`s from MMLU world-knowledge subjects.

    Parameters
    ----------
    splits : sequence of str | None
        HF splits to load per subject. Defaults to ``("test", "dev")``.
        Do **not** use ``auxiliary_train`` here — it only exists on the
        ``config="all"`` bundle and does not contain these subjects.
    limit : int | None
        Cap on total yielded examples (across all subjects/splits).
    subjects : iterable of str | None
        Subject whitelist; defaults to `DEFAULT_WORLD_SUBJECTS`.
    split : str | None
        Deprecated single-split override (for backwards compat with the
        first dry-run CLI). Prefer `splits=`.
    """
    if split is not None:
        use_splits = (split,)
    else:
        use_splits = tuple(splits or DEFAULT_WORLD_SPLITS)

    subject_list = list(subjects or DEFAULT_WORLD_SUBJECTS)
    n_yielded = 0
    per_subject: dict[str, int] = {}

    for subject in subject_list:
        subj_kept = 0
        for sp in use_splits:
            try:
                for ex in load_mmlu(split=sp, config=subject, limit=None):
                    yield McqExample(
                        question=ex.question,
                        options=list(ex.options),
                        gold_idx=ex.gold_idx,
                        source="mmlu_world",
                        macro_cat=ex.macro_cat,
                        cot=ex.cot,
                        subject=ex.subject or subject,
                        uid=ex.uid,
                    )
                    n_yielded += 1
                    subj_kept += 1
                    if limit is not None and n_yielded >= limit:
                        print(
                            f"[mmlu_world] yielded {n_yielded} "
                            f"(splits={use_splits}, subjects={len(subject_list)})."
                        )
                        return
            except Exception as e:
                print(f"[mmlu_world] {subject}/{sp}: skip ({type(e).__name__}: {e})")
        per_subject[subject] = subj_kept

    print(
        f"[mmlu_world] yielded {n_yielded} from {len(subject_list)} subjects "
        f"(splits={use_splits})."
    )
    for subj, n in sorted(per_subject.items(), key=lambda x: -x[1]):
        if n:
            print(f"[mmlu_world]   {subj}: {n}")
