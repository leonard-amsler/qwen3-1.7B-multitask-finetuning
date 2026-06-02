"""Loader for TriviaQA reformatted as 4-option multiple-choice.

Why this loader exists
----------------------
v1 had only 3 % of training rows in the `history_geo` macro_cat despite
the General Knowledge benchmark explicitly covering world knowledge,
history and geography. TriviaQA is a large pool of factual trivia
questions (~80k train items) whose answers are almost all named
entities — exactly the kind of knowledge probes the CI relies on.

Hugging Face dataset id: `mandarjoshi/trivia_qa`, config `rc.nocontext`.

Schema (per row, the bits we use):
  - question                  : the trivia question
  - answer.value              : the canonical gold answer string
  - answer.aliases            : list of acceptable variants

We **fabricate 3 distractors** from a pool of all gold answers, restricted
to entities of the same coarse type as the current gold (year vs numeric
vs short_entity vs phrase). Concretely:

  1. We walk the whole split once to build:
       - a typed pool: {option_type -> list[str]} of unique answer values
       - aliases per gold (to avoid leakage between distractors and gold)
  2. Yield one `McqExample` per row with 4 options (gold + 3 distractors).
  3. macro_cat = "history_geo" for the whole split — TriviaQA's coverage
     is dominated by history/geography/culture trivia, and we explicitly
     want this loader to fill that quota.

Edge cases:
  * If the gold is too unusual to find 3 same-type distractors, we
    progressively relax (drop the type constraint). Worst case we skip
    the row — `n_skipped` is logged.
  * We never use the question itself as a distractor.
  * We strip whitespace and drop empty rows.

The resulting examples have `cot=None` — they go through the distillation
pipeline in Phase 2.
"""

from __future__ import annotations

import random
import re
from typing import Iterator, Optional

from ..schema import McqExample
from ..augment import classify_option

DATASET_ID = "mandarjoshi/trivia_qa"
CONFIG = "rc.nocontext"

_MIN_DISTRACTORS = 3
_N_OPTIONS = 4


def _normalise(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()


def _looks_like_garbage(text: str) -> bool:
    """Reject obviously bad gold strings: empty, single-char, all-punct."""
    t = (text or "").strip()
    if not t:
        return True
    if len(t) == 1 and not t.isalnum():
        return True
    if t.lower() in {"unknown", "n/a", "none", "null"}:
        return True
    return False


def load_triviaqa(
    split: str = "train",
    limit: Optional[int] = None,
    seed: int = 42,
) -> Iterator[McqExample]:
    """Yield `McqExample`s from TriviaQA reformatted to 4-option MC.

    Parameters
    ----------
    split : str
        One of "train", "validation".
    limit : int | None
        Cap on yielded examples (post-skip).
    seed : int
        Seed driving distractor sampling (and dataset shuffling per row).
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, CONFIG, split=split)

    # Pass 1: materialise rows + build typed answer pool + alias index.
    rows: list[tuple[str, str, set[str]]] = []  # (question, gold, aliases_lower)
    pool_by_type: dict[str, list[str]] = {
        "year": [], "numeric": [], "short_entity": [], "phrase": [],
    }
    seen_pool: set[tuple[str, str]] = set()  # (type, gold_lower)

    for row in ds:
        question = (row.get("question") or "").strip()
        ans = row.get("answer") or {}
        if isinstance(ans, dict):
            gold = (ans.get("value") or "").strip()
            aliases_raw = list(ans.get("aliases") or [])
        else:
            # `rc.nocontext` always ships a dict but defensive guard anyway.
            gold = ""
            aliases_raw = []
        if not question or _looks_like_garbage(gold):
            continue
        t = classify_option(gold)
        rows.append((question, gold, {_normalise(a) for a in aliases_raw + [gold]}))
        key = (t, _normalise(gold))
        if key not in seen_pool:
            seen_pool.add(key)
            pool_by_type[t].append(gold)

    if not rows:
        print("[triviaqa] no rows kept from the dataset.")
        return

    # Shuffle pool buckets once so per-row sampling is cheap.
    rng = random.Random(seed)
    for v in pool_by_type.values():
        rng.shuffle(v)

    n_skipped = 0
    n_yielded = 0

    for i, (question, gold, alias_set) in enumerate(rows):
        if limit is not None and n_yielded >= limit:
            break
        t = classify_option(gold)
        distractors = _pick_distractors(
            gold=gold,
            gold_type=t,
            alias_set=alias_set,
            pool_by_type=pool_by_type,
            rng=rng,
        )
        if distractors is None:
            n_skipped += 1
            continue

        options = [gold] + distractors
        rng.shuffle(options)
        gold_idx = options.index(gold)

        yield McqExample(
            question=question,
            options=options,
            gold_idx=gold_idx,
            source="triviaqa",
            macro_cat="history_geo",
            cot=None,
            subject="triviaqa",
        )
        n_yielded += 1

    if n_skipped:
        print(
            f"[triviaqa] yielded {n_yielded}, skipped {n_skipped} "
            f"(could not fabricate {_MIN_DISTRACTORS} coherent distractors)."
        )
    else:
        print(f"[triviaqa] yielded {n_yielded}.")


# ---------------------------------------------------------------------------
# Distractor sampler
# ---------------------------------------------------------------------------

def _pick_distractors(
    gold: str,
    gold_type: str,
    alias_set: set[str],
    pool_by_type: dict[str, list[str]],
    rng: random.Random,
) -> Optional[list[str]]:
    """Try to fabricate `_MIN_DISTRACTORS` distractors of the same type as gold.

    Progression:
      1. Same option_type as the gold (preferred).
      2. Any option_type (fallback for tiny type pools).
    """
    target = _MIN_DISTRACTORS
    chosen: list[str] = []
    seen_lower: set[str] = set(alias_set)

    def _try_take_from(pool: list[str]) -> None:
        # Random window for cheap sampling — pool is pre-shuffled but
        # each call should not always start at index 0.
        n = len(pool)
        if n == 0:
            return
        start = rng.randrange(n)
        for off in range(n):
            if len(chosen) >= target:
                return
            cand = pool[(start + off) % n]
            cl = _normalise(cand)
            if cl in seen_lower:
                continue
            if cl in alias_set:
                continue
            # Reject distractors that contain the gold (or vice-versa).
            if cl in _normalise(gold) or _normalise(gold) in cl:
                continue
            seen_lower.add(cl)
            chosen.append(cand.strip())

    _try_take_from(pool_by_type.get(gold_type, []))
    if len(chosen) < target:
        for other_t, pool in pool_by_type.items():
            if other_t == gold_type:
                continue
            _try_take_from(pool)
            if len(chosen) >= target:
                break

    if len(chosen) < target:
        return None
    return chosen[:target]
