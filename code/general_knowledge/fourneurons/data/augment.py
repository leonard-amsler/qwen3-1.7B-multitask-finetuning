from __future__ import annotations

import random
import re
from collections import defaultdict
from typing import Iterable, Optional

from .schema import McqExample


MAX_K = 20

_SAMPLE_CANDIDATE_CAP = 512



_YEAR_RE = re.compile(r"^\s*(?:1[0-9]{3}|20[0-9]{2}|2100)\s*$")
_NUMERIC_RE = re.compile(
    r"""
    ^\s*
    [-+]?
    (?:\d+\.\d+|\d+/\d+|\d+,\d+|\d+)        # 12 / 12.3 / 12/3 / 1,234
    (?:\s*(?:%|[a-zA-Z][a-zA-Z\d/\^\-]*))?  # optional unit (kg, m^2, mol/L, %, ...)
    \s*$
    """,
    re.VERBOSE,
)


def classify_option(text: str) -> str:
    """Return one of `{year, numeric, short_entity, phrase}` for `text`."""
    t = (text or "").strip()
    if not t:
        return "phrase"
    if _YEAR_RE.match(t):
        return "year"
    if _NUMERIC_RE.match(t):
        return "numeric"

    words = t.split()
    if (
        1 <= len(words) <= 3
        and words[0][:1].isupper()
        and not t.endswith((".", "?", "!", ";"))
        and "," not in t
    ):
        return "short_entity"
    return "phrase"


def _length_bucket(text: str) -> str:
    n = len(text or "")
    if n <= 15:
        return "S"
    if n <= 60:
        return "M"
    return "L"


class DistractorPool:

    def __init__(self, length_aware: bool = True) -> None:
        self._by_macro_type: dict[
            tuple[str, str], list[str]
        ] = defaultdict(list)
        self._by_subject_type: dict[
            tuple[str, str, str], list[str]
        ] = defaultdict(list)
        self._seen: set[tuple[str, str, str]] = set()  # (macro, type, opt)
        self.length_aware = length_aware

    def ingest(self, examples: Iterable[McqExample]) -> None:
        for ex in examples:
            mc = ex.macro_cat
            sub = (ex.subject or "").strip()
            for opt in ex.options:
                opt = (opt or "").strip()
                if not opt:
                    continue
                t = classify_option(opt)
                key = (mc, t, opt)
                if key in self._seen:
                    continue
                self._seen.add(key)
                self._by_macro_type[(mc, t)].append(opt)
                if sub:
                    self._by_subject_type[(mc, sub, t)].append(opt)

    def size_by_macro(self) -> dict[str, int]:
        out: dict[str, int] = defaultdict(int)
        for (mc, _t), opts in self._by_macro_type.items():
            out[mc] += len(opts)
        return dict(out)

    def size_by_macro_type(self) -> dict[tuple[str, str], int]:
        return {k: len(v) for k, v in self._by_macro_type.items()}

    def sample(
        self,
        macro_cat: str,
        subject: Optional[str],
        gold_text: str,
        n: int,
        exclude: set[str],
        rng: random.Random,
    ) -> list[str]:
        t = classify_option(gold_text)
        gold_bucket = _length_bucket(gold_text)
        seen: set[str] = set(s.strip().lower() for s in exclude)

        def _filter_by_length(opts: Iterable[str]) -> list[str]:
            if not self.length_aware:
                return list(opts)
            same = [o for o in opts if _length_bucket(o) == gold_bucket]
            return same if len(same) >= n + 4 else list(opts)

        def _take_from_pool(pool: list[str]) -> None:
            if not pool or len(out) >= n:
                return
            candidates = _filter_by_length(pool)
            if len(candidates) > _SAMPLE_CANDIDATE_CAP:
                candidates = rng.sample(candidates, _SAMPLE_CANDIDATE_CAP)
            for o in candidates:
                if len(out) >= n:
                    break
                ol = o.strip().lower()
                if ol in seen:
                    continue
                if ol == gold_text.strip().lower():
                    continue
                seen.add(ol)
                out.append(o)

        out: list[str] = []
        if subject:
            sub_pool = self._by_subject_type.get((macro_cat, subject, t))
            if sub_pool:
                _take_from_pool(sub_pool)

        if len(out) < n:
            macro_pool = self._by_macro_type.get((macro_cat, t))
            if macro_pool:
                _take_from_pool(macro_pool)

        if len(out) < n:
            for (mc, tt), opts in self._by_macro_type.items():
                if mc != macro_cat or tt == t:
                    continue
                _take_from_pool(opts)
                if len(out) >= n:
                    break
        return out



def subsample_options(
    ex: McqExample,
    k: int,
    rng: random.Random,
) -> Optional[McqExample]:
    """Keep `k` options out of `ex.options`, always keeping the gold."""
    n = ex.n_options
    if not (2 <= k <= n) or k > MAX_K:
        return None
    if k == n:
        return _reshuffle(ex, rng, source_tag_suffix="aug_shuf")

    gold_text = ex.options[ex.gold_idx]
    distractors = [opt for i, opt in enumerate(ex.options) if i != ex.gold_idx]
    rng.shuffle(distractors)
    picked = distractors[: k - 1] + [gold_text]
    rng.shuffle(picked)
    new_gold_idx = picked.index(gold_text)
    return McqExample(
        question=ex.question,
        options=picked,
        gold_idx=new_gold_idx,
        source=f"{ex.source}_aug_sub{k}",
        macro_cat=ex.macro_cat,
        cot=None,
        subject=ex.subject,
        uid=ex.uid,
    )


def expand_options(
    ex: McqExample,
    k: int,
    pool: DistractorPool,
    rng: random.Random,
) -> Optional[McqExample]:
    """Add (k - n_orig) typed distractors so the example has `k` options."""
    n = ex.n_options
    if not (n < k <= MAX_K):
        return None
    need = k - n
    gold_text = ex.options[ex.gold_idx]
    exclude = {opt.strip() for opt in ex.options}
    extras = pool.sample(ex.macro_cat, ex.subject, gold_text, need, exclude, rng)
    if len(extras) < need:
        if not extras:
            return None
        k = n + len(extras)
    new_options = list(ex.options) + extras[: k - n]
    rng.shuffle(new_options)
    new_gold_idx = new_options.index(gold_text)
    return McqExample(
        question=ex.question,
        options=new_options,
        gold_idx=new_gold_idx,
        source=f"{ex.source}_aug_exp{k}",
        macro_cat=ex.macro_cat,
        cot=None,
        subject=ex.subject,
        uid=ex.uid,
    )


def _reshuffle(
    ex: McqExample,
    rng: random.Random,
    source_tag_suffix: str = "aug_shuf",
) -> McqExample:
    pairs = list(enumerate(ex.options))
    rng.shuffle(pairs)
    new_options = [opt for _, opt in pairs]
    old_to_new = {old: new for new, (old, _) in enumerate(pairs)}
    return McqExample(
        question=ex.question,
        options=new_options,
        gold_idx=old_to_new[ex.gold_idx],
        source=f"{ex.source}_{source_tag_suffix}",
        macro_cat=ex.macro_cat,
        cot=None,
        subject=ex.subject,
        uid=ex.uid,
    )



EXPAND_K_WEIGHTS = {
    6: 1.0,
    7: 1.0,
    8: 1.0,
    9: 1.0,
    10: 1.0,
    11: 1.2,
    12: 1.2,
    13: 1.2,
    14: 1.2,
    15: 1.2,
    16: 1.2,
    17: 1.0,
    18: 1.0,
    19: 1.0,
    20: 1.0,
}


LEGACY_K_WEIGHTS = {
    2: 2.0, 3: 2.0, 4: 0.5, 5: 0.5,
    6: 1.0, 7: 1.0, 8: 1.0, 9: 1.0, 10: 0.5,
    11: 1.5, 12: 1.5, 13: 1.5, 14: 1.5, 15: 1.5, 16: 1.5,
    17: 1.0, 18: 1.0, 19: 1.0, 20: 1.0,
}


def generate_variants(
    ex: McqExample,
    pool: DistractorPool,
    rng: random.Random,
    max_variants: int = 2,
    k_weights: Optional[dict[int, float]] = None,
    max_k: int = MAX_K,
    expand_only: bool = True,
) -> list[McqExample]:
    """Sample up to `max_variants` k-options variants for one example.

    With `expand_only=True` (v5 default): only k > n_orig expansions; native k
    is preserved. With `expand_only=False`: also allow subsampling to k < n_orig.
    """
    weights = k_weights or (EXPAND_K_WEIGHTS if expand_only else LEGACY_K_WEIGHTS)
    n = ex.n_options

    if expand_only:
        ks = [k for k in weights if n < k <= max_k]
    else:
        ks = [k for k in weights if 2 <= k <= max_k and k != n]

    if not ks:
        return []
    w = [weights[k] for k in ks]

    chosen: list[int] = []
    tries = 0
    while len(chosen) < max_variants and tries < max_variants * 3:
        tries += 1
        k = rng.choices(ks, weights=w, k=1)[0]
        if k in chosen:
            continue
        chosen.append(k)

    variants: list[McqExample] = []
    for k in chosen:
        v = expand_options(ex, k, pool, rng) if k > n else subsample_options(ex, k, rng)
        if v is not None:
            variants.append(v)
    return variants


DEFAULT_K_WEIGHTS = LEGACY_K_WEIGHTS
