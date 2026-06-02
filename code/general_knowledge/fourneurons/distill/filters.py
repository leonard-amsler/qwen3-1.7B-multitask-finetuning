

from __future__ import annotations

import re
from collections import Counter
from typing import Optional


_REFUSAL_PATTERNS = (
    "i cannot answer",
    "i can't answer",
    "i do not know",
    "i don't know",
    "as an ai",
    "as a language model",
    "i'm not able to",
    "i am not able to",
    "i am sorry, but",
    "i'm sorry, but",
)


_BAD_PATTERNS = (
    "```",            # code fences
    "<|endoftext|>",
    "</s>",
)



_LIST_LEAK_PREFIX_RE = re.compile(r"^\s*[\[\{]")
_LIST_LEAK_SEPARATOR_RE = re.compile(r"',\s*'|\",\s*\"")


_BINARY_GOLDS = {"yes", "no", "true", "false"}
_BINARY_FAMILY = {
    "yes": {"yes", "correct", "true", "right", "indeed"},
    "no":  {"no", "incorrect", "false", "wrong", "isn't", "not"},
    "true": {"true", "yes", "correct", "right"},
    "false": {"false", "no", "incorrect", "wrong"},
}


def _is_compound_binary(gold_text: str) -> bool:
    parts = [p.strip().lower() for p in gold_text.split(",")]
    return len(parts) >= 2 and all(p in _BINARY_GOLDS for p in parts)


# Pre-compiled tokenizer for content-word overlap.
_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Short list of high-frequency function words. We exclude these so that an
# overlap score is computed only on *informative* tokens. Anything <= 1 char
# is also dropped via the `min_len` parameter in `_content_words`.
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "and", "or", "but", "of", "to", "in", "on", "at", "by", "for", "from",
    "with", "as", "this", "that", "these", "those", "it", "its", "their",
    "if", "then", "than", "so", "not", "yes", "no", "true", "false",
    "into", "onto", "over", "under", "above", "below", "between", "among",
    "all", "each", "any", "some", "every", "only",
    "one", "two", "three", "four", "five",
    "such", "which", "who", "what", "where", "when", "how", "why",
    "do", "does", "did", "has", "have", "had",
    "can", "could", "would", "should", "may", "might", "will", "shall",
}


def _content_words(text: str, min_len: int = 2) -> set[str]:
    return {
        t for t in _TOKEN_RE.findall(text.lower())
        if len(t) >= min_len and t not in _STOPWORDS
    }


def _has_gold_mention(
    cot_lower: str,
    gold_text: str,
    overlap_threshold: float = 0.4,
) -> bool:
    g = gold_text.strip().lower()
    if not g:
        return True

    if _is_compound_binary(gold_text):
        return True

    if g in _BINARY_GOLDS:
        family = _BINARY_FAMILY.get(g, {g})
        return any(w in cot_lower for w in family)

    if len(g) <= 3 and g.isalnum():
        return re.search(r"\b" + re.escape(g) + r"\b", cot_lower) is not None

    if g in cot_lower or g.replace(" ", "") in cot_lower.replace(" ", ""):
        return True

    gold_words = _content_words(gold_text)
    if gold_words:
        cot_words = _content_words(cot_lower)
        overlap = len(gold_words & cot_words) / len(gold_words)
        if overlap >= overlap_threshold:
            return True

    short_tokens = _TOKEN_RE.findall(g)
    if short_tokens and len(g) <= 30 and not gold_words:
        matched = sum(
            1 for t in short_tokens
            if re.search(r"\b" + re.escape(t) + r"\b", cot_lower)
        )
        if matched / len(short_tokens) >= 0.5:
            return True

    return False


def _max_ngram_repeats(text: str, n: int = 3) -> int:
    words = text.lower().split()
    if len(words) < n + 1:
        return 0
    grams = [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]
    if not grams:
        return 0
    return Counter(grams).most_common(1)[0][1]


def quality_check(
    cot: str,
    gold_text: str,
    min_chars: int = 200,
    max_chars: int = 6000,
    max_3gram_repeats: int = 6,
) -> tuple[bool, Optional[str]]:

    cot = (cot or "").strip()
    if len(cot) < min_chars:
        return False, "too_short"
    if len(cot) > max_chars:
        return False, "too_long"

    if _LIST_LEAK_PREFIX_RE.match(cot):
        return False, "list_format_prefix"
    if _LIST_LEAK_SEPARATOR_RE.search(cot):
        return False, "list_format_separator"

    lower = cot.lower()
    for p in _REFUSAL_PATTERNS:
        if p in lower:
            return False, "refusal"
    for p in _BAD_PATTERNS:
        if p in cot:
            return False, "bad_pattern"

    if not _has_gold_mention(lower, gold_text):
        return False, "no_gold_mention"

    reps = _max_ngram_repeats(cot, n=3)
    if reps > max_3gram_repeats:
        return False, f"repetitive(reps={reps})"

    return True, None
