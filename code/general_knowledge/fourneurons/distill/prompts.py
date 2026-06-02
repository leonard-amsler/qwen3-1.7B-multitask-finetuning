from __future__ import annotations

import re

# Heuristic: treat these gold values as binary so we can ask a more
# question-style "why yes/no" prompt.
_BINARY_TOKENS = {"yes", "no", "true", "false"}


def is_binary_gold(gold_text: str) -> bool:
    return gold_text.strip().lower() in _BINARY_TOKENS


def build_user_prompt_short(question: str, gold_text: str) -> str:
    """3-5 sentence rationale. Used for sources where depth isn't needed."""
    question = question.strip()
    gold_text = gold_text.strip()

    if is_binary_gold(gold_text):
        return (
            f"Question: {question}\n\n"
            f"Correct answer: {gold_text}\n\n"
            f"In 3 to 5 sentences, explain why the correct answer to this "
            f"question is \"{gold_text}\". Use general world knowledge and "
            f"step-by-step reasoning. Do not list other possible answers or "
            f"compare with them — just justify this one. Write only the "
            f"explanation, no preamble."
        )

    return (
        f"Question: {question}\n\n"
        f"Correct answer: \"{gold_text}\"\n\n"
        "In 3 to 5 sentences, explain why this is the correct answer. "
        "Focus on the underlying facts, definitions, or reasoning. Do not "
        "compare with or mention other possible options — just justify this "
        "answer. Write only the explanation, no preamble."
    )


def build_user_prompt_long(question: str, gold_text: str) -> str:
    """Long multi-step rationale (v6)."""
    question = question.strip()
    gold_text = gold_text.strip()

    if is_binary_gold(gold_text):
        return (
            f"Question: {question}\n\n"
            f"Correct answer: {gold_text}\n\n"
            "Walk through the reasoning that supports this answer, "
            "step by step. Cover the relevant facts or definitions, the "
            "intermediate deductions, any computations or comparisons "
            "that matter, and how everything combines to justify "
            f"\"{gold_text}\". Aim for 10 to 20 sentences (roughly 300 to "
            "700 words). Stay focused on the reasoning toward this single "
            "answer — do not enumerate or compare with hypothetical "
            "alternatives. Write only the explanation, no preamble, no "
            "labels, no bullet points."
        )

    return (
        f"Question: {question}\n\n"
        f"Correct answer: \"{gold_text}\"\n\n"
        "You are a careful tutor. Walk through the reasoning that leads "
        f"to \"{gold_text}\" as the correct answer for this question. "
        "Cover the relevant concepts, facts, or definitions involved; the "
        "step-by-step intermediate deductions (including any computations, "
        "derivations, or domain-specific reasoning); and why the conclusion "
        "follows from those steps. Aim for 10 to 20 sentences (roughly 300 "
        "to 700 words) of explicit reasoning. Stay focused on the path "
        f"toward \"{gold_text}\" — do not enumerate or compare with other "
        "possible options. Write only the explanation, no preamble, no "
        "labels, no bullet points."
    )


def build_user_prompt_contrastive(question: str, gold_text: str) -> str:
    """Contrastive long rationale (v10)"""
    question = question.strip()
    gold_text = gold_text.strip()

    if is_binary_gold(gold_text):
        return (
            f"Question: {question}\n\n"
            f"Correct answer: {gold_text}\n\n"
            "Walk through the reasoning that supports this answer, step by "
            "step: the relevant facts or definitions, the intermediate "
            f"deductions, and how they combine to justify \"{gold_text}\". "
            "Then explicitly address the opposite conclusion: name the "
            "misconception or faulty assumption that would lead someone to "
            "answer the other way, and explain precisely why it is wrong. "
            "Aim for 8 to 15 sentences. Do not refer to answer letters or "
            "an option list — reason about the content only. Write only the "
            "explanation, no preamble, no labels, no bullet points."
        )

    return (
        f"Question: {question}\n\n"
        f"Correct answer: \"{gold_text}\"\n\n"
        "You are a careful tutor. First, walk through the reasoning that "
        f"leads to \"{gold_text}\" step by step: the relevant concepts, "
        "facts, or definitions; the intermediate deductions (including any "
        "computations, derivations, or domain-specific reasoning); and why "
        "the conclusion follows. Then, contrast it: identify the most "
        "tempting incorrect line of reasoning a knowledgeable person might "
        "follow on this question, and explain clearly what error or "
        "misconception makes it wrong. Aim for 10 to 18 sentences of "
        "explicit reasoning. Do not refer to answer letters or an option "
        f"list — reason about the content only, anchored on \"{gold_text}\". "
        "Write only the explanation, no preamble, no labels, no bullet "
        "points."
    )


# Public alias kept for backwards compatibility with any old caller.
build_user_prompt = build_user_prompt_short


def build_messages(
    question: str,
    gold_text: str,
    style: str = "short",
) -> list[dict]:
    if style == "long":
        content = build_user_prompt_long(question, gold_text)
    elif style == "short":
        content = build_user_prompt_short(question, gold_text)
    elif style == "contrastive":
        content = build_user_prompt_contrastive(question, gold_text)
    else:
        raise ValueError(f"Unknown reasoning style: {style!r}")
    return [{"role": "user", "content": content}]


_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def clean_teacher_output(text: str) -> str:
    text = _THINK_BLOCK_RE.sub("", text).strip()
    text = re.sub(
        r"^(explanation|reasoning|answer|justification)\s*[:\-]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
