from __future__ import annotations

from typing import Optional

from .schema import McqExample, format_prompt


_BOXED_ANSWER_LINE = "The answer is \\boxed{{{letter}}}"


def _synthetic_cot(ex: McqExample) -> str:
    gold_text = ex.options[ex.gold_idx].strip()
    return (
        f"Among the options, the one that best matches the question is "
        f'"{gold_text}", which is option {ex.gold_letter}.'
    )


def to_chat_messages(
    ex: McqExample,
    distilled_cot: Optional[str] = None,
    cot_if_missing: bool = True,
) -> Optional[list[dict]]:
    user_content = format_prompt(ex)

    if ex.cot and ex.cot.strip():
        reasoning = ex.cot.strip()
    elif distilled_cot and distilled_cot.strip():
        reasoning = distilled_cot.strip()
    elif cot_if_missing:
        reasoning = _synthetic_cot(ex)
    else:
        return None

    assistant_content = (
        f"<think>\n{reasoning}\n</think>\n\n"
        + _BOXED_ANSWER_LINE.format(letter=ex.gold_letter)
    )

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def to_training_row(
    ex: McqExample,
    is_augmented: bool = False,
    cot_if_missing: bool = True,
    distilled_cot: Optional[str] = None,
) -> Optional[dict]:
    if ex.cot and ex.cot.strip():
        cot_source = "loader"
    elif distilled_cot and distilled_cot.strip():
        cot_source = "distilled"
    elif cot_if_missing:
        cot_source = "synthetic"
    else:
        return None

    messages = to_chat_messages(
        ex, distilled_cot=distilled_cot, cot_if_missing=cot_if_missing
    )
    if messages is None:
        return None

    return {
        "messages": messages,
        "macro_cat": ex.macro_cat,
        "source": ex.source,
        "subject": ex.subject,
        "n_options": ex.n_options,
        "is_augmented": is_augmented,
        "gold_letter": ex.gold_letter,
        "uid": ex.uid,
        "cot_source": cot_source,
    }
