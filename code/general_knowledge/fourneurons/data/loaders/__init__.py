"""Per-source loaders that return iterables of `McqExample`.

Each loader exposes a `load_<source>(split=..., limit=None)` function.
"""

from .mmlu_pro_cot import load_mmlu_pro_cot
from .mmlu import load_mmlu
from .mmlu_pro import load_mmlu_pro
from .mmlu_world import load_mmlu_world
from .boolq import load_boolq
from .commonsenseqa import load_commonsenseqa
from .ecqa import load_ecqa
from .socialiqa import load_socialiqa
from .arc import load_arc_challenge
from .openbookqa import load_openbookqa
from .triviaqa import load_triviaqa

__all__ = [
    "load_mmlu_pro_cot",
    "load_mmlu",
    "load_mmlu_pro",
    "load_mmlu_world",
    "load_boolq",
    "load_commonsenseqa",
    "load_ecqa",
    "load_socialiqa",
    "load_arc_challenge",
    "load_openbookqa",
    "load_triviaqa",
]
