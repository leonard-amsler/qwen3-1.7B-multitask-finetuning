"""Mapping from fine-grained subjects to our 5 macro-categories.

Used by every loader so the downstream stratifier sees a consistent label
space. Unknown subjects fall back to "commonsense" (the safest default for
miscellaneous trivia).
"""

from __future__ import annotations

# MMLU subjects (57) -> macro category.
MMLU_SUBJECT_TO_MACRO = {
    # STEM
    "abstract_algebra": "stem",
    "anatomy": "stem",
    "astronomy": "stem",
    "clinical_knowledge": "stem",
    "college_biology": "stem",
    "college_chemistry": "stem",
    "college_computer_science": "stem",
    "college_mathematics": "stem",
    "college_medicine": "stem",
    "college_physics": "stem",
    "computer_security": "stem",
    "conceptual_physics": "stem",
    "electrical_engineering": "stem",
    "elementary_mathematics": "stem",
    "high_school_biology": "stem",
    "high_school_chemistry": "stem",
    "high_school_computer_science": "stem",
    "high_school_mathematics": "stem",
    "high_school_physics": "stem",
    "high_school_statistics": "stem",
    "human_aging": "stem",
    "machine_learning": "stem",
    "medical_genetics": "stem",
    "nutrition": "stem",
    "professional_medicine": "stem",
    "virology": "stem",
    # Humanities
    "formal_logic": "humanities",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "philosophy": "humanities",
    "professional_law": "humanities",
    "world_religions": "humanities",
    # Social sciences
    "business_ethics": "social_sciences",
    "econometrics": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_microeconomics": "social_sciences",
    "high_school_psychology": "social_sciences",
    "human_sexuality": "social_sciences",
    "management": "social_sciences",
    "marketing": "social_sciences",
    "professional_accounting": "social_sciences",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    # History & geography
    "global_facts": "history_geo",
    "high_school_european_history": "history_geo",
    "high_school_geography": "history_geo",
    "high_school_us_history": "history_geo",
    "high_school_world_history": "history_geo",
    "prehistory": "history_geo",
    "us_foreign_policy": "history_geo",
    # Misc / commonsense
    "miscellaneous": "commonsense",
}


# MMLU-Pro coarse categories (14) -> macro category.
MMLU_PRO_CATEGORY_TO_MACRO = {
    "math": "stem",
    "physics": "stem",
    "chemistry": "stem",
    "biology": "stem",
    "computer science": "stem",
    "engineering": "stem",
    "health": "stem",
    "philosophy": "humanities",
    "law": "humanities",
    "history": "history_geo",
    "business": "social_sciences",
    "economics": "social_sciences",
    "psychology": "social_sciences",
    "other": "commonsense",
}


def mmlu_subject_to_macro(subject: str) -> str:
    return MMLU_SUBJECT_TO_MACRO.get(subject.lower().strip(), "commonsense")


def mmlu_pro_category_to_macro(category: str) -> str:
    return MMLU_PRO_CATEGORY_TO_MACRO.get(
        category.lower().strip(), "commonsense"
    )
