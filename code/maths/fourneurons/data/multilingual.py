from datasets import (
    DatasetDict,
    Dataset,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
    concatenate_datasets,
    Value,
)
from tqdm import tqdm
from huggingface_hub import DatasetCard, whoami
import os
from pathlib import Path
import json
from functools import partial

from fourneurons.utils.config import load_config

# ===== XCOPA =====


def format_xcopa(xcopa_dataset: DatasetDict) -> DatasetDict:
    """
    Formats the XCOPA dataset to match our model's expected input/output format.

    Example:
    {"prompt": "This is a questions\n\nA) Option 1\nB) Option 2\nC) Option 3\nD) Option 3", "answer": "C"}
    """

    def format_example(example):
        premise = example["premise"]
        question = example["question"]
        choice1 = example["choice1"]
        choice2 = example["choice2"]
        label = example["label"]
        lang = example["lang"]

        prompt = f"{premise} {question.capitalize()} ?\n\nA) {choice1}\nB) {choice2}"
        answer = "A" if label == 0 else "B"

        return {
            "prompt": prompt,
            "answer": answer,
            "idx": f"{lang}_{example['idx']}",
        }

    xcopa_dataset = xcopa_dataset.map(
        format_example,
        batched=False,
        remove_columns=[
            "premise",
            "question",
            "choice1",
            "choice2",
            "label",
            "changed",
            "lang",
        ],
    )

    return xcopa_dataset


def prepare_xcopa(cfg: dict):
    """Fetches the XCOPA dataset for all languages, merges, and pushes to the Hugging Face Hub."""
    schema = Dataset.from_dict(
        {
            "premise": [],
            "choice1": [],
            "choice2": [],
            "question": [],
            "label": [],
            "idx": [],
            "changed": [],
            "lang": [],
        }
    )

    xcopa_dataset = DatasetDict({"validation": schema, "test": schema})

    expected_langs = cfg["multilingual"]["languages"]
    xcopa_languages = ["et", "ht", "id", "it", "qu", "sw", "ta", "th", "tr", "vi", "zh"]
    filtered_langs = [lang for lang in xcopa_languages if lang in expected_langs]

    for lang in tqdm(filtered_langs, desc="Fetching XCOPA datasets"):
        lang_dataset = load_dataset("xcopa", lang)

        lang_dataset["validation"] = lang_dataset["validation"].add_column(
            "lang", [lang] * len(lang_dataset["validation"])
        )
        lang_dataset["test"] = lang_dataset["test"].add_column(
            "lang", [lang] * len(lang_dataset["test"])
        )

        xcopa_dataset["validation"] = concatenate_datasets(
            [xcopa_dataset["validation"], lang_dataset["validation"]]
        )
        xcopa_dataset["test"] = concatenate_datasets(
            [xcopa_dataset["test"], lang_dataset["test"]]
        )

    # Format dataset
    xcopa_dataset = format_xcopa(xcopa_dataset)

    # Splits
    xcopa_dataset = create_splits(
        xcopa_dataset,
        test_size=cfg["multilingual"]["splits"]["test"],
        val_size=cfg["multilingual"]["splits"]["validation"],
    )

    # Save locally
    out_dir = Path(cfg["paths"]["data"]) / "multilingual" / "xcopa"
    os.makedirs(out_dir, exist_ok=True)
    xcopa_dataset.save_to_disk(out_dir)

    # Push dataset to Hugging Face Hub
    xcopa_dataset.push_to_hub(f"{cfg['hugging_face']['owner']}/xcopa", private=False)

    # Add a dataset card
    card = DatasetCard(
        content=f"""# XCOPA Dataset (merged across languages)
This dataset is a merged version of the XCOPA dataset across candidate languages: {', '.join(filtered_langs)}.

It contains both validation and test splits with the following fields:
- `premise`: The premise of the question.
- `choice1`: The first choice.
- `choice2`: The second choice.
- `question`: The question being asked.
- `label`: The correct answer (0 for choice1, 1 for choice2).
- `idx`: The index of the example.
- `changed`: Whether the example was changed from the original (boolean).
- `lang`: The language of the example.

## Original Dataset
The original XCOPA dataset can be found [here](https://github.com/cambridgeltl/xcopa).
        """,
    )
    card.push_to_hub(f"{cfg['hugging_face']['owner']}/xcopa")

    print(f"Pushed XCOPA dataset to the Hugging Face Hub")


def get_xcopa() -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    """Loads the merged XCOPA dataset from the Hugging Face Hub."""
    cfg = load_config()
    return load_dataset(f"{cfg['hugging_face']['owner']}/xcopa")


# ===== MMMLU =====


def format_ru_mmmlu(raw_dataset: DatasetDict) -> DatasetDict:
    """Formats the Russian MMMLU dataset to match our model's expected input/output format."""

    def format_example(example):
        question = example["question_ru"]
        choices = example["choices_ru"]
        answer = example["answer"]
        subject = example["subject"]

        prompt = f"{question}\n\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}"

        if len(choices) != 4:
            raise ValueError(
                f"Expected 4 choices, got {len(choices)} for example idx {example['idx']}"
            )

        return {
            "prompt": prompt,
            "answer": ["A", "B", "C", "D"][answer],
            "idx": f"ru_{subject}_{example['idx']}",
            "subject": subject,
        }

    formatted_dataset = raw_dataset.map(
        format_example,
        batched=False,
        remove_columns=[
            "question_en",
            "choices_en",
            "question_ru",
            "choices_ru",
            "lang",
        ],
        try_original_type=False,
        disable_nullable=True,
    )

    # Hardcoded to force str
    formatted_dataset = formatted_dataset.cast_column("answer", Value(dtype="string"))

    return formatted_dataset


def load_ru_mmmlu() -> DatasetDict:
    """Loads the Russian MMMLU dataset from NLPCoreTeam/mmlu_ru."""
    topics = [
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "business_ethics",
        "clinical_knowledge",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_medicine",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "econometrics",
        "electrical_engineering",
        "elementary_mathematics",
        "formal_logic",
        "global_facts",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_european_history",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_mathematics",
        "high_school_microeconomics",
        "high_school_physics",
        "high_school_psychology",
        "high_school_statistics",
        "high_school_us_history",
        "high_school_world_history",
        "human_aging",
        "human_sexuality",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "machine_learning",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "moral_disputes",
        "moral_scenarios",
        "nutrition",
        "philosophy",
        "prehistory",
        "professional_accounting",
        "professional_law",
        "professional_medicine",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
        "virology",
        "world_religions",
    ]

    ru_mmmlu_dataset = DatasetDict(
        {
            "test": Dataset.from_dict(
                {
                    "question_en": [],
                    "choices_en": [],
                    "answer": [],
                    "question_ru": [],
                    "choices_ru": [],
                }
            )
        }
    )

    for topic in tqdm(topics, desc="Loading Russian MMMLU topics"):
        topic_dataset = load_dataset("NLPCoreTeam/mmlu_ru", topic)

        lang_concatenated = concatenate_datasets(
            [topic_dataset["test"], topic_dataset["val"], topic_dataset["dev"]]
        )
        lang_concatenated = lang_concatenated.add_column(
            "lang", ["ru"] * len(lang_concatenated)
        )
        lang_concatenated = lang_concatenated.add_column(
            "subject", [topic] * len(lang_concatenated)
        )
        lang_concatenated = lang_concatenated.add_column(
            "idx", list(range(len(lang_concatenated)))
        )

        ru_mmmlu_dataset["test"] = concatenate_datasets(
            [ru_mmmlu_dataset["test"], lang_concatenated]
        )

    formatted_ru_mmmlu = format_ru_mmmlu(ru_mmmlu_dataset)

    return formatted_ru_mmmlu


def format_mmmlu(mmmlu_dataset: DatasetDict) -> DatasetDict:
    """
    Formats the MMMLU dataset to match our model's expected input/output format.

    Example:
    {"prompt": "This is a questions\n\nA) Option 1\nB) Option 2\nC) Option 3\nD) Option 3", "answer": "C"}
    """

    def format_example(example):
        question = example["Question"]
        choices = [example[opt] for opt in ["A", "B", "C", "D"]]
        label = example["Answer"]

        prompt = f"{question}\n\nA) {choices[0]}\nB) {choices[1]}\nC) {choices[2]}\nD) {choices[3]}"
        answer = label

        return {
            "prompt": prompt,
            "answer": answer,
            "idx": f"{example['Language']}_{example['Subject']}_{example['Unnamed: 0']}",
            "subject": example["Subject"],
        }

    mmmlu_dataset = mmmlu_dataset.map(
        format_example,
        batched=False,
        remove_columns=[
            "Unnamed: 0",
            "Question",
            "A",
            "B",
            "C",
            "D",
            "Answer",
            "Subject",
            "Language",
        ],
        disable_nullable=True,
    )

    return mmmlu_dataset


def prepare_mmmlu(cfg: dict):
    """Fetches the MMMLU dataset and pushes to the Hugging Face Hub."""
    languages = cfg["multilingual"]["languages"]

    lang_map = {
        "it": "IT_IT",
        "es": "ES_LA",
        "zh": "ZH_CN",
        "ru": None,
        "hi": "HI_IN",
    }

    mmmlu_dataset = DatasetDict(
        {
            "test": Dataset.from_dict(
                {
                    "Unnamed: 0": [],
                    "Question": [],
                    "A": [],
                    "B": [],
                    "C": [],
                    "D": [],
                    "Answer": [],
                    "Subject": [],
                    "Language": [],
                }
            )
        }
    )

    for lang in languages:
        if not lang_map.get(lang):
            continue

        lang_dataset = load_dataset("openai/MMMLU", lang_map[lang])

        lang_dataset["test"] = lang_dataset["test"].add_column(
            "Language", [lang] * len(lang_dataset["test"])
        )

        mmmlu_dataset["test"] = concatenate_datasets(
            [mmmlu_dataset["test"], lang_dataset["test"]]
        )

    mmmlu_dataset = format_mmmlu(mmmlu_dataset)

    ru_mmmlu = load_ru_mmmlu()
    mmmlu_dataset["test"] = concatenate_datasets(
        [mmmlu_dataset["test"], ru_mmmlu["test"]]
    )

    # Drop duplicates prompts (from https://github.com/huggingface/datasets/issues/2514#issuecomment-962496585)
    memory = set()
    def is_unique(elem , column: str, memory: set) -> bool:
        if elem[column] in memory:
            return False
        else:
            memory.add(elem[column])
            return True
    len_before = len(mmmlu_dataset["test"])
    mmmlu_dataset["test"] = mmmlu_dataset["test"].filter(partial(is_unique, column="prompt", memory=memory))
    len_after = len(mmmlu_dataset["test"])
    if len_before - len_after > 0:
        print(f"Dropped {len_before - len_after} duplicate prompts from MMMLU dataset")
    else:
        print("No duplicate prompts found in MMMLU dataset")

    # Check ID uniqueness
    all_ids = mmmlu_dataset["test"]["idx"]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Duplicate IDs found in the merged MMMLU dataset")
    else:
        print("ID uniqueness check passed for merged MMMLU dataset")

    # Split
    mmmlu_dataset = create_splits(
        mmmlu_dataset,
        test_size=cfg["multilingual"]["splits"]["test"],
        val_size=cfg["multilingual"]["splits"]["validation"],
    )

    # Save locally
    out_dir = Path(cfg["paths"]["data"]) / "multilingual" / "mmmlu"
    os.makedirs(out_dir, exist_ok=True)
    mmmlu_dataset.save_to_disk(out_dir)

    # Push dataset to Hugging Face Hub
    mmmlu_dataset.push_to_hub(f"{cfg['hugging_face']['owner']}/mmmlu", private=False)

    # Add a dataset card
    card = DatasetCard(
        content=f"""# MMMLU Dataset (merged across languages)
This dataset is a merged version of the MMMLU dataset across candidate languages: {', '.join(lang_map[lang] for lang in languages if lang_map.get(lang))} and the Russian MMMLU dataset from NLPCoreTeam/mmlu_ru.

## Original Datasets
The original MMMLU dataset can be found [here](https://huggingface.co/datasets/openai/MMMLU).
Additionally, we include the Russian MMMLU dataset from [NLPCoreTeam/mmlu_ru](https://huggingface.co/datasets/NLPCoreTeam/mmlu_ru).
        """,
    )
    card.push_to_hub(f"{cfg['hugging_face']['owner']}/mmmlu")

    print(f"Pushed MMMLU dataset to the Hugging Face Hub")


def prepare_mmmlu_jsonl(cfg: dict):
    """Prepares the MMMLU test dataset in the format expected by OpenCompass and saves locally."""
    mmmlu_dataset = get_mmmlu()

    train_out_dir = (
        Path(cfg["paths"]["data"])
        / "multilingual"
        / "mmmlu"
        / "splits"
        / "mmmlu_train.jsonl"
    )
    test_out_dir = (
        Path(cfg["paths"]["data"])
        / "multilingual"
        / "mmmlu"
        / "splits"
        / "mmmlu_test.jsonl"
    )
    val_out_dir = (
        Path(cfg["paths"]["data"])
        / "multilingual"
        / "mmmlu"
        / "splits"
        / "mmmlu_validation.jsonl"
    )
    os.makedirs(train_out_dir.parent, exist_ok=True)
    os.makedirs(test_out_dir.parent, exist_ok=True)
    os.makedirs(val_out_dir.parent, exist_ok=True)

    # Save as JSONL
    with open(train_out_dir, "w+", encoding="utf-8") as f:
        for example in mmmlu_dataset["train"]:
            json.dump(example, f, ensure_ascii=False)
            f.write("\n")
    with open(test_out_dir, "w+", encoding="utf-8") as f:
        for example in mmmlu_dataset["test"]:
            json.dump(example, f, ensure_ascii=False)
            f.write("\n")
    with open(val_out_dir, "w+", encoding="utf-8") as f:
        for example in mmmlu_dataset["validation"]:
            json.dump(example, f, ensure_ascii=False)
            f.write("\n")


def get_mmmlu() -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    """Loads the MMMLU dataset from the Hugging Face Hub."""
    cfg = load_config()
    return load_dataset(f"{cfg['hugging_face']['owner']}/mmmlu")


# ===== TyDi QA =====


def get_tydiqa() -> DatasetDict | Dataset | IterableDatasetDict | IterableDataset:
    # TODO : https://github.com/google-research-datasets/tydiqa
    raise NotImplementedError("TyDi QA dataset loading not implemented yet.")


# ===== Create splits =====


def create_splits(dataset: DatasetDict, test_size: float, val_size: float):
    """Splits the existing dataset into stratified train/validation/test splits based on provided ratios."""

    def extract_lang(example):
        example["lang"] = example["idx"].split("_")[0]
        return example

    dataset["test"] = dataset["test"].map(extract_lang, batched=False)

    train_splits, val_splits, test_splits = [], [], []

    # Stratifified split by language (for now everything is test)
    languages = set(dataset["test"]["lang"])
    for lang in languages:
        lang_subset = dataset["test"].filter(lambda ex: ex["lang"] == lang)
        total = len(lang_subset)

        n_train = int(total * (1 - test_size - val_size))
        n_val = int(total * val_size)

        shuffled = lang_subset.shuffle(seed=42)
        train_splits.append(shuffled.select(range(n_train)))
        val_splits.append(shuffled.select(range(n_train, n_train + n_val)))
        test_splits.append(shuffled.select(range(n_train + n_val, total)))

    train_dataset = concatenate_datasets(train_splits)
    val_dataset = concatenate_datasets(val_splits)
    test_dataset = concatenate_datasets(test_splits)

    print(
        f"Created splits with sizes: train={len(train_dataset)}, validation={len(val_dataset)}, test={len(test_dataset)}"
    )

    return DatasetDict(
        {
            "train": train_dataset,
            "validation": val_dataset,
            "test": test_dataset,
        }
    )


# ===== Main =====

if __name__ == "__main__":
    print(f"Current Hugging Face user: {whoami(cache=True)['name']}")

    config = load_config()

    # prepare_xcopa(config)
    prepare_mmmlu(config)
    prepare_mmmlu_jsonl(config)
