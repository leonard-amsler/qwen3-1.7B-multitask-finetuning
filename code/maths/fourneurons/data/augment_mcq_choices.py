# Example : python augment_mcq_choices.py multilingual mmmlu train Qwen/Qwen3-32B-AWQ mmmlu_more_qcms --cont

import argparse
from transformers import AutoTokenizer
from datasets import Dataset
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams
from itertools import groupby
import json
from pydantic import BaseModel, Field
import os
from typing import List, Dict
import numpy as np
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
from scipy.stats import gamma

SYSTEM_PROMPT = """# Task
You are given a multiple-choice question with one correct answer and several wrong answers. 
Your task is to generate additional wrong answers that are plausible but incorrect. 
The generated additional answers should not be obviously incorrect or nonsensical, and should be similar in style and content to the original choices.
In any case, the generated wrong choices must not include the correct answer or repeat any of the original wrong choices.

# Language
The question and choices can be in any of the following languages: Italian, Spanish, Chinese, Russian, Hindi.
Make sure to generate wrong choices that are in the same language as the question and original choices.

# Example
Question: What is the capital of France?
Choices: Berlin, Madrid, Paris, Rome
Additional wrong choices, 3 examples:
['Lisbon', 'Vienna', 'Brussels']
"""

USER_PROMPT_TEMPLATE = """Question: {question}
Choices: {choices}
Additional wrong choices, {n} examples:"""


def make_sampling_params(n_distractors: int) -> SamplingParams:
    class Distractors(BaseModel):
        distractors: List[str] = Field(
            min_length=n_distractors, max_length=n_distractors
        )

    return SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=4196,
        structured_outputs=StructuredOutputsParams(
            json=Distractors.model_json_schema()
        ),
        seed=42,
    )


def format_sample(sample: Dict) -> Dict:
    p = sample["prompt"]
    question = p.split("\n\nA)")[0].strip()
    choices_part = "A)" + p.split("\n\nA)")[1].strip()
    choices = []
    for line in choices_part.split("\n"):
        if line.strip():
            label = line[0]
            choice = line[2:].strip()
            if sample["answer"] == label:
                sample["answer"] = choice
            else:
                choices.append(choice)
    sample["prompt"] = question
    sample["choices"] = choices

    return sample


def augment(
    benchmark: str,
    dataset: str,
    split: str,
    model: str,
    name: str,
    continue_existing: bool,
) -> None:
    """
    Generates additional wrong choices for multiple-choice questions in a dataset using a base model and saves the augmented dataset to a file.
    """
    # Output path
    out_path = (
        f"/scratch/data/{benchmark}/{name}/splits/temp/{name}_{split}_augmented.jsonl"
    )

    if os.path.exists(out_path) and not continue_existing:
        raise FileExistsError(f"Output dataset already exists: {out_path}.")
    if continue_existing and not os.path.exists(
        f"/scratch/data/{benchmark}/{name}/splits/temp/{name}_{split}_augmented.jsonl"
    ):
        raise FileNotFoundError(
            f"No existing file found at {out_path}. Please provide --cont flag only if you want to continue from an existing file."
        )

    os.makedirs(f"/scratch/data/{benchmark}/{name}/splits/temp/", exist_ok=True)

    # Load model
    teacher = LLM(model=model, tokenizer=model, seed=42)
    tokenizer = AutoTokenizer.from_pretrained(model)

    # Load dataset
    dataset_path = f"/scratch/data/{benchmark}/{dataset}/splits/{dataset}_{split}.jsonl"
    with open(dataset_path) as f:
        raw_samples = [json.loads(line) for line in f if line.strip()]
    samples = [format_sample(p) for p in raw_samples]
    print(f"Loaded {len(samples)} prompts from {dataset_path}")

    # Define how many distractors to generate for each question
    MIN_DISTRACTORS = 2
    MAX_DISTRACTORS = 20
    CURR_DISTRACTORS = 4
    total = len(samples)

    np.random.seed(42)
    n_choices = np.random.randint(MIN_DISTRACTORS, MAX_DISTRACTORS + 1, size=total)
    for prompt, n in zip(samples, n_choices):
        prompt["n_missing_distractors"] = max(0, int(n - CURR_DISTRACTORS))

    remaining_samples = samples.copy()

    # Filter out samples that have already been augmented if continuing from an existing file
    if continue_existing and os.path.exists(out_path):
        with open(out_path) as f:
            existing_samples = [json.loads(line) for line in f if line.strip()]
        existing_ids = set(s["idx"] for s in existing_samples)
        remaining_samples = [
            s for s in remaining_samples if s["idx"] not in existing_ids
        ]
        print(
            f"Continuing from existing file. {len(existing_samples)} samples already augmented, {len(remaining_samples)} samples remaining to augment."
        )

    # Generation loop
    loops = 0
    with open(out_path, "w+" if not continue_existing else "a+") as fout:
        while remaining_samples:
            if loops > 3:  # TODO Hardcoded limit
                print(
                    f"Reached maximum number of loops. Stopping augmentation. {len(remaining_samples)} samples were not successfully augmented."
                )
                break
            loops += 1

            total = len(remaining_samples)
            rows_sorted = sorted(
                remaining_samples, key=lambda p: p["n_missing_distractors"]
            )
            remaining_samples = []

            for n, group in groupby(
                rows_sorted, key=lambda r: r["n_missing_distractors"]
            ):
                group = list(group)
                total -= len(group)

                print(
                    f"Generating {n} distractors for {len(group)} ({total} remaining) questions..."
                )

                if n == 0:
                    # Write directly without generation
                    for row in group:
                        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    continue

                # Generate additional distractors for this group
                prompts = []
                for row in group:
                    prompt = tokenizer.apply_chat_template(
                        [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {
                                "role": "user",
                                "content": USER_PROMPT_TEMPLATE.format(
                                    question=row["prompt"],
                                    choices=", ".join(row["choices"]),
                                    n=row["n_missing_distractors"],
                                ),
                            },
                        ],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    prompts.append(prompt)

                outputs = teacher.generate(
                    prompts=prompts, sampling_params=make_sampling_params(n)
                )

                # Format and save
                for row, output in zip(group, outputs, strict=True):
                    completion = output.outputs[0].text
                    try:
                        distractors = json.loads(completion)["distractors"]
                        row["choices"] += distractors
                        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    except Exception as e:
                        remaining_samples.append(row)


def balance_augmented(benchmark: str, dataset: str, split: str, model: str, name: str):
    """
    Loads the augmented dataset and adapts the number of choices to match a desired distribution
    """
    print("Balancing MCQ choices in augmented dataset...")
    augmented_path = (
        f"/scratch/data/{benchmark}/{name}/splits/temp/{name}_{split}_augmented.jsonl"
    )
    balanced_path = (
        f"/scratch/data/{benchmark}/{name}/splits/temp/{name}_{split}_balanced.jsonl"
    )

    if os.path.exists(balanced_path):
        print(
            f"Balanced dataset already exists at {balanced_path}. Skipping balancing step."
        )
        return

    with open(augmented_path) as f:
        data = [json.loads(line) for line in f if line.strip()]

    # Create MCQ choice distribution with a peak around 4 and still some samples at edges.
    mu, min_k, max_k, shape, edge_weight = 4, 2, 20, 4, 0.35
    ks = np.arange(min_k, max_k + 1)
    peak = gamma.pdf(ks, a=shape, scale=mu / (shape - 1))  # Sharp gamma peak
    peak /= peak.sum()
    uniform = np.ones(len(ks)) / len(
        ks
    )  # Uniform component (lifts all buckets equally)
    probs = (1 - edge_weight) * peak + edge_weight * uniform  # Mix
    probs /= probs.sum()
    target_probs = dict(zip(ks, probs))

    languages = set(item["lang"] for item in data)
    lang_counts = Counter(item["lang"] for item in data)
    target_counts = {
        lang: {t: int(p * lang_counts[lang]) for t, p in target_probs.items()}
        for lang in languages
    }

    remaining = {item["idx"]: item for item in data}
    result = []

    np.random.seed(42)
    duplicate_errors = 0
    for t in sorted(target_probs, reverse=True):
        for lang in languages:
            current = [
                item
                for item in remaining.values()
                if item["lang"] == lang and len(item["choices"]) + 1 >= t
            ]
            num_to_sample = min(target_counts[lang][t], len(current))
            sampled = np.random.choice(current, size=num_to_sample, replace=False)
            for item in sampled:
                if len(item["choices"]) > len(set(item["choices"])):
                    choices = list(set(item["choices"]))
                    if len(choices) < t - 1:
                        duplicate_errors += 1
                        continue
                    item["choices"] = choices
                item["ignored_choices"] = item["choices"][t - 1 :]
                item["choices"] = item["choices"][: t - 1]
                result.append(item)
                del remaining[item["idx"]]

    print(
        f"Balancing complete. {duplicate_errors} were skipped due to duplicate choices. {len(remaining)} samples were not included due to balancing constraints."
    )

    with open(balanced_path, "w") as f:
        for item in result:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Balanced dataset saved to {balanced_path}. Generating reports...")
    fig, axes = plt.subplots(
        len(languages), 1, figsize=(8, 4 * len(languages)), sharex=True
    )
    lang_choice_counts_json = {}
    for ax, lang in zip(axes, languages):
        lang_items = [item for item in result if item["lang"] == lang]
        choice_counts = Counter(len(item["choices"]) + 1 for item in lang_items)
        lang_choice_counts_json[lang] = {
            str(k): v for k, v in sorted(choice_counts.items())
        }
        xs = sorted(choice_counts.keys())
        ys = [choice_counts[x] for x in xs]
        ax.bar(xs, ys)
        ax.set_title(f"Distribution of number of choices for {lang}")
        ax.set_ylabel("Count")

    axes[-1].set_xlabel("Number of choices")
    plt.tight_layout()
    os.makedirs(f"/scratch/results/{benchmark}/{name}/{split}/", exist_ok=True)
    plt.savefig(f"/scratch/results/{benchmark}/{name}/{split}/choices_distribution.png")

    with open(
        f"/scratch/results/{benchmark}/{name}/{split}/choices_distribution.json", "w"
    ) as f:
        json.dump(lang_choice_counts_json, f, indent=2, ensure_ascii=False)

    print("Reports generated and saved.")


def format_augmented(benchmark: str, dataset: str, split: str, model: str, name: str):
    """
    Loads the augmented dataset and formats the choices back into the original format.
    """
    print("Formatting augmented dataset into original format...")
    np.random.seed(42)

    balanced_path = (
        f"/scratch/data/{benchmark}/{name}/splits/temp/{name}_{split}_balanced.jsonl"
    )
    out_path = (
        f"/scratch/data/{benchmark}/{name}/splits/temp/{name}_{split}_formatted.jsonl"
    )

    if os.path.exists(out_path):
        print(
            f"Formatted dataset already exists at {out_path}. Skipping formatting step."
        )
        return

    with open(balanced_path) as f:
        data = [json.loads(line) for line in f if line.strip()]

    def format_sample_back(example):
        prompt = f"{example['prompt']}\n\n"
        all_answers = example["choices"] + [example["answer"]]
        new_positions = np.random.permutation(range(len(all_answers)))
        answers_shuffled = [
            all_answers[i]
            for i, _ in sorted(enumerate(new_positions), key=lambda x: x[1])
        ]
        correct_answer_id = int(new_positions[-1])
        for i, ans in enumerate(answers_shuffled):
            # Format: A) answer text\n
            prompt += f"{chr(65 + i)}) {ans}"
            if i < len(answers_shuffled) - 1:
                prompt += "\n"
        answer = chr(65 + correct_answer_id)

        return {
            "prompt": prompt,
            "answer": answer,
            "idx": example["idx"],
            "lang": example["lang"],
            "subject": example["subject"],
            "n_choices": len(answers_shuffled),
            "additional_choices": example["ignored_choices"],
        }

    with open(out_path, "w") as f:
        for item in tqdm(data):
            formatted = format_sample_back(item)
            f.write(json.dumps(formatted, ensure_ascii=False) + "\n")


def augmentation_report(
    benchmark: str, dataset: str, split: str, model: str, name: str
):
    """
    Generates a report summarizing the augmentation results, such as the distribution of the number of choices per question before and after augmentation.
    """
    augmented_path = (
        f"/scratch/data/{benchmark}/{name}/splits/temp/{name}_{split}_augmented.jsonl"
    )
    original_path = (
        f"/scratch/data/{benchmark}/{dataset}/splits/{dataset}_{split}.jsonl"
    )

    print("Loading augmented dataset...")
    with open(augmented_path) as f:
        augmented_samples = [json.loads(line) for line in f if line.strip()]
    print("Loading original dataset...")
    with open(original_path) as f:
        original_samples = [json.loads(line) for line in f if line.strip()]

    print(f"Number of original samples: {len(original_samples)}")
    print(f"Number of augmented samples: {len(augmented_samples)}")

    augmented_indices = set(s["idx"] for s in augmented_samples)
    missing_samples = [s for s in original_samples if s["idx"] not in augmented_indices]

    # Print number of samples that were not successfully augmented
    print(
        f"Number of samples that were not successfully augmented: {len(original_samples) - len(augmented_samples)}"
    )

    # Print the language distribution of the samples
    original_language_counts = {}
    for s in original_samples:
        lang = s["lang"]
        original_language_counts[lang] = original_language_counts.get(lang, 0) + 1

    language_counts = {}
    for s in missing_samples:
        lang = s["lang"]
        language_counts[lang] = language_counts.get(lang, 0) + 1
    print("Language distribution of missing samples:")
    for lang, count in language_counts.items():
        print(
            f"{lang}: {count} \t(present: {original_language_counts.get(lang, 0)-count}/{original_language_counts.get(lang, 0)})"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "benchmark",
        help="Benchmark name (options: safety, multilingual, knowledge, math)",
    )
    parser.add_argument("dataset", help="Dataset name (examples: safetybench, mmlu)")
    parser.add_argument(
        "split", help="Dataset split to augment (e.g. val, test, default: val)"
    )
    parser.add_argument(
        "model",
        help="Hugging Face model name or path to use for generation (e.g. gpt-4, gpt-3.5-turbo, or a local checkpoint)",
    )
    parser.add_argument(
        "name", help="Name for the new augmented dataset (used for output file)"
    )
    parser.add_argument(
        "--cont",
        action="store_true",
        help="Whether to continue from an existing output file if it exists",
    )
    args = parser.parse_args()

    if args.benchmark not in ["safety", "multilingual", "knowledge", "math"]:
        parser.error(
            f"Invalid benchmark {args.benchmark}. Valid options: safety, multilingual, knowledge, math"
        )

    # augment(args.benchmark, args.dataset, args.split, args.model, args.name, args.cont)
    # augmentation_report(args.benchmark, args.dataset, args.split, args.model, args.name)
    balance_augmented(args.benchmark, args.dataset, args.split, args.model, args.name)
    format_augmented(args.benchmark, args.dataset, args.split, args.model, args.name)
