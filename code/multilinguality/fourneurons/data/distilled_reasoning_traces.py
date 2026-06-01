# Example : python distilled_reasoning_traces.py multilingual mmmlu_more_qcms train Qwen/Qwen3-32B-AWQ mmmlu_more_qcms --cont --n 4

import argparse
from transformers import AutoTokenizer
from datasets import Dataset
from vllm import LLM, SamplingParams
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
from itertools import batched
import pandas as pd

from fourneurons.evaluation.extract_answer import extract_boxed_answer
from fourneurons.prompts.prompt_loader import load_prompt


def generate(
    benchmark: str,
    dataset: str,
    split: str,
    model: str,
    name: str,
    continue_existing: bool,
    n: int,
) -> None:
    """
    Generates Language-Mixed CoT traces (https://arxiv.org/pdf/2510.04230)
    """
    # Output path
    out_path = f"/scratch/data/{benchmark}/{name}/splits/{name}_{split}.jsonl"

    if os.path.exists(out_path) and not continue_existing:
        raise FileExistsError(f"Output dataset already exists: {out_path}.")
    if continue_existing and not os.path.exists(out_path):
        raise FileNotFoundError(
            f"No existing file found at {out_path}. Please provide --cont flag only if you want to continue from an existing file."
        )

    os.makedirs(f"/scratch/data/{benchmark}/{name}/splits/temp/", exist_ok=True)

    # Load model
    teacher = LLM(model=model, tokenizer=model, seed=42)
    tokenizer = AutoTokenizer.from_pretrained(model)
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=4096,
        seed=42,
        repetition_penalty=1,
        n=n,
    )
    system_prompt = load_prompt("/scratch/nathan/repo/fourneurons/prompts/multilingual_cot_teacher.txt")
    print(
        f"Using system prompt:\n------------------\n{system_prompt}\n------------------"
    )

    # Load dataset
    dataset_path = f"/scratch/data/{benchmark}/{dataset}/splits/temp/{dataset}_{split}_formatted.jsonl"
    with open(dataset_path) as f:
        samples = [json.loads(line) for line in f if line.strip()]
        # Stratified downsample
        n_per_lang = 2000
        print(f"Sampling up to {n_per_lang} examples per language...")
        samples = (
            pd.DataFrame(samples)
            .groupby("lang")
            .apply(lambda x: x.sample(n=min(n_per_lang, len(x)), random_state=42))
            .reset_index(drop=True)
            .to_dict(orient="records")
        )
    print(f"Loaded {len(samples)} prompts from {dataset_path}")

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
            f"Continuing from existing file. {len(existing_samples)} samples already processed, {len(remaining_samples)} samples remaining."
        )

    # Generation loop
    loops = 0
    with open(out_path, "w+" if not continue_existing else "a+") as fout:
        while remaining_samples:
            if loops > 3:  # TODO Hardcoded limit
                print(
                    f"Reached maximum number of loops. Stopping. {len(remaining_samples)} samples were not successfully processed."
                )
                break
            loops += 1

            curr_samples = remaining_samples
            remaining_samples = []

            # Generate additional distractors for this group
            prompts = []
            for row in curr_samples:
                prompt = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": row["prompt"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompts.append(prompt)

            for batch in batched(
                zip(curr_samples, prompts, strict=True), 500
            ):  # TODO Hardcoded batch size
                batch_rows, batch_prompts = zip(*batch)
                outputs = teacher.generate(
                    prompts=batch_prompts, sampling_params=sampling_params
                )

                # Format and save
                errors = []
                n_wrong = 0
                for row, output in zip(batch_rows, outputs):
                    row_success = False
                    try:
                        for attempt in output.outputs:
                            completion = attempt.text
                            answer = extract_boxed_answer(completion)
                            if answer == row["answer"]:
                                row["completion"] = completion
                                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                                fout.flush()
                                row_success = True
                                break
                        if not row_success:
                            n_wrong += 1
                            remaining_samples.append(row)
                    except Exception as e:
                        remaining_samples.append(row)
                        errors.append((row["idx"], str(e)))
                print(
                    f"Loop {loops}: Processed {len(batch)} samples, {n_wrong} incorrect, errors: {set(e.__name__ for _, e in errors)}"
                )

    # Create Dataset and push to Hugging Face Hub
    hf_dataset = Dataset.from_json(out_path)
    hf_dataset.push_to_hub(
        f"cs-552-2026-4neurons/{dataset}_4neurons", private=True, split=split
    )

def report(
    benchmark: str,
    dataset: str,
    split: str,
    model: str,
    name: str,
    continue_existing: bool,
) -> None:
    """
    Plots the distribution of languages, subjects, and number of distractors in the generated dataset.

    Produces two files in the splits directory:
    - {name}_{split}_nchoices_by_lang.png : one subplot per language with histogram of n_choices
    - {name}_{split}_subjects_by_lang.png : histogram of subjects colored by language
    """
    data_path = f"/scratch/data/{benchmark}/{name}/splits/{name}_{split}.jsonl"
    out_dir = f"/scratch/results/{benchmark}/{dataset}/{split}/"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Generated dataset not found: {data_path}. Please run the generation step first.")

    # Load dataset
    df = pd.read_json(data_path, lines=True)

    # Ensure lang exists (fallback to first 2 chars of 'idx' if needed)
    if "lang" not in df.columns or df["lang"].isnull().any():
        # derive language as first 2 letters of idx
        df["lang"] = df.get("lang")
        df["lang"] = df["lang"].where(df["lang"].notnull(), df["idx"].astype(str).str[:2])

    # Ensure n_choices exists (fallback to 'choices' length if needed)
    if "n_choices" not in df.columns and "choices" in df.columns:
        df["n_choices"] = df["choices"].apply(lambda x: len(x) if isinstance(x, list) else None)

    os.makedirs(out_dir, exist_ok=True)

    # Plot 1: one subplot per language for n_choices histogram
    langs = sorted(df["lang"].dropna().unique())
    n_cols = 3
    n_rows = (len(langs) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), constrained_layout=True)
    axes = axes.flatten()
    for i, lang in enumerate(langs):
        ax = axes[i]
        data = df[df["lang"] == lang]["n_choices"].dropna()
        sns.histplot(data, bins=range(int(data.min() or 0), int((data.max() or 0) + 2)), ax=ax)
        ax.set_title(f"{lang} (n={len(data)})")
        ax.set_xlabel("n_choices")
    for j in range(len(langs), len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle(f"n_choices distribution per language ({dataset} {split})")
    fig_path = os.path.join(out_dir, f"reasoning_traces_nchoices_by_lang.png")
    fig.savefig(fig_path)
    plt.close(fig)

    # Plot 2: subjects distribution colored by language
    plt.figure(figsize=(10, 6))
    # Count per subject-language
    subj_lang = df.groupby(["subject", "lang"]).size().reset_index(name="count")
    # Pivot for stacked bar plot
    pivot = subj_lang.pivot(index="subject", columns="lang", values="count").fillna(0)
    pivot = pivot.sort_index()
    pivot.plot(kind="bar", stacked=True, figsize=(12, 6))
    plt.ylabel("Number of questions")
    plt.title(f"Questions per subject colored by language ({dataset} {split})")
    plt.legend(title="lang", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig2_path = os.path.join(out_dir, f"reasoning_traces_subjects_by_lang.png")
    plt.tight_layout()
    plt.savefig(fig2_path)
    plt.close()

    # Save statistics as JSON
    questions_per_subject_per_language = {}
    if len(subj_lang) > 0:
        grouped = subj_lang.groupby(["subject", "lang"])["count"].sum()
        for (subject, lang), count in grouped.items():
            questions_per_subject_per_language[f"{subject}::{lang}"] = int(count)
    answers_per_language = {}
    if "n_choices" in df.columns:
        counts = df.groupby(["lang", "n_choices"]).size()
        for (lang, n_choices), count in counts.items():
            answers_per_language.setdefault(lang, {})[int(n_choices)] = int(count)

    stats = {
        "benchmark": benchmark,
        "dataset": dataset,
        "split": split,
        "model": model,
        "continue_existing": continue_existing,
        "questions_per_subject_per_language": questions_per_subject_per_language,
        "answers_per_language": answers_per_language,
    }
    stats_path = os.path.join(out_dir, f"reasoning_traces_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

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
        "run_name", help="Name for this generation run (used for output directory)"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of reasoning traces to generate per question (default: 1)",
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

    if args.n > 8:
        parser.error("n is too large. Generating more than 8 traces per question may lead to excessively long generation times and large output files.")

    generate(
       args.benchmark, args.dataset, args.split, args.model, args.run_name, args.cont, args.n
    )
    report(
        args.benchmark, args.dataset, args.split, args.model, args.run_name, args.cont, 
    )
