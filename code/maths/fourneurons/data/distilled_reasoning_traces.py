# Example : python distilled_reasoning_traces.py multilingual mmmlu_more_qcms train Qwen/Qwen3-32B-AWQ mmmlu_more_qcms

import argparse
from transformers import AutoTokenizer
from datasets import Dataset
from vllm import LLM, SamplingParams
import json
import os
from itertools import batched
import pandas as pd

from fourneurons.evaluation.extract_answer import extract_boxed_answer


def generate(
    benchmark: str,
    dataset: str,
    split: str,
    model: str,
    name: str,
    continue_existing: bool,
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
    )
    system_prompt = open("../prompts/multilingual_cot_teacher.txt").read()
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
                    completion = output.outputs[0].text
                    try:
                        answer = extract_boxed_answer(completion)
                        if answer == row["answer"]:
                            row["completion"] = completion
                            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                        else:
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
        "--cont",
        action="store_true",
        help="Whether to continue from an existing output file if it exists",
    )
    args = parser.parse_args()

    if args.benchmark not in ["safety", "multilingual", "knowledge", "math"]:
        parser.error(
            f"Invalid benchmark {args.benchmark}. Valid options: safety, multilingual, knowledge, math"
        )

    generate(
        args.benchmark, args.dataset, args.split, args.model, args.run_name, args.cont
    )
