import os
import json
import argparse
from datetime import datetime

import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer
from fourneurons.data.format_for_sft import format_for_sft

SNAPSHOT_DIR = "/scratch/hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/"
MODEL_PATH = SNAPSHOT_DIR + os.listdir(SNAPSHOT_DIR)[0]


def load_split(data_dir, filename):
    """Load a JSONL split file."""
    examples = []
    split_path = os.path.join(data_dir, filename)
    with open(split_path, "r") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def analyze_token_distribution(data_dir, prompt_file, split_name):
    """
    Analyze token distribution for a dataset.
    
    Args:
        data_dir: Directory containing the split files
        prompt_file: Path to the prompt template file
        split_name: Name of the split file to analyze
    """
    print(f"\nLoading model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    print(f"\nLoading data from {data_dir}/{split_name}...")
    train_raw = load_split(data_dir, split_name)
    texts = [format_for_sft(ex, tokenizer, prompt_file_path=prompt_file) for ex in train_raw]
    dataset = Dataset.from_dict({"text": texts})

    print(f"\nTraining data contains {len(dataset)} samples")
    print(f"\nSample:\n{dataset[0]['text'][:300]}... \n")

    lengths = []
    for text in dataset["text"]:
        enc = tokenizer(
            text,
            add_special_tokens=True,
            return_attention_mask=False,
            return_token_type_ids=False,
            truncation=False,
        )
        lengths.append(len(enc["input_ids"]))

    lengths = np.array(lengths)

    print(f"\nToken length statistics")
    print(f"Min:   {lengths.min()}")
    print(f"P50:   {int(np.percentile(lengths, 50))}")
    print(f"P90:   {int(np.percentile(lengths, 90))}")
    print(f"P95:   {int(np.percentile(lengths, 95))}")
    print(f"P99:   {int(np.percentile(lengths, 99))}")
    print(f"Max:   {lengths.max()}")
    print(f"Mean:  {lengths.mean():.1f}")
    print(f"Std:   {lengths.std():.1f}\n")

    for cutoff in [512, 1024, 1536, 2048, 3072, 4096]:
        truncated = (lengths > cutoff).sum()
        frac = truncated / len(lengths) * 100
        print(f"max_length={cutoff:4d} -> truncates {truncated:5d}/{len(lengths)} samples ({frac:.2f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze token distribution for a dataset"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/scratch/data/safety/safetybench/cot",
        help="Directory containing the split files (default: /scratch/data/safety/safetybench/cot)",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="/scratch/nico/standard-project-m2-4neurons/prompts/sp_general_qcm_think.txt",
        help="Path to the prompt template file to use for formatting (default: /scratch/nico/standard-project-m2-4neurons/prompts/sp_general_qcm_think.txt)",
    )
    parser.add_argument(
        "--split-name",
        type=str,
        default="safetybench_train_cot.jsonl",
        help="Name of the split file to analyze (default: safetybench_train_cot.jsonl)",
    )
    
    args = parser.parse_args()
    
    analyze_token_distribution(
        data_dir=args.data_dir,
        prompt_file=args.prompt_file,
        split_name=args.split_name,
    )


if __name__ == "__main__":
    main()