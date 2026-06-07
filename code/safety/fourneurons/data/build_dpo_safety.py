import json
import argparse
import os
from collections import Counter
from transformers import AutoTokenizer


MERGED_SFT_DIR = "/scratch/results/safety/safetybench/lora-final-cot-benchmark-safetybench-think/merged"

PASS8_RESULTS = {
    "weak": "/scratch/results/safety/safetybench/pass8_weak_categories_train/val_gens_n8.jsonl",
    "all": "/scratch/results/safety/safetybench/pass8_all_categories_train/val_gens_n8.jsonl",
}

OUTPUT_FILES = {
    "weak": "/scratch/data/safety/safetybench/dpo/train_dpo_weak_categories.jsonl",
    "all": "/scratch/data/safety/safetybench/dpo/train_dpo_all_categories.jsonl",
}

DPO_MAX_LENGTH = 6144
LOOP_CHAR_THRESHOLD = 8000  # verified: no-box loops are >10k, genuine no-box are <2k


def check_lengths(pairs, tokenizer, max_length):
    print("\nChecking token lengths...")
    chosen_lengths = []
    rejected_lengths = []
    truncated = []

    for p in pairs:
        len_chosen = len(tokenizer(p["prompt"] + p["chosen"], add_special_tokens=False)["input_ids"])
        len_rejected = len(tokenizer(p["prompt"] + p["rejected"], add_special_tokens=False)["input_ids"])
        chosen_lengths.append(len_chosen)
        rejected_lengths.append(len_rejected)
        if len_chosen > max_length or len_rejected > max_length:
            truncated.append({
                "prompt": p["prompt"][:80],
                "len_chosen": len_chosen,
                "len_rejected": len_rejected,
            })

    all_lengths = sorted(chosen_lengths + rejected_lengths)
    n = len(all_lengths)
    print(f"  max_length setting : {max_length}")
    print(f"  min    : {all_lengths[0]}")
    print(f"  p50    : {all_lengths[n//2]}")
    print(f"  p90    : {all_lengths[int(n*0.90)]}")
    print(f"  p95    : {all_lengths[int(n*0.95)]}")
    print(f"  p99    : {all_lengths[int(n*0.99)]}")
    print(f"  max    : {all_lengths[-1]}")
    print(f"  pairs that would be truncated: {len(truncated)} / {len(pairs)}")

    if truncated:
        print("\n  Examples of truncated pairs:")
        for t in truncated[:3]:
            print(f"    prompt: {t['prompt']}...")
            print(f"    len_chosen={t['len_chosen']}  len_rejected={t['len_rejected']}")


def build_dpo_pairs(merged_sft_dir, all_categories=False):
    """
    Build DPO pairs from pass@8 results on the train split.

    Chosen: longest correct completion.
    Rejected: wrong completion with boxed answer (preferred),
    or short no-box completion (forgot \\boxed{}, not a loop).
    """
    key = "all" if all_categories else "weak"
    json_file = PASS8_RESULTS[key]
    output_file = OUTPUT_FILES[key]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    dpo_pairs = []
    with open(json_file) as f:
        for line in f:
            row = json.loads(line)
            correct = [c for c in row["completions"] if c["correct"]]
            wrong_with_box = [c for c in row["completions"] if not c["correct"] and c["extracted"] is not None]
            wrong_no_box = [
                c for c in row["completions"]
                if not c["correct"]
                and c["extracted"] is None
                and len(c["text"]) < LOOP_CHAR_THRESHOLD  # exclude loops, not pertinent for dpo
            ]

            if not correct:
                continue

            if wrong_with_box:
                rejected = wrong_with_box[0]["text"]
            elif wrong_no_box:
                rejected = wrong_no_box[0]["text"]
            else:
                continue

            dpo_pairs.append({
                "prompt": row["prompt"],
                "answer": row["answer"],
                "category": row["category"],
                "chosen": max(correct, key=lambda c: len(c["text"]))["text"],
                "rejected": rejected,
            })

    cat_counts = Counter(p["category"] for p in dpo_pairs)
    print(f"Categories: {'all' if all_categories else 'weak only'}")
    print(f"Total DPO pairs: {len(dpo_pairs)}")
    for cat, n in sorted(cat_counts.items(), key=lambda x: x[1]):
        print(f"  {cat:<35} {n}")

    tokenizer = AutoTokenizer.from_pretrained(merged_sft_dir)
    check_lengths(dpo_pairs, tokenizer, max_length=DPO_MAX_LENGTH)

    with open(output_file, "w") as f:
        for p in dpo_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nSaved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--merged_sft_dir",
        type=str,
        required=True,
        help="Directory containing the merged SFT model and tokenizer.",
    )
    parser.add_argument(
        "--all_categories",
        action="store_true",
        help="Use all categories. Default: only weak categories (Unfairness and Bias, Offensiveness).",
    )
    args = parser.parse_args()
    build_dpo_pairs(merged_sft_dir=args.merged_sft_dir, all_categories=args.all_categories)