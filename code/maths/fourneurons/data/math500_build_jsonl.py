# This script converts the math500 dataset into a jsonl format suitable for training.
# Path of the original dataset: https://huggingface.co/datasets/HuggingFaceH4/MATH-500
# The output jsonl files will be saved at: /scratch/data/math/math500/splits/

# Initial format of each example in the MATH-500 dataset:
# {
#   "problem": "What is 123 + 456?",
#   "solution": "To solve 123 + 456, we can add the two numbers together. 123 + 456 = $\boxed{579}$.",
#   "answer": "579",
#   "subject": "Algebra",
#   "level": 2,
#   "unique_id": "test/algebra/123456.json"
# }

# The output format for each example in the jsonl files will be:
# {
#   "prompt": "What is 123 + 456?",
#   "answer": "579",
#   "completion": "<think>\nTo solve 123 + 456, we can add the two numbers together. 123 + 456 = 579.\n</think>\n\nTherefore, the final answer is \boxed{579}.",
#   "level": "2",
#   "type": "Algebra"
# }

from pathlib import Path
import json
from datasets import load_dataset

OUT_DIR = Path("/scratch/data/math/math500/splits/")
ANSWER_FORMAT = "Therefore, the final answer is \\boxed{{{answer}}}."

def format_example(row):
    problem = str(row["problem"]).strip()
    solution = str(row["solution"]).strip()
    answer = str(row["answer"]).strip()
    subject = str(row["subject"]).strip()
    level = str(row["level"]).strip()

    completion = (
        f"<think>\n{solution}\n</think>\n\n"
        + ANSWER_FORMAT.format(answer=answer)
    )

    return {
        "prompt": problem,
        "answer": answer,
        "completion": completion,
        "level": level,
        "type": subject,
    }


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "HuggingFaceH4/MATH-500",
        split="test",
        cache_dir="/scratch/hf_cache",
    )

    ds = ds.shuffle(seed=42)

    # Write the full dataset to a single jsonl file for evaluation purposes
    write_jsonl((format_example(x) for x in ds), OUT_DIR / "math500_full.jsonl")  # Save the full dataset as well for evaluation


if __name__ == "__main__":
    main()
