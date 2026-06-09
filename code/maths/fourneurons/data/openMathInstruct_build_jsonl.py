# This scripts converts the OpenMathInstruct dataset into a jsonl format suitable for training.
# Path of the original dataset: scratch/hf_cache/hub/datasets--nvidia--OpenMathInstruct-2
# The output jsonl files will be saved at: scratch/data/math/openmathinstruct/splits/ under train.jsonl, val.jsonl, and test.jsonl

# Initial format of each example in the OpenMathInstruct dataset:
# {
#   "problem": "What is 123 + 456?",
#   "generated_solution": "To solve 123 + 456, we can add the two numbers together. 123 + 456 = 579.",
#   "expected_answer": "579",
#   "problem_source": "augmented_gsm8k"
# }

# The output format for each example in the jsonl files will be:
# {
#   "prompt": "What is 123 + 456?",
#   "answer": "579",
#   "completion": "<think>\nTo solve 123 + 456, we can add the two numbers together. 123 + 456 = 579.\n</think>\n\nTherefore, the final answer is \boxed{579}.",
#   "problem_source": "augmented_gsm8k"
# }

from pathlib import Path
import json
from datasets import load_dataset

OUT_DIR = Path("/scratch/data/math/openmathinstruct/splits/")
ANSWER_FORMAT = "Therefore, the final answer is \\boxed{{{answer}}}."

def format_example(row):
    problem = str(row["problem"]).strip()
    solution = str(row["generated_solution"]).strip()
    answer = str(row["expected_answer"]).strip()

    completion = (
        f"<think>\n{solution}\n</think>\n\n"
        + ANSWER_FORMAT.format(answer=answer)
    )

    return {
        "prompt": problem,
        "answer": answer,
        "completion": completion,
        "problem_source": row.get("problem_source"),
    }


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "nvidia/OpenMathInstruct-2",
        split="train_1M",
        cache_dir="/scratch/hf_cache",
    )

    ds = ds.shuffle(seed=42)

    # Example split ratios: 98% train, 1% val, 1% test
    # 980k train, 10k val, and 10k test samples for the 1M subset
    n = len(ds)
    n_val = int(0.01 * n)
    n_test = int(0.01 * n)

    test_ds = ds.select(range(n_test))
    val_ds = ds.select(range(n_test, n_test + n_val))
    train_ds = ds.select(range(n_test + n_val, n))

    write_jsonl((format_example(x) for x in train_ds), OUT_DIR / "openmathinstruct_train.jsonl")
    write_jsonl((format_example(x) for x in val_ds), OUT_DIR / "openmathinstruct_val.jsonl")
    write_jsonl((format_example(x) for x in test_ds), OUT_DIR / "openmathinstruct_test.jsonl")


if __name__ == "__main__":
    main()