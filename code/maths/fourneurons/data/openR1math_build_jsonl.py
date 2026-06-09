# This scripts converts the OpenMathInstruct dataset into a jsonl format suitable for training.
# Path of the original dataset: open-r1/OpenR1-Math-220k
# The output jsonl files will be saved at: scratch/data/math/openR1math/splits/

# Initial format of each example in the OpenMathInstruct dataset:
# {
#   "problem": "What is 123 + 456?",
#   "answer": "579",
#   "problem_type": "augmented_gsm8k"
#   "question_type": "math-word-problem"
#   "source": "olympiads"
#   "generations": [list of generated solutions] # Already contain the <think>...</think> tags
#   "is_reasoning_complete": [list of booleans indicating whether each generated solution is reasoning-complete]
#   "correctness_math_verify": [list of correctness labels for each generated solution]
# }

# The output format for each example in the jsonl files will be:
# {
    # "prompt": "What is 123 + 456?",
    # "answer": "579",
    # "completion": "<think>\nTo solve 123 + 456, we can add the two numbers together. 123 + 456 = 579.\n</think>\n\nTherefore, the final answer is \boxed{579}.",
    # "problem_type": "augmented_gsm8k",
    # "question_type": "math-word-problem",
    # "source": "olympiads",
    # "uuid": "some-unique-id"
# }

from pathlib import Path
import json
from datasets import load_dataset

DATASET_NAME = "openR1math"
OUT_DIR = Path(f"/scratch/data/math/{DATASET_NAME}/splits/")
ANSWER_FORMAT = "Therefore, the final answer is \\boxed{{{answer}}}."

def format_example(row):
    problem = str(row["problem"]).strip()
    answer = str(row["answer"]).strip()
    generations = row.get("generations", [])
    reasoning_completeness = row.get("is_reasoning_complete", [])
    correctness_labels = row.get("correctness_math_verify", [])

    # Find the shortest generation that is both reasoning-complete and correct, if any
    valid_solutions = [
        gen for gen, is_complete, is_correct in zip(generations, reasoning_completeness, correctness_labels)
        if is_complete and is_correct
    ]
    if valid_solutions:
        solution = str(min(valid_solutions, key=len)).strip()
    else:
        return None  # Skip examples that don't have a valid solution

    # Format the completion with the solution and the final answer
    completion = (
        solution + "\n\n" + ANSWER_FORMAT.format(answer=answer)
    )

    return {
        "prompt": row["problem"],
        "answer": row["answer"],
        "completion": completion,
        "problem_type": row["problem_type"],
        "question_type": row["question_type"],
        "source": row["source"],
        "uuid": row["uuid"]
    }


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            if row is not None:  # Skip rows that were filtered out
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("open-r1/OpenR1-Math-220k", "default", split="train")

    ds = ds.shuffle(seed=42)

    # Example split ratios: 98% train, 1% val, 1% test
    # 980k train, 10k val, and 10k test samples for the 1M subset
    n = len(ds)
    n_val = int(0.01 * n)
    n_test = int(0.01 * n)

    test_ds = ds.select(range(n_test))
    val_ds = ds.select(range(n_test, n_test + n_val))
    train_ds = ds.select(range(n_test + n_val, n))

    write_jsonl((format_example(x) for x in train_ds), OUT_DIR / f"{DATASET_NAME}_train.jsonl")
    write_jsonl((format_example(x) for x in val_ds), OUT_DIR / f"{DATASET_NAME}_val.jsonl")
    write_jsonl((format_example(x) for x in test_ds), OUT_DIR / f"{DATASET_NAME}_test.jsonl")


if __name__ == "__main__":
    main()