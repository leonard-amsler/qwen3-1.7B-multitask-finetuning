# This script converts the CompetitionMath dataset into a jsonl format suitable for training.
# Path of the original dataset: https://huggingface.co/datasets/qwedsacf/competition_math
# The output jsonl files will be saved at: /scratch/data/math/competitionmath/splits/

# Initial format of each example in the CompetitionMath dataset:
# {
#   "problem": "What is 123 + 456?",
#   "solution": "To solve 123 + 456, we can add the two numbers together. 123 + 456 = $\boxed{579}$.",
#   "level": "Level 1",
#   "type": "Algebra",
# }

# The output format for each example in the jsonl files will be:
# {
#   "prompt": "What is 123 + 456?",
#   "answer": "579",
#   "completion": "<think>\nTo solve 123 + 456, we can add the two numbers together. 123 + 456 = 579.\n</think>\n\nTherefore, the final answer is \boxed{579}.",
#   "level": "Level 1",
#   "type": "Algebra"
# }

from pathlib import Path
import json
from datasets import load_dataset

OUT_DIR = Path("/scratch/data/math/competitionmath/splits/")
ANSWER_FORMAT = "Therefore, the final answer is \\boxed{{{answer}}}."


def extract_boxed_answer(text):
    '''
    Extract the content of the last \boxed{...} in the text, handling nested braces.
    For example, if the text is "The answer is \\boxed{579}", it will return "579".
    If the text is "The answer is \\boxed{\\frac{1}{2}}", it will return "\\frac{1}{2}".
    '''
    
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None

    pos = start + len(marker)
    depth = 1
    answer_chars = []
    while pos < len(text):
        char = text[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(answer_chars).strip()

        answer_chars.append(char)
        pos += 1

    return None


def format_example(row):
    problem = str(row["problem"]).strip()
    solution = str(row["solution"]).strip()
    level = row.get("level")
    qtype = row.get("type")

    # Extract the final answer from the last \boxed{...}; answers may contain nested braces.
    answer = extract_boxed_answer(solution)
    if not answer:
        return None

    completion = (
        f"<think>\n{solution}\n</think>\n\n"
        + ANSWER_FORMAT.format(answer=answer)
    )

    return {
        "prompt": problem,
        "answer": answer,
        "completion": completion,
        "level": level,
        "type": qtype,
    }


def write_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            if row is not None:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "qwedsacf/competition_math",
        split="train",
        cache_dir="/scratch/hf_cache",
    )

    ds = ds.shuffle(seed=42)

    # Split ratios: 98% train, 1% val, 1% test.
    n = len(ds)
    n_val = int(0.01 * n)
    n_test = int(0.01 * n)

    test_ds = ds.select(range(n_test))
    val_ds = ds.select(range(n_test, n_test + n_val))
    train_ds = ds.select(range(n_test + n_val, n))

    write_jsonl((format_example(x) for x in train_ds), OUT_DIR / "competitionmath_train.jsonl")
    write_jsonl((format_example(x) for x in val_ds), OUT_DIR / "competitionmath_val.jsonl")
    write_jsonl((format_example(x) for x in test_ds), OUT_DIR / "competitionmath_test.jsonl")
    write_jsonl((format_example(x) for x in ds), OUT_DIR / "competitionmath_full.jsonl")  # Save the full dataset as well for evaluation


if __name__ == "__main__":
    main()