import json
from datasets import load_dataset

ANSWERS_PATH = "/scratch/hf_cache/datasets/SafetyBench/test_answers_en.json"
TEST_PATH    = "/scratch/hf_cache/datasets/SafetyBench/test_en.json"

ANSWER_MAP = {i: chr(ord("A") + i) for i in range(20)}  # {0: "A", 1: "B", ...}


def _format_options(options):
    return "\n".join(f"{chr(ord('A') + i)}) {opt}" for i, opt in enumerate(options))


def _format_prompt(question, options):
    return f"{question}\n\n{_format_options(options)}"


def load_safetybench_test():
    """Load the full test split (11,435 examples) from local JSON files."""
    with open(TEST_PATH) as f:
        questions = json.load(f)
    with open(ANSWERS_PATH) as f:
        answers = json.load(f)

    return [
        {
            "prompt": _format_prompt(item["question"], item["options"]),
            "answer": ANSWER_MAP[answers[str(item["id"])]["answer"]],
            "category": item["category"],
        }
        for item in questions
    ]


def load_safetybench_dev():
    """Load the small dev split (35 labeled examples) from HuggingFace."""
    ds = load_dataset("thu-coai/SafetyBench", "dev")["en"][0]
    return [
        {
            "prompt": _format_prompt(item["question"], item["options"]),
            "answer": ANSWER_MAP[item["answer"]],
            "category": category,
        }
        for category, items in ds.items()
        for item in items
    ]


if __name__ == "__main__":
    test = load_safetybench_test()
    dev  = load_safetybench_dev()
    print(f"Test: {len(test)} examples")
    print(f"Dev:  {len(dev)} examples")
    print(f"\nSample:\n  prompt: {test[0]['prompt']}")
    print(f"  answer: {test[0]['answer']}")
    print(f"  category: {test[0]['category']}")