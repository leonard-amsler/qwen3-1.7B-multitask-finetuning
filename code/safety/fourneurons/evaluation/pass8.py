import os, re, json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from collections import defaultdict

# MODEL
SNAPSHOT_DIR = "/scratch/hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/"
BASE_MODEL = SNAPSHOT_DIR + os.listdir(SNAPSHOT_DIR)[0]
MERGED_DIR = "/scratch/results/safety/safetybench/lora-final-cot-benchmark-safetybench-think/merged"
N = 8

# DATA
VAL_FILE = "/scratch/data/safety/safetybench/splits/safetybench_train.jsonl"
OUTPUT_DIR = "/scratch/results/safety/safetybench/pass8_weak_categories_train"
TARGET_CATEGORIES = {"Unfairness and Bias", "Offensiveness"}


os.makedirs(OUTPUT_DIR, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR)
llm = LLM(model=MERGED_DIR, seed=42)
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=16384, n=N)

samples = []
with open(VAL_FILE) as f:
    for line in f:
        ex = json.loads(line)
        if ex.get("category") in TARGET_CATEGORIES:
            samples.append(ex)

print(f"Evaluating {len(samples)} samples from {TARGET_CATEGORIES}")

prompts = [
    tokenizer.apply_chat_template(
        [{"role": "user", "content": ex["prompt"]}],
        tokenize=False, add_generation_prompt=True
    )
    for ex in samples
]

outputs = llm.generate(prompts, sampling_params)

def extract_boxed(text):
    m = re.search(r"\\boxed\{([A-Z])\}", text)
    return m.group(1) if m else None

stats = defaultdict(lambda: {"pass1": [], "pass8": []})
gens_rows = []

for ex, output in zip(samples, outputs):
    cat = ex["category"]
    gold = ex["answer"]
    completions = [o.text for o in output.outputs]
    correct_flags = [extract_boxed(c) == gold for c in completions]

    pass1 = int(correct_flags[0])
    pass8 = int(any(correct_flags))

    stats[cat]["pass1"].append(pass1)
    stats[cat]["pass8"].append(pass8)

    gens_rows.append({
        "prompt": ex["prompt"],
        "answer": gold,
        "category": cat,
        "pass1": pass1,
        "pass8": pass8,
        "completions": [
            {"text": c, "extracted": extract_boxed(c), "correct": correct_flags[i]}
            for i, c in enumerate(completions)
        ],
    })

gens_file = os.path.join(OUTPUT_DIR, "val_gens_n8.jsonl")
with open(gens_file, "w") as f:
    for row in gens_rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
print(f"Generations saved to {gens_file}")

summary = {}
print(f"\n{'Category':<25} {'pass@1':>8} {'pass@8':>8} {'Gap':>8} {'N':>5}")
print("-" * 55)
for cat, s in sorted(stats.items()):
    p1 = sum(s["pass1"]) / len(s["pass1"])
    p8 = sum(s["pass8"]) / len(s["pass8"])
    summary[cat] = {"pass@1": round(p1, 4), "pass@8": round(p8, 4), "gap": round(p8 - p1, 4), "n": len(s["pass1"])}
    print(f"{cat:<25} {p1:>8.4f} {p8:>8.4f} {p8-p1:>+8.4f} {len(s['pass1']):>5}")

summary_file = os.path.join(OUTPUT_DIR, "summary.json")
with open(summary_file, "w") as f:
    json.dump(summary, f, indent=2)
print(f"\nSummary saved to {summary_file}")