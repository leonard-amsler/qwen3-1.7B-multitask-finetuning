import os
import re
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from prompts.prompt_loader import load_prompt


MODEL_PATH = "/scratch/hf_cache/hub/models--Qwen--Qwen3-32B-AWQ/snapshots/"
MODEL_PATH = MODEL_PATH + os.listdir(MODEL_PATH)[0]

INPUT_FILE = "/scratch/data/safety/safetybench/splits/safetybench_train.jsonl"
OUTPUT_FILE = "/scratch/data/safety/safetybench/cot/safetybench_train_cot.jsonl"
SYSTEM_PROMPT_FILE = "/scratch/nico/standard-project-m2-4neurons/prompts/sp_general_qcm_think.txt"

MAX_ATTEMPTS = 8
TEMPERATURE = 0.7
TOP_P = 0.9
MAX_TOKENS = 4096
CHUNK_SIZE = 500


def extract_boxed(text):
    match = re.search(r"\\boxed\{([A-Z])\}", text)
    return match.group(1) if match else None


def build_prompt(sample, tokenizer, prompt_file=None):
    if prompt_file is not None:
        system_prompt = load_prompt(prompt_file)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample["prompt"]},
        ]
    else:
        messages = [
            {"role": "user", "content": sample["prompt"]},
        ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def iter_chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i:i + size]


print(f"Loading tokenizer from {MODEL_PATH}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print(f"Loading vLLM model from {MODEL_PATH}...")
llm = LLM(
    model=MODEL_PATH,
    quantization="awq",
    seed=42,
    max_model_len=8192,
)

sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    top_p=TOP_P,
    max_tokens=MAX_TOKENS,
    n=1,
    seed=42,
)

samples = []
with open(INPUT_FILE) as f:
    for line in f:
        samples.append(json.loads(line))

print(f"Loaded {len(samples)} samples from {INPUT_FILE}")

done_prompts = set()
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE) as f:
        for line in f:
            row = json.loads(line)
            done_prompts.add(row["prompt"])
    print(f"Resuming: found {len(done_prompts)} already processed prompts in {OUTPUT_FILE}")

pending = [s for s in samples if s["prompt"] not in done_prompts]
print(f"{len(pending)} prompts remaining to process")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

kept = 0
discarded_final = 0

with open(OUTPUT_FILE, "a", encoding="utf-8") as fout:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if not pending:
            print("\nAll prompts processed successfully.")
            break

        print(f"\nAttempt {attempt}/{MAX_ATTEMPTS} — generating for {len(pending)} prompts")
        next_pending = []
        newly_kept = 0

        for chunk_idx, chunk in enumerate(iter_chunks(pending, CHUNK_SIZE), start=1):
            prompts = [build_prompt(sample, tokenizer, SYSTEM_PROMPT_FILE) for sample in chunk]
            # prompts = [build_prompt(sample, tokenizer, SYSTEM_PROMPT_FILE) for sample in chunk]
            outputs = llm.generate(prompts, sampling_params)

            for sample, output in zip(chunk, outputs):
                text = output.outputs[0].text
                pred = extract_boxed(text)

                if pred == sample["answer"]:
                    row = {
                        "prompt": sample["prompt"],
                        "answer": sample["answer"],
                        "category": sample["category"],
                        "completion": text,
                    }
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    newly_kept += 1
                    kept += 1
                else:
                    next_pending.append(sample)

            fout.flush()
            os.fsync(fout.fileno())

            print(
                f"  chunk {chunk_idx}: processed {len(chunk)}, "
                f"attempt_kept {newly_kept}, still_pending_so_far {len(next_pending)}"
            )

        print(
            f"Attempt {attempt}: kept {newly_kept}, "
            f"still pending {len(next_pending)}, "
            f"cumulative kept {kept}"
        )

        pending = next_pending

    discarded_final = len(pending)

print("\nDone.")
print(f"Kept prompts: {kept}")
print(f"Discarded prompts: {discarded_final}")
if kept + discarded_final > 0:
    print(f"Retention rate: {100 * kept / (kept + discarded_final):.2f}%")
print(f"Output written to: {OUTPUT_FILE}")