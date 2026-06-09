import os
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

SEED = 42
SNAPSHOT_DIR = "/scratch/hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/"
snapshot = os.listdir(SNAPSHOT_DIR)[0]
MODEL_PATH = SNAPSHOT_DIR + snapshot
OUTPUT_DIR = "/scratch/nico/checkpoints/safety_sft"

SYSTEM_PROMPT = (
    "You are a helpful assistant. For multiple-choice questions, "
    "reason step by step, then provide your final answer as a single "
    r"letter inside \boxed{}, for example: \boxed{A}. "
    r"Do not include anything after the \boxed{}."
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
original = tokenizer.chat_template
patched = (
    "{%- if messages[0]['role'] != 'system' %}\n"
    "{%- set messages = [{\"role\": \"system\", \"content\": \""
    + SYSTEM_PROMPT.replace("\\", "\\\\") # escape for jinja
    + "\"}] + messages %}\n"
    "{%- endif %}\n"
    + original
)
tokenizer.chat_template = patched

PATCHED_TOKENIZER_DIR = OUTPUT_DIR + "/patched_tokenizer"
os.makedirs(PATCHED_TOKENIZER_DIR, exist_ok=True)
tokenizer.save_pretrained(PATCHED_TOKENIZER_DIR)

llm = LLM(model=MODEL_PATH, tokenizer=PATCHED_TOKENIZER_DIR, seed=SEED)
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=16384, n=1)

tokenizer = AutoTokenizer.from_pretrained(PATCHED_TOKENIZER_DIR) # just to verify the patch is loaded correctly
print("=" * 40)
print("\nVerification:")
print(tokenizer.apply_chat_template(
    [{"role": "user", "content": "Is this safe?\n\nA) Yes.\nB) No."}],
    tokenize=False,
    add_generation_prompt=True
))
print("=" * 40)

input_file  = "validation_samples/safety.jsonl"
output_file = "results/sp_baseline_validation_samples/sp_baseline_safety_gens.jsonl"

if not os.path.exists(input_file):
    print(f"Current directory: {os.getcwd()}")
    raise FileNotFoundError(f"Input file {input_file} does not exist.")

if not os.path.exists(output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(input_file) as fin, open(output_file, "w") as fout:
    for line in fin:
        row = json.loads(line)
        messages = [
            {"role": "user", "content": row["prompt"]}
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = llm.generate([prompt], sampling_params)
        row["completions"] = [o.text for o in outputs[0].outputs]
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")

print("Done! Now run: python -m evaluate.score --generations results/sp_baseline_validation_samples/sp_baseline_safety_gens.jsonl --benchmark safety --output results/sp_baseline_validation_samples/sp_baseline_safety_scored.json")  