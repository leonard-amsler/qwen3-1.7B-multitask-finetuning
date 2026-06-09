import os
import json
import wandb
import random
import torch
from datetime import datetime
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from fourneurons.data.format_for_sft import format_for_sft

for env_name in list(os.environ):
    if env_name.startswith("WANDB_") and os.environ[env_name] == "":
        del os.environ[env_name]




# Set up Weights & Biases for experiment tracking:
RUN_NAME = os.getenv("FOURNEURONS_RUN_NAME") or "qwen3-1.7b-lora-math-mixed-16k"
run_id = os.getenv("FOURNEURONS_RUN_ID") or RUN_NAME + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
WANDB_PROJECT = os.getenv("WANDB_PROJECT") or "math-sft"
WANDB_NAME = os.getenv("WANDB_NAME") or run_id
WANDB_RUN_ID = os.getenv("WANDB_RUN_ID") or None
WANDB_RESUME = os.getenv("WANDB_RESUME") or ("allow" if WANDB_RUN_ID else None)

SNAPSHOT_DIR = "/scratch/hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/"
MODEL_PATH = SNAPSHOT_DIR + os.listdir(SNAPSHOT_DIR)[0]

SPLITS_DIR_1 = "/scratch/data/math/openmathinstruct/splits/"
SPLITS_DIR_2 = "/scratch/data/math/openR1math/splits/"

OUTPUT_DIR = os.getenv("OUTPUT_DIR") or f"/scratch/checkpoints/math/{run_id}"
RESUME_FROM_CHECKPOINT = os.getenv("RESUME_FROM_CHECKPOINT") or None
PROMPT_FILE = "fourneurons/prompts/math.txt"


def resolve_resume_checkpoint(checkpoint):
    if checkpoint is None:
        return None

    if checkpoint.lower() in {"latest", "last", "auto"}:
        if not os.path.isdir(OUTPUT_DIR):
            raise FileNotFoundError(f"Cannot resolve latest checkpoint: {OUTPUT_DIR} does not exist")

        candidates = []
        for name in os.listdir(OUTPUT_DIR):
            path = os.path.join(OUTPUT_DIR, name)
            if not os.path.isdir(path) or not name.startswith("checkpoint-"):
                continue
            try:
                step = int(name.rsplit("-", 1)[1])
            except ValueError:
                continue
            candidates.append((step, path))

        if not candidates:
            raise FileNotFoundError(f"No checkpoint-* directories found under {OUTPUT_DIR}")

        checkpoint = max(candidates)[1]

    required_files = ["trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth"]
    missing_files = [name for name in required_files if not os.path.isfile(os.path.join(checkpoint, name))]
    if missing_files:
        raise FileNotFoundError(
            f"Checkpoint {checkpoint} is missing required resume state files: {', '.join(missing_files)}"
        )

    return checkpoint


RESUME_FROM_CHECKPOINT = resolve_resume_checkpoint(RESUME_FROM_CHECKPOINT)

wandb.login(key=os.getenv("WANDB_KEY"))
wandb.init(
    project=WANDB_PROJECT,
    name=WANDB_NAME,
    id=WANDB_RUN_ID,
    resume=WANDB_RESUME,
)

print(f"Run name: {run_id}")
print(f"Output directory: {OUTPUT_DIR}")
if RESUME_FROM_CHECKPOINT:
    print(f"Resuming from checkpoint: {RESUME_FROM_CHECKPOINT}")
if WANDB_RUN_ID:
    print(f"Resuming W&B run id: {WANDB_RUN_ID} with resume={WANDB_RESUME}")

N_EPOCHS = 4
MAX_TRAINING_SAMPLES = 50000
MAX_VALIDATION_SAMPLES = 256

# Data loading:
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
MAX_CONTEXT_LENGTH = 16384


def collect_formatted_examples(filename, splits_dir, target_count, *, max_length=MAX_CONTEXT_LENGTH):
    """Collect target_count examples after formatting and token-length filtering."""
    path = os.path.join(splits_dir, filename)
    kept = []
    scanned = 0
    dropped_overlong = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            scanned += 1
            sample = json.loads(line)
            formatted = format_for_sft(sample, tokenizer, prompt_file_path=PROMPT_FILE)
            token_count = len(tokenizer(formatted, add_special_tokens=False)["input_ids"])

            if token_count <= max_length:
                kept.append(formatted)
                if len(kept) >= target_count:
                    break
            else:
                dropped_overlong += 1

    if len(kept) < target_count:
        print(
            f"WARNING: only collected {len(kept)}/{target_count} valid examples from {path}. "
            f"Scanned={scanned}, dropped_overlong={dropped_overlong}."
        )
    else:
        print(
            f"Collected {len(kept)} valid examples from {path}. "
            f"Scanned={scanned}, dropped_overlong={dropped_overlong}."
        )

    return kept, dropped_overlong, scanned


train_target_per_dataset = MAX_TRAINING_SAMPLES // 2
val_target_per_dataset = MAX_VALIDATION_SAMPLES // 2

print(f"Formatting/filtering OpenMathInstruct train data from {SPLITS_DIR_1}...")
train_texts_1, dropped_train_1, scanned_train_1 = collect_formatted_examples(
    "openmathinstruct_train.jsonl", SPLITS_DIR_1, train_target_per_dataset
)
print(f"Formatting/filtering OpenMathInstruct validation data from {SPLITS_DIR_1}...")
validation_texts_1, dropped_val_1, scanned_val_1 = collect_formatted_examples(
    "openmathinstruct_val.jsonl", SPLITS_DIR_1, val_target_per_dataset
)

print(f"Formatting/filtering OpenR1Math train data from {SPLITS_DIR_2}...")
train_texts_2, dropped_train_2, scanned_train_2 = collect_formatted_examples(
    "openR1math_train.jsonl", SPLITS_DIR_2, train_target_per_dataset
)
print(f"Formatting/filtering OpenR1Math validation data from {SPLITS_DIR_2}...")
validation_texts_2, dropped_val_2, scanned_val_2 = collect_formatted_examples(
    "openR1math_val.jsonl", SPLITS_DIR_2, val_target_per_dataset
)

train_texts = train_texts_1 + train_texts_2
validation_texts = validation_texts_1 + validation_texts_2

random.seed(42)
random.shuffle(train_texts)
random.shuffle(validation_texts)

print(
    "Dropped overlong examples before applying sample caps: "
    f"openmath train={dropped_train_1}/{scanned_train_1}, "
    f"openmath val={dropped_val_1}/{scanned_val_1}, "
    f"openr1 train={dropped_train_2}/{scanned_train_2}, "
    f"openr1 val={dropped_val_2}/{scanned_val_2}"
)

print(f"Building train dataset from {len(train_texts)} formatted examples...")
train_dataset = Dataset.from_dict({"text": train_texts})

print(f"Building validation dataset from {len(validation_texts)} formatted examples...")
validation_dataset = Dataset.from_dict({"text": validation_texts})

print(f"\nTraining on {len(train_dataset)} samples")
print(f"Validating on {len(validation_dataset)} samples")

print(f"\nSample:\n{train_dataset[0]['text']} ... \n")

# LoRA training:
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules="all-linear",
    lora_dropout=0.05,
    bias="none",
)

model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=N_EPOCHS,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,     # effective batch size = 16
    learning_rate=2e-4,                 # default for LoRA fine-tuning
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",              # saves checkpoint after each epoch                  
    save_total_limit=10,                 # keep only the 10 most recent checkpoints
    max_length=MAX_CONTEXT_LENGTH,       # truncate inputs to fit model context window
    gradient_checkpointing=True,
    dataset_text_field="text",
    report_to="wandb",
    run_name=WANDB_NAME,
    eval_strategy="steps",
    eval_steps=500,
    per_device_eval_batch_size=1,
)

def disable_bitsandbytes_lora_dispatch():
    import peft.tuners.lora.model as peft_lora_model

    peft_lora_model.is_bnb_available = lambda: False
    peft_lora_model.is_bnb_4bit_available = lambda: False

disable_bitsandbytes_lora_dispatch()

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    peft_config=lora_config,
    args=training_args,
)

trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)
trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
print(f"\nDone. Checkpoints saved to {OUTPUT_DIR}")