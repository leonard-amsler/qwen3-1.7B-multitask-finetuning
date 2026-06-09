import os
import json
import wandb
import torch
from datetime import datetime
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from fourneurons.data.format_for_sft import format_for_sft

# Set up Weights & Biases for experiment tracking:
wandb.login(key=os.getenv("WANDB_KEY"))
wandb.init(project="math-sft", name="qwen3-1.7b-lora")

SNAPSHOT_DIR = "/scratch/hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/"
MODEL_PATH = SNAPSHOT_DIR + os.listdir(SNAPSHOT_DIR)[0]
SPLITS_DIR = "/scratch/data/math/openmathinstruct/splits/"
run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
OUTPUT_DIR = f"/scratch/checkpoints/math/{run_id}"
PROMPT_FILE = "fourneurons/prompts/math.txt"

MAX_TRAINING_SAMPLES = 250000
MAX_VALIDATION_SAMPLES = 256

# Data loading:
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def load_split(filename):
    examples = []
    with open(os.path.join(SPLITS_DIR, filename)) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples

print(f"Loading training data from {SPLITS_DIR}...")
train_raw = load_split("openmathinstruct_train.jsonl")
val_raw = load_split("openmathinstruct_val.jsonl")

# Keep only a subset of the data for this example:
train_raw = train_raw[:MAX_TRAINING_SAMPLES]
val_raw = val_raw[:MAX_VALIDATION_SAMPLES]

print(f"Formatting {len(train_raw)} training examples for SFT...")
train_texts = [format_for_sft(ex, tokenizer, prompt_file_path=PROMPT_FILE) for ex in train_raw]
train_dataset = Dataset.from_dict({"text": train_texts})

print(f"Formatting {len(val_raw)} validation examples for SFT...")
validation_texts = [format_for_sft(ex, tokenizer, prompt_file_path=PROMPT_FILE) for ex in val_raw]
validation_dataset = Dataset.from_dict({"text": validation_texts})

print(f"\nTraining on {len(train_dataset)} samples")
print(f"Validating on {len(validation_dataset)} samples")

print(f"\nSample:\n{train_dataset[0]['text'][:1000]} ... \n")

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
    num_train_epochs=4,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,     # effective batch size = 16
    learning_rate=2e-4,                 # default for LoRA fine-tuning
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",              # saves checkpoint after each epoch
    save_total_limit=4,                 # keep only the 4 epoch checkpoints
    max_length=4096,                    # truncate inputs to fit model context window
    gradient_checkpointing=True,
    dataset_text_field="text",
    report_to="wandb",
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

trainer.train()
trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
print(f"\nDone. Checkpoints saved to {OUTPUT_DIR}")