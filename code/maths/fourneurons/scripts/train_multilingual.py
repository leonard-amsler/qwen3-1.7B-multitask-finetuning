import os
import json
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from fourneurons.data.format_for_sft import format_for_sft

SNAPSHOT_DIR = "/scratch/hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/"
MODEL_PATH = SNAPSHOT_DIR + os.listdir(SNAPSHOT_DIR)[0]
SPLITS_DIR = "/scratch/data/multilingual/mmmlu_more_qcms/splits"
OUTPUT_DIR = "/scratch/checkpoints/multilingual"

# Data loading:
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)


def load_split(filename):
    examples = []
    with open(os.path.join(SPLITS_DIR, filename)) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


train_raw = load_split("mmmlu_more_qcms_train.jsonl")
texts = [format_for_sft(ex, tokenizer) for ex in train_raw]
dataset = Dataset.from_dict({"text": texts})

print(f"Training on {len(dataset)} samples")
print(f"\nSample:\n{dataset[0]['text']}\n")

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
    num_train_epochs=2,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # effective batch size = 16
    learning_rate=2e-4,  # default for LoRA fine-tuning
    bf16=True,
    logging_steps=50,
    save_strategy="epoch",  # saves checkpoint after each epoch
    save_total_limit=2,  # keep only the 2 epoch checkpoints
    max_length=2048,
    dataset_text_field="text",
    report_to="wandb",
)

trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    train_dataset=dataset,
    peft_config=lora_config,
    args=training_args,
)

trainer.train()
trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
print(f"\nDone. Checkpoints saved to {OUTPUT_DIR}")
print("Epoch checkpoints: checkpoint-1, checkpoint-2")
print("Final model: final/")
