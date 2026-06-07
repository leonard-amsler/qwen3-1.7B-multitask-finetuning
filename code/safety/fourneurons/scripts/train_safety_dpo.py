import os
import json
import argparse
import wandb
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig
from prompts.prompt_loader import load_prompt

OUTPUT_DIR = "/scratch/checkpoints/safety_dpo"

DPO_FILES = {
    "weak": "/scratch/data/safety/safetybench/dpo/train_dpo_weak_categories.jsonl",
    "all": "/scratch/data/safety/safetybench/dpo/train_dpo_all_categories.jsonl",
}

BETA = 0.1
LEARNING_RATE = 5e-6
NUM_EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 8
MAX_LENGTH = 4096  # !! must match DPO_MAX_LENGTH in build_safety_dpo.py

SYSTEM_PROMPT = load_prompt("/scratch/nico/standard-project-m2-4neurons/prompts/sp_general_qcm_think.txt")


def load_dpo_pairs(all_categories=False):
    key = "all" if all_categories else "weak"
    path = DPO_FILES[key]
    pairs = []
    with open(path) as f:
        for line in f:
            pairs.append(json.loads(line))
    print(f"Loaded {len(pairs)} DPO pairs from {path}")
    return pairs


def format_prompt(prompt, tokenizer):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def main(merged_sft_dir, all_categories=False):
    run_name = f"dpo-{'all' if all_categories else 'weak'}-categories"

    wandb.init(
        project="safety-dpo",
        name=run_name,
        config={
            "beta": BETA,
            "learning_rate": LEARNING_RATE,
            "num_epochs": NUM_EPOCHS,
            "effective_batch_size": BATCH_SIZE * GRAD_ACCUM,
            "max_length": MAX_LENGTH,
            "categories": "all" if all_categories else "weak",
            "base_model": merged_sft_dir,
        },
    )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(merged_sft_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    pairs = load_dpo_pairs(all_categories=all_categories)
    dataset = Dataset.from_dict({
        "prompt": [format_prompt(p["prompt"], tokenizer) for p in pairs],
        "chosen": [p["chosen"] for p in pairs],
        "rejected": [p["rejected"] for p in pairs],
    })

    print(f"Dataset size: {len(dataset)}")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules="all-linear",
        lora_dropout=0.05,
        bias="none",
    )

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        merged_sft_dir,
        dtype="bfloat16",
    )

    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        bf16=True,
        beta=BETA,
        max_length=MAX_LENGTH,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=3,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        report_to="wandb",
        run_name=run_name,
        remove_unused_columns=False,
        precompute_ref_log_probs=True,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("Starting DPO training...")
    trainer.train()

    print("Saving final model...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))

    wandb.finish()
    print(f"Done. Checkpoint saved to {OUTPUT_DIR}/final")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--merged_sft_dir",
        type=str,
        required=True,
        help="Directory containing the merged SFT model and tokenizer.",
    )
    parser.add_argument(
        "--all_categories",
        action="store_true",
        help="Train on all categories. Default: weak categories only.",
    )
    args = parser.parse_args()
    main(merged_sft_dir=args.merged_sft_dir, all_categories=args.all_categories)