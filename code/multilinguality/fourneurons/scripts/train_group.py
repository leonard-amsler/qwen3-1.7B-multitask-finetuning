# python fourneurons/scripts/train_group.py --run_name group_sft1 --epochs 6 --single_dataset_batches

import os
import json
import torch
import argparse
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTTrainer, SFTConfig, GRPOTrainer, GRPOConfig
from peft import LoraConfig

from fourneurons.data.format_for_sft import format_for_sft
from fourneurons.model.language_grouped_trainer import LanguageGroupedSFTTrainer, EpochShuffleCallback
from fourneurons.evaluation.extract_answer import extract_boxed_answer

MODEL_PATH = "/scratch/checkpoints/group/base_patched"
PROMPT_PATH = "/scratch/nathan/repo/fourneurons/prompts/sp_group_think.txt"

def sft(run_name, base, epochs, start_lr, single_dataset_batches, cont):
    """
    SFT training function.

    Args:
        run_name:               WandB + checkpoint run name
        base:                   Path to checkpoint to fine-tune from (can be base model or previous SFT checkpoint)
        epochs:                 Number of training epochs
        start_lr:               Starting learning rate for fine-tuning (e.g. 2e-4)
        single_dataset_batches: Whether to create batches with examples from only one dataset (default: False). 
    """
    splits_dir = f"/scratch/data/group/mixed/splits"
    output_dir = f"/scratch/checkpoints/group/{run_name}"
    os.environ["WANDB_PROJECT"] = run_name

    # Data loading:
    tokenizer = AutoTokenizer.from_pretrained(base)

    def load_split(filename):
        examples = []
        with open(os.path.join(splits_dir, filename)) as f:
            for line in f:
                examples.append(json.loads(line))
        return examples


    train_raw = load_split(f"mixed_train.jsonl")
    # format_for_sft now returns a dict {"messages": [...]}
    examples = [format_for_sft(ex, tokenizer, prompt_file_path=PROMPT_PATH) for ex in train_raw]
    dataset = Dataset.from_list(examples)

    print(f"Training on {len(dataset)} samples")
    print(f"\nSample messages:\n{dataset[0]['messages']}\n")

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
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,  # effective batch size = 16
        learning_rate=start_lr,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",  # saves checkpoint after each epoch
        save_total_limit=100,
        max_length=6000,
        assistant_only_loss=True,
        report_to="wandb",
        truncation_mode="keep_end",
    )

    if single_dataset_batches:
        print("Using single-dataset batches for training.")
        dataset_ds = [ex["dataset"] for ex in train_raw]
        trainer = LanguageGroupedSFTTrainer(
            model=model,
            processing_class=tokenizer,
            languages=dataset_ds,
            train_dataset=dataset,
            peft_config=lora_config,
            args=training_args,
        )
        trainer.add_callback(EpochShuffleCallback(trainer))
    else:
        trainer = SFTTrainer(
            model=model,
            processing_class=tokenizer,
            train_dataset=dataset,
            peft_config=lora_config,
            args=training_args,
        )

    trainer.train(resume_from_checkpoint=base if cont else None)
    trainer.save_model(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    print(f"\nDone. Checkpoints saved to {output_dir}")
    print("Epoch checkpoints: checkpoint-1, checkpoint-2")
    print("Final model: final/")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train a group model")
    argparser.add_argument("--run_name", required=True, type=str, help="Name of the training run (used for output directory and WandB logging)")
    argparser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to the checkpoint to fine-tune from (default: base group model)")
    argparser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    argparser.add_argument("--start_lr", type=float, default=2e-4, help="Starting learning rate for fine-tuning")
    argparser.add_argument("--single_dataset_batches", action="store_true", help="Whether to create batches with examples from only one dataset (default: False)")
    argparser.add_argument("--cont", action="store_true", help="Whether to continue training from the provided model_path checkpoint (default: False, i.e. start a new SFT run from the base model or previous checkpoint)")

    args = argparser.parse_args()

    sft(run_name=args.run_name, base=args.model_path, epochs=args.epochs, start_lr=args.start_lr, single_dataset_batches=args.single_dataset_batches, cont=args.cont)