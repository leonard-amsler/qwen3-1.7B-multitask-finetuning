# python fourneurons/scripts/train_multilingual.py sft --run_name mmmlu_sft3_long2 --model_path /scratch/checkpoints/multilingual/mmmlu_sft3_long2/final --epochs 12
# python fourneurons/scripts/train_multilingual.py grpo --run_name mmmlu_grpo1 --model_path /scratch/results/multilingual/mmmlu_prox/mmmlu_prox_mmmlu_sft3_long2_6875/merged --num_generations 8

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

MODEL_PATH = "/scratch/checkpoints/multilingual/base"
PROMPT_PATH = "/scratch/nathan/repo/fourneurons/prompts/multilingual_cot_teacher.txt"

def sft(run_name, base, epochs, start_lr, single_language_batches):
    """
    SFT training function.

    Args:
        run_name:               WandB + checkpoint run name
        base:                   Path to checkpoint to fine-tune from (can be base model or previous SFT checkpoint)
        epochs:                 Number of training epochs
        start_lr:               Starting learning rate for fine-tuning (e.g. 2e-4)
        single_language_batches: Whether to create batches with examples from only one language (default: False). 
    """
    splits_dir = f"/scratch/data/multilingual/mmmlu_more_qcms/splits"
    output_dir = f"/scratch/checkpoints/multilingual/{run_name}"
    os.environ["WANDB_PROJECT"] = run_name

    # Data loading:
    tokenizer = AutoTokenizer.from_pretrained(base)

    def load_split(filename):
        examples = []
        with open(os.path.join(splits_dir, filename)) as f:
            for line in f:
                examples.append(json.loads(line))
        return examples


    train_raw = load_split(f"mmmlu_more_qcms_train.jsonl")
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
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,  # effective batch size = 16
        learning_rate=start_lr,
        bf16=True,
        logging_steps=10,
        save_strategy="epoch",  # saves checkpoint after each epoch
        save_total_limit=100,
        max_length=3000,
        assistant_only_loss=True,
        report_to="wandb",
        truncation_mode="keep_end",
    )

    if single_language_batches:
        print("Using single-language batches for training.")
        dataset_languages = [ex["idx"].split("_")[0] for ex in train_raw]
        trainer = LanguageGroupedSFTTrainer(
            model=model,
            processing_class=tokenizer,
            languages=dataset_languages,
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

    trainer.train(resume_from_checkpoint=base or None)
    trainer.save_model(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    print(f"\nDone. Checkpoints saved to {output_dir}")
    print("Epoch checkpoints: checkpoint-1, checkpoint-2")
    print("Final model: final/")

def grpo(run_name, base, num_generations=8):
    """
    GRPO training function.
    
    Args:
        run_name:               WandB + checkpoint run name
        base:                   Path to merged SFT checkpoint (or base model)
        num_generations:        Rollouts per problem (G). Higher = more stable gradients but slower.
    """
    splits_dir = f"/scratch/data/multilingual/mmmlu_more_qcms/splits"
    output_dir = f"/scratch/checkpoints/multilingual/{run_name}"
    os.environ["WANDB_PROJECT"] = run_name

    # ── Data loading ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(base)

    def load_split(filename):
        examples = []
        with open(os.path.join(splits_dir, filename)) as f:
            for line in f:
                examples.append(json.loads(line))
        return examples

    # Point this at your problems file
    train_raw = load_split("mmmlu_more_qcms_grpo_train.jsonl")

    # Reuse format_for_sft — GRPO needs the prompt only (no answer in messages)
    # GRPOTrainer expects a "prompt" key with the conversation so far
    def format_for_grpo(ex):
        formatted = format_for_sft(ex, tokenizer, prompt_file_path=PROMPT_PATH)
        return {
            "prompt": formatted["messages"][:-1],  # strip the assistant turn — model must generate it
            "answer": ex["answer"],                 # kept for reward fn, not seen by model
        }

    examples = [format_for_grpo(ex) for ex in train_raw]
    dataset = Dataset.from_list(examples)
    print(f"GRPO training on {len(dataset)} problems")
    print(f"\nSample prompt:\n{dataset[0]['prompt']}\n")

    # ── Reward function ──────────────────────────────────────────────────────────
    def reward_fn(completions, answer, **kwargs):
        rewards = []
        for completion, ans in zip(completions, answer):
            # completion is [{'role': 'assistant', 'content': '...'}]
            text = completion[0]["content"]
            extracted = extract_boxed_answer(text)
            rewards.append(1.0 if extracted == ans else 0.0)
        return rewards

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16)

    # ── GRPO config ──────────────────────────────────────────────────────────────
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        max_steps=250,
        num_generations=num_generations,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=5e-6,
        bf16=True,
        logging_steps=1,
        save_steps=50,
        save_total_limit=100,
        max_completion_length=4096,
        report_to="wandb",
        # KL penalty, following DeepSeek-R1's 0.001 recommendation
        beta=0.001,
        # Temperature for rollout sampling
        temperature=0.9,
    )

    # ── Trainer ──────────────────────────────────────────────────────────────────
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_fn,
        train_dataset=dataset,
        args=grpo_config,
    )

    trainer.train()
    trainer.save_model(os.path.join(output_dir, "final"))
    tokenizer.save_pretrained(os.path.join(output_dir, "final"))
    print(f"\nDone. Checkpoints saved to {output_dir}")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Train a multilingual model")
    subparsers = argparser.add_subparsers(dest="mode", required=True)
    # ── SFT ──────────────────────────────────────────────────────────────────────
    sft_parser = subparsers.add_parser("sft", help="Supervised fine-tuning with LoRA")
    sft_parser.add_argument("--run_name", required=True, type=str, help="Name of the training run (used for output directory and WandB logging)")
    sft_parser.add_argument("--model_path", type=str, default=MODEL_PATH, help="Path to the checkpoint to fine-tune from (default: base multilingual model)")
    sft_parser.add_argument("--epochs", type=int, default=6, help="Number of training epochs")
    sft_parser.add_argument("--start_lr", type=float, default=2e-4, help="Starting learning rate for fine-tuning")
    sft_parser.add_argument("--single_language_batches", action="store_true", help="Whether to create batches with examples from only one language (default: False)")

    # ── GRPO ─────────────────────────────────────────────────────────────────────
    grpo_parser = subparsers.add_parser("grpo", help="GRPO reinforcement learning")
    grpo_parser.add_argument("--run_name", required=True, type=str)
    grpo_parser.add_argument("--model_path", type=str, default=MODEL_PATH)
    grpo_parser.add_argument("--num_generations", type=int, default=8, help="Rollouts per problem (G). Higher = more stable gradients but slower.")

    args = argparser.parse_args()

    if args.mode == "sft":
        sft(run_name=args.run_name, base=args.model_path, epochs=args.epochs, start_lr=args.start_lr, single_language_batches=args.single_language_batches)
    elif args.mode == "grpo":
        grpo(run_name=args.run_name, base=args.model_path, num_generations=args.num_generations)