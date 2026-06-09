import json
import os
import random
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import wandb
from datasets import Dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

from evaluate.benchmarks import extract_benchmark_answer, is_correct_benchmark_answer


for env_name in list(os.environ):
    if env_name.startswith("WANDB_") and os.environ[env_name] == "":
        del os.environ[env_name]


BEST_MATH500_16K_CHECKPOINT = Path(
    "/scratch/checkpoints/math/qwen3-1.7b-lora-math-mixed_20260526-220009/checkpoint-4458"
)
SNAPSHOT_DIR = Path("/scratch/hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots")
DEFAULT_PROMPT_FILE = Path("fourneurons/prompts/math.txt")
DEFAULT_FRONTIER_TRAIN_FILE = Path("/scratch/data/math/rl_frontier/splits/rl_frontier_train.jsonl")

RUN_NAME = os.getenv("FOURNEURONS_RUN_NAME") or "qwen3-1.7b-lora-math-rl"
run_id = os.getenv("FOURNEURONS_RUN_ID") or RUN_NAME + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
WANDB_PROJECT = os.getenv("WANDB_PROJECT") or "math-rl"
WANDB_NAME = os.getenv("WANDB_NAME") or run_id
WANDB_RUN_ID = os.getenv("WANDB_RUN_ID") or None
WANDB_RESUME = os.getenv("WANDB_RESUME") or ("allow" if WANDB_RUN_ID else None)

BASE_MODEL_PATH = Path(os.getenv("BASE_MODEL_PATH") or (SNAPSHOT_DIR / sorted(os.listdir(SNAPSHOT_DIR))[0]))
INIT_CHECKPOINT = Path(os.getenv("MATH_RL_INIT_CHECKPOINT") or BEST_MATH500_16K_CHECKPOINT)
TRAIN_FILE = Path(os.getenv("MATH_RL_TRAIN_FILE") or DEFAULT_FRONTIER_TRAIN_FILE)
EVAL_FILE = os.getenv("MATH_RL_EVAL_FILE")
EVAL_FILE = Path(EVAL_FILE) if EVAL_FILE else None
PROMPT_FILE = Path(os.getenv("MATH_RL_PROMPT_FILE") or DEFAULT_PROMPT_FILE)
OUTPUT_DIR = os.getenv("OUTPUT_DIR") or f"/scratch/checkpoints/math/{run_id}"
RESUME_FROM_CHECKPOINT = os.getenv("RESUME_FROM_CHECKPOINT") or None

SEED = int(os.getenv("MATH_RL_SEED", "42"))
N_EPOCHS = float(os.getenv("MATH_RL_NUM_EPOCHS", "1"))
MAX_STEPS = int(os.getenv("MATH_RL_MAX_STEPS", "-1"))
MAX_TRAINING_SAMPLES = int(os.getenv("MATH_RL_MAX_TRAINING_SAMPLES", "4096"))
MAX_VALIDATION_SAMPLES = int(os.getenv("MATH_RL_MAX_VALIDATION_SAMPLES", "0"))
# Leave room for prompt tokens above the 16k completion budget.
MAX_CONTEXT_LENGTH = int(os.getenv("MATH_RL_MAX_CONTEXT_LENGTH", "20000"))
MAX_COMPLETION_LENGTH = int(os.getenv("MATH_RL_MAX_COMPLETION_LENGTH", "16384"))

NUM_GENERATIONS = int(os.getenv("MATH_RL_NUM_GENERATIONS", "8"))
NUM_GENERATIONS_EVAL = int(os.getenv("MATH_RL_NUM_GENERATIONS_EVAL", "2"))
PER_DEVICE_TRAIN_BATCH_SIZE = int(os.getenv("MATH_RL_PER_DEVICE_TRAIN_BATCH_SIZE", "1"))
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("MATH_RL_GRADIENT_ACCUMULATION_STEPS", "8"))
GENERATION_BATCH_SIZE = int(os.getenv("MATH_RL_GENERATION_BATCH_SIZE", str(NUM_GENERATIONS)))
PER_DEVICE_EVAL_BATCH_SIZE = int(os.getenv("MATH_RL_PER_DEVICE_EVAL_BATCH_SIZE", str(NUM_GENERATIONS_EVAL)))
LEARNING_RATE = float(os.getenv("MATH_RL_LEARNING_RATE", "1e-6"))
BETA = float(os.getenv("MATH_RL_KL_BETA", "0.001"))
LOSS_TYPE = os.getenv("MATH_RL_LOSS_TYPE") or "dapo"
EVAL_STEPS = int(os.getenv("MATH_RL_EVAL_STEPS", "50"))
SAVE_STEPS = int(os.getenv("MATH_RL_SAVE_STEPS", "50"))
LOGGING_STEPS = int(os.getenv("MATH_RL_LOGGING_STEPS", "1"))
TRAINER_BF16 = os.getenv("MATH_RL_TRAINER_BF16", "0").lower() in {"1", "true", "yes"}
TEMPERATURE = float(os.getenv("MATH_RL_TEMPERATURE", "0.7"))
TOP_P = float(os.getenv("MATH_RL_TOP_P", "0.9"))
TOP_K = int(os.getenv("MATH_RL_TOP_K", "0"))
USE_VLLM = os.getenv("MATH_RL_USE_VLLM", "1").lower() in {"1", "true", "yes"}
MASK_TRUNCATED_COMPLETIONS = os.getenv("MATH_RL_MASK_TRUNCATED_COMPLETIONS", "0").lower() in {
    "1",
    "true",
    "yes",
}
VLLM_MODE = os.getenv("MATH_RL_VLLM_MODE") or "colocate"
VLLM_MAX_MODEL_LENGTH = int(os.getenv("MATH_RL_VLLM_MAX_MODEL_LENGTH", str(MAX_CONTEXT_LENGTH)))
VLLM_TENSOR_PARALLEL_SIZE = int(os.getenv("MATH_RL_VLLM_TENSOR_PARALLEL_SIZE", "1"))
VLLM_ENABLE_SLEEP_MODE = os.getenv("MATH_RL_VLLM_ENABLE_SLEEP_MODE", "0").lower() in {
    "1",
    "true",
    "yes",
}
VLLM_IMPORTANCE_SAMPLING_CORRECTION = os.getenv(
    "MATH_RL_VLLM_IMPORTANCE_SAMPLING_CORRECTION",
    "0",
).lower() in {"1", "true", "yes"}
VLLM_GPU_MEMORY_UTILIZATION = float(os.getenv("MATH_RL_VLLM_GPU_MEMORY_UTILIZATION", "0.3"))

CORRECT_REWARD = float(os.getenv("MATH_RL_CORRECT_REWARD", "1.0"))
INCORRECT_BOXED_REWARD = float(os.getenv("MATH_RL_INCORRECT_BOXED_REWARD", "-0.25"))
MISSING_BOX_REWARD = float(os.getenv("MATH_RL_MISSING_BOX_REWARD", "-0.75"))
OVERLONG_REWARD_PENALTY = float(os.getenv("MATH_RL_OVERLONG_REWARD_PENALTY", "-0.05"))


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


def disable_bitsandbytes_lora_dispatch():
    import peft.tuners.lora.model as peft_lora_model

    peft_lora_model.is_bnb_available = lambda: False
    peft_lora_model.is_bnb_4bit_available = lambda: False


def load_prompt_file(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def build_prompt(tokenizer, system_prompt, row):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": row["prompt"]})

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def collect_rl_examples(path, tokenizer, system_prompt, target_count):
    if target_count <= 0:
        return []

    examples = []
    scanned = 0
    dropped_missing = 0
    dropped_overlong = 0
    max_prompt_tokens = MAX_CONTEXT_LENGTH - MAX_COMPLETION_LENGTH
    if max_prompt_tokens <= 0:
        raise ValueError("MATH_RL_MAX_CONTEXT_LENGTH must be larger than MATH_RL_MAX_COMPLETION_LENGTH")

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            scanned += 1
            row = json.loads(line)
            if "prompt" not in row or "answer" not in row:
                dropped_missing += 1
                continue

            prompt = build_prompt(tokenizer, system_prompt, row)
            token_count = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
            if token_count > max_prompt_tokens:
                dropped_overlong += 1
                continue

            examples.append(
                {
                    "prompt": prompt,
                    "answer": str(row["answer"]),
                    "raw_prompt": row["prompt"],
                    "source": str(row.get("source", "unknown")),
                    "rl_correct_count": row.get("rl_correct_count"),
                    "rl_boxed_count": row.get("rl_boxed_count"),
                    "prompt_tokens": token_count,
                }
            )
            if len(examples) >= target_count:
                break

    if len(examples) < target_count:
        print(
            f"WARNING: only collected {len(examples)}/{target_count} RL examples from {path}. "
            f"Scanned={scanned}, dropped_missing={dropped_missing}, dropped_overlong={dropped_overlong}."
        )
    else:
        print(
            f"Collected {len(examples)} RL examples from {path}. "
            f"Scanned={scanned}, dropped_missing={dropped_missing}, dropped_overlong={dropped_overlong}."
        )

    return examples


def split_train_eval(examples):
    random.Random(SEED).shuffle(examples)
    if MAX_VALIDATION_SAMPLES <= 0 or len(examples) < 2:
        return examples[:MAX_TRAINING_SAMPLES], []

    eval_count = min(MAX_VALIDATION_SAMPLES, max(1, len(examples) // 10))
    eval_examples = examples[:eval_count]
    train_examples = examples[eval_count : eval_count + MAX_TRAINING_SAMPLES]
    return train_examples, eval_examples


def completion_to_text(completion):
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts = []
        for message in completion:
            if isinstance(message, dict):
                content = message.get("content", "")
                if isinstance(content, str):
                    parts.append(content)
                else:
                    parts.append(str(content))
            else:
                parts.append(str(message))
        return "".join(parts)
    return str(completion)


def math_verifier_reward(prompts, completions, answer, completion_ids=None, log_extra=None, log_metric=None, **kwargs):
    rewards = []
    extracted_values = []
    boxed_values = []
    correct_values = []
    length_values = []

    if completion_ids is None:
        completion_ids = [None] * len(completions)

    for completion, reference, ids in zip(completions, answer, completion_ids):
        completion_text = completion_to_text(completion)
        extracted = extract_benchmark_answer(completion_text, "boxed", str(reference))
        boxed = extracted is not None
        correct = is_correct_benchmark_answer(extracted, str(reference), "boxed")

        if correct:
            reward = CORRECT_REWARD
        elif boxed:
            reward = INCORRECT_BOXED_REWARD
        else:
            reward = MISSING_BOX_REWARD

        completion_length = len(ids) if ids is not None else 0
        if ids is not None and completion_length >= MAX_COMPLETION_LENGTH:
            reward += OVERLONG_REWARD_PENALTY

        rewards.append(reward)
        extracted_values.append(extracted if extracted is not None else "")
        boxed_values.append(boxed)
        correct_values.append(correct)
        length_values.append(completion_length)

    if log_extra is not None:
        log_extra("extracted", extracted_values)
        log_extra("boxed", boxed_values)
        log_extra("correct", correct_values)

    if log_metric is not None and rewards:
        log_metric("reward/mean", sum(rewards) / len(rewards))
        log_metric("reward/boxed_rate", sum(boxed_values) / len(boxed_values))
        log_metric("reward/correct_rate", sum(correct_values) / len(correct_values))
        if length_values:
            log_metric("reward/mean_completion_tokens", sum(length_values) / len(length_values))

        if len(correct_values) % NUM_GENERATIONS == 0:
            group_counts = [
                sum(correct_values[i : i + NUM_GENERATIONS])
                for i in range(0, len(correct_values), NUM_GENERATIONS)
            ]
            group_hist = Counter(group_counts)
            for count in range(NUM_GENERATIONS + 1):
                log_metric(f"reward/correct_count_{count}_of_{NUM_GENERATIONS}", group_hist[count] / len(group_counts))

            unique_counts = []
            for i in range(0, len(extracted_values), NUM_GENERATIONS):
                unique = {value for value in extracted_values[i : i + NUM_GENERATIONS] if value}
                unique_counts.append(len(unique))
            log_metric("reward/unique_boxed_answers", sum(unique_counts) / len(unique_counts))

    return rewards


def load_policy_model():
    disable_bitsandbytes_lora_dispatch()
    if (INIT_CHECKPOINT / "adapter_config.json").is_file():
        print(f"Loading base model from {BASE_MODEL_PATH}")
        print(f"Loading trainable LoRA adapter from {INIT_CHECKPOINT}")
        base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(base_model, INIT_CHECKPOINT, is_trainable=True)
    else:
        print(f"Loading full model from {INIT_CHECKPOINT}")
        model = AutoModelForCausalLM.from_pretrained(INIT_CHECKPOINT, torch_dtype=torch.bfloat16)

    model.config.use_cache = False
    return model


def main():
    global RESUME_FROM_CHECKPOINT

    if NUM_GENERATIONS != 8:
        raise ValueError("Math RL should use MATH_RL_NUM_GENERATIONS=8 to match pass@8 training groups.")
    if GENERATION_BATCH_SIZE % NUM_GENERATIONS != 0:
        raise ValueError("MATH_RL_GENERATION_BATCH_SIZE must be divisible by MATH_RL_NUM_GENERATIONS.")
    if NUM_GENERATIONS_EVAL < 1:
        raise ValueError("MATH_RL_NUM_GENERATIONS_EVAL must be at least 1.")
    if PER_DEVICE_EVAL_BATCH_SIZE % NUM_GENERATIONS_EVAL != 0:
        raise ValueError("MATH_RL_PER_DEVICE_EVAL_BATCH_SIZE must be divisible by MATH_RL_NUM_GENERATIONS_EVAL.")
    if not TRAIN_FILE.is_file():
        raise FileNotFoundError(f"Missing RL train file: {TRAIN_FILE}")
    if EVAL_FILE is not None and not EVAL_FILE.is_file():
        raise FileNotFoundError(f"Missing RL eval file: {EVAL_FILE}")
    if not PROMPT_FILE.is_file():
        raise FileNotFoundError(f"Missing prompt file: {PROMPT_FILE}")
    if not INIT_CHECKPOINT.is_dir():
        raise FileNotFoundError(f"Missing init checkpoint: {INIT_CHECKPOINT}")

    RESUME_FROM_CHECKPOINT = resolve_resume_checkpoint(RESUME_FROM_CHECKPOINT)

    wandb_key = os.getenv("WANDB_KEY") or os.getenv("WANDB_API_KEY")
    if wandb_key:
        wandb.login(key=wandb_key)
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_NAME,
        id=WANDB_RUN_ID,
        resume=WANDB_RESUME,
        config={
            "run_id": run_id,
            "train_file": str(TRAIN_FILE),
            "eval_file": str(EVAL_FILE) if EVAL_FILE else None,
            "init_checkpoint": str(INIT_CHECKPOINT),
            "num_generations": NUM_GENERATIONS,
            "num_generations_eval": NUM_GENERATIONS_EVAL,
            "max_completion_length": MAX_COMPLETION_LENGTH,
            "learning_rate": LEARNING_RATE,
            "beta": BETA,
            "loss_type": LOSS_TYPE,
            "trainer_bf16": TRAINER_BF16,
            "use_vllm": USE_VLLM,
            "vllm_mode": VLLM_MODE,
            "vllm_importance_sampling_correction": VLLM_IMPORTANCE_SAMPLING_CORRECTION,
            "mask_truncated_completions": MASK_TRUNCATED_COMPLETIONS,
        },
    )

    print(f"Run name: {run_id}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Train file: {TRAIN_FILE}")
    print(f"Eval file: {EVAL_FILE if EVAL_FILE else 'split from train file'}")
    print(f"Prompt file: {PROMPT_FILE}")
    if RESUME_FROM_CHECKPOINT:
        print(f"Resuming from checkpoint: {RESUME_FROM_CHECKPOINT}")
    if WANDB_RUN_ID:
        print(f"Resuming W&B run id: {WANDB_RUN_ID} with resume={WANDB_RESUME}")

    tokenizer = AutoTokenizer.from_pretrained(INIT_CHECKPOINT if (INIT_CHECKPOINT / "tokenizer.json").is_file() else BASE_MODEL_PATH)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    system_prompt = load_prompt_file(PROMPT_FILE)
    target_count = MAX_TRAINING_SAMPLES + (0 if EVAL_FILE else MAX_VALIDATION_SAMPLES)
    examples = collect_rl_examples(TRAIN_FILE, tokenizer, system_prompt, target_count)
    if EVAL_FILE:
        train_examples = examples[:MAX_TRAINING_SAMPLES]
        eval_examples = collect_rl_examples(EVAL_FILE, tokenizer, system_prompt, MAX_VALIDATION_SAMPLES)
    else:
        train_examples, eval_examples = split_train_eval(examples)

    train_dataset = Dataset.from_list(train_examples)
    eval_dataset = Dataset.from_list(eval_examples) if eval_examples else None
    if len(train_dataset) == 0:
        raise ValueError(f"No trainable RL examples were collected from {TRAIN_FILE}")

    print(f"\nTraining on {len(train_dataset)} RL prompts")
    print(f"Evaluating on {len(eval_dataset) if eval_dataset else 0} RL prompts")
    print(f"\nSample prompt:\n{train_dataset[0]['prompt'][:1200]} ... \n")

    model = load_policy_model()

    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=N_EPOCHS,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        generation_batch_size=GENERATION_BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        bf16=TRAINER_BF16,
        logging_steps=LOGGING_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=4,
        gradient_checkpointing=True,
        report_to="wandb",
        run_name=WANDB_NAME,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=EVAL_STEPS,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        num_generations=NUM_GENERATIONS,
        num_generations_eval=NUM_GENERATIONS_EVAL,
        max_completion_length=MAX_COMPLETION_LENGTH,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        top_k=TOP_K,
        beta=BETA,
        loss_type=LOSS_TYPE,
        # Keep this false when using OVERLONG_REWARD_PENALTY: masked truncated
        # completions do not contribute policy-gradient loss, so the penalty
        # would be logged but not learned from.
        mask_truncated_completions=MASK_TRUNCATED_COMPLETIONS,
        use_vllm=USE_VLLM,
        vllm_mode=VLLM_MODE,
        vllm_max_model_length=VLLM_MAX_MODEL_LENGTH,
        vllm_tensor_parallel_size=VLLM_TENSOR_PARALLEL_SIZE,
        vllm_enable_sleep_mode=VLLM_ENABLE_SLEEP_MODE,
        vllm_importance_sampling_correction=VLLM_IMPORTANCE_SAMPLING_CORRECTION,
        vllm_gpu_memory_utilization=VLLM_GPU_MEMORY_UTILIZATION,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=math_verifier_reward,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final"))
    print(f"\nDone. RL checkpoints saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
