from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import load_from_disk
from peft import LoraConfig
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainingArguments,
)
from trl import SFTTrainer

try:
    from trl import SFTConfig  
    _HAS_SFTCONFIG = True
except ImportError:  
    SFTConfig = None  
    _HAS_SFTCONFIG = False


BASE_MODEL = "Qwen/Qwen3-1.7B"


def _build_formatting_func(tokenizer):
    def _fmt(row):
        return tokenizer.apply_chat_template(
            row["messages"], tokenize=False, add_generation_prompt=False
        )
    return _fmt


def _assert_chat_template_is_sane(tokenizer, dataset_train) -> None:
    row = dataset_train[0]
    text = tokenizer.apply_chat_template(
        row["messages"], tokenize=False, add_generation_prompt=False
    )
    n_thinks = text.count("<think>")
    n_close = text.count("</think>")
    assert n_thinks == 1 and n_close == 1, (
        "Chat-template sanity check failed: expected exactly one <think> "
        f"opener and one </think> closer per row, got {n_thinks} / {n_close}. "
        "Inspect the template and the assistant content."
    )
    assert "\\boxed{" in text, (
        "First training row has no `\\boxed{...}` in the rendered string. "
        "Check `to_chat_messages`."
    )
    print("[sanity] chat-template OK: 1× <think>, 1× </think>, 1× \\boxed{}.")
    # Print a snippet so we can eyeball.
    snippet = text[-400:]
    print(f"[sanity] last 400 chars of rendered row 0:\n{snippet}\n")


def _build_training_args(
    *,
    output_dir: Path,
    learning_rate: float,
    per_device_batch_size: int,
    grad_accum: int,
    num_epochs: int,
    eval_steps: int,
    save_steps: int,
    logging_steps: int,
    warmup_steps: int,
    bf16: bool,
    seed: int,
    max_seq_length: int,
    optim: str,
):

    common = dict(
        output_dir=str(output_dir),
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=num_epochs,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        logging_steps=logging_steps,
        report_to=["tensorboard"],
        optim=optim,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        bf16=bf16,
        fp16=not bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataloader_num_workers=2,
        seed=seed,
        remove_unused_columns=False,
    )
    if _HAS_SFTCONFIG:

        for k in ("max_length", "max_seq_length"):
            try:
                return SFTConfig(**common, **{k: max_seq_length}), {"kind": "sftconfig", "max_seq_kwarg": k}
            except TypeError:
                continue

        return SFTConfig(**common), {"kind": "sftconfig", "max_seq_kwarg": None}
    return TrainingArguments(**common), {"kind": "training_args", "max_seq_kwarg": None}



def train_model(
    dataset_dir: Path,
    output_dir: Path,
    final_model_dir: Path,
    base_model: str = BASE_MODEL,
    num_epochs: int = 1,
    learning_rate: float = 2e-4,
    per_device_batch_size: int = 2,
    grad_accum: int = 8,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    max_seq_length: int = 2048,
    warmup_steps: int = 50,
    eval_steps: int = 200,
    save_steps: int = 200,
    logging_steps: int = 20,
    early_stop_patience: int = 4,
    seed: int = 42,
    bf16: bool = False,
    optim: str = "adamw_torch_fused",
) -> None:
    print("=" * 80)
    print("Qwen3-1.7B LoRA SFT — General Knowledge")
    print("=" * 80)

    print(f"[1/7] Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print(f"[2/7] Loading dataset from disk: {dataset_dir}")
    dsd = load_from_disk(str(dataset_dir))
    if "train" not in dsd or "test" not in dsd:
        raise SystemExit(
            f"Dataset at {dataset_dir} must contain `train` and `test` splits. "
            f"Got: {list(dsd.keys())}"
        )
    train_ds = dsd["train"]
    eval_ds = dsd["test"]
    print(f"      train={len(train_ds)}  eval={len(eval_ds)}")
    print(f"      train columns: {train_ds.column_names}")

    _assert_chat_template_is_sane(tokenizer, train_ds)

    print(f"[3/7] Loading base model: {base_model}")
    dtype = torch.bfloat16 if bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        dtype=dtype,
        trust_remote_code=True,
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"      {model.config.model_type}, {n_params:.2f}B params, dtype={dtype}")

    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    print(f"[4/7] Configuring LoRA: r={lora_r}, alpha={lora_alpha}")
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    print("[5/7] Building TrainingArguments / SFTConfig")
    output_dir.mkdir(parents=True, exist_ok=True)
    training_args, ta_meta = _build_training_args(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_batch_size=per_device_batch_size,
        grad_accum=grad_accum,
        num_epochs=num_epochs,
        eval_steps=eval_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        warmup_steps=warmup_steps,
        bf16=bf16,
        seed=seed,
        max_seq_length=max_seq_length,
        optim=optim,
    )
    print(f"      using {ta_meta['kind']} (max_seq via "
          f"{'config.' + ta_meta['max_seq_kwarg'] if ta_meta['max_seq_kwarg'] else 'trainer-arg or unused'})")

    print("[6/7] Initializing SFTTrainer")
    sft_kwargs: dict = dict(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        peft_config=lora_config,
        args=training_args,
        formatting_func=_build_formatting_func(tokenizer),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=early_stop_patience,
                early_stopping_threshold=1e-3,
            )
        ],
    )


    trainer = None
    last_err: Exception | None = None
    for tok_kwarg in ("processing_class", "tokenizer"):

        attempt_kwargs = {tok_kwarg: tokenizer}
        if ta_meta["kind"] == "training_args":
            attempt_kwargs["max_seq_length"] = max_seq_length
        try:
            trainer = SFTTrainer(**sft_kwargs, **attempt_kwargs)
            print(f"      SFTTrainer accepted `{tok_kwarg}`.")
            break
        except TypeError as exc:
            last_err = exc
            continue
    if trainer is None:
        raise RuntimeError(
            "Could not initialize SFTTrainer with either `processing_class` "
            f"or `tokenizer`. Last error: {last_err}"
        )

    print("\n" + "=" * 80)
    print(
        f"Training: {num_epochs} epoch(s), eff. batch="
        f"{per_device_batch_size * grad_accum}, LR={learning_rate}"
    )
    print("=" * 80 + "\n")

    try:
        trainer.train()
        print("\n[train] done.")
    except KeyboardInterrupt:
        print("\n[train] interrupted by user. Saving current state.")
    except Exception:
        print("\n[train] failed.")
        raise

    print(f"\n[7/7] Saving LoRA adapter to {final_model_dir}")
    final_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    with open(final_model_dir / "training_config.json", "w") as f:
        json.dump(
            {
                "base_model": base_model,
                "lora_r": lora_r,
                "lora_alpha": lora_alpha,
                "lora_dropout": lora_dropout,
                "learning_rate": learning_rate,
                "per_device_batch_size": per_device_batch_size,
                "grad_accum": grad_accum,
                "num_epochs": num_epochs,
                "max_seq_length": max_seq_length,
                "warmup_steps": warmup_steps,
                "seed": seed,
                "dataset_dir": str(dataset_dir),
            },
            f,
            indent=2,
        )
    print(f"[done] adapter at {final_model_dir}")
    print(
        "Next step: merge with the base model and bake thinking-on:\n"
        f"  python -m fourneurons.scripts.merge_lora "
        f"--adapter_dir {final_model_dir} --output_dir <vllm_dir>"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--dataset_dir", type=Path, required=True,
                   help="HF DatasetDict saved by build_train (must have train/ and test/).")
    p.add_argument("--output_dir", type=Path, required=True,
                   help="Where TrainingArguments writes checkpoints / logs.")
    p.add_argument("--final_model_dir", type=Path, required=True,
                   help="Directory in which to save the final LoRA adapter.")
    p.add_argument("--base_model", default=BASE_MODEL)
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--per_device_batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--max_seq_length", type=int, default=2048)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=200)
    p.add_argument("--logging_steps", type=int, default=20)
    p.add_argument("--early_stop_patience", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true",
                   help="Use bf16 instead of fp16 (recommended on A100/H100).")
    p.add_argument("--optim", default="adamw_torch_fused",
                   help=(
                       "Optimizer for HF Trainer. Defaults to "
                       "`adamw_torch_fused` (pure-CUDA, no bitsandbytes). "
                       "Use `paged_adamw_8bit` only if bnb is correctly "
                       "installed (it isn't, on this pod)."
                   ))
    args = p.parse_args(argv)

    train_model(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        final_model_dir=args.final_model_dir,
        base_model=args.base_model,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        per_device_batch_size=args.per_device_batch_size,
        grad_accum=args.grad_accum,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_seq_length=args.max_seq_length,
        warmup_steps=args.warmup_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        early_stop_patience=args.early_stop_patience,
        seed=args.seed,
        bf16=args.bf16,
        optim=args.optim,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
