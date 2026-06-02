from __future__ import annotations

import argparse
import inspect
import json
import math
import random
import sys
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from trl import DPOConfig, DPOTrainer
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "trl with DPOTrainer is required. `pip install -U trl`."
    ) from exc


def _split_kwargs_for(cls, kwargs: dict) -> tuple[dict, dict]:

    try:
        sig = inspect.signature(cls.__init__)
    except (TypeError, ValueError):
        return {}, dict(kwargs)
    if any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    ):
        # `**kwargs` somewhere — assume the class accepts everything.
        return dict(kwargs), {}
    accepted_names = {
        name for name, p in sig.parameters.items() if name != "self"
    }
    accepted = {k: v for k, v in kwargs.items() if k in accepted_names}
    leftover = {k: v for k, v in kwargs.items() if k not in accepted_names}
    return accepted, leftover


def _load_pairs(path: Path) -> Dataset:

    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append(
                {
                    "prompt": row["prompt"],
                    "chosen": row["chosen"],
                    "rejected": row["rejected"],
                }
            )
    if not rows:
        raise SystemExit(f"No pairs found in {path}")
    print(f"[data] loaded {len(rows)} pairs from {path}")
    return Dataset.from_list(rows)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base_model", required=True, type=Path,
                   help="Path to the merged v2 checkpoint (`gk_v2/vllm`).")
    p.add_argument("--pairs", required=True, type=Path,
                   help="JSONL produced by build_dpo_pairs.py.")
    p.add_argument("--output_dir", required=True, type=Path,
                   help="Where TrainingArguments writes checkpoints / logs.")
    p.add_argument("--final_model_dir", required=True, type=Path,
                   help="Where to save the final DPO LoRA adapter.")
    p.add_argument("--num_epochs", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=5e-6,
                   help="DPO needs a much smaller LR than SFT (default 5e-6).")
    p.add_argument("--per_device_batch_size", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--beta", type=float, default=0.1,
                   help="DPO regularisation strength. Higher = closer to ref.")
    p.add_argument("--loss_type", default="sigmoid",
                   choices=["sigmoid", "hinge", "ipo"],
                   help="DPO loss flavour. `sigmoid` is the original Rafailov "
                   "formulation and the safest default.")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--max_prompt_length", type=int, default=1024)
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument("--logging_steps", type=int, default=10)
    p.add_argument("--eval_steps", type=int, default=100)
    p.add_argument("--save_steps", type=int, default=100)
    p.add_argument("--eval_fraction", type=float, default=0.05,
                   help="Fraction of pairs held out for eval.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bf16", action="store_true",
                   help="Use bf16 (recommended on A100).")
    p.add_argument("--optim", default="adamw_torch_fused",
                   help="Optimizer for HF Trainer (no bitsandbytes by default).")
    args = p.parse_args(argv)

    ds = _load_pairs(args.pairs)
    rng = random.Random(args.seed)
    indices = list(range(len(ds)))
    rng.shuffle(indices)
    n_eval = max(16, int(round(args.eval_fraction * len(ds))))
    eval_idx = set(indices[:n_eval])
    train_rows = [ds[i] for i in range(len(ds)) if i not in eval_idx]
    eval_rows = [ds[i] for i in range(len(ds)) if i in eval_idx]
    train_ds = Dataset.from_list(train_rows)
    eval_ds = Dataset.from_list(eval_rows)
    print(f"[data] split: train={len(train_ds)} eval={len(eval_ds)}")

    print(f"[tok] loading from {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(args.base_model), trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[model] loading base policy from {args.base_model}")
    dtype = torch.bfloat16 if args.bf16 else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        str(args.base_model),
        device_map="auto",
        dtype=dtype,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    common_kwargs = dict(
        output_dir=str(args.output_dir),
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.num_epochs,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=3,
        logging_steps=args.logging_steps,
        report_to=["tensorboard"],
        optim=args.optim,
        warmup_steps=args.warmup_steps,
        weight_decay=0.0,
        bf16=args.bf16,
        fp16=not args.bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=args.seed,
        remove_unused_columns=False,
    )

    dpo_specific = dict(
        beta=args.beta,
        loss_type=args.loss_type,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )

    cfg_accepted, leftover = _split_kwargs_for(DPOConfig, {**common_kwargs, **dpo_specific})
    config = DPOConfig(**cfg_accepted)
    if leftover:
        print(f"[trainer] DPOConfig didn't accept: {sorted(leftover)}; "
              "will try passing to DPOTrainer.")

    print("[trainer] building DPOTrainer")
    trainer_specific = leftover  # what DPOConfig refused, try here
    trainer = None
    last_err: Exception | None = None
    for tok_kwarg in ("processing_class", "tokenizer"):
        for trainer_kwargs in (trainer_specific, {}):
            try:
                trainer = DPOTrainer(
                    model=model,
                    ref_model=None,  # adapter-disabled base policy = ref
                    args=config,
                    train_dataset=train_ds,
                    eval_dataset=eval_ds,
                    peft_config=lora_config,
                    **{tok_kwarg: tokenizer},
                    **trainer_kwargs,
                )
                if trainer_kwargs:
                    print(f"[trainer] passed to DPOTrainer: {sorted(trainer_kwargs)}")
                else:
                    if trainer_specific:
                        print(f"[trainer] dropped (unsupported): "
                              f"{sorted(trainer_specific)}")
                print(f"[trainer] DPOTrainer accepted `{tok_kwarg}`.")
                break
            except TypeError as exc:
                last_err = exc
                continue
        if trainer is not None:
            break
    if trainer is None:
        raise SystemExit(
            "Could not initialize DPOTrainer with the available kwargs. "
            f"Last error: {last_err}"
        )

    print("\n" + "=" * 70)
    print(f"DPO LoRA  |  β={args.beta}  loss={args.loss_type}  "
          f"lr={args.learning_rate}  bs={args.per_device_batch_size}×"
          f"{args.grad_accum}  epochs={args.num_epochs}")
    print("=" * 70 + "\n")
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\n[train] interrupted; saving current state.")
    print("\n[train] done.")

    args.final_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(args.final_model_dir))
    tokenizer.save_pretrained(str(args.final_model_dir))
    with open(args.final_model_dir / "dpo_config.json", "w") as f:
        json.dump(
            {
                "base_model": str(args.base_model),
                "pairs": str(args.pairs),
                "beta": args.beta,
                "loss_type": args.loss_type,
                "lr": args.learning_rate,
                "num_epochs": args.num_epochs,
                "per_device_batch_size": args.per_device_batch_size,
                "grad_accum": args.grad_accum,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "max_length": args.max_length,
                "max_prompt_length": args.max_prompt_length,
                "n_train": len(train_ds),
                "n_eval": len(eval_ds),
                "seed": args.seed,
            },
            f,
            indent=2,
        )
    print(f"[done] adapter at {args.final_model_dir}")
    print(
        "Next: merge with the v2 base policy:\n"
        f"  python -m fourneurons.scripts.merge_lora "
        f"--adapter_dir {args.final_model_dir} "
        f"--output_dir /scratch/checkpoints/gk_v3/vllm "
        f"--base_model {args.base_model}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
