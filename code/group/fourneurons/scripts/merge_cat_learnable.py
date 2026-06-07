"""
Learnable CAT — LoRA Soups (COLING 2025)
Freezes all LoRA adapters, then trains one scalar alpha per adapter on a small calibration mix (5% of each domain's data, 1 epoch).

Usage:
    python -m fourneurons.scripts.merge_cat_learnable \\
        --adapters /path/to/safety/adapter /path/to/math/adapter /path/to/multilingual/adapter /path/to/gk/adapter
"""
import os
import json
import random
import argparse
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import PeftModel

BASE_MODEL = "/scratch/hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

# Default adapter paths (from the report's experiments)
DEFAULT_ADAPTERS = {
    "safety": "/scratch/checkpoints/safety/20260518-215854/final",
    "math": "/scratch/checkpoints/math/qwen3-1.7b-lora-math-rl_20260602-064411/checkpoint-200/",
    "multilingual": "/scratch/checkpoints/multilingual/mmmlu_sft3_long2/checkpoint-6875",
    "general_knowledge": "/scratch/checkpoints/gk_v11b/adapter",
}

# Calibration data: one .jsonl file per domain (with a "text" or "prompt"+"chosen" field)
DEFAULT_CALIB_DATA = {
    "safety": "/scratch/data/safety/safetybench/cot/safetybench_train_cot.jsonl",
    "math": [
        "/scratch/data/math/openR1math/splits/openR1math_train.jsonl",
        "/scratch/data/math/openmathinstruct/splits/openmathinstruct_train.jsonl",
    ],
    "multilingual": "/scratch/data/multilingual/mmmlu_more_qcms/splits/mmmlu_more_qcms_train.jsonl",
    "general_knowledge": "/scratch/data/train_v9b.jsonl",
}

CALIB_FRACTION = 0.05 # 5% of each dataset as per paper
MAX_PER_DOMAIN = 500 # cap each domain at 500 for balance and speed
MAX_LENGTH = 512
BATCH_SIZE = 2
GRAD_ACCUM = 8
LR = 5e-3
NUM_EPOCHS = 1



def load_calibration_texts(data_paths, fraction, max_per_domain=500):
    texts = []
    for domain, path in data_paths.items():
        paths = [path] if isinstance(path, str) else path
        lines = []
        for p in paths:
            with open(p) as f:
                lines += f.readlines()
        k = min(max_per_domain, max(1, int(len(lines) * fraction)))
        sampled = random.sample(lines, k)
        for line in sampled:
            row = json.loads(line)
            if "text" in row:
                texts.append(row["text"])
            elif "messages" in row:
                user_msgs = [m["content"] for m in row["messages"] if m["role"] == "user"]
                if user_msgs:
                    texts.append(user_msgs[0])
            elif "chosen" in row:
                texts.append(row["prompt"] + " " + row["chosen"])
            elif "prompt" in row:
                texts.append(row["prompt"])
        print(f"  {domain}: {k}/{len(lines)} examples loaded")
    random.shuffle(texts)
    return texts


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.encodings = tokenizer(
            texts, truncation=True, padding="max_length",
            max_length=max_length, return_tensors="pt"
        )

    def __len__(self):
        return self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


class LearnableCATModel(nn.Module):
    """
    Wraps a PeftModel that has multiple adapters loaded.
    Replaces the static equal weights with one trainable scalar per adapter per layer.
    All original parameters are frozen; only alphas are trained.
    """
    def __init__(self, peft_model, adapter_names):
        super().__init__()
        self.model = peft_model
        self.adapter_names = adapter_names

        # 4 simple scalars -> one per adapter, init to 1/N
        init = 1.0 / len(adapter_names)
        self.alphas = nn.ParameterList([
            nn.Parameter(torch.tensor(init))
            for _ in adapter_names
        ])

        # freeze everything except alphas
        for param in self.model.parameters():
            param.requires_grad = False

        print(f"Trainable params: {len(adapter_names)} alpha scalars")


    def _apply_alphas(self):
        pass  # handled in forward pass


    def forward(self, input_ids, attention_mask=None, labels=None):
        # Base model output (no adapters)
        self.model.disable_adapter_layers()
        base_out = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
        base_logits = base_out.logits

        # Weighted sum of per-adapter deltas
        delta_logits = torch.zeros_like(base_logits)
        self.model.enable_adapter_layers()
        for i, adapter in enumerate(self.adapter_names):
            self.model.set_adapter(adapter)
            out_i = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=None)
            delta_logits = delta_logits + self.alphas[i].abs() * (out_i.logits - base_logits)

        logits = base_logits + delta_logits

        # Cross-entropy loss with padding masked out
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        shift_labels[attention_mask[..., 1:] == 0] = -100  # mask padding tokens

        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        base_out.loss   = loss
        base_out.logits = logits
        return base_out


def main(adapters=None, calib_data=None, output_path=None):
    if adapters is None:
        adapters = DEFAULT_ADAPTERS
    if calib_data is None:
        calib_data = DEFAULT_CALIB_DATA
    if output_path is None:
        output_path = "/scratch/checkpoints/group_model/learnable_cat_5e3"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    random.seed(42)
    torch.manual_seed(42)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading calibration data...")
    texts = load_calibration_texts(calib_data, CALIB_FRACTION)
    print(f"Total calibration examples: {len(texts)}")

    dataset = TextDataset(texts, tokenizer, MAX_LENGTH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    print("Loading base model + adapters...")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(base, list(adapters.values())[0], adapter_name=list(adapters.keys())[0])
    for name, path in list(adapters.items())[1:]:
        model.load_adapter(path, adapter_name=name)

    cat_model = LearnableCATModel(model, list(adapters.keys()))
    device = "cuda"
    cat_model = cat_model.to(device)

    optimizer = torch.optim.AdamW(cat_model.alphas.parameters(), lr=LR)
    total_steps = (len(loader) // GRAD_ACCUM) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    print(f"Training alphas for {NUM_EPOCHS} epoch(s) on {device}...")
    cat_model.train()
    optimizer.zero_grad()

    for epoch in range(NUM_EPOCHS):
        for step, batch in enumerate(loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = input_ids.clone()

            out = cat_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss / GRAD_ACCUM
            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                if (step + 1) % (GRAD_ACCUM * 10) == 0:
                    learned = [f"{a.abs().item():.3f}" for a in cat_model.alphas]
                    print(f"  step {step+1}/{len(loader)} loss={loss.item()*GRAD_ACCUM:.4f} "
                          f"alphas(mean)={learned}")

    # final merge with learned alphas
    print("\nBuilding final merged model with learned alphas...")
    learned_weights = [a.abs().item() for a in cat_model.alphas]
    print(f"Final alpha means: { {k: f'{w:.4f}' for k, w in zip(adapters.keys(), learned_weights)} }")

    model.add_weighted_adapter(
        adapters = list(adapters.keys()),
        weights = learned_weights,
        adapter_name = "learnable_cat_final",
        combination_type = "cat",
    )
    model.set_adapter("learnable_cat_final")

    print(f"Saving to {output_path}...")
    merged = model.merge_and_unload()
    merged.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Done → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapters using Learnable CAT method")
    parser.add_argument(
        "--adapters",
        nargs=4,
        default=None,
        metavar=("SAFETY", "MATH", "MULTILINGUAL", "GK"),
        help="Paths to adapters: safety math multilingual general_knowledge (default: use report defaults)"
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output path for merged model (default: /scratch/checkpoints/group_model/learnable_cat_5e3)"
    )
    args = parser.parse_args()
    
    adapters = None
    if args.adapters:
        adapters = {
            "safety": args.adapters[0],
            "math": args.adapters[1],
            "multilingual": args.adapters[2],
            "general_knowledge": args.adapters[3],
        }
    
    main(adapters=adapters, calib_data=None, output_path=args.output)
