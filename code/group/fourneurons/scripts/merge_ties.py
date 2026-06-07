"""
TIES (TrIm, Elect, Merge) — Yadav et al. 2024
Prunes redundant weights, resolves sign conflicts, averages sign-consistent params.

Usage:
    python -m fourneurons.scripts.merge_ties \\
        --adapters /path/to/safety/adapter /path/to/math/adapter /path/to/multilingual/adapter /path/to/gk/adapter
"""
import torch
import argparse
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "/scratch/hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"

# Default adapter paths (from the report's experiments)
DEFAULT_ADAPTERS = {
    "safety": "/scratch/checkpoints/safety/20260518-215854/final",
    "math": "/scratch/checkpoints/math/qwen3-1.7b-lora-math-rl_20260602-064411/checkpoint-200/",
    "multilingual": "/scratch/checkpoints/multilingual/mmmlu_sft3_long2/checkpoint-6875",
    "general_knowledge": "/scratch/checkpoints/gk_v11b/adapter",
}

DENSITY = 0.7   # fraction of weights to keep per adapter (0.0–1.0)

def main(adapters=None, output_path=None):
    if adapters is None:
        adapters = DEFAULT_ADAPTERS
    
    if output_path is None:
        output_path = "/scratch/checkpoints/group_model/ties_merged"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    weights = [1.0] * len(adapters)   # equal contribution per domain

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    print("Loading adapters...")
    model = PeftModel.from_pretrained(model, list(adapters.values())[0], adapter_name=list(adapters.keys())[0])
    for name, path in list(adapters.items())[1:]:
        model.load_adapter(path, adapter_name=name)

    print("Merging with TIES...")
    model.add_weighted_adapter(
        adapters = list(adapters.keys()),
        weights = weights,
        adapter_name = "ties_merged",
        combination_type = "ties",
        density = DENSITY,
    )
    model.set_adapter("ties_merged")

    print("Saving merged model...")
    merged = model.merge_and_unload()
    merged.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Done → {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge LoRA adapters using TIES method")
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
        help="Output path for merged model (default: /scratch/checkpoints/group_model/ties_merged)"
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
    
    main(adapters=adapters, output_path=args.output)