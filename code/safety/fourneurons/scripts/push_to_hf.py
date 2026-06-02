import os
import shutil
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

def main(hf_repo, checkpoint_dir):
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(checkpoint_dir, dtype="bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)

    print("\nVerification of input after chat template application:\n")
    print(tokenizer.apply_chat_template(
        [{"role": "user", "content": "Is this safe?\n\nA) Yes.\nB) No."}],
        tokenize=False,
        add_generation_prompt=True
    ))

    print(f"\nPushing to {hf_repo}...")
    model.push_to_hub(hf_repo)
    tokenizer.push_to_hub(hf_repo)

    GenerationConfig(
        bos_token_id=151643,
        do_sample=True,
        eos_token_id=[151645, 151643],
        pad_token_id=151643,
        temperature=0.7,
        top_k=20,
        top_p=0.9,
        transformers_version="4.51.0",
    ).push_to_hub(hf_repo)

    print(f"\nDone! https://huggingface.co/{hf_repo}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "hf_repo", 
        help="Hugging Face repository name (e.g., 'cs-552-2026-4neurons/<model_name>')",
    )
    parser.add_argument(
        "checkpoint_dir",
        help="Path to the LoRA checkpoint directory to push. If not provided, it will push the base model.",
    )
    args = parser.parse_args()
    main(args.hf_repo, args.checkpoint_dir)