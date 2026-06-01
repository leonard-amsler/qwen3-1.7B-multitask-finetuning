from transformers import AutoTokenizer
import argparse
import os
from typing import Optional
from pathlib import Path

from fourneurons.prompts.prompt_loader import load_prompt

def patch_chat_template(checkpoint_dir: str, prompt_file_path: str, output_dir: Optional[str] = None):
    """
    Load tokenizer from checkpoint, inject system prompt as default,
    force thinking OFF, and save back.
    """
    if not Path(checkpoint_dir).exists():
        print(f"Error: Checkpoint directory {checkpoint_dir} does not exist.")
        return
    
    tok = AutoTokenizer.from_pretrained(checkpoint_dir)

    system_prompt = load_prompt(prompt_file_path)

    # Inject default system prompt + thinking OFF into the template
    original = tok.chat_template
    patched = (
        f"{{%- set enable_thinking = true %}}\n"
        "{%- if messages[0]['role'] != 'system' %}\n"
        f"{{% set messages = [{{\"role\": \"system\", \"content\": \"{system_prompt}\"}}] + messages %}}\n"
        "{%- endif %}\n"
        + original
    )
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    tok.chat_template = patched
    tok.save_pretrained(output_dir or checkpoint_dir)
    print(f"Patched chat template saved to {output_dir or checkpoint_dir}")

    # Verify
    print("\nVerification:")
    print(tok.apply_chat_template(
        [{"role": "user", "content": "Is this safe?\n\nA) Yes.\nB) No."}],
        tokenize=False,
        add_generation_prompt=True
    ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch chat template in a LoRA checkpoint.")
    
    parser.add_argument("checkpoint_dir", type=str, help="Path to the LoRA checkpoint directory.")
    parser.add_argument("prompt_file_path", type=str, help="Path to the system prompt text file to inject into the chat template.")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional directory to save the patched tokenizer. If not provided, it will overwrite the original checkpoint.")
    
    args = parser.parse_args()

    patch_chat_template(args.checkpoint_dir, args.prompt_file_path, output_dir=args.output_dir)