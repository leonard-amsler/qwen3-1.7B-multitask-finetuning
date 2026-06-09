from transformers import AutoTokenizer
import argparse
import json
import os
from pathlib import Path


def load_prompt(prompt_file_path: str) -> str:
    with open(prompt_file_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def patch_chat_template(checkpoint_dir: str, prompt_file_path: str, thinking: bool = False, output_dir: str = None):
    """
    Load tokenizer from checkpoint, inject system prompt as default,
    force thinking OFF, and save back.
    """
    if not Path(checkpoint_dir).exists():
        print(f"Error: Checkpoint directory {checkpoint_dir} does not exist.")
        return
    
    tok = AutoTokenizer.from_pretrained(checkpoint_dir)

    system_prompt = load_prompt(prompt_file_path)

    if prompt_file_path.strip('.txt').endswith("_nothink") and thinking:
        raise ValueError("Prompt name suggests thinking OFF, but thinking=True was passed. Please check your arguments.")
    if prompt_file_path.strip('.txt').endswith("_think") and not thinking:
        raise ValueError("Prompt name suggests thinking ON, but thinking=False was passed. Please check your arguments.")

    # Inject default system prompt and force the selected Qwen thinking mode.
    original = tok.chat_template
    encoded_prompt = json.dumps(system_prompt)
    patched = (
        f"{{%- set enable_thinking = {'true' if thinking else 'false'} %}}\n"
        f"{{%- set default_system_prompt = {encoded_prompt} %}}\n"
        "{%- if messages[0]['role'] != 'system' %}\n"
        "{%- set messages = [{'role': 'system', 'content': default_system_prompt}] + messages %}\n"
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
    parser.add_argument("--thinking", action="store_true", help="Whether to set thinking ON or OFF in the template. Default is OFF.")
    parser.add_argument("--output_dir", type=str, default=None, help="Optional directory to save the patched tokenizer. If not provided, it will overwrite the original checkpoint.")
    
    args = parser.parse_args()

    patch_chat_template(args.checkpoint_dir, args.prompt_file_path, thinking=args.thinking, output_dir=args.output_dir)