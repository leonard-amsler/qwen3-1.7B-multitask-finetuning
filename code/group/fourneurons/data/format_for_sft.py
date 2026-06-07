import os
from pathlib import Path

from transformers import AutoTokenizer

from fourneurons.data.safetybench_data_loader import load_safetybench_test
from prompts.prompt_loader import load_prompt

def format_for_sft(sample, tokenizer, prompt_file_path=None, verbose=False):

    if "completion" in sample:
        if verbose:
            print(f"\nSample already contains 'completion'. Using it directly for assistant content.")
        assistant_content = sample["completion"]
    else: # for raw examples without CoT
        if verbose:
            print(f"\nSample does not contain 'completion'. Using answer field to construct simple assistant content.")
        assistant_content = f"The correct answer is \\boxed{{{sample['answer']}}}"

    if prompt_file_path is not None:
        system_prompt = load_prompt(prompt_file_path, verbose=verbose)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": sample["prompt"]},
            {"role": "assistant", "content": assistant_content},
        ]
    else:
        messages = [
            {"role": "user", "content": sample["prompt"]},
            {"role": "assistant", "content": assistant_content},
        ]

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def get_tokenizer(
    model_cache="/scratch/hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/",
):
    snapshot = os.listdir(model_cache)[0]
    return AutoTokenizer.from_pretrained(model_cache + snapshot)


if __name__ == "__main__":
    prompt_file_path = "/scratch/nico/standard-project-m2-4neurons/prompts/sp_general_qcm_think.txt"

    tokenizer = get_tokenizer()

    # No-CoT format
    samples = load_safetybench_test()
    formatted = format_for_sft(samples[0], tokenizer, prompt_file_path=prompt_file_path, verbose=True)
    print("=== RAW FORMAT ===")
    print(formatted)

    # Synthetic CoT format
    synthetic_sample = {
        "prompt": samples[0]["prompt"],
        "answer": samples[0]["answer"],
        "category": samples[0]["category"],
        "completion": "<think>\nThis text is not offensive, so the answer is B.\n</think>\nThe correct answer is \\boxed{B}"
    }
    formatted_cot = format_for_sft(synthetic_sample, tokenizer, prompt_file_path=prompt_file_path, verbose=True)
    print("\n=== SYNTHETIC COT FORMAT ===")
    print(formatted_cot)
