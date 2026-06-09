import os
import json
import argparse
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SNAPSHOT_DIR = "/scratch/hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/"
BASE_MODEL = SNAPSHOT_DIR + os.listdir(SNAPSHOT_DIR)[0]


def count_requested_rows(val_file, max_num_samples=None):
    count = 0
    with open(val_file, encoding="utf-8") as fin:
        for i, _ in enumerate(fin):
            if max_num_samples and i >= max_num_samples:
                break
            count += 1
    return count


def is_complete_generation_row(row, expected_generations):
    completions = row.get("completions")
    return isinstance(completions, list) and len(completions) == expected_generations


def existing_complete_prefix(output_file, expected_generations):
    completed = 0
    last_good_offset = 0

    if not os.path.exists(output_file):
        return completed, last_good_offset, 0

    with open(output_file, "rb") as fin:
        while True:
            line = fin.readline()
            if not line:
                break
            try:
                row = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                break
            if not is_complete_generation_row(row, expected_generations):
                break
            completed += 1
            last_good_offset = fin.tell()

    return completed, last_good_offset, os.path.getsize(output_file)


def truncate_file(path, offset):
    with open(path, "ab") as fout:
        fout.truncate(offset)


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


def write_generation_batch(fout, rows, outputs):
    for row, output in zip(rows, outputs):
        completions = [o.text for o in output.outputs]
        row["completions"] = completions
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    fout.flush()
    os.fsync(fout.fileno())


def disable_bitsandbytes_lora_dispatch():
    import peft.tuners.lora.model as peft_lora_model

    peft_lora_model.is_bnb_available = lambda: False
    peft_lora_model.is_bnb_4bit_available = lambda: False


def load_prompt_file(prompt_file_path):
    with open(prompt_file_path, "r", encoding="utf-8") as f:
        return f.read()


def main(
    benchmark,
    dataset,
    dataset_split,
    checkpoint_dir,
    run_name,
    base_only=False,
    prompt_file_path=None,
    num_generations=1,
    max_num_samples=None,
    max_tokens=4096,
    temperature=0.7,
    top_p=0.9,
    top_k=None,
    merged_model_dir=None,
    generation_batch_size=None,
    resume_generation=False,
):
    """
    Evaluate a LoRA checkpoint on a specified benchmark and dataset split.
    """
    output_dir = f"/scratch/results/{benchmark}/{dataset}/{run_name}"

    print(f"Evaluating on {benchmark} - {dataset} ({dataset_split} split)")
    print(f"Model checkpoint: {checkpoint_dir}" if checkpoint_dir else "Base model evaluation (no LoRA checkpoint provided)")
    print(f"System prompt file: {prompt_file_path}" if prompt_file_path else "No system prompt file provided")

    if base_only:
        print("Evaluating base model (no LoRA)...")
        model_dir = BASE_MODEL
    else:
        # Merge LoRA into base
        model_dir = merged_model_dir or os.path.join(output_dir, "merged")
        if not os.path.exists(model_dir):
            print("Merging LoRA weights...")
            disable_bitsandbytes_lora_dispatch()
            base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.bfloat16)
            model = PeftModel.from_pretrained(base, checkpoint_dir)
            model.merge_and_unload().save_pretrained(model_dir)
            AutoTokenizer.from_pretrained(checkpoint_dir).save_pretrained(model_dir)
            print(f"Merged model saved to {model_dir}")

    llm = LLM(model=model_dir, tokenizer=model_dir, seed=42, dtype="bfloat16")
    sampling_kwargs = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "seed": 42,
        "n": num_generations,
    }
    if top_k is not None:
        sampling_kwargs["top_k"] = top_k
    sampling_params = SamplingParams(**sampling_kwargs)
    print(
        "Sampling: "
        f"temperature={temperature}, top_p={top_p}, "
        f"top_k={top_k if top_k is not None else 'default'}, "
        f"n={num_generations}, max_tokens={max_tokens}"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    system_prompt = load_prompt_file(prompt_file_path) if prompt_file_path else None

    val_file = (
        f"/scratch/data/{benchmark}/{dataset}/splits/{dataset}_{dataset_split}.jsonl"
    )
    if not os.path.exists(val_file):
        if not os.path.exists(f"/scratch/data/{benchmark}"):
            raise FileNotFoundError(f"Benchmark not found: /scratch/data/{benchmark}/")
        elif not os.path.exists(f"/scratch/data/{benchmark}/{dataset}"):
            raise FileNotFoundError(
                f"Dataset not found: /scratch/data/{benchmark}/{dataset}/. The available datasets for {benchmark} are: {os.listdir(f'/scratch/data/{benchmark}/')}"
            )
        else:
            raise FileNotFoundError(
                f"Validation split not found: {val_file}. The available splits in /scratch/data/{benchmark}/{dataset}/splits/ are: {os.listdir(f'/scratch/data/{benchmark}/{dataset}/splits/')}"
            )

    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{dataset_split}_gens.jsonl"
    total_requested = count_requested_rows(val_file, max_num_samples)
    completed_rows = 0
    file_mode = "w"

    if resume_generation:
        completed_rows, last_good_offset, output_size = existing_complete_prefix(
            output_file,
            num_generations,
        )
        if completed_rows > total_requested:
            raise ValueError(
                f"Existing generations contain {completed_rows} complete rows, "
                f"but this run only requested {total_requested}. Use a different "
                "run_name or remove the old generations file."
            )
        if output_size != last_good_offset:
            print(
                f"Truncating incomplete generations file from {output_size} to "
                f"{last_good_offset} bytes: {output_file}",
                flush=True,
            )
            truncate_file(output_file, last_good_offset)
        if completed_rows:
            print(
                f"Resuming generation from row {completed_rows} of {total_requested}",
                flush=True,
            )
        if completed_rows == total_requested:
            print(f"Generations already complete: {output_file}", flush=True)
            return
        file_mode = "a"

    with open(val_file, encoding="utf-8") as fin, open(
        output_file,
        file_mode,
        encoding="utf-8",
    ) as fout:
        with tqdm(
            total=total_requested,
            initial=completed_rows,
            desc="Generating samples",
            unit="sample",
            dynamic_ncols=True,
        ) as progress:
            batch_rows = []
            batch_prompts = []

            def run_batch():
                nonlocal completed_rows, batch_rows, batch_prompts
                if not batch_rows:
                    return
                batch_start = completed_rows
                batch_end = completed_rows + len(batch_rows)
                progress.set_postfix_str(
                    f"rows {batch_start}-{batch_end - 1}",
                    refresh=True,
                )
                outputs = llm.generate(batch_prompts, sampling_params, use_tqdm=False)
                write_generation_batch(fout, batch_rows, outputs)
                completed_rows = batch_end
                batch_rows = []
                batch_prompts = []
                progress.update(batch_end - batch_start)

            for i, line in enumerate(fin):
                if max_num_samples and i >= max_num_samples:
                    break
                if i < completed_rows:
                    continue

                row = json.loads(line)
                batch_rows.append(row)
                batch_prompts.append(build_prompt(tokenizer, system_prompt, row))

                if generation_batch_size and len(batch_rows) >= generation_batch_size:
                    run_batch()

            run_batch()

    print(f"Generations saved to {output_file}")

    score_output_file = f"{output_dir}/{dataset_split}_scored.json"
    print(f"Now score with:")
    print(f"\npython -m evaluate.score --generations {output_file} --benchmark {benchmark} --output {score_output_file}")
    print(f"\nOr with W&B logging:")
    print(f"\npython -m evaluate.score_wandb --generations {output_file} --benchmark {benchmark} --output {score_output_file} --run_name {run_name}_scoring")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "benchmark",
        help="Benchmark name (options: safety, multilingual, knowledge, math)",
    )
    parser.add_argument("dataset", help="Dataset name (examples: safetybench, mmlu)")
    parser.add_argument(
        "split", help="Dataset split to evaluate (e.g. val, test, default: val)"
    )
    parser.add_argument(
        "run_name", help="Name for this evaluation run (used for output directory)"
    )
    parser.add_argument(
        "--checkpoint",
        required=False,
        default=None,
        help="Path to LoRA checkpoint dir. Required unless --base is set.",
    )
    parser.add_argument(
        "--base", action="store_true", help="Evaluate base model without LoRA"
    )
    parser.add_argument(
        "--prompt_file_path",
        required=False,
        default=None,
        help="Path to the file containing the system prompt for evaluation. If not provided, the tokenizer's chat template will not be modified.",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        required=False,
        default=1,
        help="Number of generations to produce per prompt (default: 1)",
    )
    parser.add_argument(
        "--max_num_samples",
        type=int,
        required=False,
        default=None,
        help="Maximum number of samples to evaluate (default: all)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        required=False,
        default=4096,
        help="Maximum number of tokens to generate per prompt (default: 4096)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=False,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        required=False,
        default=0.9,
        help="Nucleus sampling top_p (default: 0.9)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        required=False,
        default=None,
        help="Sampling top_k. Omit to use vLLM's default.",
    )
    parser.add_argument(
        "--merged_model_dir",
        required=False,
        default=None,
        help=(
            "Optional shared path for the merged full model. Useful when running "
            "multiple decoding settings for the same LoRA checkpoint."
        ),
    )
    parser.add_argument(
        "--generation_batch_size",
        type=int,
        required=False,
        default=None,
        help=(
            "Number of prompts to generate per llm.generate call. Omit to "
            "generate all remaining prompts in one call."
        ),
    )
    parser.add_argument(
        "--resume_generation",
        action="store_true",
        help=(
            "Append to an existing complete prefix of <split>_gens.jsonl and "
            "continue from the first missing row."
        ),
    )
    args = parser.parse_args()

    if args.benchmark not in ["safety", "multilingual", "knowledge", "math"]:
        parser.error(
            f"Invalid benchmark {args.benchmark}. Valid options: safety, multilingual, knowledge, math"
        )

    if not args.base and args.checkpoint is None:
        parser.error("--checkpoint is required unless --base is set")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")
    if not (0 < args.top_p <= 1):
        parser.error("--top_p must be in (0, 1]")
    if args.top_k is not None and args.top_k <= 0:
        parser.error("--top_k must be positive when provided")
    if args.generation_batch_size is not None and args.generation_batch_size <= 0:
        parser.error("--generation_batch_size must be positive when provided")

    main(
        args.benchmark,
        args.dataset,
        args.split,
        args.checkpoint,
        args.run_name,
        args.base,
        args.prompt_file_path,
        args.num_generations,
        args.max_num_samples,
        args.max_tokens,
        args.temperature,
        args.top_p,
        args.top_k,
        args.merged_model_dir,
        args.generation_batch_size,
        args.resume_generation,
    )
