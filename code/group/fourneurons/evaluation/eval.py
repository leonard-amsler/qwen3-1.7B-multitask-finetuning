import os
import json
import argparse

from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from fourneurons.prompts.prompt_loader import load_prompt


SNAPSHOT_DIR = "/scratch/hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots/"
BASE_MODEL = SNAPSHOT_DIR + os.listdir(SNAPSHOT_DIR)[0]


def main(
    benchmark,
    dataset,
    dataset_split,
    checkpoint_dir,
    run_name,
    base_only=False,
    merged=False,
    prompt_file_path=None,
    n=1,
):
    """
    Evaluate a LoRA checkpoint on a specified benchmark and dataset split.
    """
    output_dir = f"/scratch/results/{benchmark}/{dataset}/{run_name}"

    print(f"Evaluating on {benchmark} - {dataset} ({dataset_split} split)")
    print(
        f"Model checkpoint: {checkpoint_dir}"
        if checkpoint_dir
        else "Base model evaluation (no LoRA checkpoint provided)"
    )
    print(
        f"System prompt file: {prompt_file_path}"
        if prompt_file_path
        else "No system prompt file provided"
    )

    if prompt_file_path:
        system_prompt = load_prompt(prompt_file_path, verbose=True)
    else:
        system_prompt = None

    if base_only:
        print("Evaluating base model (no LoRA)...")
        model_dir = BASE_MODEL
    elif merged is True:
        print("Using already merged model...")
        model_dir = checkpoint_dir
    else:
        model_dir = os.path.join(output_dir, "merged")
        if not os.path.exists(model_dir):
            print("Merging LoRA weights...")
            base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype="bfloat16")
            model = PeftModel.from_pretrained(base, checkpoint_dir)
            model.merge_and_unload().save_pretrained(model_dir)
            AutoTokenizer.from_pretrained(checkpoint_dir).save_pretrained(model_dir)
            print(f"Merged model saved to {model_dir}")

    llm = LLM(model=model_dir, tokenizer=model_dir, seed=42)
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=16384,
        seed=42,
        n=n,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    val_file = f"/scratch/data/{benchmark}/{dataset}/splits/{dataset}_{dataset_split}.jsonl"
    if not os.path.exists(val_file):
        if not os.path.exists(f"/scratch/data/{benchmark}"):
            raise FileNotFoundError(f"Benchmark not found: /scratch/data/{benchmark}/")
        elif not os.path.exists(f"/scratch/data/{benchmark}/{dataset}"):
            raise FileNotFoundError(
                f"Dataset not found: /scratch/data/{benchmark}/{dataset}/. "
                f"The available datasets for {benchmark} are: {os.listdir(f'/scratch/data/{benchmark}/')}"
            )
        else:
            raise FileNotFoundError(
                f"Validation split not found: {val_file}. "
                f"The available splits in /scratch/data/{benchmark}/{dataset}/splits/ are: "
                f"{os.listdir(f'/scratch/data/{benchmark}/{dataset}/splits/')}"
            )

    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/{dataset_split}_gens.jsonl"

    with open(val_file) as fin, open(output_file, "w") as fout:
        rows = []
        prompts = []

        for line in fin:
            row = json.loads(line)
            rows.append(row)

            messages = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})

            messages.append({"role": "user", "content": row["prompt"]})

            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(prompt)

        outputs = llm.generate(prompts, sampling_params)

        for row, output in zip(rows, outputs):
            completions = [o.text for o in output.outputs]
            row["completions"] = completions
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Generations saved to {output_file}")

    score_output_file = f"{output_dir}/{dataset_split}_scored.json"
    print("Now score with:")
    print(
        f"\npython -m evaluate.score --generations {output_file} "
        f"--benchmark {benchmark} --output {score_output_file}"
    )
    print("\nOr with W&B logging:")
    print(
        f"\npython -m evaluate.score_wandb --generations {output_file} "
        f"--benchmark {benchmark} --output {score_output_file} "
        f"--run_name {run_name}_scoring"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "benchmark",
        help="Benchmark name (options: safety, multilingual, knowledge, math, group)",
    )
    parser.add_argument(
        "dataset",
        help="Dataset name (examples: safetybench, mmlu)",
    )
    parser.add_argument(
        "split",
        help="Dataset split to evaluate (e.g. val, test, default: val)",
    )
    parser.add_argument(
        "run_name",
        help="Name for this evaluation run (used for output directory)",
    )
    parser.add_argument(
        "--checkpoint",
        required=False,
        default=None,
        help="Path to LoRA checkpoint dir. Required unless --base is set.",
    )
    parser.add_argument(
        "--base",
        action="store_true",
        help="Evaluate base model without LoRA",
    )
    parser.add_argument(
        "--merged",
        action="store_true",
        help="Use already merged model instead of merging on the fly",
    )
    parser.add_argument(
        "--prompt_file_path",
        required=False,
        default=None,
        help="Path to the file containing the system prompt for evaluation.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of completions to generate per problem (default: 1)",
    )
    args = parser.parse_args()

    if args.benchmark not in ["safety", "multilingual", "knowledge", "math", "group"]:
        parser.error(
            f"Invalid benchmark {args.benchmark}. "
            "Valid options: safety, multilingual, knowledge, math, group"
        )

    if not args.base and args.checkpoint is None:
        parser.error("--checkpoint is required unless --base is set")

    main(
        args.benchmark,
        args.dataset,
        args.split,
        args.checkpoint,
        args.run_name,
        args.base,
        args.merged,
        args.prompt_file_path,
        args.n,
    )