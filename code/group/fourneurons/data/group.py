import gc
import json
import random
import shutil
import tempfile
from pathlib import Path
import argparse

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from vllm import LLM, SamplingParams
import optuna

from fourneurons.evaluation.score import score_generations

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_MODEL = "/scratch/checkpoints/multilingual/base_patched"

MULTILINGUAL_DATA_PATH = "/scratch/data/multilingual/mmmlu/splits/mmmlu_test.jsonl"
MATH_DATA_PATH         = "/scratch/data/math/competitionmath/splits/competitionmath_full.jsonl"
SAFETY_DATA_PATH       = "/scratch/data/safety/safetybench/splits/safetybench_val.jsonl"
GK_DATA_PATH           = "/scratch/noe/standard-project-m2-4neurons/validation_samples/general_knowledge_dev_full.jsonl"

MULTILINGUAL_TRAIN_DATA_PATH = "/scratch/data/multilingual/mmmlu_more_qcms/splits/mmmlu_more_qcms_train.jsonl"
MATH_DATA_TRAIN_PATH         = ["/scratch/data/math/openR1math/splits/openR1math_train.jsonl", "/scratch/data/math/openmathinstruct/splits/openmathinstruct_train.jsonl"]
SAFETY_DATA_TRAIN_PATH       = "/scratch/data/safety/safetybench/cot/safetybench_train_cot.jsonl"
GK_DATA_TRAIN_PATH           = "/scratch/data/train_v9b.jsonl"

DATASET_MAP = {
    MULTILINGUAL_DATA_PATH: "multilingual",
    MATH_DATA_PATH:         "math",
    SAFETY_DATA_PATH:       "safety",
    GK_DATA_PATH:           "general_knowledge",
}

DATASET_TRAIN_MAP = {
    MULTILINGUAL_TRAIN_DATA_PATH: "multilingual",
    MATH_DATA_TRAIN_PATH[0]:         "math",
    MATH_DATA_TRAIN_PATH[1]:         "math",
    SAFETY_DATA_TRAIN_PATH:       "safety",
    GK_DATA_TRAIN_PATH:           "general_knowledge",
}


ADAPTERS = {
    "multilingual": "/scratch/checkpoints/multilingual/mmmlu_sft3_long/checkpoint-3750",
    "math":         "/scratch/checkpoints/math/20260526-161659/checkpoint-6250",
    "safety":       "/scratch/checkpoints/safety/20260518-215854/final",
    "general":      "/scratch/checkpoints/gk_v1/adapter",
}

MIXED_DATA_PATH       = "/scratch/data/group/mixed/splits/mixed_test.jsonl"
SHORT_MIXED_DATA_PATH = "/scratch/data/group/mixed/splits/mixed_quicktest.jsonl"

# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------
def build_eval_dataset(n_total: int, split_name: str) -> None:
    """Sample evenly from each task dataset and write a mixed split."""
    random.seed(42)
    out_path = Path(f"/scratch/data/group/mixed/splits/mixed_{split_name}.jsonl")

    samples = []
    for path, dataset_name in DATASET_MAP.items():
        with open(path) as f:
            data = [json.loads(line) for line in f]
        for sample in random.sample(data, n_total // 4):
            samples.append({
                "prompt":  sample["prompt"],
                "answer":  sample["answer"],
                "dataset": dataset_name,
            })

    random.shuffle(samples)

    for sample in samples:
        for col in ("prompt", "answer"):
            if col not in sample:
                raise ValueError(f"Sample missing '{col}': {sample}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"Built {split_name} split: {len(samples)} samples → {out_path}")

def build_train_dataset():
    """Build a training dataset with the same format as the eval mixed dataset."""
    random.seed(42)

    samples = []
    for path in [MULTILINGUAL_TRAIN_DATA_PATH] + MATH_DATA_TRAIN_PATH + [SAFETY_DATA_TRAIN_PATH, GK_DATA_TRAIN_PATH]:
        with open(path) as f:
            dataset_data = [json.loads(line) for line in f]
            for sample in dataset_data:
                dataset_name = DATASET_TRAIN_MAP[path]
                sample["dataset"] = dataset_name
                if dataset_name == "general_knowledge":
                    sample["prompt"] = sample["messages"][0]["content"]
                    sample["completion"] = sample["messages"][1]["content"]
                    if len(sample["messages"]) > 2:
                        raise ValueError(f"Unexpected message format in general knowledge sample: {sample}")
                    del sample["messages"]
        print(f"Loaded {len(dataset_data)} samples from {path}")
        samples.extend(random.sample(dataset_data, 3000 if dataset_name != "math" else 1500))
    random.shuffle(samples)

    out_path = Path("/scratch/data/group/mixed/splits/mixed_train.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"Built training dataset: {len(samples)} samples → {out_path}")

# ---------------------------------------------------------------------------
# Merge adapters to a temp dir on CPU (no GPU memory used)
# ---------------------------------------------------------------------------
def merge_to_tmpdir(params: dict) -> str:
    """
    Load base + adapters on CPU, merge with given weights, save to a temp
    directory, and return its path. Caller is responsible for cleanup.
    CPU-only so this never touches GPU memory.
    """
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16,
        device_map="cpu",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    peft_model = PeftModel.from_pretrained(base, ADAPTERS["multilingual"], adapter_name="multilingual")
    peft_model.load_adapter(ADAPTERS["safety"],  adapter_name="safety")
    peft_model.load_adapter(ADAPTERS["general"], adapter_name="general")
    peft_model.load_adapter(ADAPTERS["math"],    adapter_name="math")

    peft_model.add_weighted_adapter(
        adapters=["multilingual", "safety", "general", "math"],
        weights=[params["multilingual"], params["safety"], params["general"], params["math"]],
        combination_type="dare_ties",
        density=params["density"],
        majority_sign_method="frequency",
        adapter_name="merged",
    )
    peft_model.set_adapter("merged")

    merged = peft_model.merge_and_unload()

    tmp_dir = tempfile.mkdtemp(prefix="lora_trial_")
    merged.save_pretrained(tmp_dir)
    tokenizer.save_pretrained(tmp_dir)

    # Free CPU memory before vLLM loads onto GPU
    del merged, peft_model, base
    gc.collect()

    return tmp_dir

# ---------------------------------------------------------------------------
# Evaluate a merged checkpoint with vLLM
# ---------------------------------------------------------------------------
def evaluate_with_vllm(model_dir: str, n: int, trial=None) -> float:
    """
    Spin up a vLLM engine on the merged checkpoint, generate, score, then
    destroy the engine and free GPU memory before returning.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    with open(SHORT_MIXED_DATA_PATH) as f:
        rows = [json.loads(line) for line in f]

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": r["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for r in rows
    ]

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=512,
        seed=42,
        n=n,
    )

    llm = LLM(
        model=model_dir,
        dtype="float16",
        gpu_memory_utilization=0.90,  # leave headroom for merge overhead
        max_model_len=4096,           # cap context to save KV cache memory
    )

    outputs = llm.generate(prompts, sampling_params)

    for row, output in zip(rows, outputs):
        row["completions"] = [o.text for o in output.outputs]

    results = score_generations(rows, "boxed")
    metrics = results["metrics"]

    score = metrics["pass@1"]
    print(f"  pass@1={metrics['pass@1']:.4f}  score={score:.4f}")

    if trial is not None and "per_dataset" in metrics:
        for task, task_score in metrics["per_dataset"].items():
            trial.set_user_attr(f"score_{task}", task_score)

    # Destroy vLLM engine and release GPU memory before next trial
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    return score

# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------
def optuna_objective(trial: optuna.Trial) -> float:
    params = {
        "multilingual": trial.suggest_float("multilingual", 0.5, 2.0),
        "safety":       trial.suggest_float("safety",       0.5, 2.0),
        "general":      trial.suggest_float("general",      0.5, 2.0),
        "math":         trial.suggest_float("math",         0.5, 2.0),
        "density":      trial.suggest_float("density",      0.5, 0.9),
    }

    tmp_dir = merge_to_tmpdir(params)
    try:
        score = evaluate_with_vllm(tmp_dir, n=1, trial=trial)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)  # always clean up

    return score

# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------
def make_json_callback(path: str):
    json_path = Path(path)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    def callback(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        record = {
            "trial":  trial.number,
            "value":  trial.value,
            "params": trial.params,
            "attrs":  trial.user_attrs,
        }
        with open(json_path, "a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"  → trial {trial.number} saved to {json_path}")

    return callback


def load_completed_trials(path: str) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Study runner
# ---------------------------------------------------------------------------
def run_study(n_trials: int = 50, study_log: str = "lora_merge_study.jsonl") -> optuna.Study:
    completed = load_completed_trials(study_log)
    remaining = n_trials - len(completed)

    if completed:
        print(f"Resuming: {len(completed)} trials already done, {remaining} remaining.")

    DISTRIBUTIONS = {
        "multilingual": optuna.distributions.FloatDistribution(0.5, 2.0),
        "safety":       optuna.distributions.FloatDistribution(0.5, 2.0),
        "general":      optuna.distributions.FloatDistribution(0.5, 2.0),
        "math":         optuna.distributions.FloatDistribution(0.5, 2.0),
        "density":      optuna.distributions.FloatDistribution(0.5, 0.9),
    }

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    # Re-add completed trials so TPE samples from full history
    for t in completed:
        study.add_trial(optuna.trial.create_trial(
            params=t["params"],
            distributions=DISTRIBUTIONS,
            value=t["value"],
        ))

    if remaining > 0:
        study.optimize(
            optuna_objective,
            n_trials=remaining,
            callbacks=[make_json_callback(study_log)],
        )

    return study

# ---------------------------------------------------------------------------
# Save best merged adapter
# ---------------------------------------------------------------------------
def save_best_adapter(study: optuna.Study, output_dir: str) -> None:
    best = study.best_trial
    print(f"\nBest trial #{best.number}  score={best.value:.4f}")
    print(f"Best params: {best.params}")

    tmp_dir = merge_to_tmpdir(best.params)
    try:
        shutil.copytree(tmp_dir, output_dir, dirs_exist_ok=True)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"Saved merged model → {output_dir}")

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build_data", action="store_true",
                        help="Rebuild mixed splits before running")
    parser.add_argument("--n_trials",   type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="/scratch/checkpoints/merged_best")
    parser.add_argument("--study_log",  type=str, default="/scratch/results/group/lora_merge_study.jsonl",
                        help="JSONL file for trial persistence — safe to resume from")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.study_log).parent.mkdir(parents=True, exist_ok=True)

    if args.build_data:
        build_train_dataset()
        build_eval_dataset(n_total=1500, split_name="test")
        build_eval_dataset(n_total=300,  split_name="quicktest")

    print("Starting Optuna study...")
    #study = run_study(n_trials=args.n_trials, study_log=args.study_log)

    #save_best_adapter(study, output_dir=args.output_dir)