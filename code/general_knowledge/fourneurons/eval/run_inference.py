from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable


def _read_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {e}")
    return items


def _resolve_sampling_params(
    model_path: str,
    n: int,
    max_tokens: int,
    *,
    temperature_override: float | None = None,
    top_p_override: float | None = None,
    top_k_override: int | None = None,
):

    from vllm import SamplingParams  # local import to keep CPU-only paths cheap

    temperature = 0.7
    top_p = 0.9
    top_k = -1

    gen_cfg_path = os.path.join(model_path, "generation_config.json")
    if os.path.isfile(gen_cfg_path):
        with open(gen_cfg_path) as f:
            cfg = json.load(f)
        temperature = float(cfg.get("temperature", temperature))
        top_p = float(cfg.get("top_p", top_p))
        top_k = int(cfg.get("top_k", top_k))

    if temperature_override is not None:
        temperature = temperature_override
    if top_p_override is not None:
        top_p = top_p_override
    if top_k_override is not None:
        top_k = top_k_override

    return SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        seed=42,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        required=True,
        help="HF model id or local path (e.g. Qwen/Qwen3-1.7B or ./final_gk_model_vllm).",
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input JSONL with {prompt, answer, ...} per line.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output JSONL with `completions` appended to each row.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Number of sampled completions per problem. n=1 for pass@1, "
        "n=8 to also report pass@8.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=4096,
        help="Max new tokens per completion. Matches the CI cap (max_model_len=4096).",
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=4096,
        help="vLLM max_model_len. Should mirror the CI value.",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        help="vLLM dtype (matches the CI's FP16 inference).",
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.90,
        help="vLLM GPU memory utilization fraction.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of problems (debugging).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Override sampling temperature (use 0.0 for greedy decoding). "
        "Defaults to whatever the model's generation_config.json says.",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=None,
        help="Override top_p. Defaults to generation_config.json value.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Override top_k. Defaults to generation_config.json value.",
    )
    args = parser.parse_args(argv)

    items = _read_jsonl(args.input)
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        raise SystemExit(f"No examples found in {args.input}.")

    from transformers import AutoTokenizer
    from vllm import LLM

    print(f"[infer] loading tokenizer from {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"[infer] loading model into vLLM (dtype={args.dtype}, "
          f"max_model_len={args.max_model_len})...")
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )

    sampling_params = _resolve_sampling_params(
        args.model,
        n=args.n,
        max_tokens=args.max_tokens,
        temperature_override=args.temperature,
        top_p_override=args.top_p,
        top_k_override=args.top_k,
    )
    print(
        f"[infer] sampling: n={args.n}, T={sampling_params.temperature}, "
        f"top_p={sampling_params.top_p}, top_k={sampling_params.top_k}, "
        f"max_tokens={args.max_tokens}"
    )

    prompts: list[str] = []
    for it in items:
        messages = [{"role": "user", "content": it["prompt"]}]
        prompts.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )

    print(f"[infer] generating {len(prompts)} prompts × n={args.n}...")
    outputs = llm.generate(prompts, sampling_params)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as fout:
        for it, out in zip(items, outputs):
            row = dict(it)
            row["completions"] = [o.text for o in out.outputs]
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[infer] wrote {len(items)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
