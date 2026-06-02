from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


_THINKING_ON_PREAMBLE = (
    "{# Bake thinking mode = ON. The CI does not pass enable_thinking as a "
    "kwarg, so we force it at the top of the template. #}\n"
    "{%- set enable_thinking = true %}\n"
)


def _bake_thinking_on(tokenizer) -> None:
    tpl = tokenizer.chat_template
    if tpl is None:
        print("[warn] tokenizer has no chat_template; skipping.")
        return
    if "set enable_thinking = true" in tpl:
        print("[bake] chat template already forces thinking ON. Nothing to do.")
        return
    tokenizer.chat_template = _THINKING_ON_PREAMBLE + tpl
    print("[bake] thinking mode ON forced via chat template preamble.")


def _write_generation_config(
    output_dir: Path,
    base_gen_cfg_path: Path | None,
    temperature: float,
    top_p: float,
    top_k: int,
) -> None:

    cfg: dict = {}
    if base_gen_cfg_path is not None and base_gen_cfg_path.exists():
        with open(base_gen_cfg_path, "r") as f:
            cfg = json.load(f)
    else:
        cfg = {
            "bos_token_id": 151643,
            "eos_token_id": [151645, 151643],
            "pad_token_id": 151643,
            "do_sample": True,
            "transformers_version": "4.51.0",
        }
    cfg["do_sample"] = True
    cfg["temperature"] = temperature
    cfg["top_p"] = top_p
    cfg["top_k"] = top_k
    out_path = output_dir / "generation_config.json"
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[gen_cfg] wrote {out_path}: T={temperature} top_p={top_p} top_k={top_k}")


def merge(
    adapter_dir: Path,
    output_dir: Path,
    base_model: str,
    temperature: float,
    top_p: float,
    top_k: int,
    device: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading base model: {base_model}")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
    )

    print(f"[2/5] Loading LoRA adapter from {adapter_dir}")
    model = PeftModel.from_pretrained(base, str(adapter_dir))

    print("[3/5] Merging adapter into base weights")
    model = model.merge_and_unload()

    print(f"[4/5] Saving merged model to {output_dir}")
    model.save_pretrained(str(output_dir), safe_serialization=True)

    # The adapter dir holds the tokenizer we trained against (or its config).
    # If for some reason it doesn't, fall back to the base model's tokenizer.
    print("[5/5] Saving tokenizer with thinking-on chat template")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(adapter_dir), trust_remote_code=True
        )
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True
        )
    _bake_thinking_on(tokenizer)
    tokenizer.save_pretrained(str(output_dir))

    # Generation config — try to grab the base model's if present in the
    # local HF cache; otherwise emit a sensible default.
    base_gen_cfg_path = None
    try:
        from huggingface_hub import hf_hub_download

        base_gen_cfg_path = Path(
            hf_hub_download(repo_id=base_model, filename="generation_config.json")
        )
    except Exception:
        base_gen_cfg_path = None
    _write_generation_config(
        output_dir,
        base_gen_cfg_path=base_gen_cfg_path,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )

    print(f"\n[done] vLLM-ready checkpoint at: {output_dir}")
    print(
        "Next: smoke-test with the project's run_inference + dev set.\n"
        "Then push to HF Hub:\n"
        f"  huggingface-cli upload "
        f"cs-552-2026-<your-org>/general_knowledge_model {output_dir} ."
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--adapter_dir", type=Path, required=True,
                   help="Directory containing the LoRA adapter (output of train.py).")
    p.add_argument("--output_dir", type=Path, required=True,
                   help="Where to write the merged, vLLM-loadable checkpoint.")
    p.add_argument("--base_model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--device", default="cpu",
                   help="Device for the merge (CPU is fine and avoids OOM).")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=20)
    args = p.parse_args(argv)

    merge(
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
        base_model=args.base_model,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
