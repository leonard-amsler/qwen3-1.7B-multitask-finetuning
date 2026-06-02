from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator, Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fourneurons.data.schema import McqExample, stable_uid
from fourneurons.data.loaders import (
    load_boolq,
    load_commonsenseqa,
    load_ecqa,
    load_mmlu,
    load_mmlu_pro_cot,
    load_mmlu_world,
    load_socialiqa,
    load_triviaqa,
)
from fourneurons.distill.prompts import build_messages, clean_teacher_output
from fourneurons.distill.filters import quality_check



DISTILL_SOURCES = {
    "mmlu":          (load_mmlu,          {"split": "validation"}),
    "mmlu_world":    (load_mmlu_world,    {}),                       # v5: per-subject test+dev
    "triviaqa":      (load_triviaqa,      {"split": "train"}),       # v5: history_geo bulk
    "boolq":         (load_boolq,         {"split": "train"}),
    "socialiqa":     (load_socialiqa,     {"split": "train"}),
    "commonsenseqa": (load_commonsenseqa, {"split": "train"}),
    "mmlu_pro_cot":  (load_mmlu_pro_cot,  {"split": "train"}),
    "ecqa":          (load_ecqa,          {"split": "train"}),
}

COT_LESS_SOURCES = DISTILL_SOURCES

LONG_REASONING_SOURCES = frozenset({"mmlu", "mmlu_world", "mmlu_pro_cot"})

_MIN_CHARS_LONG = 1000
_MIN_CHARS_SHORT = 200  # v5 default, kept for non-deep sources


def _reasoning_style_for(source: str, override: Optional[str] = None) -> str:

    if override and override != "auto":
        return override
    return "long" if source in LONG_REASONING_SOURCES else "short"


def _min_chars_for(source: str, style: Optional[str] = None) -> int:
    if style in ("long", "contrastive"):
        return _MIN_CHARS_LONG
    if style == "short":
        return _MIN_CHARS_SHORT
    return _MIN_CHARS_LONG if source in LONG_REASONING_SOURCES else _MIN_CHARS_SHORT



def _read_cache(path: Path) -> set[str]:
    if not path.exists():
        return set()
    uids: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = row.get("uid")
            if uid:
                uids.add(uid)
    return uids


def _load_blocklist(paths: Iterable[Path]) -> set[str]:
    blocked: set[str] = set()
    for p in paths:
        if not p.exists():
            print(f"[blocklist] {p}: missing, skipping")
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prompt = row.get("prompt") or ""
                stem = prompt.split("\n\nChoices:")[0].split("\nChoices:")[0]
                blocked.add(stable_uid(stem))
    print(f"[blocklist] {len(blocked)} dev question hashes loaded.")
    return blocked

def _collect_targets(
    sources: list[str],
    blocklist: set[str],
    skip_uids: set[str],
    per_source_cap: Optional[int],
) -> list[McqExample]:
    out: list[McqExample] = []
    seen = set(blocklist) | set(skip_uids)
    for name in sources:
        if name not in DISTILL_SOURCES:
            print(f"[collect] unknown source {name}, skipping")
            continue
        loader, kw = DISTILL_SOURCES[name]
        print(f"[collect] {name} {kw}...")
        kept = 0
        for ex in loader(**kw):
            if ex.uid in seen:
                continue
            seen.add(ex.uid)
            out.append(ex)
            kept += 1
            if per_source_cap and kept >= per_source_cap:
                break
        print(f"[collect]   needed from {name}: {kept}")
    return out


def _build_prompts(
    examples: list[McqExample],
    tokenizer,
    enable_thinking: bool = False,
    style_override: Optional[str] = None,
) -> list[str]:
    prompts: list[str] = []
    for ex in examples:
        msgs = build_messages(
            question=ex.question,
            gold_text=ex.options[ex.gold_idx],
            style=_reasoning_style_for(ex.source, style_override),
        )
        try:
            text = tokenizer.apply_chat_template(
                msgs,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        prompts.append(text)
    return prompts


def _run_teacher(
    model_id: str,
    prompts: list[str],
    max_tokens: int,
    temperature: float,
    top_p: float,
    max_model_len: int,
    gpu_mem: float,
    quantization: Optional[str],
    seed: int,
) -> list[str]:
    from vllm import LLM, SamplingParams

    print(f"[teacher] loading vLLM model {model_id}...")
    llm_kwargs = dict(
        model=model_id,
        dtype="float16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem,
        trust_remote_code=True,
    )
    if quantization is not None:
        llm_kwargs["quantization"] = quantization

    t0 = time.time()
    llm = LLM(**llm_kwargs)
    print(f"[teacher] loaded in {time.time() - t0:.1f}s")

    sp = SamplingParams(
        n=1,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=seed,
    )

    print(f"[teacher] generating {len(prompts)} prompts (max_tokens={max_tokens})...")
    t0 = time.time()
    outs = llm.generate(prompts, sp)
    print(f"[teacher] generated in {time.time() - t0:.1f}s")

    return [o.outputs[0].text for o in outs]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--teacher",
        default="Qwen/Qwen3-14B-AWQ",
        help="HF id (or local path) of the teacher model.",
    )
    p.add_argument(
        "--quantization",
        default="awq_marlin",
        help="vLLM quantization backend. Try 'awq_marlin' (default) or 'awq'. "
        "Set to '' to disable (e.g. for an FP16 teacher).",
    )
    p.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Cache JSONL. Resumed automatically on re-run.",
    )
    p.add_argument(
        "--failed_log",
        type=Path,
        default=None,
        help="Optional sidecar JSONL for rejected outputs (defaults to "
        "<output>.failed.jsonl).",
    )
    p.add_argument(
        "--sources",
        nargs="+",
        default=list(DISTILL_SOURCES.keys()),
        help="Which sources to distill from. Defaults to everything in "
        "DISTILL_SOURCES (CoT-less + CoT-bearing).",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap on the total number of questions to distill this run. "
        "Useful for a pilot (e.g. --limit 100).",
    )
    p.add_argument(
        "--per_source_cap",
        type=int,
        default=None,
        help="Cap per source after dedup.",
    )
    p.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Cap on teacher new tokens per CoT. v5 used 1024 (3-5 sentence "
        "rationales fit easily). v6 bumps to 2048 so the long-reasoning "
        "prompts (10-20 sentences, 300-700 words) have headroom.",
    )
    p.add_argument(
        "--max_model_len",
        type=int,
        default=4096,
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=0.7,
    )
    p.add_argument(
        "--top_p",
        type=float,
        default=0.9,
    )
    p.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.90,
    )
    p.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Run the teacher in Qwen3 thinking mode. We strip the "
        "<think> block automatically. Slower; not usually needed.",
    )
    p.add_argument(
        "--reasoning_style",
        choices=("auto", "short", "long", "contrastive"),
        default="auto",
        help="Override the per-source reasoning style for the whole run. "
        "'auto' (default) = v6 source routing (long for STEM, short else). "
        "'contrastive' = v10 prompt (derive + refute tempting wrong answer, "
        "option-agnostic). Use with `--sources mmlu mmlu_pro_cot` and a "
        "fresh `--output` cache for the v10 STEM pass.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    p.add_argument(
        "--dev_blocklist",
        nargs="+",
        type=Path,
        default=[
            Path("validation_samples/general_knowledge.jsonl"),
            Path("validation_samples/general_knowledge_dev_small.jsonl"),
            Path("validation_samples/general_knowledge_dev_full.jsonl"),
            Path("validation_samples/ood_dev.jsonl"),  # v5 OOD set
        ],
    )
    args = p.parse_args(argv)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    failed_log = args.failed_log or args.output.with_suffix(".failed.jsonl")
    failed_log.parent.mkdir(parents=True, exist_ok=True)

    blocklist = _load_blocklist(args.dev_blocklist)
    existing = _read_cache(args.output)
    print(f"[cache] {len(existing)} uids already in {args.output}")

    targets = _collect_targets(
        sources=args.sources,
        blocklist=blocklist,
        skip_uids=existing,
        per_source_cap=args.per_source_cap,
    )
    if args.limit is not None:
        targets = targets[: args.limit]
    print(f"[collect] {len(targets)} new questions to distill.")
    if not targets:
        print("[collect] nothing to do; exiting.")
        return 0

    from transformers import AutoTokenizer

    print(f"[teacher] loading tokenizer for {args.teacher}...")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher, trust_remote_code=True)
    prompts = _build_prompts(
        targets,
        tokenizer,
        enable_thinking=args.enable_thinking,
        style_override=args.reasoning_style,
    )
    if args.reasoning_style != "auto":
        print(f"[prompts] reasoning_style override = {args.reasoning_style}")

    quant = args.quantization or None
    if quant == "":
        quant = None
    raw_outputs = _run_teacher(
        model_id=args.teacher,
        prompts=prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        max_model_len=args.max_model_len,
        gpu_mem=args.gpu_memory_utilization,
        quantization=quant,
        seed=args.seed,
    )

    n_ok = 0
    n_bad = 0
    bad_reasons: Counter[str] = Counter()

    with open(args.output, "a", encoding="utf-8") as fout, \
         open(failed_log, "a", encoding="utf-8") as ferr:
        for ex, raw in zip(targets, raw_outputs):
            cleaned = clean_teacher_output(raw)
            gold_text = ex.options[ex.gold_idx]
            ok, reason = quality_check(
                cleaned,
                gold_text,
                min_chars=_min_chars_for(
                    ex.source, _reasoning_style_for(ex.source, args.reasoning_style)
                ),
            )
            row = {
                "uid": ex.uid,
                "source": ex.source,
                "subject": ex.subject,
                "question": ex.question,
                "gold_text": gold_text,
                "cot": cleaned if ok else None,
                "raw": raw,
                "teacher": args.teacher,
            }
            if ok:
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                fout.flush()
                n_ok += 1
            else:
                row["reject_reason"] = reason
                ferr.write(json.dumps(row, ensure_ascii=False) + "\n")
                ferr.flush()
                n_bad += 1
                bad_reasons[reason] += 1

    print()
    print(f"[done] kept   : {n_ok}")
    print(f"[done] dropped: {n_bad}")
    for r, c in bad_reasons.most_common():
        print(f"[done]   {r:20s}: {c}")
    print(f"[done] cache  : {args.output}  (total uids in cache now: "
          f"{len(_read_cache(args.output))})")
    print(f"[done] failed : {failed_log}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
