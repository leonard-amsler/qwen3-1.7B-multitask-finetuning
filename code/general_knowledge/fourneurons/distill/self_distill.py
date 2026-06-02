from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Optional

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))



_THINK_CLOSE_RE = re.compile(r"</think>", re.IGNORECASE)

_BOXED_RE = re.compile(
    r"\\?boxed\s*\{\s*\\?\(?\s*([A-Za-z])\s*\\?\)?\s*\}"
)


_BOXED_TEXT_RE = re.compile(
    r"\\?boxed\s*\{\s*\\?text\s*\{\s*\(?\s*([A-Za-z])\b",
    re.IGNORECASE,
)

_ANSWER_IS_X_RE = re.compile(
    r"(?:^|[\s\.,;:!\?\*\(\-])"
    r"(?:the\s+|my\s+|our\s+|so\s+the\s+|so\s+my\s+)?"
    r"(?:final|correct|right|best|chosen|selected)?\s*"
    r"answer"
    r"(?:\s+is)?"              
    r"\s*[:\-=→]*\s*"            
    r"[\*\s\\]*"                
    r"\(?\s*([A-Za-z])\b",       
    re.IGNORECASE,
)

_OPTION_X_VERB_RE = re.compile(
    r"(?:^|[\s\.\n\(])(?:option|choice|selection)\s+"
    r"\*{0,2}\s*\(?\s*([A-Za-z])\s*\)?\s*\*{0,2}"
    r"\s+(?:is|would|should|appears|seems|matches|fits|will)\b",
    re.IGNORECASE,
)

_LETTER_IS_CORRECT_RE = re.compile(
    r"(?:^|[\s\.,;:!\?\*\(\-])"
    r"\*{0,2}\s*\(?\s*([A-Za-z])\s*\)?\s*\*{0,2}"
    r"\s+is\s+(?:the\s+)?(?:correct|right|best|true|most\s+correct)"
    r"\s+(?:answer|choice|option|selection|response)",
    re.IGNORECASE,
)

_BOLD_OPTION_LABEL_RE = re.compile(
    r"\*\*\s*\(?\s*([A-Za-z])\s*[\.\):]\s+[^\*\n]{2,}?\*\*",
)

_BOLD_LETTER_RE = re.compile(
    r"\*\*\s*(?:option\s+|choice\s+)?"
    r"\(?\s*([A-Za-z])\s*\)?\s*\*\*"
)


def _extract(completion: str) -> tuple[str, Optional[str]]:
    
    parts = _THINK_CLOSE_RE.split(completion, maxsplit=1)
    if len(parts) == 2:
        thinking, after = parts
    else:
        return completion.strip(), None

    after = after.strip()
    if not after:
        return thinking.strip(), None

    matches = _BOXED_RE.findall(after)
    if matches:
        return thinking.strip(), matches[-1].upper()

    matches = _BOXED_TEXT_RE.findall(after)
    if matches:
        return thinking.strip(), matches[-1].upper()

    matches = _ANSWER_IS_X_RE.findall(after)
    if matches:
        return thinking.strip(), matches[-1].upper()

    matches = _OPTION_X_VERB_RE.findall(after)
    if matches:
        return thinking.strip(), matches[-1].upper()

    matches = _LETTER_IS_CORRECT_RE.findall(after)
    if matches:
        return thinking.strip(), matches[-1].upper()

    tail600 = after[-600:]
    matches = _BOLD_OPTION_LABEL_RE.findall(tail600)
    if matches:
        return thinking.strip(), matches[-1].upper()

    tail400 = after[-400:]
    matches = _BOLD_LETTER_RE.findall(tail400)
    if matches:
        return thinking.strip(), matches[-1].upper()

    return thinking.strip(), None


def _select_sample(
    samples: list[dict],
    select_best: bool,
    max_thinking_chars: int,
    max_3gram_repeats: int = 6,
) -> Optional[dict]:
    if not samples:
        return None
    if not select_best:
        return samples[0]

    from fourneurons.distill.filters import _max_ngram_repeats

    clean = [
        s
        for s in samples
        if len(s["thinking"]) <= max_thinking_chars
        and _max_ngram_repeats(s["thinking"], 3) <= max_3gram_repeats
    ]
    if not clean:
        return None
    return min(clean, key=lambda s: len(s["thinking"]))


def _build_assistant_content(thinking: str, gold_letter: str) -> str:
    thinking = thinking.strip()
    return (
        f"<think>\n{thinking}\n</think>\n\n"
        f"The answer is \\boxed{{{gold_letter}}}"
    )


def _matches_filter(source: Optional[str], patterns: list[str]) -> bool:
    
    if not patterns:
        return True
    if not source:
        return False
    return any(source == p or source.startswith(p) for p in patterns)




def _read_cache_counts(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    counts: Counter[str] = Counter()
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
                counts[uid] += 1
    return dict(counts)


def _iter_cache_rows(path: Path):
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue



def _generate_chunked(
    *,
    teacher: str,
    quantization: Optional[str],
    prompts_by_idx: list[tuple[int, str]],  
    items_by_idx: dict[int, dict],          
    cache_path: Path,
    chunk_size: int,
    n_samples: int,
    max_tokens: int,
    max_model_len: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    gpu_mem: float,
    tensor_parallel_size: int,
    seed: int,
) -> None:
    
    from vllm import LLM, SamplingParams

    if not prompts_by_idx:
        print("[generate] nothing to generate (cache already complete).")
        return

    print(f"[teacher] loading vLLM model {teacher}...")
    t0 = time.time()
    llm_kwargs = dict(
        model=teacher,
        dtype="float16",
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_mem,
        trust_remote_code=True,
    )
    if quantization:
        llm_kwargs["quantization"] = quantization
    if tensor_parallel_size and tensor_parallel_size > 1:
        llm_kwargs["tensor_parallel_size"] = tensor_parallel_size
    llm = LLM(**llm_kwargs)
    print(f"[teacher] loaded in {time.time() - t0:.1f}s")

    sp_kwargs = dict(
        n=n_samples,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_tokens=max_tokens,
        seed=seed,
    )
    
    if min_p and min_p > 0:
        sp_kwargs["min_p"] = min_p
    sp = SamplingParams(**sp_kwargs)

    n_total = len(prompts_by_idx)
    n_chunks = (n_total + chunk_size - 1) // chunk_size
    print(
        f"[generate] {n_total} prompts pending in {n_chunks} chunks of "
        f"~{chunk_size}  (n_samples={n_samples}, max_tokens={max_tokens})"
    )

    t_start = time.time()
    for chunk_idx in range(n_chunks):
        start = chunk_idx * chunk_size
        chunk = prompts_by_idx[start : start + chunk_size]
        chunk_prompts = [prompt for _, prompt in chunk]

        t_chunk = time.time()
        outs = llm.generate(chunk_prompts, sp)
        dt = time.time() - t_chunk

        chunk_path = cache_path.with_name(cache_path.name + f".part{chunk_idx}")
        n_correct_chunk = 0
        n_samples_chunk = 0
        with open(chunk_path, "w", encoding="utf-8") as fchunk:
            for (global_idx, _prompt), out in zip(chunk, outs):
                meta = items_by_idx[global_idx]
                gold = meta["gold_letter"]
                for sample_idx, sample in enumerate(out.outputs):
                    text = sample.text
                    thinking, pred = _extract(text)
                    is_correct = pred is not None and pred == gold
                    if is_correct:
                        n_correct_chunk += 1
                    n_samples_chunk += 1
                    fchunk.write(
                        json.dumps(
                            {
                                "uid": meta["uid"],
                                "split": meta["split"],
                                "row_idx": meta["row_idx"],
                                "sample_idx": sample_idx,
                                "thinking": thinking,
                                "extracted_letter": pred,
                                "gold_letter": gold,
                                "is_correct": is_correct,
                                "completion": text,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
            fchunk.flush()
       
        with open(chunk_path, "r", encoding="utf-8") as fchunk, \
             open(cache_path, "a", encoding="utf-8") as fcache:
            for line in fchunk:
                fcache.write(line)
            fcache.flush()
        chunk_path.unlink(missing_ok=True)

        done = start + len(chunk)
        elapsed = time.time() - t_start
        eta = elapsed * (n_total - done) / max(done, 1)
        print(
            f"[chunk] {chunk_idx + 1}/{n_chunks}  "
            f"prompts={len(chunk)}  gen={dt:.1f}s "
            f"({dt / max(1, len(chunk)):.2f}s/prompt)  "
            f"correct={n_correct_chunk}/{n_samples_chunk}  "
            f"cum_done={done}/{n_total}  "
            f"eta={eta / 60:.1f}min"
        )



def _assemble_dataset(
    *,
    cache_path: Path,
    source_dsd,
    cot_source_tag: str,
    keep_raw_completion: bool,
    fallback_to_source: bool,
    output_dir: Path,
    failed_log: Path,
    filter_sources: Optional[list[str]] = None,
    select_best: bool = False,
    max_thinking_chars: int = 4000,
):
    
    print(f"[assemble] reading cache {cache_path}")
    correct_samples: dict[str, list[dict]] = {}
    samples_per_uid: Counter[str] = Counter()
    correct_per_uid: Counter[str] = Counter()

    extract_status: Counter[str] = Counter()
    for row in _iter_cache_rows(cache_path):
        uid = row.get("uid")
        if not uid:
            continue
        samples_per_uid[uid] += 1

        completion = row.get("completion", "")
        gold = (row.get("gold_letter") or "").strip().upper()
        thinking, pred = _extract(completion)
        is_correct = pred is not None and pred == gold

        if "</think>" not in completion:
            extract_status["truncated_no_close"] += 1
        elif pred is None:
            extract_status["no_answer"] += 1
        elif not is_correct:
            extract_status["wrong_answer"] += 1
        else:
            extract_status["correct"] += 1

        if is_correct:
            correct_per_uid[uid] += 1
            correct_samples.setdefault(uid, []).append(
                {
                    "thinking": thinking,
                    "gold_letter": gold,
                    "completion": completion,
                }
            )

    first_correct: dict[str, dict] = {}
    n_no_clean = 0
    for uid, samples in correct_samples.items():
        chosen = _select_sample(samples, select_best, max_thinking_chars)
        if chosen is not None:
            first_correct[uid] = chosen
        else:
            n_no_clean += 1
    if select_best:
        print(
            f"[assemble] select_best=ON (max_thinking_chars={max_thinking_chars}): "
            f"{len(first_correct)} uids kept a clean correct sample, "
            f"{n_no_clean} uids had only loopy/over-long correct samples "
            f"(-> fallback to source CoT)."
        )

    attempted_uids = set(samples_per_uid)
    print(
        f"[assemble] cache covers {len(attempted_uids)} uids "
        f"with {sum(samples_per_uid.values())} samples total "
        f"({len(first_correct)} uids have >= 1 correct sample)."
    )

    if extract_status:
        total = sum(extract_status.values())
        print("[assemble] per-sample extraction status (re-scored):")
        for status, n in extract_status.most_common():
            print(f"            {status:20s}: {n} ({100 * n / total:.1f}%)")

    if samples_per_uid:
        n_dist = Counter(samples_per_uid.values())
        print("[assemble] samples-per-uid distribution:")
        for k in sorted(n_dist):
            print(f"            {k} samples: {n_dist[k]} uids")
        c_dist = Counter(correct_per_uid[uid] for uid in samples_per_uid)
        print("[assemble] correct-samples-per-uid distribution:")
        for k in sorted(c_dist):
            n_uids = c_dist[k]
            pct = 100 * n_uids / len(samples_per_uid)
            print(f"            {k}/N correct: {n_uids} uids ({pct:.1f}%)")

    kept: dict[str, list[dict]] = {name: [] for name in source_dsd.keys()}
    n_dropped = 0
    n_in_scope = 0           
    n_replaced = 0       
    n_fallback = 0     
    n_passthrough = 0       
    has_filter = bool(filter_sources)
    failed_log.parent.mkdir(parents=True, exist_ok=True)
    with open(failed_log, "w", encoding="utf-8") as ferr:
        for split_name, split in source_dsd.items():
            for row in split:
                uid = row.get("uid")
                if not uid:
                    continue
                if uid not in attempted_uids:
                    if has_filter:
                        kept[split_name].append(dict(row))
                        n_passthrough += 1
                    continue
                n_in_scope += 1

                rec = first_correct.get(uid)
                if rec is not None:
                    user_msg = next(
                        (m for m in row["messages"] if m["role"] == "user"),
                        None,
                    )
                    if user_msg is None:
                        n_dropped += 1
                        continue
                    new_assistant = _build_assistant_content(
                        rec["thinking"], rec["gold_letter"]
                    )
                    new_row = dict(row)
                    new_row["messages"] = [
                        {"role": "user", "content": user_msg["content"]},
                        {"role": "assistant", "content": new_assistant},
                    ]
                    new_row["cot_source"] = cot_source_tag
                    if keep_raw_completion:
                        new_row["raw_completion"] = rec["completion"]
                    kept[split_name].append(new_row)
                    n_replaced += 1
                else:
                    ferr.write(
                        json.dumps(
                            {
                                "uid": uid,
                                "gold_letter": row.get("gold_letter"),
                                "source": row.get("source"),
                                "macro_cat": row.get("macro_cat"),
                                "n_samples": samples_per_uid.get(uid, 0),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    if fallback_to_source:
                        kept[split_name].append(dict(row))
                        n_fallback += 1
                    else:
                        n_dropped += 1

    n_kept = sum(len(v) for v in kept.values())
    print(
        f"[assemble] in-scope (attempted) uids: {n_in_scope}  "
        f"teacher-replaced: {n_replaced}  "
        f"fallback-to-source: {n_fallback}  "
        f"passthrough (out-of-filter): {n_passthrough}  "
        f"dropped: {n_dropped}"
    )
    print(
        f"[assemble] kept {n_kept} / {n_in_scope + n_passthrough} rows  "
        f"({100 * n_kept / max(1, n_in_scope + n_passthrough):.1f}% kept of total)"
    )
    print(f"[assemble] failed log: {failed_log}")

    from datasets import Dataset, DatasetDict

    out_dsd = DatasetDict(
        {name: Dataset.from_list(rows) for name, rows in kept.items() if rows}
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    out_dsd.save_to_disk(str(output_dir))
    print(f"[save] wrote {output_dir}  splits={list(out_dsd.keys())}")

    return {
        "n_in_scope": n_in_scope,
        "n_replaced": n_replaced,
        "n_fallback": n_fallback,
        "n_passthrough": n_passthrough,
        "n_dropped": n_dropped,
        "n_kept": n_kept,
        "kept_pct_of_attempted": round(100 * n_kept / max(1, n_in_scope), 2),
        "replaced_pct_of_attempted": round(
            100 * n_replaced / max(1, n_in_scope), 2
        ),
        "by_split": {name: len(rows) for name, rows in kept.items()},
    }



def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])

    p.add_argument(
        "--teacher",
        default="Qwen/Qwen3-14B-AWQ",
        help="HF id of the teacher model. v7-alt: Qwen3-14B-AWQ. "
        "v7 self-distill: Qwen/Qwen3-1.7B.",
    )
    p.add_argument(
        "--quantization",
        default="awq_marlin",
        help="vLLM quantization backend. 'awq_marlin' (default) for AWQ "
        "models. Pass empty string '' to disable (e.g. raw FP16 baseline).",
    )
    p.add_argument(
        "--enable_thinking",
        action="store_true",
        help="Render the teacher prompt with `enable_thinking=True` so "
        "the assistant turn begins with an open <think> tag. REQUIRED "
        "for v7-style distillation.",
    )

    p.add_argument(
        "--source_dataset",
        type=Path,
        required=True,
        help="HF DatasetDict on disk (typically /scratch/data/train_v6).",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        required=True,
        help="Where to write the new HF DatasetDict (train_v7).",
    )
    p.add_argument(
        "--cot_source_tag",
        default="qwen3_14b_thinking",
        help="Value written to `cot_source` on each row. Use "
        "'self_distill_baseline' if teacher is the 1.7B model.",
    )
    p.add_argument(
        "--cache_path",
        type=Path,
        default=None,
        help="JSONL cache for resumable generation. "
        "Defaults to <output_dir>/self_distill_cache.jsonl.",
    )
    p.add_argument(
        "--failed_log",
        type=Path,
        default=None,
        help="JSONL sidecar for uids with no correct sample. "
        "Defaults to <output_dir>/self_distill_failed.jsonl.",
    )

    p.add_argument("--n_samples", type=int, default=2)
    p.add_argument("--max_tokens", type=int, default=3000)
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--top_k", type=int, default=20)
    p.add_argument(
        "--min_p",
        type=float,
        default=0.0,
        help="vLLM min_p. Qwen3 docs recommend 0 for thinking mode.",
    )

    p.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    p.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="vLLM TP degree. Set to >1 only if multiple GPUs available.",
    )
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--chunk_size",
        type=int,
        default=1500,
        help="Number of distinct prompts per vLLM batch / cache flush. "
        "Smaller = more checkpoints (resilient to kills) but slightly "
        "slower amortization of vLLM overhead.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on total prompts (debug pilots).",
    )
    p.add_argument(
        "--filter_sources",
        nargs="*",
        default=[],
        help="Phase B: only re-distill rows whose `source` matches one of "
        "these prefixes (e.g. `mmlu_pro_cot mmlu_world`). Rows that "
        "don't match are passed through verbatim from the source "
        "dataset (= they keep their existing CoT). Empty list = no "
        "filter, all rows attempted.",
    )
    p.add_argument(
        "--keep_raw_completion",
        action="store_true",
        help="Store the teacher's raw completion in a `raw_completion` "
        "column (useful for debug, bloats the dataset).",
    )
    p.add_argument(
        "--no_fallback_to_source",
        action="store_true",
        help="Drop rows where no sample was correct (default behaviour: "
        "fall back to the source row's existing CoT, so the output "
        "dataset is strictly >= the input on coverage).",
    )
    p.add_argument(
        "--phase",
        choices=("all", "generate", "assemble"),
        default="all",
        help="Run the generate phase, the assemble phase, or both. "
        "Generate is resumable; assemble is fast.",
    )
    p.add_argument(
        "--select_best",
        action="store_true",
        help="v9b: at assembly, among the correct samples per uid, drop "
        "loopy/over-long traces and keep the SHORTEST clean one (instead "
        "of the first correct). uids with no clean correct sample fall "
        "back to the source CoT. Fixes V9's looping/truncation.",
    )
    p.add_argument(
        "--max_thinking_chars",
        type=int,
        default=4000,
        help="With --select_best, max chars of a thinking trace to be "
        "eligible (default 4000 ~= v6's p99). Longer traces are treated "
        "as loopy and skipped.",
    )

    args = p.parse_args(argv)

    from datasets import load_from_disk
    from transformers import AutoTokenizer

    args.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = args.cache_path or args.output_dir / "self_distill_cache.jsonl"
    failed_log = args.failed_log or args.output_dir / "self_distill_failed.jsonl"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    quantization = args.quantization or None
    if quantization == "":
        quantization = None

    print(f"[load] {args.source_dataset}")
    dsd = load_from_disk(str(args.source_dataset))
    if "train" not in dsd:
        raise SystemExit(
            f"Source dataset has no `train` split. Got: {list(dsd.keys())}"
        )
    splits = {name: dsd[name] for name in dsd.keys() if name in ("train", "test")}
    for name, split in splits.items():
        print(
            f"[load]   {name}: {len(split)} rows  cols={split.column_names}"
        )

    if args.phase in ("all", "generate"):
        print(f"[tokenizer] loading {args.teacher}")
        tokenizer = AutoTokenizer.from_pretrained(
            args.teacher, trust_remote_code=True
        )

        all_items: list[dict] = []
        all_prompts: list[str] = []
        n_filtered_out = 0
        for split_name, split in splits.items():
            for i, row in enumerate(split):
                if args.limit is not None and len(all_items) >= args.limit:
                    break
                uid = row.get("uid")
                if not uid:
                    continue
                if args.filter_sources and not _matches_filter(
                    row.get("source"), args.filter_sources
                ):
                    n_filtered_out += 1
                    continue
                gold = (row.get("gold_letter") or "").strip().upper()
                user_msg = next(
                    (m for m in row["messages"] if m["role"] == "user"), None
                )
                if user_msg is None:
                    continue
                try:
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": user_msg["content"]}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=args.enable_thinking,
                    )
                except TypeError:
                    prompt = tokenizer.apply_chat_template(
                        [{"role": "user", "content": user_msg["content"]}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                all_items.append(
                    {
                        "uid": uid,
                        "split": split_name,
                        "row_idx": i,
                        "gold_letter": gold,
                    }
                )
                all_prompts.append(prompt)

        if args.filter_sources:
            print(
                f"[prompts] filter_sources={args.filter_sources}: "
                f"{len(all_prompts)} matched, {n_filtered_out} filtered out."
            )
        print(f"[prompts] {len(all_prompts)} prompts in scope.")

        cache_counts = _read_cache_counts(cache_path)
        if cache_counts:
            done_uids = {
                uid for uid, c in cache_counts.items() if c >= args.n_samples
            }
            partial_uids = {
                uid for uid, c in cache_counts.items() if c < args.n_samples
            }
            print(
                f"[resume] cache has {len(cache_counts)} uids "
                f"({len(done_uids)} complete, {len(partial_uids)} partial)."
            )
            if partial_uids:
                print(
                    "[resume] partial uids will be redone "
                    f"(first 3: {list(partial_uids)[:3]})."
                )
        else:
            done_uids = set()
            print("[resume] no existing cache, starting fresh.")

        prompts_by_idx: list[tuple[int, str]] = []
        items_by_idx: dict[int, dict] = {}
        for idx, (item, prompt) in enumerate(zip(all_items, all_prompts)):
            if item["uid"] in done_uids:
                continue
            prompts_by_idx.append((idx, prompt))
            items_by_idx[idx] = item

        if not prompts_by_idx:
            print("[generate] all uids already done; skipping vLLM load.")
        else:
            _generate_chunked(
                teacher=args.teacher,
                quantization=quantization,
                prompts_by_idx=prompts_by_idx,
                items_by_idx=items_by_idx,
                cache_path=cache_path,
                chunk_size=args.chunk_size,
                n_samples=args.n_samples,
                max_tokens=args.max_tokens,
                max_model_len=args.max_model_len,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                gpu_mem=args.gpu_memory_utilization,
                tensor_parallel_size=args.tensor_parallel_size,
                seed=args.seed,
            )

    if args.phase in ("all", "assemble"):
        if not cache_path.exists():
            raise SystemExit(
                f"Cache {cache_path} missing — run --phase generate first."
            )
        stats = _assemble_dataset(
            cache_path=cache_path,
            source_dsd=splits,
            cot_source_tag=args.cot_source_tag,
            keep_raw_completion=args.keep_raw_completion,
            fallback_to_source=not args.no_fallback_to_source,
            output_dir=args.output_dir,
            failed_log=failed_log,
            filter_sources=args.filter_sources or None,
            select_best=args.select_best,
            max_thinking_chars=args.max_thinking_chars,
        )

        summary = {
            **stats,
            "cache_path": str(cache_path),
            "failed_log": str(failed_log),
            "args": {
                k: str(v) if isinstance(v, Path) else v
                for k, v in vars(args).items()
            },
        }
        with open(args.output_dir / "self_distill_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(json.dumps(summary, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
