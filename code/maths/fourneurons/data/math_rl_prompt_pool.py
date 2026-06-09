"""Build a math RL prompt pool from the main high-signal math datasets.

The output intentionally contains only prompt-level information:

    {"prompt": "...", "answer": "...", "source": "..."}

This matches the verifier-first RL plan in docs/math_rl.md. The model should
sample its own completions; supervised solutions are kept out of the pool.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

import requests
from datasets import load_dataset
from huggingface_hub import hf_hub_url


DEFAULT_OUT_DIR = Path("/scratch/data/math/rl_prompt_pool/splits")
DEFAULT_EXISTING_PATHS = {
    "openmathinstruct": Path("/scratch/data/math/openmathinstruct/splits/openmathinstruct_train.jsonl"),
    "openR1math": Path("/scratch/data/math/openR1math/splits/openR1math_train.jsonl"),
}

SOURCE_ORDER = (
    "openmathinstruct",
    "openR1math",
    "numinamath_1_5",
    "nemotron_math_v2",
)

DEFAULT_SOURCES = SOURCE_ORDER


def _as_clean_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _has_image_marker(text: str) -> bool:
    return (
        "![](" in text
        or re.search(r"<img\b", text, flags=re.IGNORECASE) is not None
        or re.search(r"<image[_\s-]*\d*>", text, flags=re.IGNORECASE) is not None
        or re.search(r"<img[_\s-]*\d+>", text, flags=re.IGNORECASE) is not None
    )


def _is_bad_answer(answer: str) -> bool:
    normalized = answer.strip().casefold()
    if not normalized:
        return True
    return normalized in {"proof", "none", "null", "nan", "n/a", "unknown"}


def _is_yes(value: Any) -> bool:
    return _as_clean_str(value).casefold() in {"yes", "true", "1"}


def _prompt_key(prompt: str) -> str:
    normalized = re.sub(r"\s+", " ", prompt).strip().casefold()
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def _safe_row(prompt: Any, answer: Any, source: str, **metadata: Any) -> dict[str, Any] | None:
    prompt = _as_clean_str(prompt)
    answer = _as_clean_str(answer)
    if not prompt or _is_bad_answer(answer):
        return None
    if _has_image_marker(prompt):
        return None

    row: dict[str, Any] = {
        "prompt": prompt,
        "answer": answer,
        "source": source,
    }
    for key, value in metadata.items():
        if value is not None and value != "":
            row[key] = value
    return row


def iter_existing_jsonl(path: Path, source: str) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            row = _safe_row(
                item.get("prompt"),
                item.get("answer"),
                source,
                original_source=item.get("source") or item.get("problem_source"),
                problem_type=item.get("problem_type") or item.get("type"),
                question_type=item.get("question_type"),
                level=item.get("level"),
                uuid=item.get("uuid"),
                source_line=line_no,
            )
            if row is not None:
                yield row


def iter_numinamath_1_5(cache_dir: str | None) -> Iterator[dict[str, Any]]:
    ds = load_dataset(
        "AI-MO/NuminaMath-1.5",
        split="train",
        cache_dir=cache_dir,
    )
    for item in ds:
        if item.get("problem_is_valid") is not None and not _is_yes(item.get("problem_is_valid")):
            continue
        if item.get("solution_is_valid") is not None and not _is_yes(item.get("solution_is_valid")):
            continue
        if _as_clean_str(item.get("question_type")).casefold() == "proof":
            continue
        row = _safe_row(
            item.get("problem"),
            item.get("answer"),
            "numinamath_1_5",
            original_source=item.get("source"),
            problem_type=item.get("problem_type"),
            question_type=item.get("question_type"),
            synthetic=item.get("synthetic"),
        )
        if row is not None:
            yield row


def iter_nemotron_math_v2(cache_dir: str | None) -> Iterator[dict[str, Any]]:
    del cache_dir  # Nemotron is streamed directly from its JSONL shard.
    url = hf_hub_url(
        "nvidia/Nemotron-Math-v2",
        "data/medium.jsonl",
        repo_type="dataset",
    )
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            item = json.loads(line)
            if item.get("changed_answer_to_majority") is True:
                continue
            if item.get("tools"):
                continue
            row = _safe_row(
                item.get("problem"),
                item.get("expected_answer"),
                "nemotron_math_v2",
                original_source=item.get("data_source"),
                uuid=item.get("uuid"),
                license=item.get("license"),
                used_in=item.get("used_in"),
            )
            if row is not None:
                yield row


def iter_source_rows(source: str, cache_dir: str | None, existing_paths: dict[str, Path]) -> Iterator[dict[str, Any]]:
    if source in existing_paths:
        path = existing_paths[source]
        if not path.exists():
            raise FileNotFoundError(f"Missing existing split for {source}: {path}")
        yield from iter_existing_jsonl(path, source)
        return

    if source == "numinamath_1_5":
        yield from iter_numinamath_1_5(cache_dir)
    elif source == "nemotron_math_v2":
        yield from iter_nemotron_math_v2(cache_dir)
    else:
        raise ValueError(f"Unknown source: {source}")


def take_rows(
    rows: Iterable[dict[str, Any]],
    *,
    limit: int | None,
    seen_prompts: set[str],
) -> tuple[list[dict[str, Any]], int, int]:
    kept: list[dict[str, Any]] = []
    scanned = 0
    duplicates = 0
    for row in rows:
        scanned += 1
        key = _prompt_key(row["prompt"])
        if key in seen_prompts:
            duplicates += 1
            continue
        seen_prompts.add(key)
        kept.append(row)
        if limit is not None and len(kept) >= limit:
            break
    return kept, scanned, duplicates


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_source_limits(values: list[str]) -> dict[str, int]:
    limits: dict[str, int] = {}
    for value in values:
        if "=" not in value:
            raise argparse.ArgumentTypeError(
                f"Expected SOURCE=N for --source-limit, got {value!r}"
            )
        source, raw_limit = value.split("=", 1)
        limits[source] = int(raw_limit)
    return limits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        nargs="+",
        default=list(DEFAULT_SOURCES),
        choices=SOURCE_ORDER,
        help="Sources to include in the prompt pool.",
    )
    parser.add_argument(
        "--source-limit",
        action="append",
        default=[],
        metavar="SOURCE=N",
        help="Optional per-source cap after filtering and deduplication.",
    )
    parser.add_argument("--cache_dir", default="/scratch/hf_cache")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--split_name", default="train")
    parser.add_argument("--dataset_name", default="rl_prompt_pool")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_shuffle", action="store_true")
    args = parser.parse_args()

    source_limits = parse_source_limits(args.source_limit)
    unknown_limits = set(source_limits) - set(SOURCE_ORDER)
    if unknown_limits:
        parser.error(f"Unknown --source-limit source(s): {sorted(unknown_limits)}")

    seen_prompts: set[str] = set()
    all_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"sources": {}, "seed": args.seed}

    for source in args.sources:
        limit = source_limits.get(source)
        rows, scanned, duplicates = take_rows(
            iter_source_rows(source, args.cache_dir, DEFAULT_EXISTING_PATHS),
            limit=limit,
            seen_prompts=seen_prompts,
        )
        all_rows.extend(rows)
        summary["sources"][source] = {
            "kept": len(rows),
            "scanned": scanned,
            "duplicates": duplicates,
            "limit": limit,
        }
        print(
            f"{source}: kept={len(rows)}, scanned={scanned}, "
            f"duplicates={duplicates}, limit={limit}"
        )

    if not args.no_shuffle:
        random.Random(args.seed).shuffle(all_rows)

    output_path = args.out_dir / f"{args.dataset_name}_{args.split_name}.jsonl"
    write_jsonl(all_rows, output_path)
    summary["total_kept"] = len(all_rows)
    summary["output_path"] = str(output_path)

    summary_path = output_path.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(all_rows)} rows to {output_path}")
    print(f"Wrote summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
