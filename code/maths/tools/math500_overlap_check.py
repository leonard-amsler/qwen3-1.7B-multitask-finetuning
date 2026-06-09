#!/usr/bin/env python3
"""Check overlap between sampled math SFT train rows and local MATH-500."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import unicodedata
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer

from fourneurons.data.format_for_sft import format_for_sft


OPENMATH_PATH = Path("/scratch/data/math/openmathinstruct/splits/openmathinstruct_train.jsonl")
OPENR1_PATH = Path("/scratch/data/math/openR1math/splits/openR1math_train.jsonl")
MATH500_PATH = Path("/scratch/data/math/math500/splits/math500_full.jsonl")
SNAPSHOT_DIR = Path("/scratch/hf_cache/hub/models--Qwen--Qwen3-1.7B/snapshots")
PROMPT_FILE = "fourneurons/prompts/math.txt"


def normalize_problem(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = text.lower()
    text = text.replace("\u2212", "-").replace("\u00a0", " ")
    text = re.sub(r"\\(?:left|right|,|;|!| )", "", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s*([=+\-*/^_{}()[\],.:;<>])\s*", r"\1", text)
    text = re.sub(r"[\$`]", "", text)
    return text.strip()


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if line.strip():
                yield line_no, json.loads(line)


def load_tokenizer() -> AutoTokenizer:
    snapshots = sorted(p for p in SNAPSHOT_DIR.iterdir() if p.is_dir())
    if not snapshots:
        raise FileNotFoundError(f"No tokenizer snapshot found under {SNAPSHOT_DIR}")
    return AutoTokenizer.from_pretrained(str(snapshots[0]), local_files_only=True)


def collect_actual_training_rows(
    path: Path,
    dataset: str,
    target_count: int,
    tokenizer: AutoTokenizer,
    max_length: int,
):
    kept = []
    scanned = 0
    dropped_overlong = 0

    for line_no, row in iter_jsonl(path):
        scanned += 1
        formatted = format_for_sft(row, tokenizer, prompt_file_path=PROMPT_FILE)
        token_count = len(tokenizer(formatted, add_special_tokens=False)["input_ids"])
        if token_count > max_length:
            dropped_overlong += 1
            continue

        prompt = str(row["prompt"]).strip()
        kept.append(
            {
                "dataset": dataset,
                "line_no": line_no,
                "prompt": prompt,
                "normalized": normalize_problem(prompt),
                "answer": row.get("answer"),
                "token_count": token_count,
            }
        )
        if len(kept) >= target_count:
            break

    return kept, {"dataset": dataset, "scanned": scanned, "kept": len(kept), "dropped_overlong": dropped_overlong}


def load_math500(path: Path):
    rows = []
    for line_no, row in iter_jsonl(path):
        prompt = str(row["prompt"]).strip()
        rows.append(
            {
                "math500_index": len(rows),
                "line_no": line_no,
                "prompt": prompt,
                "normalized": normalize_problem(prompt),
                "answer": row.get("answer"),
                "level": row.get("level"),
                "type": row.get("type"),
            }
        )
    return rows


def exact_overlaps(train_rows, math500_rows):
    by_norm = defaultdict(list)
    for row in train_rows:
        by_norm[row["normalized"]].append(row)

    matches = []
    for m in math500_rows:
        for t in by_norm.get(m["normalized"], []):
            matches.append({"math500": m, "train": t, "similarity": 1.0})
    return matches


def char_ngrams(text: str, n: int = 5):
    compact = re.sub(r"\s+", " ", text)
    if len(compact) <= n:
        return {compact}
    return {compact[i : i + n] for i in range(len(compact) - n + 1)}


def minhash_signature(tokens: set[str], num_perm: int = 64):
    if not tokens:
        return tuple([2**64 - 1] * num_perm)
    mins = [2**64 - 1] * num_perm
    for token in tokens:
        encoded = token.encode("utf-8")
        for seed in range(num_perm):
            digest = hashlib.blake2b(encoded, digest_size=8, person=seed.to_bytes(4, "little")).digest()
            value = int.from_bytes(digest, "little")
            if value < mins[seed]:
                mins[seed] = value
    return tuple(mins)


def minhash_overlaps(
    train_rows,
    math500_rows,
    threshold: float,
    candidate_pairs=None,
    num_perm: int = 64,
    bands: int = 16,
):
    if candidate_pairs is not None:
        train_tokens = {}
        math_tokens = {}
        matches = []
        for m_idx, t_idx in candidate_pairs:
            if t_idx not in train_tokens:
                train_tokens[t_idx] = char_ngrams(train_rows[t_idx]["normalized"])
            if m_idx not in math_tokens:
                math_tokens[m_idx] = char_ngrams(math500_rows[m_idx]["normalized"])
            tokens = train_tokens[t_idx]
            other = math_tokens[m_idx]
            union = tokens | other
            score = len(tokens & other) / len(union) if union else 1.0
            if score >= threshold:
                matches.append({"math500": math500_rows[m_idx], "train": train_rows[t_idx], "similarity": score})
        matches.sort(key=lambda x: x["similarity"], reverse=True)
        return matches

    rows_per_band = num_perm // bands
    math_tokens = [char_ngrams(row["normalized"]) for row in math500_rows]
    math_sigs = [minhash_signature(tokens, num_perm=num_perm) for tokens in math_tokens]

    buckets = defaultdict(list)
    for idx, sig in enumerate(math_sigs):
        for band in range(bands):
            start = band * rows_per_band
            buckets[(band, sig[start : start + rows_per_band])].append(idx)

    matches = []
    seen = set()
    for t_idx, train in enumerate(train_rows):
        tokens = char_ngrams(train["normalized"])
        sig = minhash_signature(tokens, num_perm=num_perm)
        candidates = set()
        for band in range(bands):
            start = band * rows_per_band
            candidates.update(buckets.get((band, sig[start : start + rows_per_band]), []))
        for m_idx in candidates:
            key = (t_idx, m_idx)
            if key in seen:
                continue
            seen.add(key)
            union = tokens | math_tokens[m_idx]
            score = len(tokens & math_tokens[m_idx]) / len(union) if union else 1.0
            if score >= threshold:
                matches.append({"math500": math500_rows[m_idx], "train": train, "similarity": score})
    matches.sort(key=lambda x: x["similarity"], reverse=True)
    return matches


def embedding_overlaps(train_rows, math500_rows, threshold: float, top_k: int):
    train_texts = [row["normalized"] for row in train_rows]
    math_texts = [row["normalized"] for row in math500_rows]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1, norm="l2")
    train_vecs = vectorizer.fit_transform(train_texts)
    math_vecs = vectorizer.transform(math_texts)

    nn = NearestNeighbors(n_neighbors=min(top_k, len(train_rows)), metric="cosine", algorithm="brute")
    nn.fit(train_vecs)
    distances, indices = nn.kneighbors(math_vecs)

    matches = []
    nearest = []
    candidate_pairs = []
    for m_idx, (row_distances, row_indices) in enumerate(zip(distances, indices, strict=True)):
        for rank, (distance, t_idx) in enumerate(zip(row_distances, row_indices, strict=True), start=1):
            score = float(1.0 - distance)
            item = {"math500": math500_rows[m_idx], "train": train_rows[int(t_idx)], "similarity": score, "rank": rank}
            candidate_pairs.append((m_idx, int(t_idx)))
            if rank == 1:
                nearest.append(item)
            if score >= threshold:
                matches.append(item)
    matches.sort(key=lambda x: x["similarity"], reverse=True)
    nearest.sort(key=lambda x: x["similarity"], reverse=True)
    return matches, nearest, candidate_pairs


def slim_match(match):
    return {
        "similarity": round(float(match["similarity"]), 6),
        "rank": match.get("rank"),
        "math500_index": match["math500"]["math500_index"],
        "math500_type": match["math500"].get("type"),
        "math500_level": match["math500"].get("level"),
        "math500_prompt": match["math500"]["prompt"],
        "train_dataset": match["train"]["dataset"],
        "train_line_no": match["train"]["line_no"],
        "train_token_count": match["train"].get("token_count"),
        "train_prompt": match["train"]["prompt"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-per-dataset", type=int, default=125_000)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--embedding-threshold", type=float, default=0.90)
    parser.add_argument("--minhash-threshold", type=float, default=0.80)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/math500_overlap_report.json"))
    args = parser.parse_args()

    tokenizer = load_tokenizer()
    openmath_rows, openmath_stats = collect_actual_training_rows(
        OPENMATH_PATH, "openmathinstruct", args.target_per_dataset, tokenizer, args.max_length
    )
    openr1_rows, openr1_stats = collect_actual_training_rows(
        OPENR1_PATH, "openR1math", args.target_per_dataset, tokenizer, args.max_length
    )
    train_rows = openmath_rows + openr1_rows
    math500_rows = load_math500(MATH500_PATH)

    exact = exact_overlaps(train_rows, math500_rows)
    embedding, nearest, candidate_pairs = embedding_overlaps(train_rows, math500_rows, args.embedding_threshold, args.top_k)
    minhash = minhash_overlaps(train_rows, math500_rows, args.minhash_threshold, candidate_pairs=candidate_pairs)

    exact_pairs = {(m["math500"]["math500_index"], m["train"]["dataset"], m["train"]["line_no"]) for m in exact}
    embedding_pairs = {(m["math500"]["math500_index"], m["train"]["dataset"], m["train"]["line_no"]) for m in embedding}
    minhash_pairs = {(m["math500"]["math500_index"], m["train"]["dataset"], m["train"]["line_no"]) for m in minhash}

    report = {
        "config": {
            "target_per_dataset": args.target_per_dataset,
            "max_length": args.max_length,
            "normalization": "NFKC, lowercase, whitespace/punctuation spacing canonicalization, remove common LaTeX spacing and dollar/backtick delimiters",
            "embedding_method": "sklearn TF-IDF sparse character n-gram embedding, analyzer=char_wb, ngram_range=(3,5), cosine similarity",
            "embedding_threshold": args.embedding_threshold,
            "minhash_method": "exact normalized character 5-gram Jaccard on TF-IDF top-k candidate pairs; equivalent final score used for MinHash verification without exhaustive LSH scan",
            "minhash_threshold": args.minhash_threshold,
            "minhash_candidate_pairs": len(candidate_pairs),
        },
        "stats": {
            "sampled_training_rows": len(train_rows),
            "math500_rows": len(math500_rows),
            "openmathinstruct": openmath_stats,
            "openR1math": openr1_stats,
        },
        "summary": {
            "exact_pairs": len(exact),
            "exact_math500_items": len({m["math500"]["math500_index"] for m in exact}),
            "embedding_pairs_at_threshold": len(embedding),
            "embedding_math500_items_at_threshold": len({m["math500"]["math500_index"] for m in embedding}),
            "minhash_pairs_at_threshold": len(minhash),
            "minhash_math500_items_at_threshold": len({m["math500"]["math500_index"] for m in minhash}),
            "union_pairs": len(exact_pairs | embedding_pairs | minhash_pairs),
            "union_math500_items": len({idx for idx, _, _ in (exact_pairs | embedding_pairs | minhash_pairs)}),
            "best_embedding_similarity": round(float(nearest[0]["similarity"]), 6) if nearest else None,
            "nearest_embedding_ge_0_80": sum(1 for m in nearest if m["similarity"] >= 0.80),
            "nearest_embedding_ge_0_70": sum(1 for m in nearest if m["similarity"] >= 0.70),
        },
        "exact_matches": [slim_match(m) for m in exact],
        "embedding_matches": [slim_match(m) for m in embedding[:200]],
        "minhash_matches": [slim_match(m) for m in minhash[:200]],
        "top_25_nearest_embedding": [slim_match(m) for m in nearest[:25]],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps({"stats": report["stats"], "summary": report["summary"], "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
