"""Select frontier RL examples from evaluate.score JSON output."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_scored(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        scored = json.load(f)
    detailed = scored.get("detailed_results")
    if not isinstance(detailed, list):
        raise SystemExit(f"{path} does not look like evaluate.score output.")
    return detailed


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True, help="Original prompt pool JSONL.")
    parser.add_argument("--scored", type=Path, required=True, help="JSON from evaluate.score --output.")
    parser.add_argument("--output", type=Path, required=True, help="Selected frontier JSONL.")
    parser.add_argument(
        "--include-correct",
        default="1-7",
        help="Correct-count buckets to include, e.g. 1-7 or 1,2,3,4.",
    )
    parser.add_argument("--max_rows", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    pool_rows = read_jsonl(args.pool)
    detailed = load_scored(args.scored)
    if len(pool_rows) != len(detailed):
        raise SystemExit(
            f"Pool/scored length mismatch: {len(pool_rows)} rows vs {len(detailed)} scored items."
        )

    include_counts = parse_count_spec(args.include_correct)
    selected = []
    correct_hist = Counter()
    boxed_hist = Counter()
    source_hist = Counter()
    source_correct_hist: dict[str, Counter] = defaultdict(Counter)

    for row, result in zip(pool_rows, detailed):
        correct_count = int(result["c"])
        boxed_count = int(result.get("boxed", 0))
        source = str(row.get("source", "unknown"))
        correct_hist[correct_count] += 1
        boxed_hist[boxed_count] += 1
        source_correct_hist[source][correct_count] += 1
        if correct_count not in include_counts:
            continue
        out_row = dict(row)
        out_row["rl_correct_count"] = correct_count
        out_row["rl_boxed_count"] = boxed_count
        out_row["rl_num_generations"] = int(result["n"])
        selected.append(out_row)
        source_hist[source] += 1

    random.Random(args.seed).shuffle(selected)
    if args.max_rows is not None:
        selected = selected[: args.max_rows]

    write_jsonl(selected, args.output)
    summary = {
        "pool": str(args.pool),
        "scored": str(args.scored),
        "output": str(args.output),
        "include_correct": sorted(include_counts),
        "selected_rows": len(selected),
        "correct_count_histogram": dict(sorted(correct_hist.items())),
        "boxed_count_histogram": dict(sorted(boxed_hist.items())),
        "selected_source_histogram": dict(sorted(source_hist.items())),
        "source_correct_count_histogram": {
            source: dict(sorted(hist.items()))
            for source, hist in sorted(source_correct_hist.items())
        },
    }
    summary_path = args.output.with_suffix(".summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(selected)} selected rows to {args.output}")
    print(f"Wrote summary to {summary_path}")
    return 0


def parse_count_spec(spec: str) -> set[int]:
    counts: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            counts.update(range(int(start), int(end) + 1))
        else:
            counts.add(int(part))
    invalid = [count for count in counts if count < 0]
    if invalid:
        raise argparse.ArgumentTypeError(f"Correct counts must be non-negative: {invalid}")
    return counts


if __name__ == "__main__":
    raise SystemExit(main())
