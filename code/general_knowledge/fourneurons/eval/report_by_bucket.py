"""Score GK generations with CI knowledge extraction and per-bucket reports."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluate.benchmarks import extract_benchmark_answer, is_correct_benchmark_answer


METHOD = "knowledge"  # General Knowledge benchmark extraction policy.


def _read_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
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


def _k_bucket(n_options: int) -> str:
    if n_options <= 2:
        return "2"
    if n_options == 3:
        return "3"
    if n_options == 4:
        return "4"
    if n_options == 5:
        return "5"
    if n_options <= 10:
        return "6-10"
    return "11-20"


def _score(items: list[dict]) -> dict:
    """Returns a dict with per-row stats and aggregate buckets."""
    per_row: list[dict] = []
    n_completions = None
    n_boxed = 0
    n_total_comps = 0

    for i, it in enumerate(items):
        completions = it.get("completions") or []
        if not completions:
            raise SystemExit(f"Row {i}: empty `completions`.")
        if n_completions is None:
            n_completions = len(completions)
        elif len(completions) != n_completions:
            raise SystemExit(
                f"Row {i}: inconsistent n_completions "
                f"({len(completions)} vs {n_completions})."
            )

        reference = str(it.get("answer") or it.get("reference") or "")
        c = 0
        for comp in completions:
            text = str(comp)
            n_total_comps += 1
            if "\\boxed{" in text:
                n_boxed += 1
            extracted = extract_benchmark_answer(text, METHOD, reference)
            if is_correct_benchmark_answer(extracted, reference, METHOD):
                c += 1
        per_row.append({
            "i": i,
            "n": n_completions,
            "c": c,
            "meta": it.get("meta") or {},
        })

    n_problems = len(per_row)
    n = n_completions or 0
    pass_at_1 = sum(r["c"] / max(1, r["n"]) for r in per_row) / max(1, n_problems)

    pass_at_8 = None
    if n >= 8:
        from math import comb

        def pk_row(c: int, n: int, k: int) -> float:
            if n - c < k:
                return 1.0
            return 1.0 - comb(n - c, k) / comb(n, k)

        pass_at_8 = sum(pk_row(r["c"], r["n"], 8) for r in per_row) / max(1, n_problems)

    def _agg(key_fn):
        by: dict[str, list[float]] = defaultdict(list)
        for r in per_row:
            by[key_fn(r)].append(r["c"] / max(1, r["n"]))
        return {k: {"n": len(v), "pass@1": sum(v) / len(v)} for k, v in by.items()}

    by_source = _agg(lambda r: r["meta"].get("source", "?"))
    by_macro = _agg(lambda r: r["meta"].get("macro_cat", "?"))
    by_k = _agg(lambda r: _k_bucket(int(r["meta"].get("n_options") or 0)))

    boxed_compliance = (n_boxed / n_total_comps) if n_total_comps else 0.0

    return {
        "n_problems": n_problems,
        "n_completions": n,
        "pass@1": pass_at_1,
        "pass@8": pass_at_8,
        "boxed_compliance": boxed_compliance,
        "by_source": by_source,
        "by_macro_cat": by_macro,
        "by_n_options_bucket": by_k,
    }


def _print_report(report: dict) -> None:
    print("=" * 60)
    print(f"n_problems   : {report['n_problems']}")
    print(f"n_completions: {report['n_completions']}")
    print(f"pass@1       : {report['pass@1']:.4f}")
    if report["pass@8"] is not None:
        print(f"pass@8       : {report['pass@8']:.4f}")
    if report.get("boxed_compliance") is not None:
        print(f"boxed comply : {report['boxed_compliance']:.4f} "
              f"({100*report['boxed_compliance']:.1f}%)")

    def _print_block(title: str, data: dict, key_order=None) -> None:
        if not data:
            return
        print()
        print(f"-- {title} --")
        keys = key_order or sorted(data.keys())
        for k in keys:
            if k not in data:
                continue
            v = data[k]
            print(f"  {k:18s} n={v['n']:4d}  pass@1={v['pass@1']:.4f}")

    _print_block("by source", report["by_source"])
    _print_block(
        "by macro_cat",
        report["by_macro_cat"],
        key_order=["stem", "humanities", "social_sciences", "history_geo", "commonsense"],
    )
    _print_block(
        "by n_options bucket",
        report["by_n_options_bucket"],
        key_order=["2", "3", "4", "5", "6-10", "11-20"],
    )
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--generations",
        required=True,
        type=Path,
        help="Scoreable JSONL (same as `evaluate.score`), ideally with `meta`.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write the full report JSON.",
    )
    args = parser.parse_args(argv)

    items = _read_jsonl(args.generations)
    report = _score(items)
    _print_report(report)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[report] wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
