import json
import argparse
from pathlib import Path
from collections import defaultdict


def main(scored_path, gens_path, output_path=None):
    with open(scored_path) as f:
        scored = json.load(f)

    prompt_to_category = {}
    with open(gens_path) as f:
        for line in f:
            row = json.loads(line)
            prompt_to_category[row["prompt"]] = row.get("category", "Unknown")

    category_stats = defaultdict(lambda: {"correct": 0, "total": 0, "no_format": 0})

    for item in scored["detailed_results"]:
        cat = prompt_to_category.get(item["prompt"], "Unknown")
        category_stats[cat]["total"] += 1
        category_stats[cat]["correct"] += item["c"]
        if not item["completions"][0].get("has_format_compliance", True):
            category_stats[cat]["no_format"] += 1

    print(f"\nOverall pass@1 : {scored['metrics']['pass@1']:.4f}")
    print(f"Total problems : {scored['n_problems']}\n")

    header = f"{'Category':<35} {'Correct':>8} {'Total':>7} {'pass@1':>8} {'No-box':>7}"
    print(header)
    print("-" * len(header))

    rows = []
    for cat, stats in sorted(category_stats.items()):
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        rows.append((cat, stats["correct"], stats["total"], acc, stats["no_format"]))

    for cat, correct, total, acc, no_fmt in sorted(rows, key=lambda x: x[3]):
        flag = " ← WEAK" if acc < 0.80 else ""
        print(f"{cat:<35} {correct:>8} {total:>7} {acc:>8.4f} {no_fmt:>7}{flag}")

    if output_path:
        output = {
            "overall": scored["metrics"],
            "by_category": {
                cat: {"correct": c, "total": t, "pass@1": round(a, 4), "no_format": nf}
                for cat, c, t, a, nf in rows
            }
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored",  required=True, help="Path to val_scored.json")
    parser.add_argument("--gens",    required=True, help="Path to val_gens.jsonl")
    parser.add_argument("--output",  default=None,  help="Optional path to save JSON breakdown")
    args = parser.parse_args()
    main(args.scored, args.gens, args.output)