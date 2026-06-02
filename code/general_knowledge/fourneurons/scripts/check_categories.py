from datasets import load_dataset
from collections import Counter

print("\n[1] MMLU-Pro-CoT Categories:")
print("-" * 50)
mmlu_dataset = load_dataset("UW-Madison-Lee-Lab/MMLU-Pro-CoT-Train-Labeled", split="train")
mmlu_categories = Counter([ex['category'].lower() for ex in mmlu_dataset])
print(f"Total unique categories: {len(mmlu_categories)}")
for cat, count in mmlu_categories.most_common(20):
    print(f"  {cat:30s}: {count:6d} examples")

print("\n[2] ECQA Categories:")
print("-" * 50)
ecqa_dataset = load_dataset("tasksource/ecqa", split="train")
# ECQA doesn't have explicit categories, but check available fields
print(f"ECQA fields: {ecqa_dataset.column_names}")
print(f"ECQA examples: {len(ecqa_dataset)}")
print("(ECQA is all commonsense QA)")


print("""
The category distribution issue comes from:
1. MMLU has many specific subjects (math, physics, etc.)
2. We're grouping them too broadly into 3 macro-categories
3. "commonsense" gets lumped into world_knowledge
4. History/geography are underrepresented in MMLU

Better approach:
- Keep more granular categories (science, history, world_knowledge, commonsense)
- Or better yet, DON'T force categorization - let natural distribution emerge
- For training, categories are mostly for tracking, not critical
""")
