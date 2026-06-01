import json
import pandas as pd

reasoning_samples_path = "/scratch/data/multilingual/mmmlu_more_qcms/splits/mmmlu_more_qcms_train.jsonl"
train_samples_path = "/scratch/data/multilingual/mmmlu/splits/mmmlu_train.jsonl"

print("Loading reasoning samples to exclude from training set...")
reasoning_ids = []
with open(reasoning_samples_path) as f:
    for line in f:
        if line.strip():
            sample = json.loads(line)
            reasoning_ids.append(sample["idx"])
print(f"Loaded {len(reasoning_ids)} reasoning samples.")

print("Loading training samples and excluding reasoning samples...")
samples = pd.read_json(train_samples_path, lines=True)
samples = samples[~samples["idx"].isin(reasoning_ids)]
print(f"Loaded {len(samples)} training samples after exclusion.")

# Stratified downsample (n per language) at random
n_per_lang = 1000
print(f"Sampling up to {n_per_lang} examples per language...")
final_samples = []
for lang, group in samples.groupby("lang"):
    if len(group) > n_per_lang:
        sampled_group = group.sample(n_per_lang, random_state=42)
    else:
        print(f"Language {lang} has only {len(group)} samples, keeping all.")
        sampled_group = group
    final_samples.append(sampled_group)

final_samples = pd.concat(final_samples)
print(f"Final dataset has {len(final_samples)} samples.")

output_path = "/scratch/data/multilingual/mmmlu_more_qcms/splits/mmmlu_more_qcms_grpo_train.jsonl"
print(f"Saving final dataset to {output_path}...")
with open(output_path, "w") as f:
    # Shuffle the final samples before saving
    final_samples = final_samples.sample(frac=1, random_state=42).reset_index(drop=True)
    for _, row in final_samples.iterrows():
        json.dump(row.to_dict(), f, ensure_ascii=False)
        f.write("\n")
print("Done.")

print("Starting health check of the saved file (comparing agaisnt original reasoning samples)...")
with open(output_path) as f:
    saved_ids = set()
    for line in f:
        if line.strip():
            sample = json.loads(line)
            saved_ids.add(sample["idx"])
overlap = set(reasoning_ids) & saved_ids
if overlap:
    print(f"Error: Found {len(overlap)} overlapping samples between reasoning and final dataset!")
else:
    print("Health check passed: No overlap between reasoning samples and final dataset.")