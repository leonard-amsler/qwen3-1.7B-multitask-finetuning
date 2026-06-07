import json
import os
import random
from fourneurons.data.safetybench_data_loader import load_safetybench_test

SEED = 42
VAL_RATIO = 0.10
OUTPUT_DIR = "/scratch/data/safety/safetybench/splits"

def build_splits():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_safetybench_test()
    random.seed(SEED)
    random.shuffle(data)

    cut = int(len(data) * (1 - VAL_RATIO))
    splits = {"train": data[:cut], "val": data[cut:]}

    for name, split in splits.items():
        path = os.path.join(OUTPUT_DIR, f"safetybench_{name}.jsonl")
        with open(path, "w") as f:
            for ex in split:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"{name}: {len(split)} samples saved to {path}")
        print(f"Example from {name} split:")
        print(json.dumps(split[0], indent=2, ensure_ascii=False))

if __name__ == "__main__":
    build_splits()



