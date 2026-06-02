import os
import re
import json
from datasets import load_dataset, concatenate_datasets, ClassLabel, Value

CATEGORY_MAPPING = {
    "science": ["math", "physics", "chemistry", "biology", "computer science", "engineering", "health"],
    "history": ["history", "geography"],
    "social_sciences": ["psychology", "sociology", "anthropology"],
    "humanities": ["philosophy", "law", "business", "economics"],
    "commonsense": ["other", "commonsense"]  # ECQA + other
}

def clean_answer(answer: str) -> str:
    return re.sub(r'\s+', ' ', str(answer).strip().lower())

def parse_chain_of_thoughts(cot_input) -> str:
    try:
        if isinstance(cot_input, list):
            return "\n".join(str(item) for item in cot_input)
        elif isinstance(cot_input, str):
            try:
                cot_list = json.loads(cot_input)
                return "\n".join(str(item) for item in cot_list)
            except:
                return cot_input
    except Exception as e:
        return str(cot_input)

def map_category_to_macro(category: str) -> str:
    category_lower = category.lower().strip()
    
    for macro_cat, subjects in CATEGORY_MAPPING.items():
        if category_lower in subjects:
            return macro_cat
    
    print(f"⚠️ Unknown category '{category}', defaulting to 'world_knowledge'")
    return "world_knowledge"

def format_mmlu_pro_cot(example):
    question = example['question']
    answer_letter = example['answer']
    cot_text = parse_chain_of_thoughts(example['chain_of_thoughts'])
    macro_category = map_category_to_macro(example['category'])
    
    lines = question.split('\n')
    choices = []
    for line in lines:
        line = line.strip()
        if line and len(line) > 2 and line[0] in 'ABCDEFGHIJ' and line[1] == '.':
            choice_text = line[3:].strip()
            choices.append(choice_text)
    
    question_part = '\n'.join([l for l in lines if not (len(l.strip()) > 2 and l.strip()[0] in 'ABCDEFGHIJ' and l.strip()[1] == '.')])
    
    options_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
    content = f"Answer the question clearly.\n{question_part}\nOptions:\n{options_str}"
    
    response = f"<think>\n{cot_text}\n</think>\n\nThe answer is \\boxed{{{answer_letter}}}"
    
    messages = [
        {"role": "user", "content": content},
        {"role": "assistant", "content": response}
    ]
    
    return {
        "messages": messages, 
        "macro_category": macro_category,
        "gold_letter": answer_letter,
        "category": example['category']
    }

def format_ecqa(example):
    question = example['q_text']
    choices = [example['q_op1'], example['q_op2'], example['q_op3'], example['q_op4'], example['q_op5']]
    answer_text = example['q_ans']
    explanation = example['taskB']
    
    answer_idx = 0
    for i, choice in enumerate(choices):
        if choice.lower() == answer_text.lower():
            answer_idx = i
            break
    
    answer_letter = chr(65 + answer_idx)
    
    options_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
    content = f"Answer the question clearly.\n{question}\nOptions:\n{options_str}"
    
    response = f"<think>\n{explanation}\n</think>\n\nThe answer is \\boxed{{{answer_letter}}}"
    
    messages = [
        {"role": "user", "content": content},
        {"role": "assistant", "content": response}
    ]
    
    return {
        "messages": messages, 
        "macro_category": "commonsense",  # ECQA is all commonsense QA
        "gold_letter": answer_letter,
        "category": "commonsense"
    }

def print_category_distribution(split_dataset):
    
    category_feature = split_dataset['train'].features['macro_category']
    category_names = category_feature.names  # ["history", "science", "world_knowledge"]
    
    for split_name, dataset in [("Train", split_dataset['train']), ("Test", split_dataset['test'])]:
        print(f"\n{split_name} Set:")
        print("-" * 50)
        
        # Count categories
        categories = {}
        for example in dataset:
            cat_idx = example['macro_category']  # This is an integer
            cat_name = category_names[cat_idx]   # Convert to name
            categories[cat_name] = categories.get(cat_name, 0) + 1
        
        total = len(dataset)
        
        for cat_name in sorted(categories.keys()):
            count = categories[cat_name]
            percentage = (count / total) * 100
            print(f"  {cat_name:20s}: {count:6d} examples ({percentage:6.2f}%)")
        
        print(f"  {'-'*40}")
        print(f"  {'TOTAL':20s}: {total:6d} examples (100.00%)")
    
    print("="*80 + "\n")

def prepare_data(samples_per_category=50000, save_to_disk=False):
    """Prepare data using MMLU-Pro-CoT and ECQA datasets."""
    
    print("Loading MMLU-Pro-CoT dataset...")
    mmlu_dataset = load_dataset("UW-Madison-Lee-Lab/MMLU-Pro-CoT-Train-Labeled", split="train")
    print(f"Loaded MMLU-Pro-CoT with {len(mmlu_dataset)} examples")
    
    mmlu_formatted = mmlu_dataset.map(format_mmlu_pro_cot, remove_columns=mmlu_dataset.column_names)
    
    print("Loading ECQA dataset...")
    ecqa_dataset = load_dataset("tasksource/ecqa", split="train")
    print(f"Loaded ECQA with {len(ecqa_dataset)} examples")
    
    ecqa_formatted = ecqa_dataset.map(format_ecqa, remove_columns=ecqa_dataset.column_names)
    
    print("Combining datasets...")
    combined_dataset = concatenate_datasets([mmlu_formatted, ecqa_formatted])
    print(f"Combined dataset with {len(combined_dataset)} examples")
    
    if len(combined_dataset) > samples_per_category:
        combined_dataset = combined_dataset.shuffle(seed=42).select(range(samples_per_category))
        print(f"Limited to {len(combined_dataset)} examples")
    
    all_categories = list(CATEGORY_MAPPING.keys())
    combined_dataset = combined_dataset.cast_column("macro_category", ClassLabel(names=all_categories))
    
    split_dataset = combined_dataset.train_test_split(
        test_size=0.1, 
        seed=42,
        stratify_by_column="macro_category"
    )
    
    print_category_distribution(split_dataset)
    
    if save_to_disk:
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        os.makedirs(data_dir, exist_ok=True)
        
        train_path = os.path.join(data_dir, "train_data")
        test_path = os.path.join(data_dir, "test_data")
        
        print(f"\nSaving datasets to {data_dir}...")
        split_dataset['train'].save_to_disk(train_path)
        split_dataset['test'].save_to_disk(test_path)
        print(f"✅ Train data saved to {train_path}")
        print(f"✅ Test data saved to {test_path}")
    
    return split_dataset

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true", help="Save processed datasets to disk")
    parser.add_argument("--samples", type=int, default=50000, help="Number of samples to use")
    args = parser.parse_args()
    
    data = prepare_data(samples_per_category=args.samples, save_to_disk=args.save)
    print(f"\nTrain size: {len(data['train'])}")
    print(f"Test size: {len(data['test'])}")
    print("\nSample format:")
    print(data['train'][0]['messages'])