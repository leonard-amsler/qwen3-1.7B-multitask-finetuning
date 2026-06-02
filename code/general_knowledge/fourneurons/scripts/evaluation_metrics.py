import re
import collections

def extract_boxed_answer(text: str) -> str:
    match = re.search(r'\\boxed{([^}]*)}', text)
    if match:
        return match.group(1).strip()
    return None # Format compliance failed

def compute_exact_match(prediction: str, truth: str) -> int:
    return int(prediction == truth)

def compute_f1(prediction: str, truth: str) -> float:
    pred_tokens = prediction.split()
    truth_tokens = truth.split()
    
    common = collections.Counter(pred_tokens) & collections.Counter(truth_tokens)
    num_same = sum(common.values())
    
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    if num_same == 0:
        return 0.0
        
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

# Example Usage
if __name__ == "__main__":
    generated_text = "Reasoning: Columbus sailed in 1492. Final Answer: \\boxed{1492}"
    ground_truth = "1492"
    
    extracted = extract_boxed_answer(generated_text)
    
    if extracted is not None:
        print(f"Format Compliant: True")
        print(f"Exact Match: {compute_exact_match(extracted, ground_truth)}")
        print(f"F1 Score: {compute_f1(extracted, ground_truth)}")
    else:
        print(f"Format Compliant: False")