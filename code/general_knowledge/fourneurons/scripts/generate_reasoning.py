import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
INPUT_FILE = "../data/train_data.json"  
OUTPUT_FILE = "../data/train_data_with_reasoning.json" 

def main():
    print(f"Loading model {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float32, 
        device_map="cpu",        
        low_cpu_mem_usage=True
    )

    with open(INPUT_FILE, "r") as f:
        data = [json.loads(line) for line in f]

    print(f"Processing {len(data[:5])} items for testing...")
    
    with open(OUTPUT_FILE, "w") as f_out:
        for i, item in enumerate(data[:5]): 
            question_content = item["messages"][0]["content"]
            gold_letter = item["gold_letter"]
            gold_text = item["gold_text"]

            prompt = (
                f"{question_content}\n\n"
                f"The correct answer to this question is {gold_letter} ({gold_text}). "
                "Write a 2-5 sentence explanation of why it is correct, and briefly why the others are wrong."
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True
            )
            
            reasoning = tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:], 
                skip_special_tokens=True
            )

            item["teacher_reasoning"] = reasoning.strip()
            f_out.write(json.dumps(item) + "\n")
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i + 1}/{len(data)}")

    print(f"Finished. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()