import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from vllm import LLM, SamplingParams
import torch

def merge_adapter_to_cache():
    ADAPTER_PATH = os.path.join(os.path.dirname(__file__), "final_gk_model")
    BASE_MODEL = "Qwen/Qwen3-1.7B"
    MERGED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "final_gk_model_vllm")
    
    if not os.path.exists(ADAPTER_PATH):
        raise FileNotFoundError(f"Adapter not found at {ADAPTER_PATH}. Please train the model first using train.py")
    
    if not os.path.exists(MERGED_MODEL_PATH):
        print(f"Merging LoRA adapter with base model...")
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="cpu",  # Use CPU for merging to avoid CUDA issues
            dtype=torch.float16
        )
        
        print(f"Loading LoRA adapter from: {ADAPTER_PATH}")
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        
        print(f"Merging and saving to: {MERGED_MODEL_PATH}")
        model = model.merge_and_unload()
        model.save_pretrained(MERGED_MODEL_PATH)
        
        tokenizer = AutoTokenizer.from_pretrained(ADAPTER_PATH)
        tokenizer.save_pretrained(MERGED_MODEL_PATH)
        print("✅ Merge complete!")
    else:
        print(f"Using pre-merged model from: {MERGED_MODEL_PATH}")
    
    return MERGED_MODEL_PATH

def generate_with_vllm(model_path):
    print(f"\nLoading model with vLLM...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(model=model_path, dtype="float16")
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=16384, n=1)
    
    input_file  = "../../validation_samples/general_knowledge.jsonl"
    output_file = "../../results/my_general_knowledge_gens.jsonl"
    
    print(f"Reading from: {input_file}")
    print(f"Writing to: {output_file}")
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(input_file) as fin, open(output_file, "w") as fout:
        count = 0
        for line in fin:
            line = line.strip()
            if not line:
                continue
            
            row = json.loads(line)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. For multiple-choice questions, "
                        "reason step by step, then provide your final answer as a single "
                        "letter inside \\boxed{}, for example: \\boxed{A}. "
                        "Do not include anything after the \\boxed{}."
                    )
                },
                {"role": "user", "content": row["prompt"]}
            ]        
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Generate using vLLM (fast!)
            outputs = llm.generate([prompt], sampling_params)
            row["completions"] = [o.text for o in outputs[0].outputs]
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            
            count += 1
            if count % 10 == 0:
                print(f"Processed {count} examples...")
    
    print(f"\n✅ Done! Processed {count} examples.")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    # Merge adapter first
    model_path = merge_adapter_to_cache()
    
    # Then generate with vLLM
    generate_with_vllm(model_path)