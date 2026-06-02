Return 

V5

python -m fourneurons.distill.distill \
    --teacher  Qwen/Qwen3-14B-AWQ \
    --output   /scratch/data/distilled_cot_v5.jsonl \
    --max_tokens 1024 \
    --gpu_memory_utilization 0.85

python -m fourneurons.data.build_train \
    --output_dir          /scratch/data/train_v5 \
    --total               30000 \
    --max_variants        1 \
    --distilled_cot_cache /scratch/data/distilled_cot_v5.jsonl \
    --seed                42

python -m fourneurons.scripts.train \
    --dataset_dir       /scratch/data/train_v5 \
    --output_dir        /scratch/checkpoints/gk_v5 \
    --final_model_dir   /scratch/checkpoints/gk_v5/adapter \
    --num_epochs        1 \
    --learning_rate     2e-4 \
    --per_device_batch_size 2 \
    --grad_accum        8 \
    --lora_r            64 \
    --lora_alpha        128 \
    --max_seq_length    4096 \
    --eval_steps        200 \
    --save_steps        200 \
    --logging_steps     20 \
    --bf16

python -m fourneurons.scripts.merge_lora \
    --adapter_dir /scratch/checkpoints/gk_v5/adapter \
    --output_dir  /scratch/checkpoints/gk_v5/vllm \
    --base_model  Qwen/Qwen3-1.7B \
    --device      cpu \
    --temperature 0.7 \
    --top_p       0.9 \
    --top_k       20

V6
python -m fourneurons.distill.distill \
    --teacher Qwen/Qwen3-14B-AWQ \
    --output  /scratch/data/distilled_cot_v6_long.jsonl \
    --sources mmlu mmlu_world mmlu_pro_cot \
    --max_tokens 2048

python -m fourneurons.data.build_train \
    --output_dir          /scratch/data/train_v6 \
    --total               30000 \
    --max_variants        1 \
    --distilled_cot_cache /scratch/data/distilled_cot_v5.jsonl /scratch/data/distilled_cot_v6_long.jsonl \
    --seed                42

python -m fourneurons.scripts.train \
    --dataset_dir       /scratch/data/train_v6 \
    --output_dir        /scratch/checkpoints/gk_v6 \
    --final_model_dir   /scratch/checkpoints/gk_v6/adapter \
    --num_epochs        1 \
    --learning_rate     2e-4 \
    --per_device_batch_size 2 \
    --grad_accum        8 \
    --lora_r            64 \
    --lora_alpha        128 \
    --max_seq_length    4096 \
    --eval_steps        200 \
    --save_steps        200 \
    --logging_steps     20 \
    --bf16

python -m fourneurons.scripts.merge_lora \
    --adapter_dir /scratch/checkpoints/gk_v6/adapter \
    --output_dir  /scratch/checkpoints/gk_v6/vllm \
    --base_model  Qwen/Qwen3-1.7B \
    --device      cpu


V7

python -m fourneurons.distill.self_distill \
    --teacher          Qwen/Qwen3-14B-AWQ \
    --quantization     awq_marlin \
    --enable_thinking \
    --source_dataset   /scratch/data/train_v6 \
    --output_dir       /scratch/data/train_v7 \
    --cot_source_tag   qwen3_14b_thinking \
    --n_samples        2 \
    --max_tokens       4500 \
    --max_model_len    6144 \
    --temperature      0.6 \
    --top_p            0.95 \
    --top_k            20 \
    --gpu_memory_utilization 0.92 \
    --chunk_size       1000 \
    --phase            generate

python -m fourneurons.scripts.train \
    --dataset_dir       /scratch/data/train_v7 \
    --output_dir        /scratch/checkpoints/gk_v7 \
    --final_model_dir   /scratch/checkpoints/gk_v7/adapter \
    --num_epochs        1 \
    --learning_rate     1e-4 \
    --per_device_batch_size 2 \
    --grad_accum        8 \
    --lora_r            32 \
    --lora_alpha        64 \
    --max_seq_length    6144 \
    --eval_steps        200 \
    --save_steps        200 \
    --logging_steps     20 \
    --bf16

python -m fourneurons.scripts.merge_lora \
    --adapter_dir /scratch/checkpoints/gk_v7/adapter \
    --output_dir  /scratch/checkpoints/gk_v7/vllm \
    --base_model  Qwen/Qwen3-1.7B \
    --device      cpu

V8
python -m fourneurons.distill.self_distill \
    --teacher          Qwen/Qwen3-1.7B \
    --quantization     "" \
    --enable_thinking \
    --source_dataset   /scratch/data/train_v6 \
    --output_dir       /scratch/data/train_v8 \
    --cot_source_tag   self_distill_baseline \
    --n_samples        4 \
    --max_tokens       3000 \
    --max_model_len    4096 \
    --temperature      0.6 \
    --top_p            0.95 \
    --top_k            20 \
    --gpu_memory_utilization 0.92 \
    --chunk_size       2000 \
    --no_fallback_to_source \
    --phase            generate

python -m fourneurons.distill.self_distill \
    --teacher          Qwen/Qwen3-1.7B \
    --source_dataset   /scratch/data/train_v6 \
    --output_dir       /scratch/data/train_v8 \
    --cot_source_tag   self_distill_baseline \
    --n_samples        4 \
    --no_fallback_to_source \
    --phase            assemble

python -m fourneurons.scripts.train \
    --dataset_dir       /scratch/data/train_v8 \
    --output_dir        /scratch/checkpoints/gk_v8 \
    --final_model_dir   /scratch/checkpoints/gk_v8/adapter \
    --num_epochs        1 \
    --learning_rate     5e-5 \
    --per_device_batch_size 2 \
    --grad_accum        8 \
    --lora_r            8 \
    --lora_alpha        16 \
    --max_seq_length    4096 \
    --eval_steps        200 \
    --save_steps        200 \
    --logging_steps     20 \
    --bf16

python -m fourneurons.scripts.merge_lora \
    --adapter_dir /scratch/checkpoints/gk_v8/adapter \
    --output_dir  /scratch/checkpoints/gk_v8/vllm \
    --base_model  Qwen/Qwen3-1.7B \
    --device      cpu

V9

python -m fourneurons.scripts.train \
    --dataset_dir       /scratch/data/train_v8 \
    --output_dir        /scratch/checkpoints/gk_v9 \
    --final_model_dir   /scratch/checkpoints/gk_v9/adapter \
    --num_epochs 1 --learning_rate 2e-4 \
    --lora_r 64 --lora_alpha 128 --max_seq_length 4096 --bf16

python -m fourneurons.scripts.merge_lora \
    --adapter_dir /scratch/checkpoints/gk_v9/adapter \
    --output_dir  /scratch/checkpoints/gk_v9/vllm \
    --base_model  Qwen/Qwen3-1.7B \
    --device      cpu

V9b

python -m fourneurons.distill.self_distill \
    --teacher Qwen/Qwen3-1.7B --quantization "" \
    --source_dataset /scratch/data/train_v6 \
    --output_dir /scratch/data/train_v9b \
    --cache_path /scratch/data/train_v8/self_distill_cache.jsonl \
    --cot_source_tag self_distill_clean \
    --n_samples 4 \
    --select_best --max_thinking_chars 4000 \
    --phase assemble

python -m fourneurons.scripts.train \
    --dataset_dir       /scratch/data/train_v9b \
    --output_dir        /scratch/checkpoints/gk_v9b \
    --final_model_dir   /scratch/checkpoints/gk_v9b/adapter \
    --num_epochs 1 --learning_rate 2e-4 \
    --lora_r 64 --lora_alpha 128 --max_seq_length 4096 --bf16

python -m fourneurons.scripts.merge_lora \
    --adapter_dir /scratch/checkpoints/gk_v9b/adapter \
    --output_dir  /scratch/checkpoints/gk_v9b/vllm \
    --base_model  Qwen/Qwen3-1.7B --device cpu

V10

python -m fourneurons.distill.distill \
    --teacher Qwen/Qwen3-14B-AWQ --quantization awq_marlin \
    --output /scratch/data/distilled_cot_v10_contrastive.jsonl \
    --sources mmlu mmlu_pro_cot --reasoning_style contrastive \
    --max_tokens 2048 --max_model_len 4096

python -m fourneurons.data.build_train \
    --output_dir          /scratch/data/train_v10 \
    --total               30000 \
    --max_variants        1 \
    --distilled_cot_cache /scratch/data/distilled_cot_v5.jsonl \
                          /scratch/data/distilled_cot_v6_long.jsonl \
                          /scratch/data/distilled_cot_v10_contrastive.jsonl \
    --seed                42

python -m fourneurons.scripts.train \
    --dataset_dir       /scratch/data/train_v10 \
    --output_dir        /scratch/checkpoints/gk_v10 \
    --final_model_dir   /scratch/checkpoints/gk_v10/adapter \
    --num_epochs 1 --learning_rate 2e-4 \
    --lora_r 64 --lora_alpha 128 --max_seq_length 4096 --bf16

python -m fourneurons.scripts.merge_lora \
    --adapter_dir /scratch/checkpoints/gk_v10/adapter \
    --output_dir  /scratch/checkpoints/gk_v10/vllm \
    --base_model  Qwen/Qwen3-1.7B --device cpu


V11
python -m fourneurons.scripts.build_dpo_pairs \
  --model /scratch/checkpoints/gk_v9b/vllm \
  --dataset_dir /scratch/data/train_v9b \
  --output /scratch/data/dpo_pairs_v9b.jsonl \
  --n_examples 4000 --n_per_prompt 8 --max_tokens 2048

python -m fourneurons.scripts.train_dpo \
  --base_model /scratch/checkpoints/gk_v9b/vllm \
  --pairs /scratch/data/dpo_pairs_v9b.jsonl \
  --output_dir /scratch/checkpoints/gk_v11 \
  --final_model_dir /scratch/checkpoints/gk_v11/adapter \
  --num_epochs 1 --beta 0.1 --bf16

python -m fourneurons.scripts.merge_lora \
  --adapter_dir /scratch/checkpoints/gk_v11/adapter \
  --output_dir /scratch/checkpoints/gk_v11/vllm \
  --base_model /scratch/checkpoints/gk_v9b/vllm

V11b
python -m fourneurons.scripts.train_dpo \
  --base_model /scratch/checkpoints/gk_v9b/vllm \
  --pairs /scratch/data/dpo_pairs_v9b.jsonl \
  --output_dir /scratch/checkpoints/gk_v11b \
  --final_model_dir /scratch/checkpoints/gk_v11b/adapter \
  --num_epochs 3 --learning_rate 1e-5 --beta 0.05 --bf16

python -m fourneurons.scripts.merge_lora \
  --adapter_dir /scratch/checkpoints/gk_v11b/adapter \
  --output_dir /scratch/checkpoints/gk_v11b/vllm \
  --base_model /scratch/checkpoints/gk_v9b/vll


Test
for V in v6 v9b v10; do
  python -m fourneurons.eval.run_inference \
    --model /scratch/checkpoints/gk_$V/vllm \
    --input validation_samples/general_knowledge_dev_full.jsonl \
    --output /scratch/eval_$V/devfull_${V}_generations.jsonl \
    --n 1 --max_tokens 4096 --max_model_len 4096 --gpu_memory_utilization 0.90
  python -m fourneurons.eval.report_by_bucket \
    --generations /scratch/eval_$V/devfull_${V}_generations.jsonl \
    --output /scratch/eval_$V/devfull_${V}_report.json
done