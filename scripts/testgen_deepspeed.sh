#!/bin/bash

# DeepSpeed Inference TestGen Script
# This script runs test case generation with DeepSpeed tensor parallelism

data_path="Dataset"
data="testgeneval"
baseline_prompt="graph_tr"
llm_model="Qwen/Qwen2.5-Coder-7B-Instruct"
max_seq_len=8192
model_name="qwen2_5-7b"
name="qwen25_7b_deepspeed_testgen"
output_dir="./results/models/"
gnn_hidden_size=16
model_weight_path="./results/models/your_model.pt"  # SET YOUR MODEL PATH HERE
batch_size=4
tensor_parallel_size=2
dtype="bf16"
branch_limit=10000  # Limit number of branches per module

# DeepSpeed configuration
deepspeed_inference_config="configs/deepspeed_inference.json"

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Are NVIDIA drivers installed?"
    exit 1
fi

# Get the number of GPUs
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

echo "=========================================="
echo "DeepSpeed Inference TestGen Configuration"
echo "=========================================="
echo "GPUs detected: $gpu_count"
echo "Tensor parallel size: $tensor_parallel_size"
echo "Batch size: $batch_size"
echo "Max sequence length: $max_seq_len"
echo "Data type: $dtype"
echo "Model: $llm_model"
echo "Model weights: $model_weight_path"
echo "Branch limit: $branch_limit"
echo "=========================================="

# Validate configuration
if [ "$gpu_count" -lt "$tensor_parallel_size" ]; then
    echo "ERROR: Not enough GPUs! Required: $tensor_parallel_size, Available: $gpu_count"
    exit 1
fi

if [ ! -f "$model_weight_path" ]; then
    echo "WARNING: Model weight path does not exist: $model_weight_path"
    echo "Please set the correct path in the script."
    exit 1
fi

# Update DeepSpeed config with current tensor_parallel_size
if [ -f "$deepspeed_inference_config" ]; then
    tmp_config=$(mktemp)
    python3 -c "
import json
with open('$deepspeed_inference_config', 'r') as f:
    config = json.load(f)
config['tensor_parallel']['tp_size'] = $tensor_parallel_size
config['dtype'] = '$dtype'
with open('$tmp_config', 'w') as f:
    json.dump(config, f, indent=2)
"
    deepspeed_inference_config=$tmp_config
    echo "Updated DeepSpeed config with tp_size=$tensor_parallel_size"
fi

echo ""
echo "Starting DeepSpeed inference test case generation..."
echo ""

# Run testgen with DeepSpeed inference
python main.py --mode testgen \
    --seed 42 \
    --data_path $data_path \
    --data $data \
    --baseline_prompt $baseline_prompt \
    --llm_model $llm_model \
    --max_seq_length $max_seq_len \
    --batch_size $batch_size \
    --model_name $model_name \
    --name $name \
    --n_hidden $gnn_hidden_size \
    --graph_sampling \
    --num_gpu $tensor_parallel_size \
    --model_weight_path $model_weight_path \
    --dtype $dtype \
    --use_deepspeed_inference \
    --tensor_parallel_size $tensor_parallel_size \
    --deepspeed_inference_config $deepspeed_inference_config \
    --max_new_tokens 512 \
    --do_generate \
    --branch_limit $branch_limit

# Clean up temporary config if created
if [ -n "$tmp_config" ] && [ -f "$tmp_config" ]; then
    rm "$tmp_config"
fi

echo ""
echo "=========================================="
echo "DeepSpeed test case generation completed!"
echo "Results saved in: $output_dir"
echo "=========================================="
