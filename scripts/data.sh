export CUDA_VISIBLE_DEVICES=0

#!/bin/bash

checkpoint_path="" # set this if continue training.
data_path="Dataset"
data="testgeneval"
baseline_prompt="graph_tr" # other prompts: ['code', 'code_tr', 'graph', 'code_baseline']
llm_model="Qwen/Qwen2.5-Coder-7B-Instruct"
max_seq_len=28000
gradient_accumulation_steps=32
model_name="qwen2_5-7b" # name to save the prompts and data for fast running next time
name="qwen25_7b_graph_tr_3_epochs" # name to log to wandb so that we can keep track of the running
output_dir="./results/models/" # path to save the trained models
learning_rate=1.48e-4
max_grad_norm=3.0
num_train_epochs=3
gnn_hidden_size=16
lora_rank=32
start_fuzz_layer_index=7
end_fuzz_layer_index=8

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi not found. Are NVIDIA drivers installed?"
    exit 1
fi

# Get the number of GPUs
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)


python main.py --mode data \
    --seed 42 \
    --data_path $data_path \
    --data $data \
    --batch_size 1 \
    --name $name