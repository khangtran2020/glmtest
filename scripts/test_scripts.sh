#!/bin/bash

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
model_weight_path="" # set the trained model weight
start_fuzz_layer_index=7
end_fuzz_layer_index=8

# Check if nvidia-smi is available
if ! command -v nvidia-smi &> /dev/null; then
    echo "nvidia-smi not found. Are NVIDIA drivers installed?"
    exit 1
fi

# Get the number of GPUs
gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

# Evaluate the count
if [ "$gpu_count" -eq 0 ]; then
    echo "No GPUs detected."
elif [ "$gpu_count" -eq 1 ]; then
    python main.py --mode test \
        --seed 42 \
        --data_path $data_path \
        --data $data \
        --baseline_prompt $baseline_prompt \
        --llm_model $llm_model \
        --max_seq_len $max_seq_len \
        --batch_size 1 \
        --model_name $model_name \
        --name $name \
        --n_hidden $gnn_hidden_size \
        --use_accelerate \
        --graph_sampling \
        --num_gpu 1 \
        --model_weight_path $model_weight_path # \
        # --fuzz_model \
        # --start_fuzz_layer_index $start_fuzz_layer_index \
        # --end_fuzz_layer_index $end_fuzz_layer_index \ # uncomment if you want fuzzing model
else
    accelerate launch --debug --num_processes "$gpu_count"  main.py --mode test \
        --seed 42 \
        --data_path $data_path \
        --data $data \
        --baseline_prompt $baseline_prompt \
        --llm_model $llm_model \
        --max_seq_len $max_seq_len \
        --batch_size 1 \
        --model_name $model_name \
        --name $name \
        --n_hidden $gnn_hidden_size \
        --use_accelerate \
        --graph_sampling \
        --num_gpu "$gpu_count" \
        --model_weight_path $model_weight_path # \
        # --fuzz_model \
        # --start_fuzz_layer_index $start_fuzz_layer_index \
        # --end_fuzz_layer_index $end_fuzz_layer_index \ # uncomment if you want fuzzing model
fi



# accelerate launch main.py --mode test \
#     --seed 42 \
#     --data_path Dataset \
#     --data testgeneval \
#     --baseline_prompt graph \
#     --llm_model "HuggingFaceTB/SmolLM2-135M-Instruct" \
#     --max_seq_len 16384 \
#     --batch_size 1 \
#     --gradient_accumulation_steps 16 \
#     --save_steps 1000 \
#     --validating_steps 1000 \
#     --num_gpu 1 \
#     --name "testing_small_graph_accelerate" \
#     --output_dir "./results/models/" \
#     --overwrite_output_dir \
#     --do_train \
#     --do_eval \
#     --n_hidden 16 \
#     --learning_rate 5e-5 \
#     --max_grad_norm 1.0 \
#     --num_train_epochs 3 \
#     --dtype bfloat16 \
#     --use_lora \
#     --use_accelerate \
#     --graph_sampling \
#     --model_weight_path results/models/testing_small_graph_accelerate/final_model \
#     --do_test