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

# Evaluate the count
if [ "$gpu_count" -eq 0 ]; then
    echo "No GPUs detected."
elif [ "$gpu_count" -eq 1 ]; then
    python main.py --mode train \
        --seed 42 \
        --data_path $data_path \
        --data $data \
        --baseline_prompt $baseline_prompt \
        --llm_model $llm_model \
        --max_seq_len $max_seq_len \
        --batch_size 1 \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --save_steps 320 \
        --validating_steps 5000 \
        --model_name $model_name \
        --name $name \
        --output_dir $output_dir \
        --overwrite_output_dir \
        --do_train \
        --do_eval \
        --num_gpu 1 \
        --n_hidden $gnn_hidden_size \
        --learning_rate $learning_rate \
        --max_grad_norm $max_grad_norm \
        --num_train_epochs $num_train_epochs \
        --dtype bf16 \
        --use_lora \
        --lora_r $lora_rank \
        --use_accelerate \
        --graph_sampling # \
        # --fuzz_model \
        # --start_fuzz_layer_index $start_fuzz_layer_index \
        # --end_fuzz_layer_index $end_fuzz_layer_index \ # uncomment if you want fuzzing model
        # --continue_training \
        # --checkpoint_path $checkpoint_path # uncomment if you want to continue training
else
    accelerate launch --debug --num_processes "$gpu_count"  main.py --mode train \
        --seed 42 \
        --data_path $data_path \
        --data $data \
        --baseline_prompt $baseline_prompt \
        --llm_model $llm_model \
        --max_seq_len $max_seq_len \
        --batch_size 1 \
        --gradient_accumulation_steps $gradient_accumulation_steps \
        --save_steps 320 \
        --validating_steps 5000 \
        --model_name $model_name \
        --name $name \
        --output_dir $output_dir \
        --overwrite_output_dir \
        --do_train \
        --do_eval \
        --num_gpu "$gpu_count" \
        --n_hidden $gnn_hidden_size \
        --learning_rate $learning_rate \
        --max_grad_norm $max_grad_norm \
        --num_train_epochs $num_train_epochs \
        --dtype bf16 \
        --use_lora \
        --lora_r $lora_rank \
        --use_accelerate \
        --graph_sampling \
        # --fuzz_model \
        # --start_fuzz_layer_index $start_fuzz_layer_index \
        # --end_fuzz_layer_index $end_fuzz_layer_index \ # uncomment if you want fuzzing model
        # --continue_training \
        # --checkpoint_path $checkpoint_path # uncomment if you want to continue training
fi
