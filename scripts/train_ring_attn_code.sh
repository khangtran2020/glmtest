master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=${master_addr:-"127.0.0.1"}
export CURRENT_RANK=${SLURM_PROCID:-"0"}
export OMP_NUM_THREADS=1
worker_list=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | tr '\n' ' ')
n_node=${SLURM_JOB_NUM_NODES:-1}

echo "MASTER_ADDR="$MASTER_ADDR
echo "JobID: $SLURM_JOB_ID | Full list: $worker_list"

model_max_length=32768
rope_theta=3580165449

accelerate launch --multi_gpu --num_processes 2 main.py --mode train \
    --seed 42 \
    --data_path Dataset \
    --data testgeneval \
    --baseline_prompt code \
    --llm_model "Qwen/CodeQwen1.5-7B-Chat" \
    --max_seq_len 16384 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --num_gpu 1 \
    --output_dir "./results/models/" \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --num_train_epochs 1 \
    --model_max_length $model_max_length \
    --rope_theta $rope_theta \
    --dtype bfloat16 \
    --debug \
    --use_lora \
    --longlora 