head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)

accelerate launch \
    --machine_rank 0 \
    --num_processes 8 \
    --num_machines 2 \
    --rdzv_backend c10d \
    --main_process_ip $head_node_ip \
    --main_process_port 29500 \
    --mixed_precision fp16 \
    main.py --mode train \
    --seed 42 \
    --data_path Dataset \
    --data testgeneval \
    --baseline_prompt graph \
    --llm_model "Qwen/CodeQwen1.5-7B-Chat" \
    --max_seq_len 16384 \
    --batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_gpu 8 \
    --output_dir "./results/models/" \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --num_train_epochs 1 \
    --dtype bfloat16 \
    --debug \
    --use_lora