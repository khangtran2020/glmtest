accelerate launch --num_processes 2 main.py --mode train \
    --seed 42 \
    --data_path Dataset \
    --data testgeneval \
    --baseline_prompt graph \
    --llm_model "Qwen/CodeQwen1.5-7B-Chat" \
    --max_seq_len 16384 \
    --batch_size 1 \
    --gradient_accumulation_steps 1 \
    --num_gpu 2 \
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

export CUDA_VISIBLE_DEVICES=0

accelerate launch --debug --num_processes 4  main.py --mode train \
    --seed 42 \
    --data_path Dataset \
    --data testgeneval \
    --baseline_prompt graph_tr \
    --llm_model "Qwen/Qwen2.5-Coder-1.5B" \
    --max_seq_len 32768 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_steps 10000 \
    --validating_steps 10000 \
    --num_gpu 4 \
    --model_name "qwen2_5-1_5b"\
    --name "testing_qwen25_15_graph_tr_accelerate" \
    --output_dir "./results/models/" \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --n_hidden 16 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --num_train_epochs 3 \
    --dtype bf16 \
    --use_lora \
    --use_accelerate \
    --graph_sampling \
    --do_test