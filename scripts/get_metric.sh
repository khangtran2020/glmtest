accelerate launch \
    --num_processes 1 \
    --num_machines 1 \
    main.py --mode metric \
    --seed 42 \
    --data_path Dataset \
    --data testgeneval \
    --baseline_prompt graph_tr \
    --llm_model "Qwen/Qwen2.5-Coder-3B-Instruct" \
    --max_seq_len 28000 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --save_steps 1200 \
    --validating_steps 1200 \
    --num_gpu 1 \
    --gen_file_path "./results/generated/testing_Qwen2.5_3B_graph_tr_accelerate_4096_2GPUs.json" \
    --name "testing_Qwen2.5_3B_graph_tr_accelerate" \
    --output_dir "./results/models/" \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --model_name "qwen2_5-3b"\
    --n_hidden 16 \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --num_train_epochs 3 \
    --dtype bf16 \
    --use_lora \
    --use_accelerate \
    --graph_sampling \
    --do_test \
    --max_new_token 4096 \
    --debug
    # --continue_training \
    
    
    
