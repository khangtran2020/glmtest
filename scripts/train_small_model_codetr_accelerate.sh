export CUDA_VISIBLE_DEVICES=0

accelerate launch main.py --mode train \
    --seed 42 \
    --data_path Dataset \
    --data testgeneval \
    --baseline_prompt graph \
    --llm_model "HuggingFaceTB/SmolLM2-135M-Instruct" \
    --max_seq_len 16384 \
    --batch_size 1 \
    --gradient_accumulation_steps 1 \
    --save_steps 1 \
    --num_gpu 1 \
    --name "testing_small_graph_accelerate" \
    --output_dir "./results/models/" \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --learning_rate 5e-5 \
    --max_grad_norm 1.0 \
    --num_train_epochs 1 \
    --dtype bfloat16 \
    --debug \
    --use_lora \
    --use_accelerate \
    --graph_sampling \
    --do_test