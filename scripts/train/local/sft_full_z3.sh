#!/bin/bash

datetime=$(date +"%Y_%m_%d_%H_%M_%S")


output_dir="/mnt/workspace/xiaoxi/model_logs/qwen2_vl-7b/sft_cmmcot_${datetime}"

run_name="${output_dir##*/}"


vllm="/mnt/workspace/xiaoxi/model_pretrained/qwen_lab/Qwen2-VL-7B-Instruct"


deepspeed --include="localhost:0,1" --master_port 22115 src/train.py \
    --model_name_or_path ${vllm} \
    --stage sft \
    --do_train True \
    --finetuning_type full \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --dataset birds_to_word_2k \
    --new_special_tokens "<IMG>,</IMG>" \
    --template qwen2_vl \
    --cutoff_len 4096 \
    --max_samples 1024 \
    --overwrite_cache True \
    --preprocessing_num_workers 32 \
    --output_dir ${output_dir} \
    --logging_steps 1 \
    --save_steps 10 \
    --plot_loss True \
    --overwrite_output_dir True \
    --run_name ${run_name} \
    --report_to wandb \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1.0e-5 \
    --num_train_epochs 1.0 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --bf16 True \
    --gradient_checkpointing 
