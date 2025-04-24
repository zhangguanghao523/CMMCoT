#!/bin/bash

datetime=$(date +"%Y_%m_%d_%H_%M_%S")

# deepspeed config
deepspeed_config='examples/deepspeed/ds_z3_config.json'

# vllm model
vllm="/mnt/workspace/xiaoxi/model_pretrained/qwen_lab/Qwen2-VL-7B-Instruct"

# output dir
output_dir="/mnt/workspace/xiaoxi/model_logs/model_cmmcot_${datetime}"
run_name="${output_dir##*/}"


user_params="--deepspeed=${deepspeed_config} \
    --model_name_or_path=${vllm} \
    --stage=sft \
    --do_train=True \
    --finetuning_type=full \
    --dataset=rs_birds_to_word_2k,rs_caption_grit_dual_20k,rs_caption_grit_quad_10k,rs_caption_grit_triple_20k,rs_coregerence_shikra_dual_10k,rs_coregerence_vocot_dual_50k,rs_coregerence_vocot_quad_10k,rs_coregerence_vocot_triple_30k,rs_dreamsim_15k,rs_imagecode_16k_merge_v1,rs_cos_174k,rs_nextqa_4k,rs_nlvr2_86k,rs_spotthediff_2k,rs_star_3k,rs_vist,rs_vocot_80k_single \
    --mix_strategy=interleave_over \
    --interleave_probs=0.007247,0.057426,0.028713,0.057421,0.037577,0.141972,0.028624,0.085798,0.036991,0.041646,0.143566,0.011112,0.183730,0.013073,0.008706,0.001545,0.114853 \
    --new_special_tokens=\"<IMG>,</IMG>\" \
    --template=qwen2_vl \
    --cutoff_len=8192 \
    --dataloader_num_workers=1 \
    --streaming \
    --buffer_size=256 \
    --preprocessing_batch_size=256 \
    --dispatch_batches=False \
    --max_steps=3000 \
    --overwrite_cache=True \
    --output_dir=${output_dir} \
    --logging_steps=1 \
    --save_steps=50 \
    --overwrite_output_dir=True \
    --run_name=${run_name} \
    --report_to=wandb \
    --per_device_train_batch_size=1 \
    --gradient_accumulation_steps=8 \
    --learning_rate=1.0e-5 \
    --lr_scheduler_type=cosine \
    --warmup_ratio=0.1 \
    --bf16=True \
    " 

run_script='src/train.py'

nebula_project=nebula_project
algo_name='pytorch220'

queue=queue

# 256
worker_count=32


UserId=UserId
AccessKeyId=AccessKeyId
AccessKey=AccessKey
BucketName=BucketName
Endpoint=Endpoint
NAS_ADDR=NAS_ADDR


echo "${user_params}"
nebulactl run mdl --nebula_project="$nebula_project" \
                  --queue="$queue" \
                  --entry="$run_script" \
                  --algo_name="$algo_name" \
                  --worker_count="$worker_count" \
                  --user_params="$user_params" \
                  --file.cluster_file=./cluster.json \
                  --job_name=vllm_prompt_pre \
                  --user_id="$UserId" \
                  --oss_access_id="$AccessKeyId" \
                  --oss_access_key="$AccessKey" \
                  --oss_bucket="$BucketName" \
                  --oss_endpoint="$Endpoint" \
                  --enable_oss_cache=true \
                  --nas_file_system_id="$NAS_ADDR" \
                  --env="DS_SKIP_CUDA_CHECK=1" \
