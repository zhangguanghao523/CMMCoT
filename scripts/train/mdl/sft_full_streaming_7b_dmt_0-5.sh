#!/bin/bash

datetime=$(date +"%Y_%m_%d_%H_%M_%S")

# deepspeed config
deepspeed_config='examples/deepspeed/ds_z3_config.json'

# vllm model
vllm="/mnt/workspace/xiaoxi/model_logs/model_cmmcot-7b_stage1"

# output dir
output_dir="/mnt/workspace/xiaoxi/model_logs/model_cmmcot-7b_stage2"
run_name="${output_dir##*/}"

user_params="--deepspeed=${deepspeed_config} \
    --model_name_or_path=${vllm} \
    --stage=sft \
    --do_train=True \
    --finetuning_type=full \
    --dataset=CLEVR_Math_MathV360K,FigureQA_MathV360K,GeoQA_Plus_MathV360K,Geometry3K_MathV360K,IconQA_MathV360K,MapQA_MathV360K,PMC-VQA_MathV360K,Super_CLEVR_MathV360K,TabMWP_MathV360K,UniGeo_MathV360K,VisualWebInstruct_filtered,VizWiz_MathV360K,ai2d_cauldron,ai2d_gpt4v,ai2d_internvl,allava_instruct_laion4v,allava_instruct_vflan4v,aokvqa_cauldron,chart2text_cauldron,chartqa_cauldron,chrome_writting,clevr_cauldron,diagram_image_to_text_cauldron,dvqa_cauldron,figureqa_cauldron,geo170k_align,geo170k_qa,geo3k,geomverse_cauldron,hateful_memes_cauldron,hitab_cauldron,hme100k,iam_cauldron,iconqa_cauldron,iiit5k,image_textualization_filtered,infographic_gpt4v,infographic_vqa,infographic_vqa_llava_format,intergps_cauldron,k12_printing,llavar_gpt4_20k,lrv_chart,lrv_normal_filtered,mavis_math_metagen,mavis_math_rule_geo,multihiertt_cauldron,orand_car_a,raven_cauldron,robut_sqa_cauldron,robut_wikisql_cauldron,robut_wtq_cauldron,scienceqa_cauldron,scienceqa_nona_context,screen2words_cauldron,sharegpt4v_coco,sharegpt4v_knowledge,sharegpt4v_llava,sharegpt4v_sam,sroie,st_vqa_cauldron,tabmwp_cauldron,tallyqa_cauldron,textcaps,textocr_gpt4v,ureader_cap,ureader_ie,ureader_kg,ureader_qa,vistext_cauldron,visual7w_cauldron,visualmrc_cauldron,vqarad_cauldron,vsr_cauldron,websight_cauldron,rs_birds_to_word_2k,rs_caption_grit_dual_20k,rs_caption_grit_quad_10k,rs_caption_grit_triple_20k,rs_coregerence_shikra_dual_10k,rs_coregerence_vocot_dual_50k,rs_coregerence_vocot_quad_10k,rs_coregerence_vocot_triple_30k,rs_dreamsim_15k,rs_imagecode_16k_merge_v1,rs_cos_174k,rs_nextqa_4k,rs_nlvr2_86k,rs_spotthediff_2k,rs_star_3k,rs_vist,rs_vocot_80k_single \
    --mix_strategy=interleave_over \
    --interleave_probs=0.000955,0.003180,0.003193,0.001758,0.004084,0.000945,0.006499,0.001562,0.004059,0.002160,0.047656,0.001194,0.000439,0.000879,0.002242,0.009038,0.003614,0.002989,0.004873,0.003301,0.001595,0.012654,0.000053,0.036157,0.018078,0.010891,0.012262,0.000378,0.001681,0.001536,0.000451,0.013468,0.001023,0.004936,0.000360,0.018002,0.000358,0.000794,0.000382,0.000231,0.046398,0.003578,0.000321,0.001897,0.015792,0.018077,0.001377,0.000361,0.007592,0.001538,0.013556,0.006914,0.000899,0.003473,0.002843,0.009043,0.000359,0.005422,0.001625,0.006077,0.003117,0.004107,0.017840,0.003967,0.004539,0.016530,0.003132,0.006789,0.045732,0.001801,0.002596,0.000546,0.000056,0.000389,0.001807,0.003624,0.028713,0.014357,0.028710,0.018788,0.070986,0.014312,0.042899,0.018496,0.020823,0.071783,0.005556,0.091865,0.006537,0.004353,0.000772,0.057426 \
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
