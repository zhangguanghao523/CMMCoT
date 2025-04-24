#!/bin/bash


IMAGE_FOLDER="oss://shaoquan-data/xiaoxi-all/"
CKPT_NAME=cmmcot

CUDA_VISIBLE_DEVICES=0 python -m evaluation.cmmcot_inference \
     --model-path /local/path/to/your/model \
     --question-file oss://mvap-public-data/mantis.json \
     --image-folder ${IMAGE_FOLDER} \
     --answers-file /mnt/workspace/${CKPT_NAME}.json 
