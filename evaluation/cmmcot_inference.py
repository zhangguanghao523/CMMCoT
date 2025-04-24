import argparse
import os
import argparse
from typing import Any, Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

import json
from tqdm import tqdm
import pandas as pd
from PIL import Image
import fs
import traceback

# 新的模型引用
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor, Qwen2VLModel
from transformers.generation.utils import GenerationMixin
from transformers.cache_utils import StaticCache, DynamicCache
from qwen_vl_utils import process_vision_info
from evaluation.qwen2_vl_cmmcot import CustomQwen2VLForConditionalGeneration
import re
import time



# 自定义数据集类
class CustomDataset:
    def __init__(self, questions, image_folder, processor):
        self.questions = questions
        self.image_folder = image_folder
        self.processor = processor

    def __getitem__(self, index):
        line = self.questions[index]
        image_files = line["image"]
        qs = line["conversations"][0]["value"]
        image_content = [{"type": "image", "image": os.path.join(self.image_folder, image_file)} for image_file in image_files]
        messages = [
            {
                "role": "user",
                "content": image_content + [{"type": "text", "text": qs}],
            }
        ]
        print("messages:", messages)
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)

        return {
            'input_ids': text,
            'image_inputs': image_inputs,
            'line': line
        }

    def __len__(self):
        return len(self.questions)


def create_data_loader(questions, image_folder, processor, batch_size=1, num_workers=4):
    dataset = CustomDataset(questions, image_folder, processor)
    return dataset


def eval_model(args):
    model = CustomQwen2VLForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype="auto", 
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(args.model_path)

    questions = fs.read_json(args.question_file)
    for question in questions:
        question['ID'] = len(questions)
        
    data_loader = create_data_loader(questions, args.image_folder, processor)

    output_result = []
    total_tokens_generated = 0
    total_inference_time = 0
    TOKEN_TO_EXCLUDE = 151655 # Define the token ID to exclude
    
    for item in tqdm(data_loader):
        try:
            input_ids = item['input_ids']
            image_inputs = item['image_inputs']

            model.reset_state() # 重新初始化q k v

            # 更新图像输入和处理器
            model.set_image_inputs(image_inputs)
            model.set_processor(processor)
            
            inputs = processor(
                text=[input_ids],
                images=image_inputs,
                padding=True,
                return_tensors="pt",
            ).to("cuda")


            # Start timer
            start_time = time.time()

            # 新的生成逻辑
            generated_ids = model.generate(
                **inputs, 
                max_new_tokens=4096,
                temperature=0.8,
                top_k=50,
                top_p=0.95,
                do_sample=True,
                repetition_penalty=1.05,
            )

            # End timer
            end_time = time.time()
            # Calculate time taken
            inference_time = end_time - start_time
            total_inference_time += inference_time

            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            print("generated_ids_trimmed:", generated_ids_trimmed)
            num_tokens_generated = sum(len(ids) for ids in generated_ids_trimmed)

            num_tokens_generated = sum(
                len(ids) - sum(1 for id in ids if id == TOKEN_TO_EXCLUDE)
                for ids in generated_ids_trimmed
            )
            print("num_tokens_generated:", num_tokens_generated)
            total_tokens_generated += num_tokens_generated

            output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=False, clean_up_tokenization_spaces=False)[0]
            print("output_text:", output_text)
            output_result.append({
                "question_id": item['line']["id"],
                "prompt": item['line']["conversations"][0]["value"],
                "target": item['line']["conversations"][1]["value"],
                "gen": output_text.strip(),
            })
        except Exception:
            traceback.print_exc()

    fs.write_json(output_result, args.answers_file, indent=4)

    # Calculate average inference time per token
    if total_tokens_generated > 0:
        average_time_per_token = total_inference_time / total_tokens_generated
    else:
        average_time_per_token = 0

    print(f"Total tokens generated (excluding token {TOKEN_TO_EXCLUDE}): {total_tokens_generated}")
    print(f"Total inference time: {total_inference_time:.4f} seconds")
    print(f"Average inference time per token: {average_time_per_token:.6f} seconds/token")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="/mnt/workspace/xiaoxi/xiaoxi-all/model_logs/vllm_factory/qwen2_vl-7b/mcot/full_sft_mcot-interleave-noimglabel-singleturn_2024_12_19_23_39_59/checkpoint-1000")
    parser.add_argument("--image-folder", type=str, default="oss://shaoquan-data/xiaoxi-all/")
    parser.add_argument("--question-file", type=str, default="oss://mvap-public-data/shufangxun/dataset/llvm/SFT/VQA/Mantis-Instruct/txt_test/nlvr2_test.json")
    parser.add_argument("--answers-file", type=str, default="/mnt/workspace/xiaoxi/code/moe-vllm_bench/benchmark/nlvr2_test/debug.json")
    args = parser.parse_args()
    eval_model(args)
