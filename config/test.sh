#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,6


accelerate launch --num_processes=3 --main_process_port=12345 -m lmms_eval \
    --model qwen2_5_vl_interleave \
    --model_args=pretrained='/data/yuansheng/checkpoint/Qwen2.5-VL-7B-Instruct',max_pixels=12845056,min_pixels=3136 \
    --gen_kwargs=max_new_tokens=2048 \
    --tasks olympiadbench_test_en_oe \
    --verbosity=DEBUG \
    --batch_size 3 \
    --output_path ./eval_results/ \
    --log_samples
