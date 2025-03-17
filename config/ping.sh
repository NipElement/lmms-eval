#!/bin/bash
export MODELSCOPE_CACHE='/map-vepfs/huggingface'
export HF_HOME='/map-vepfs/huggingface'
export CUDA_VISIBLE_DEVICES=1,2,3

# accelerate launch --num_processes=1 --gpu_ids 0,1,2,3,4,5,6,7 --main_process_port=12345 -m lmms_eval \
#     --model qwen2_5_vl_interleave \
#     --model_args=pretrained='/map-vepfs/huggingface/models/Qwen2.5-VL-7B-Instruct',max_pixels=12845056,min_pixels=3136 \
#     --gen_kwargs=max_new_tokens=1024 \
#     --tasks olympiadbench_test_en_oe_cot_num_10 \
#     --verbosity=DEBUG \
#     --batch_size 1 \
#     --output_path ./eval_results/ \
#     --log_samples \
#     --limit 4

python -m lmms_eval \
    --model qwen2_5_vl_interleave_api \
    --tasks olympiadbench_test_en_oe_cot_num_10 \
    --verbosity=DEBUG \
    --batch_size 32 \
    --output_path ./eval_results/olympiadbench_test_en_oe_cot_num_10 \
    --log_samples 


python -m lmms_eval \
    --model qwen2_5_vl_interleave_api \
    --tasks olympiadbench_test_cn_oe \
    --verbosity=DEBUG \
    --batch_size 32 \
    --output_path ./eval_results/olympiadbench_test_cn_oe \
    --log_samples

vllm serve /map-vepfs/huggingface/models/Qwen2.5-VL-7B-Instruct --tensor-parallel-size 3 --limit-mm-per-prompt image=20 
