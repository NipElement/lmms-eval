#!/bin/bash
export HF_HOME='/data/yuansheng/cache'
export CUDA_VISIBLE_DEVICES=0,5,6,7
# pip install qwen_vl_utils

# en
python -m lmms_eval \
    --model qwen2_5_vl_interleave_api \
    --tasks olympiadbench_test_en_oe \
    --verbosity=DEBUG \
    --batch_size 32 \
    --output_path ./eval_results/olympiadbench_test_en_oe \
    --log_samples

python -m lmms_eval \
    --model qwen2_5_vl_interleave_api \
    --tasks olympiadbench_test_en_oe_cot_num_10 \
    --verbosity=DEBUG \
    --batch_size 32 \
    --output_path ./eval_results/olympiadbench_test_en_oe_cot_num_10 \
    --log_samples 

# cn
python -m lmms_eval \
    --model qwen2_5_vl_interleave_api \
    --tasks olympiadbench_test_cn_oe \
    --verbosity=DEBUG \
    --batch_size 32 \
    --output_path ./eval_results/olympiadbench_test_cn_oe \
    --log_samples \
    --limit 233

python -m lmms_eval \
    --model qwen2_5_vl_interleave_api \
    --tasks olympiadbench_test_cn_oe_cot_num_10 \
    --verbosity=DEBUG \
    --batch_size 32 \
    --output_path ./eval_results/olympiadbench_test_cn_oe_cot_num_10 \
    --log_samples \
    --limit 233

vllm serve /data/yuansheng/checkpoint/Qwen2.5-VL-7B-Instruct --tensor-parallel-size 4 --limit-mm-per-prompt image=20 --gpu-memory-utilization 0.8

# change config.json rope_scaling type "dynamic" to "mrope"
vllm serve /data/yuansheng/checkpoint/mammoth_mix_60K_icl_28K_example_num_10/checkpoint-3602 --tensor-parallel-size 4 --limit-mm-per-prompt image=20 --gpu-memory-utilization 0.8

vllm serve /data/yuansheng/checkpoint/mammoth_mix_86K_icl_27K_max_token/checkpoint-5434 --tensor-parallel-size 4 --limit-mm-per-prompt image=20 --gpu-memory-utilization 0.8