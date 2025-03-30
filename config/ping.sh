#!/bin/bash
export HF_HOME='/data/yuansheng/cache'
export CUDA_VISIBLE_DEVICES=0,1
# pip install qwen_vl_utils
# pip install antlr4-python3-runtime==4.11

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

export CUDA_VISIBLE_DEVICES=0,1,2,3
vllm serve /data/yuansheng/checkpoint/Qwen2.5-VL-7B-Instruct --tensor-parallel-size 4 --limit-mm-per-prompt image=15 --gpu-memory-utilization 0.9

# change config.json rope_scaling type "dynamic" to "mrope"
vllm serve /data/yuansheng/checkpoint/mammoth_mix_60K_icl_28K_example_num_10/checkpoint-3602 --tensor-parallel-size 4 --limit-mm-per-prompt image=15 --gpu-memory-utilization 0.9
vllm serve /data/yuansheng/checkpoint/mammoth_mix_86K_icl_27K_max_token/checkpoint-5434 --tensor-parallel-size 4 --limit-mm-per-prompt image=15 --gpu-memory-utilization 0.9

export CUDA_VISIBLE_DEVICES=4,5
vllm serve /data/yuansheng/checkpoint/mammoth_mix_57K_icl_26K_multi_turn_example_num_10 --tensor-parallel-size 2 --limit-mm-per-prompt image=15 --gpu-memory-utilization 0.9 --port 8001
vllm serve /data/yuansheng/checkpoint/mammoth_mix_88K_icl_27K_default --tensor-parallel-size 2 --limit-mm-per-prompt image=15 --gpu-memory-utilization 0.9 --port 8002


# mathvista cot num 10

python -m lmms_eval \
    --model qwen2_5_vl_interleave_api \
    --tasks mathvista_testmini_cot_num_10 \
    --verbosity=DEBUG \
    --batch_size 32 \
    --output_path ./eval_results/Qwen2.5-VL-7B-Instruct/mathvista_testmini_cot_num_10 \
    --log_samples


python -m lmms_eval \
    --model qwen2_5_vl_interleave_api \
    --tasks mathvista_testmini_cot \
    --verbosity=DEBUG \
    --batch_size 16 \
    --output_path ./eval_results/mammoth_mix_57K_icl_26K_multi_turn_example_num_10/mathvista_testmini_cot \
    --log_samples