
# pip3 install vllm
# pip3 install qwen_vl_utils

# cd ~/prod/lmms-eval-public
# pip3 install -e .
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=18000000
export NCCL_DEBUG=DEBUG
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
python3 -m lmms_eval \
    --model vllm \
    --model_args=model_version='/map-vepfs/huggingface/models/Qwen2.5-VL-7B-Instruct',max_pixels=12845056,min_pixels=3136,tensor_parallel_size=4 \
    --tasks olympiadbench_test_en_oe \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix vllm \
    --output_path ./logs \
    --verbosity=DEBUG

vllm serve /map-vepfs/huggingface/models/Qwen2.5-VL-7B-Instruct \
    --tensor-parallel-size 4