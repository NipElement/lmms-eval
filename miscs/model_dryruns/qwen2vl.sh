# Run and exactly reproduce qwen2vl results!
# mme as an example
# pip3 install qwen_vl_utils
export CUDA_VISIBLE_DEVICES=0
accelerate launch --num_processes=1 --main_process_port=12345 -m lmms_eval \
    --model qwen2_5_vl_interleave \
    --model_args=pretrained=/data/yuansheng/checkpoint/Qwen2.5-VL-7B-Instruct,max_pixels=2359296 \
    --tasks olympiadbench_test_en_oe \
    --batch_size 1 --log_samples --log_samples_suffix reproduce --output_path ./logs/