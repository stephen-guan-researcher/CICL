#!/bin/bash
# Mask physical GPUs per DDP rank so each worker sees only its own device.
# Needed because bitsandbytes 4-bit quantization places weights on the
# current CUDA device and ignores `device_map` during the quant step. Without
# masking, both ranks pile their full model onto cuda:0 and OOM.
#
# Used by `accelerate launch --multi_gpu --no_python ... launch_train_rank.sh`.
export CUDA_VISIBLE_DEVICES="${LOCAL_RANK:-0}"
exec python -u -m cicl_agent.learning.train_qwen_judge "$@"
