#!/bin/bash

MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
ENABLE_LLM42="${ENABLE_LLM42:-3}"
LLM42_WINDOW_SIZE="${LLM42_WINDOW_SIZE:-64}"
LLM42_VERIFY_BATCH_SIZE="${LLM42_VERIFY_BATCH_SIZE:-8}"
ENABLE_SGLANG_DETERMINISM="${ENABLE_SGLANG_DETERMINISM:-0}"
TP_SIZE="${TP_SIZE:-1}"

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../gpu_utils.sh"
_GPU_TYPE="$GPU_SHORT_NAME"

python -m sglang.launch_server \
    --model-path "$MODEL" \
    --tp "$TP_SIZE" \
    --enable-deterministic-inference "$ENABLE_SGLANG_DETERMINISM" \
    --enable-llm42 "$ENABLE_LLM42" \
    --llm42-window-size "$LLM42_WINDOW_SIZE" \
    --llm42-verify-batch-size "$LLM42_VERIFY_BATCH_SIZE" \
    --attention-backend "$ATTENTION_BACKEND" \
    --disable-radix-cache \
    --disable-chunked-prefix-cache \
    --disable-overlap-schedule \
    --chunked-prefill-size -1 \
    --random-seed 42 \
    $([ "${DISABLE_CUSTOM_ALL_REDUCE:-0}" = "1" ] && echo "--disable-custom-all-reduce" || true) \
    $([ "${ENABLE_TORCH_SYMM_MEM:-0}" = "1" ] && echo "--enable-torch-symm-mem" || true)