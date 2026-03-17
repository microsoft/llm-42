#!/bin/bash

MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}
ENABLE_SGLANG_DETERMINISM=${ENABLE_SGLANG_DETERMINISM:-0}
ENABLE_LLM42=${ENABLE_LLM42:-3}
LLM42_WINDOW_SIZE=${LLM42_WINDOW_SIZE:-64}
LLM42_VERIFY_BATCH_SIZE=${LLM42_VERIFY_BATCH_SIZE:-8}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-triton}
MEM_FRACTION_STATIC=${MEM_FRACTION_STATIC:-0.8}
TP_SIZE=${TP_SIZE:-1}
SYMM_MEM=${ENABLE_SYMM_MEM:-0}

HOST="${SGLANG_HOST:-0.0.0.0}"
PORT="${SGLANG_PORT:-30005}"

python -m sglang.launch_server \
    --host "$HOST" \
    --port "$PORT" \
    --model-path "$MODEL" \
    --enable-deterministic-inference "$ENABLE_SGLANG_DETERMINISM" \
    --enable-llm42 "$ENABLE_LLM42" \
    --llm42-window-size "$LLM42_WINDOW_SIZE" \
    --llm42-verify-batch-size "$LLM42_VERIFY_BATCH_SIZE" \
    --llm42-skip-mismatch 0 \
    --attention-backend "$ATTENTION_BACKEND" \
    --disable-radix-cache \
    --disable-chunked-prefix-cache \
    --disable-overlap-schedule \
    --chunked-prefill-size -1 \
    --random-seed 42 \
    $( [[ "$SYMM_MEM" == "1" ]] && echo "--enable-symm-mem" ) \
    --tensor-parallel-size "$TP_SIZE"


#--model-path meta-llama/Llama-3.1-8B-Instruct \
