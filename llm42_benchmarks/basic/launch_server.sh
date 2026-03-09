#!/bin/bash
python -m sglang.launch_server \
    --model-path meta-llama/Llama-3.1-8B-Instruct \
    --enable-llm42 3 \
    --llm42-window-size 64 \
    --llm42-verify-batch-size 8 \
    --attention-backend fa3 \
    --disable-radix-cache \
    --disable-chunked-prefix-cache \
    --disable-overlap-schedule \
    --chunked-prefill-size -1 \
    --random-seed 42