#!/bin/bash

# Launch SGLang server with temperature-based batch-invariant switching
# This uses bit 512 for temperature-based switching + bit 1 for ThinkingMachine matmul = 513

set -e

# Model and server configuration
MODEL_PATH="meta-llama/Meta-Llama-3.1-8B-Instruct"
HOST="0.0.0.0"
PORT=30000
TP_SIZE=1
ATTENTION_BACKEND="flashinfer"

# Determine Python command
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: Python not found"
    exit 1
fi

echo "Starting server..."
# nsys profile --trace-fork-before-exec=true --cuda-graph-trace=node -o sglang.out --delay 60 --duration 70  \
$PYTHON_CMD -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --tp "$TP_SIZE" \
    --attention-backend $ATTENTION_BACKEND \
    --disable-radix-cache \
    --disable-chunked-prefix-cache \
    --disable-overlap-schedule \
    --disable-cuda-graph \
    --det-step-size 10 \
    --enable-det-infer 1
