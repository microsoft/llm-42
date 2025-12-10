#!/bin/bash

# Launch SGLang server for batch invariance testing
# This script starts a server with deterministic inference enabled for batch invariance verification

set -e

# Model and server configuration
MODEL_PATH="${SGLANG_TEST_MODEL:-Qwen/Qwen3-30B-A3B}"
HOST="${SGLANG_HOST:-0.0.0.0}"
PORT="${SGLANG_PORT:-30005}"
TP_SIZE="${SGLANG_TP_SIZE:-4}"
ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-flashinfer}"

# Determine Python command
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: Python not found"
    exit 1
fi

echo "=============================================="
echo "Starting SGLang Server for Batch Invariance Testing"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Host: $HOST"
echo "Port: $PORT"
echo "TP Size: $TP_SIZE"
echo "Attention Backend: $ATTENTION_BACKEND"
echo "=============================================="
echo ""

# Start the server with deterministic inference enabled
$PYTHON_CMD -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --tp "$TP_SIZE" \
    --attention-backend "$ATTENTION_BACKEND" \
    --disable-radix-cache \
    --disable-chunked-prefix-cache \
    --chunked-prefill-size -1 \
    --disable-overlap-schedule \
    --enable-metrics \
    # --min-det-step-size 64 \
    # --enable-det-infer 3 \
    # --max-det-verify-batch-size 1
    # --enable-selective-determinism 1 \
    # --enable-deterministic-inference 2 \
    # --min-det-step-size 10 \
    # --enable-det-infer 1
    # --enable-selective-determinism 1 \
    # --enable-deterministic-inference 1 \
    # --disable-cuda-graph \
