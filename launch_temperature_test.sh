#!/bin/bash

# Launch SGLang server with temperature-based batch-invariant switching
# This uses bit 512 for temperature-based switching + bit 1 for ThinkingMachine matmul = 513

set -e

echo "=========================================="
echo "Starting SGLang Server with Temperature-Based Switching"
echo "=========================================="
echo ""
echo "Mode: Deterministic flag = 513 (512 + 1)"
echo "  - Bit 512: Temperature-based switching enabled"
echo "  - Bit 1: ThinkingMachine matmul kernel"
echo ""
echo "Behavior:"
echo "  - When temperature = 0: Use batch-invariant (deterministic)"
echo "  - When temperature > 0 (all requests): Use non-deterministic"
echo "  - Mixed batch: Use batch-invariant"
echo ""
echo "=========================================="
echo ""

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
echo "Command: $PYTHON_CMD -m sglang.launch_server \\"
echo "  --model-path $MODEL_PATH \\"
echo "  --host $HOST \\"
echo "  --port $PORT \\"
echo "  --tp $TP_SIZE \\"
echo "  --enable-deterministic-inference 1"
echo ""

$PYTHON_CMD -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --tp "$TP_SIZE" \
    --enable-deterministic-inference 1 \
    --attention-backend $ATTENTION_BACKEND \
    --disable-radix-cache \
    --disable-cuda-graph
