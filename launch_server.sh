#!/bin/bash

# SGLang Server Launch Script
# Usage: ./launch_server.sh [MODE] [DET_MODE]
#
# MODE options:
#   non-deterministic (default) - Standard inference
#   deterministic               - Deterministic inference with mode 1 (default)
#   det-1, det-2, det-3, etc.   - Specific deterministic mode
#
# DET_MODE (optional, only used with "deterministic"):
#   Bitmask value for deterministic inference mode (default: 1)
#
# Deterministic Mode Bitmask:
#   1   = Full deterministic (default): det matmul, det rmsnorm, det attention
#   2   = Use kernel matmul
#   3   = Kernel matmul (1+2)
#   4   = Split-stream matmul
#   5   = Split-stream matmul (1+4)
#   6   = Kernel + split-stream (2+4)
#   32  = Non-det matmul only
#   64  = Non-det rmsnorm only
#   96  = Non-det matmul + rmsnorm (32+64)
#   128 = Non-det attention only
#   160 = Non-det matmul + attention (32+128)
#   192 = Non-det rmsnorm + attention (64+128)
#   194 = Non-det all + kernel matmul (2+32+64+128)
#   224 = All non-deterministic (32+64+128)
#
# Examples:
#   ./launch_server.sh non-deterministic
#   ./launch_server.sh deterministic
#   ./launch_server.sh deterministic 1
#   ./launch_server.sh deterministic 2
#   ./launch_server.sh det-1
#   ./launch_server.sh det-194

MODE=${1:-"non-deterministic"}
DET_MODE=${2:-1}

# Common parameters
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
    echo "Error: Python not found. Please install Python."
    exit 1
fi

# Parse det-N shorthand format
if [[ $MODE =~ ^det-([0-9]+)$ ]]; then
    DET_MODE="${BASH_REMATCH[1]}"
    MODE="deterministic"
fi

echo "================================================"
echo "SGLang Server Launcher"
echo "================================================"
echo "Mode: $MODE"
if [ "$MODE" = "deterministic" ]; then
    echo "Deterministic Mode: $DET_MODE"
fi
echo "Model: $MODEL_PATH"
echo "Host: $HOST"
echo "Port: $PORT"
echo "TP Size: $TP_SIZE"
echo "Attention Backend: $ATTENTION_BACKEND"
echo "================================================"

if [ "$MODE" = "deterministic" ]; then
    echo "Launching in DETERMINISTIC mode (mode=$DET_MODE)..."
    echo "Note: Radix cache is disabled (not supported with FlashInfer + deterministic mode)"
    $PYTHON_CMD -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --host $HOST \
        --port $PORT \
        --tp-size $TP_SIZE \
        --attention-backend $ATTENTION_BACKEND \
        --disable-radix-cache \
        --enable-deterministic-inference $DET_MODE
elif [ "$MODE" = "non-deterministic" ]; then
    echo "Launching in NON-DETERMINISTIC mode..."
    $PYTHON_CMD -m sglang.launch_server \
        --model-path $MODEL_PATH \
        --host $HOST \
        --port $PORT \
        --tp-size $TP_SIZE \
        --attention-backend $ATTENTION_BACKEND \
        --disable-radix-cache 
else
    echo "Error: Invalid mode '$MODE'"
    echo ""
    echo "Usage: $0 [MODE] [DET_MODE]"
    echo ""
    echo "MODE options:"
    echo "  non-deterministic (default) - Standard inference"
    echo "  deterministic               - Deterministic inference"
    echo "  det-<N>                     - Deterministic with specific mode N"
    echo ""
    echo "DET_MODE (optional, for 'deterministic' mode):"
    echo "  1   = Full deterministic (default)"
    echo "  2   = Kernel matmul"
    echo "  194 = Non-det all + kernel matmul"
    echo "  ... (see script header for all modes)"
    echo ""
    echo "Examples:"
    echo "  $0 non-deterministic"
    echo "  $0 deterministic"
    echo "  $0 deterministic 2"
    echo "  $0 det-194"
    exit 1
fi
