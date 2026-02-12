#!/bin/bash

# Launch multiple SGLang servers for parallel batch invariance testing with different step sizes
# This script starts N servers (one per GPU) with deterministic inference enabled
# Each server uses a different llm42-window-size

set -e

# Configuration
NUM_GPUS="${NUM_GPUS:-4}"
MODEL_PATH="${SGLANG_TEST_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
HOST="${SGLANG_HOST:-0.0.0.0}"
BASE_PORT="${SGLANG_BASE_PORT:-30005}"
TP_SIZE="${SGLANG_TP_SIZE:-1}"
ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-fa3}"
LOG_DIR="${LOG_DIR:-./server_logs_multi_step_size}"

# Step sizes for each server (comma-separated)
STEP_SIZES="${STEP_SIZES:-32,64,128,256}"

# Determine Python command
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: Python not found"
    exit 1
fi

# Convert step sizes to array
IFS=',' read -ra STEP_SIZE_ARRAY <<< "$STEP_SIZES"

# Validate we have enough GPUs for step sizes
if [ ${#STEP_SIZE_ARRAY[@]} -gt $NUM_GPUS ]; then
    echo "Error: More step sizes (${#STEP_SIZE_ARRAY[@]}) than GPUs ($NUM_GPUS)"
    exit 1
fi

NUM_SERVERS=${#STEP_SIZE_ARRAY[@]}

# Create log directory
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "Starting $NUM_SERVERS SGLang Servers with Different Step Sizes"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Host: $HOST"
echo "Base Port: $BASE_PORT"
echo "TP Size per server: $TP_SIZE"
echo "Attention Backend: $ATTENTION_BACKEND"
echo "Log Directory: $LOG_DIR"
echo ""
echo "Step Size Configuration:"
for ((i=0; i<NUM_SERVERS; i++)); do
    PORT=$((BASE_PORT + i))
    echo "  GPU $i (port $PORT): step_size=${STEP_SIZE_ARRAY[$i]}"
done
echo "=============================================="
echo ""

# Array to store PIDs
declare -a PIDS=()

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down servers..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "Killing server with PID $pid"
            kill "$pid" 2>/dev/null || true
        fi
    done
    wait
    echo "All servers stopped."
}

trap cleanup EXIT INT TERM

# Launch servers with different step sizes
for ((i=0; i<NUM_SERVERS; i++)); do
    PORT=$((BASE_PORT + i))
    GPU_ID=$i
    STEP_SIZE=${STEP_SIZE_ARRAY[$i]}
    LOG_FILE="$LOG_DIR/server_gpu${GPU_ID}_port${PORT}_step${STEP_SIZE}.log"
    
    echo "Starting server on GPU $GPU_ID, port $PORT with step_size=$STEP_SIZE..."
    echo "Log file: $LOG_FILE"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID SGLANG_LOG_LEVEL=DEBUG SGLANG_DEBUG_SAMPLING=1 $PYTHON_CMD -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --host "$HOST" \
        --port "$PORT" \
        --tp "$TP_SIZE" \
        --attention-backend "$ATTENTION_BACKEND" \
        --disable-radix-cache \
        --disable-chunked-prefix-cache \
        --disable-overlap-schedule \
        --enable-metrics \
        --random-seed 42 \
        --chunked-prefill-size -1 \
        --llm42-window-size "$STEP_SIZE" \
        --enable-llm42 3 \
        --llm42-verify-batch-size 1 \
        > "$LOG_FILE" 2>&1 &
    
    SERVER_PID=$!
    PIDS+=($SERVER_PID)
    
    echo "  → Server started with PID $SERVER_PID on GPU $GPU_ID (port $PORT, step_size=$STEP_SIZE)"
    
    # Small delay between launches to avoid race conditions
    sleep 2
done

echo ""
echo "=============================================="
echo "All $NUM_SERVERS servers launched successfully!"
echo "=============================================="
echo "Server URLs and Step Sizes:"
for ((i=0; i<NUM_SERVERS; i++)); do
    PORT=$((BASE_PORT + i))
    STEP_SIZE=${STEP_SIZE_ARRAY[$i]}
    echo "  GPU $i: http://$HOST:$PORT (step_size=$STEP_SIZE)"
done
echo ""
echo "To use with run_compare_mismatches_multi_step_size.sh:"
URLS=""
for ((i=0; i<NUM_SERVERS; i++)); do
    PORT=$((BASE_PORT + i))
    if [ $i -eq 0 ]; then
        URLS="http://127.0.0.1:$PORT"
    else
        URLS="$URLS,http://127.0.0.1:$PORT"
    fi
done
echo "  ./run_compare_mismatches_multi_step_size.sh"
echo ""
echo "Press Ctrl+C to stop all servers..."
echo "=============================================="

# Wait for all background processes
wait
