#!/bin/bash

# Launch multiple SGLang servers for parallel batch invariance testing
# This script starts N servers (one per GPU) with deterministic inference enabled

set -e

# Configuration
NUM_GPUS="${NUM_GPUS:-4}"
MODEL_PATH="${SGLANG_TEST_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
HOST="${SGLANG_HOST:-0.0.0.0}"
BASE_PORT="${SGLANG_BASE_PORT:-30005}"
TP_SIZE="${SGLANG_TP_SIZE:-1}"
ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-flashinfer}"
LOG_DIR="${LOG_DIR:-./server_logs}"

# Determine Python command
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: Python not found"
    exit 1
fi

# Create log directory
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "Starting $NUM_GPUS SGLang Servers for Parallel Batch Invariance Testing"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Host: $HOST"
echo "Base Port: $BASE_PORT"
echo "TP Size per server: $TP_SIZE"
echo "Attention Backend: $ATTENTION_BACKEND"
echo "Log Directory: $LOG_DIR"
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

# Launch servers
for ((i=0; i<NUM_GPUS; i++)); do
    PORT=$((BASE_PORT + i))
    GPU_ID=$i
    LOG_FILE="$LOG_DIR/server_gpu${GPU_ID}_port${PORT}.log"
    
    echo "Starting server on GPU $GPU_ID, port $PORT..."
    echo "Log file: $LOG_FILE"
    
    CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_CMD -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --host "$HOST" \
        --port "$PORT" \
        --tp "$TP_SIZE" \
        --attention-backend "$ATTENTION_BACKEND" \
        --disable-radix-cache \
        --disable-chunked-prefix-cache \
        --disable-overlap-schedule \
        --enable-metrics \
        --min-det-step-size 128 \
        --enable-det-infer 3 \
        --max-det-verify-batch-size 1 \
        --skip-server-warmup \
        > "$LOG_FILE" 2>&1 &
    
    SERVER_PID=$!
    PIDS+=($SERVER_PID)
    
    echo "  → Server started with PID $SERVER_PID on GPU $GPU_ID (port $PORT)"
    
    # Small delay between launches to avoid race conditions
    sleep 2
done

echo ""
echo "=============================================="
echo "All $NUM_GPUS servers launched successfully!"
echo "=============================================="
echo "Server URLs:"
for ((i=0; i<NUM_GPUS; i++)); do
    PORT=$((BASE_PORT + i))
    echo "  GPU $i: http://$HOST:$PORT"
done
echo ""
echo "To use with run_compare_mismatches_multi_qps.sh:"
URLS=""
for ((i=0; i<NUM_GPUS; i++)); do
    PORT=$((BASE_PORT + i))
    if [ $i -eq 0 ]; then
        URLS="http://127.0.0.1:$PORT"
    else
        URLS="$URLS,http://127.0.0.1:$PORT"
    fi
done
echo "  ./run_compare_mismatches_multi_qps.sh"
echo "  (or with custom QPS: QPS_VALUES=\"1,3,6,10\" ./run_compare_mismatches_multi_qps.sh)"
echo ""
echo "Press Ctrl+C to stop all servers..."
echo "=============================================="

# Wait for all background processes
wait
