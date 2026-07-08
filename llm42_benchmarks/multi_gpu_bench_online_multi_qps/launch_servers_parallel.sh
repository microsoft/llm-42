#!/bin/bash

# Launch SGLang server(s) for batch invariance testing with dense model
# Number of servers is calculated as NUM_GPUS / TP_SIZE

set -e

# Configuration
NUM_GPUS="${NUM_GPUS:-4}"
TP_SIZE="${SGLANG_TP_SIZE:-4}"
NUM_SERVERS=$((NUM_GPUS / TP_SIZE))  # Calculate servers based on available GPUs and TP size
MODEL_PATH="${MODEL:-Qwen/Qwen3-30B-A3B-Instruct-2507}"
HOST="${SGLANG_HOST:-0.0.0.0}"
BASE_PORT="${SGLANG_BASE_PORT:-30005}"
ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-fa3}"
ENABLE_SGLANG_DETERMINISM="${ENABLE_SGLANG_DETERMINISM:-0}"
ENABLE_LLM42="${ENABLE_LLM42:-3}"
LLM42_WINDOW_SIZE="${LLM42_WINDOW_SIZE:-64}"
LLM42_VERIFY_BATCH_SIZE="${LLM42_VERIFY_BATCH_SIZE:-8}"
ENABLE_SYMM_MEM="${ENABLE_SYMM_MEM:-0}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.8}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="${LOG_DIR:-./server_logs/TP_${TP_SIZE}_${TIMESTAMP}}"

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
echo "Starting $NUM_SERVERS SGLang Server(s) (NUM_GPUS=$NUM_GPUS, TP=$TP_SIZE)"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Host: $HOST"
echo "Base Port: $BASE_PORT"
echo "Num GPUs: $NUM_GPUS"
echo "TP Size: $TP_SIZE"
echo "Num Servers: $NUM_SERVERS (= $NUM_GPUS / $TP_SIZE)"
echo "Attention Backend: $ATTENTION_BACKEND"
echo "Enable SGLang Determinism: $ENABLE_SGLANG_DETERMINISM"
echo "Enable LLM42: $ENABLE_LLM42"
echo "LLM42 Window Size: $LLM42_WINDOW_SIZE"
echo "LLM42 Verify Batch Size: $LLM42_VERIFY_BATCH_SIZE"
echo "Enable Symm Mem: $ENABLE_SYMM_MEM"
echo "Mem Fraction Static: $MEM_FRACTION_STATIC"
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

# Launch server(s)
for ((i=0; i<NUM_SERVERS; i++)); do
    PORT=$((BASE_PORT + i))
    LOG_FILE="$LOG_DIR/server_${i}_port${PORT}.log"
    
    # Calculate GPU range for this server (e.g., server 0 gets GPUs 0-3, server 1 gets GPUs 4-7)
    START_GPU=$((i * TP_SIZE))
    END_GPU=$((START_GPU + TP_SIZE - 1))
    GPU_IDS=$(seq -s, $START_GPU $END_GPU)
    
    echo "Starting server $i on GPUs $GPU_IDS, port $PORT with TP=$TP_SIZE..."
    echo "Log file: $LOG_FILE"
    
    # Assign specific GPUs for this server
    CUDA_VISIBLE_DEVICES=$GPU_IDS $PYTHON_CMD -m sglang.launch_server \
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
        --enable-deterministic-inference "$ENABLE_SGLANG_DETERMINISM" \
        --llm42-window-size "$LLM42_WINDOW_SIZE" \
        --enable-llm42 "$ENABLE_LLM42" \
        --llm42-verify-batch-size "$LLM42_VERIFY_BATCH_SIZE" \
        $( [[ "$ENABLE_SYMM_MEM" == "1" ]] && echo "--enable-symm-mem" ) \
        --mem-fraction-static "$MEM_FRACTION_STATIC" \
        > "$LOG_FILE" 2>&1 &
    
    SERVER_PID=$!
    PIDS+=($SERVER_PID)
    
    echo "  → Server $i started with PID $SERVER_PID (GPUs $GPU_IDS, port $PORT)"
    
    # Small delay between launches to avoid race conditions
    sleep 2
done

echo ""
echo "=============================================="
echo "$NUM_SERVERS server(s) launched successfully!"
echo "=============================================="
echo "Server URL(s):"
for ((i=0; i<NUM_SERVERS; i++)); do
    PORT=$((BASE_PORT + i))
    echo "  Server $i: http://$HOST:$PORT"
done
echo ""
echo "To use with run_compare_mismatches_multi_qps.sh:"
URLS=""
for ((i=0; i<NUM_SERVERS; i++)); do
    PORT=$((BASE_PORT + i))
    if [ $i -eq 0 ]; then
        URLS="http://127.0.0.1:$PORT"
    else
        URLS="$URLS,http://127.0.0.1:$PORT"
    fi
done
echo "  BASE_URLS=\"$URLS\" ./run_compare_mismatches_multi_qps.sh"
echo "  (QPS values will be run sequentially on the single server)"
echo ""
echo "Press Ctrl+C to stop all servers..."
echo "=============================================="

# Wait for all background processes
wait


        # --llm42-window-size 32 \
        # --enable-llm42 3 \
        # --llm42-verify-batch-size 32 \