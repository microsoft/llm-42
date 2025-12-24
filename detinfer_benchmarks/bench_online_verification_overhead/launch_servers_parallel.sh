#!/bin/bash

# Launch multiple SGLang servers for parallel testing with different step sizes
# All servers have skip_mismatch enabled to measure pure verification overhead
# This script starts N servers (one per GPU) with different step sizes:
#   config1: detinfer_step_size_32 (step_size 32, skip mismatch)
#   config2: detinfer_step_size_128 (step_size 128, skip mismatch)
#   config3: detinfer_step_size_512 (step_size 512, skip mismatch)
#   config4: detinfer_step_size_1024 (step_size 1024, skip mismatch)

set -e

# Configuration
NUM_GPUS="${NUM_GPUS:-4}"
MODEL_PATH="${SGLANG_TEST_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
HOST="${SGLANG_HOST:-0.0.0.0}"
BASE_PORT="${SGLANG_BASE_PORT:-30010}"
TP_SIZE="${SGLANG_TP_SIZE:-1}"
ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-fa3}"
LOG_DIR="${LOG_DIR:-./server_logs_verification_overhead}"

# Config names (comma-separated)
CONFIG_NAMES="${CONFIG_NAMES:-detinfer_step_size_32,detinfer_step_size_128,detinfer_step_size_512,detinfer_step_size_1024}"

# Determine Python command
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: Python not found"
    exit 1
fi

# Convert config names to array
IFS=',' read -ra CONFIG_ARRAY <<< "$CONFIG_NAMES"

# Validate we have enough GPUs for configs
if [ ${#CONFIG_ARRAY[@]} -gt $NUM_GPUS ]; then
    echo "Error: More configs (${#CONFIG_ARRAY[@]}) than GPUs ($NUM_GPUS)"
    exit 1
fi

NUM_SERVERS=${#CONFIG_ARRAY[@]}

# Create log directory
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "Starting $NUM_SERVERS SGLang Servers for Verification Overhead Benchmark"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Host: $HOST"
echo "Base Port: $BASE_PORT"
echo "TP Size per server: $TP_SIZE"
echo "Attention Backend: $ATTENTION_BACKEND"
echo "Log Directory: $LOG_DIR"
echo ""
echo "NOTE: All servers have --det-skip-mismatch enabled (no rollback)"
echo ""
echo "Configuration Mapping:"
for ((i=0; i<NUM_SERVERS; i++)); do
    PORT=$((BASE_PORT + i))
    echo "  GPU $i (port $PORT): ${CONFIG_ARRAY[$i]}"
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

# Function to get config-specific arguments
# All configs use --det-skip-mismatch 0.0 to measure pure verification overhead (skip all mismatches)
get_config_args() {
    local config_name="$1"
    case "$config_name" in
        "detinfer_step_size_32")
            echo "--min-det-step-size 32 --enable-det-infer 3 --max-det-verify-batch-size 1 --det-skip-mismatch 0.0"
            ;;
        "detinfer_step_size_64")
            echo "--min-det-step-size 64 --enable-det-infer 3 --max-det-verify-batch-size 1 --det-skip-mismatch 0.0"
            ;;
        "detinfer_step_size_128")
            echo "--min-det-step-size 128 --enable-det-infer 3 --max-det-verify-batch-size 1 --det-skip-mismatch 0.0"
            ;;
        "detinfer_step_size_256")
            echo "--min-det-step-size 256 --enable-det-infer 3 --max-det-verify-batch-size 1 --det-skip-mismatch 0.0"
            ;;
        "detinfer_step_size_512")
            echo "--min-det-step-size 512 --enable-det-infer 3 --max-det-verify-batch-size 1 --det-skip-mismatch 0.0"
            ;;
        "detinfer_step_size_1024")
            echo "--min-det-step-size 1024 --enable-det-infer 3 --max-det-verify-batch-size 1 --det-skip-mismatch 0.0"
            ;;
        *)
            echo "Error: Unknown config name: $config_name" >&2
            exit 1
            ;;
    esac
}

# Launch servers with different configurations
for ((i=0; i<NUM_SERVERS; i++)); do
    PORT=$((BASE_PORT + i))
    GPU_ID=$i
    CONFIG_NAME=${CONFIG_ARRAY[$i]}
    CONFIG_ARGS=$(get_config_args "$CONFIG_NAME")
    LOG_FILE="$LOG_DIR/server_gpu${GPU_ID}_port${PORT}_${CONFIG_NAME}.log"
    
    echo "Starting server on GPU $GPU_ID, port $PORT with config=$CONFIG_NAME..."
    echo "  Config args: $CONFIG_ARGS"
    echo "  Log file: $LOG_FILE"
    
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
        $CONFIG_ARGS \
        > "$LOG_FILE" 2>&1 &
    
    SERVER_PID=$!
    PIDS+=($SERVER_PID)
    
    echo "  → Server started with PID $SERVER_PID on GPU $GPU_ID (port $PORT, config=$CONFIG_NAME)"
    
    # Small delay between launches to avoid race conditions
    sleep 2
done

echo ""
echo "=============================================="
echo "All $NUM_SERVERS servers launched successfully!"
echo "=============================================="
echo "Server URLs and Configurations:"
for ((i=0; i<NUM_SERVERS; i++)); do
    PORT=$((BASE_PORT + i))
    CONFIG_NAME=${CONFIG_ARRAY[$i]}
    echo "  GPU $i: http://$HOST:$PORT ($CONFIG_NAME)"
done
echo ""
echo "To run verification overhead benchmark:"
echo "  ./run_all_verification_overhead.sh"
echo ""
echo "Press Ctrl+C to stop all servers..."
echo "=============================================="

# Wait for all background processes
wait
