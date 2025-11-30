#!/bin/bash
# Automated sweep for varying min-det-step-size
# Creates two plots:
#   1. Rollbacks vs min_det_step_size (vary batch size, fix max_tokens)
#   2. Rollbacks vs min_det_step_size (vary max_tokens, fix batch_size)

set -e

MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
PORT=30003
STEP_SIZES=(5 10 20 50 100)
RESULTS_DIR="rollback_results"
TP_SIZE="${SGLANG_TP_SIZE:-1}"
ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-flashinfer}"

# Test configuration
BATCH_SIZES="8 16 32 64"
MAX_TOKENS_LIST="32 64 128 256"
FIXED_MAX_TOKENS=128
FIXED_BATCH_SIZE=32
N_PROMPTS=64
NUM_BATCHES=1  # Number of batched requests per config (each batch sends batch_size prompts in ONE request)
TEMPERATURE=0.0

# Track PIDs for cleanup
SERVER_PID=""
SCRIPT_PID=""

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up..."
    # Disable trap to prevent repeated calls
    trap - SIGINT SIGTERM
    
    if [ -n "$SCRIPT_PID" ] && kill -0 $SCRIPT_PID 2>/dev/null; then
        echo "Killing Python script (PID: $SCRIPT_PID)..."
        kill $SCRIPT_PID 2>/dev/null || true
        sleep 1
        kill -9 $SCRIPT_PID 2>/dev/null || true
    fi
    
    if [ -n "$SERVER_PID" ] && kill -0 $SERVER_PID 2>/dev/null; then
        echo "Killing server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true
        sleep 2
        kill -9 $SERVER_PID 2>/dev/null || true
    fi
    
    # Also kill any remaining sglang processes on this port
    pkill -f "sglang.launch_server.*--port $PORT" 2>/dev/null || true
    
    echo "Done."
    exit 1
}

# Trap Ctrl+C and cleanup
trap cleanup SIGINT SIGTERM

mkdir -p "$RESULTS_DIR"

# Function to run one experiment
run_experiment() {
    local step_size=$1
    
    echo ""
    echo "=========================================="
    echo "Running: min_det_step_size=$step_size"
    echo "=========================================="
    
    # Start server
    echo "Starting server..."
    python -m sglang.launch_server \
        --model-path "$MODEL" \
        --port "$PORT" \
        --tp "$TP_SIZE" \
        --attention-backend "$ATTENTION_BACKEND" \
        --disable-radix-cache \
        --disable-chunked-prefix-cache \
        --disable-overlap-schedule \
        --enable-metrics \
        --enable-det-infer 1 \
        --min-det-step-size "$step_size" \
        > "$RESULTS_DIR/server_step_${step_size}.log" 2>&1 &
    
    local server_pid=$!
    SERVER_PID=$server_pid
    echo "Server PID: $server_pid"
    
    # Wait for server (longer timeout for model loading)
    # Wait at least 40 seconds before first check
    echo "Waiting 40s for server to initialize..."
    sleep 40
    
    echo "Checking server status..."
    for i in {1..60}; do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "✓ Server ready"
            break
        fi
        sleep 5
        echo "  Still waiting... ($((40 + i*5))s)"
    done
    
    if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "✗ Server failed to start"
        kill $server_pid 2>/dev/null || true
        return 1
    fi
    
    # Run the rollback collection script in foreground (so we can track it)
    echo "Running rollback collection..."
    python vllm_online_batch_invariance_multitest.py \
        --min-det-step-size "$step_size" \
        --batch-sizes $BATCH_SIZES \
        --max-tokens-list $MAX_TOKENS_LIST \
        --fixed-max-tokens $FIXED_MAX_TOKENS \
        --fixed-batch-size $FIXED_BATCH_SIZE \
        --n-prompts $N_PROMPTS \
        --num-batches $NUM_BATCHES \
        --temperature $TEMPERATURE \
        --output-dir "$RESULTS_DIR" &
    
    SCRIPT_PID=$!
    wait $SCRIPT_PID
    SCRIPT_PID=""
    
    # Stop server
    echo "Stopping server..."
    kill $server_pid 2>/dev/null || true
    sleep 3
    # Force kill if still running
    kill -9 $server_pid 2>/dev/null || true
    
    # Clear PIDs
    SERVER_PID=""
    
    echo "✓ Completed: min_det_step_size=$step_size"
}

# Run experiments for each step size
for step_size in "${STEP_SIZES[@]}"; do
    run_experiment "$step_size"
done

# Generate plots
echo ""
echo "=========================================="
echo "Generating plots..."
echo "=========================================="

# Convert step sizes array to space-separated string
STEP_SIZES_STR="${STEP_SIZES[*]}"

python vllm_online_batch_invariance_multitest.py \
    --plot-only \
    --output-dir "$RESULTS_DIR" \
    --step-sizes-to-plot $STEP_SIZES_STR

echo ""
echo "=========================================="
echo "✓ Sweep complete!"
echo "=========================================="
echo "Results in: $RESULTS_DIR/"
echo ""
echo "Output files:"
echo "  - step_N.json: Raw data for each min_det_step_size"
echo "  - plot_vary_batch_size.png: Rollbacks vs step_size (varying batch size)"
echo "  - plot_vary_max_tokens.png: Rollbacks vs step_size (varying max_tokens)"
echo ""
ls -la "$RESULTS_DIR"
