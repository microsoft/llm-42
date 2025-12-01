#!/bin/bash
# Run per-request rollback benchmark sweep for different det_step_sizes
# and generate CDF plots.
#
# Usage:
#   ./run_cdf_sweep.sh [options]
#
# This script will:
#   1. Run benchmarks for det_step_size = 10, 20, 50, 100
#   2. Collect per-request rollback stats from server logs  
#   3. Generate CDF plots comparing the distributions

set -e

MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
PORT="${PORT:-30003}"
TP_SIZE="${SGLANG_TP_SIZE:-1}"
ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-flashinfer}"

# Benchmark configuration
NUM_REQUESTS="${NUM_REQUESTS:-100}"
BATCH_SIZE="${BATCH_SIZE:-32}"
MAX_TOKENS="${MAX_TOKENS:-128}"

# Step sizes to sweep
STEP_SIZES=(10 20 50 100)

OUTPUT_DIR="cdf_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$OUTPUT_DIR/run_$TIMESTAMP"

# Track server PID for cleanup
SERVER_PID=""

cleanup() {
    echo ""
    echo "Cleaning up..."
    trap - SIGINT SIGTERM
    
    if [ -n "$SERVER_PID" ] && kill -0 $SERVER_PID 2>/dev/null; then
        echo "Stopping server (PID: $SERVER_PID)..."
        kill $SERVER_PID 2>/dev/null || true
        sleep 2
        kill -9 $SERVER_PID 2>/dev/null || true
    fi
    
    pkill -f "sglang.launch_server.*--port $PORT" 2>/dev/null || true
    echo "Done."
}

trap cleanup SIGINT SIGTERM EXIT

# Create output directory
mkdir -p "$RUN_DIR"

echo "=========================================="
echo "Per-Request Rollback CDF Sweep"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Model:          $MODEL"
echo "  Port:           $PORT"
echo "  Step sizes:     ${STEP_SIZES[*]}"
echo "  Num requests:   $NUM_REQUESTS"
echo "  Batch size:     $BATCH_SIZE"
echo "  Max tokens:     $MAX_TOKENS"
echo "  Output dir:     $RUN_DIR"
echo ""

# Function to run one experiment
run_experiment() {
    local step_size=$1
    
    echo ""
    echo "=========================================="
    echo "Running: det_step_size=$step_size"
    echo "=========================================="
    
    local LOG_FILE="$RUN_DIR/server_step${step_size}.log"
    local RESULTS_FILE="$RUN_DIR/results_step${step_size}.json"
    
    # Start server with logging
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
        2>&1 | tee "$LOG_FILE" &
    
    SERVER_PID=$!
    echo "Server PID: $SERVER_PID"
    
    # Wait for server
    echo "Waiting for server to initialize..."
    sleep 30
    
    for i in {1..60}; do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "✓ Server ready"
            break
        fi
        sleep 5
        echo "  Still waiting... ($((30 + i*5))s)"
    done
    
    if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "✗ Server failed to start for step_size=$step_size"
        kill $SERVER_PID 2>/dev/null || true
        SERVER_PID=""
        return 1
    fi
    
    # Run benchmark
    echo "Running benchmark..."
    python bench_per_request_rollbacks.py \
        --port "$PORT" \
        --num-requests "$NUM_REQUESTS" \
        --batch-size "$BATCH_SIZE" \
        --max-tokens "$MAX_TOKENS" \
        --step-size "$step_size" \
        --log-file "$LOG_FILE" \
        --output "$RESULTS_FILE"
    
    # Wait for logs to flush
    sleep 3
    
    # Re-analyze to get final stats
    python bench_per_request_rollbacks.py \
        --analyze-only \
        --log-file "$LOG_FILE" \
        --output "$RESULTS_FILE" \
        --num-requests "$NUM_REQUESTS" \
        --batch-size "$BATCH_SIZE" \
        --max-tokens "$MAX_TOKENS" \
        --step-size "$step_size"
    
    # Stop server
    echo "Stopping server..."
    kill $SERVER_PID 2>/dev/null || true
    sleep 2
    kill -9 $SERVER_PID 2>/dev/null || true
    SERVER_PID=""
    
    # Clean up GPU memory - wait longer and kill any lingering processes
    echo "Cleaning up GPU memory..."
    pkill -9 -f "sglang.launch_server.*--port $PORT" 2>/dev/null || true
    sleep 5
    
    # Force Python garbage collection by running a small script
    python -c "import torch; torch.cuda.empty_cache(); print('GPU cache cleared')" 2>/dev/null || true
    sleep 3
    
    echo "✓ Completed: det_step_size=$step_size"
    echo ""
}

# Run experiments for each step size
for step_size in "${STEP_SIZES[@]}"; do
    run_experiment "$step_size"
done

# Generate CDF plots
echo ""
echo "=========================================="
echo "Generating CDF plots..."
echo "=========================================="

# Build list of result files
RESULT_FILES=""
for step_size in "${STEP_SIZES[@]}"; do
    RESULT_FILES="$RESULT_FILES $RUN_DIR/results_step${step_size}.json"
done

python plot_rollback_cdf.py \
    --results $RESULT_FILES \
    --output rollback \
    --output-dir "$RUN_DIR" \
    --plot-type all \
    --no-show

echo ""
echo "=========================================="
echo "✓ Sweep complete!"
echo "=========================================="
echo ""
echo "Results in: $RUN_DIR/"
echo ""
echo "Output files:"
for step_size in "${STEP_SIZES[@]}"; do
    echo "  - server_step${step_size}.log: Server log"
    echo "  - results_step${step_size}.json: Per-request stats"
done
echo ""
echo "Plots:"
echo "  - rollback_cdf_rollbacks.png: CDF of rollbacks per request"
echo "  - rollback_cdf_tokens.png: CDF of tokens rolled back per request"
echo "  - rollback_cdf_combined.png: Both CDFs side by side"
echo "  - rollback_summary.png: Bar chart summary"
echo ""
ls -la "$RUN_DIR"
