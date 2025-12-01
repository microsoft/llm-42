#!/bin/bash
# Run per-request rollback benchmark with automatic server management and log capture
#
# Usage:
#   ./run_per_request_bench.sh [options]
#
# Options:
#   --step-size N     min_det_step_size value (default: 10)
#   --num-requests N  number of requests (default: 100)
#   --batch-size N    prompts per request (default: 1)
#   --max-tokens N    max tokens to generate (default: 128)
#
# Example:
#   ./run_per_request_bench.sh --step-size 20 --num-requests 50

set -e

# Default values
MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
PORT="${PORT:-30003}"
TP_SIZE="${SGLANG_TP_SIZE:-1}"
ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-flashinfer}"
STEP_SIZE=10
NUM_REQUESTS=100
BATCH_SIZE=64
MAX_TOKENS=128
OUTPUT_DIR="per_request_results"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --step-size)
            STEP_SIZE="$2"
            shift 2
            ;;
        --num-requests)
            NUM_REQUESTS="$2"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --step-size N      min_det_step_size value (default: 10)"
            echo "  --num-requests N   number of requests (default: 100)"
            echo "  --batch-size N     prompts per request (default: 1)"
            echo "  --max-tokens N     max tokens to generate (default: 128)"
            echo "  --output-dir DIR   output directory (default: per_request_results)"
            echo "  --port N           server port (default: 30003)"
            echo "  --model PATH       model path (default: meta-llama/Meta-Llama-3.1-8B-Instruct)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Track PIDs for cleanup
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
mkdir -p "$OUTPUT_DIR"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$OUTPUT_DIR/server_step${STEP_SIZE}_${TIMESTAMP}.log"
RESULTS_FILE="$OUTPUT_DIR/results_step${STEP_SIZE}_${TIMESTAMP}.json"

echo "=========================================="
echo "Per-Request Rollback Benchmark"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Model:          $MODEL"
echo "  Port:           $PORT"
echo "  Step size:      $STEP_SIZE"
echo "  Num requests:   $NUM_REQUESTS"
echo "  Batch size:     $BATCH_SIZE"
echo "  Max tokens:     $MAX_TOKENS"
echo "  Log file:       $LOG_FILE"
echo "  Results file:   $RESULTS_FILE"
echo ""

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
    --min-det-step-size "$STEP_SIZE" \
    2>&1 | tee "$LOG_FILE" &

SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Wait for server to be ready
echo ""
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
    echo "✗ Server failed to start"
    echo "Check log file: $LOG_FILE"
    exit 1
fi

# Run benchmark
echo ""
echo "Running benchmark..."
python bench_per_request_rollbacks.py \
    --port "$PORT" \
    --num-requests "$NUM_REQUESTS" \
    --batch-size "$BATCH_SIZE" \
    --max-tokens "$MAX_TOKENS" \
    --log-file "$LOG_FILE" \
    --output "$RESULTS_FILE" \
    --verbose

# Give time for final logs to be written
sleep 2

# Re-analyze log file to capture any final stats
echo ""
echo "Final log analysis..."
python bench_per_request_rollbacks.py \
    --analyze-only \
    --log-file "$LOG_FILE" \
    --output "$RESULTS_FILE" \
    --num-requests "$NUM_REQUESTS" \
    --batch-size "$BATCH_SIZE" \
    --max-tokens "$MAX_TOKENS"

echo ""
echo "=========================================="
echo "✓ Benchmark complete!"
echo "=========================================="
echo ""
echo "Output files:"
echo "  Log file:     $LOG_FILE"
echo "  Results:      $RESULTS_FILE"
echo ""

# Stop server
echo "Stopping server..."
kill $SERVER_PID 2>/dev/null || true
SERVER_PID=""
sleep 2

echo "Done."
