#!/bin/bash
# Automated sweep for varying min-det-step-size

set -e

MODEL="${MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
PORT=30000
STEP_SIZES=(1 5 10 20 50)
DURATION=120  # seconds per experiment
RESULTS_DIR="sweep_results"
TP_SIZE="${SGLANG_TP_SIZE:-1}"
ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-flashinfer}"

# Track PIDs for cleanup
SERVER_PID=""
COLLECTOR_PID=""

# Cleanup function
cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ -n "$COLLECTOR_PID" ]; then
        kill $COLLECTOR_PID 2>/dev/null || true
    fi
    if [ -n "$SERVER_PID" ]; then
        kill $SERVER_PID 2>/dev/null || true
        sleep 2
        kill -9 $SERVER_PID 2>/dev/null || true
    fi
    exit 1
}

# Trap Ctrl+C and cleanup
trap cleanup SIGINT SIGTERM

mkdir -p "$RESULTS_DIR"

# Add metrics (one-time)
echo "Adding rollback metrics..."
python add_rollback_metrics.py

# Function to run one experiment
run_experiment() {
    local step_size=$1
    local output_dir="$RESULTS_DIR/step_${step_size}"
    mkdir -p "$output_dir"
    
    echo ""
    echo "=========================================="
    echo "Running: step_size=$step_size"
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
        --enable-deterministic-inference 1 \
        --min-det-step-size "$step_size" \
        --disable-radix-cache \
        > "$output_dir/server.log" 2>&1 &
    
    local server_pid=$!
    SERVER_PID=$server_pid
    echo "Server PID: $server_pid"
    
    # Wait for server (longer timeout for model loading)
    echo "Waiting for server..."
    for i in {1..120}; do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "✓ Server ready"
            break
        fi
        sleep 5
        if [ $((i % 6)) -eq 0 ]; then
            echo "  Still waiting... (${i}0s)"
        fi
    done
    
    if ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "✗ Server failed to start"
        kill $server_pid 2>/dev/null || true
        return 1
    fi
    
    # Start stats collection
    echo "Collecting stats for ${DURATION}s..."
    python collect_rollback_stats.py \
        --url "http://localhost:$PORT" \
        --duration "$DURATION" \
        --output "$output_dir/stats.json" &
    
    local collector_pid=$!
    COLLECTOR_PID=$collector_pid
    
    # Run your benchmark here
    echo "Running benchmark..."
    python3 vllm_online_batch_invariance_multitest.py
    
    # For now, just wait for stats collection
    echo "Waiting for stats collection to complete..."
    sleep "$DURATION"
    
    # Wait for collector
    wait $collector_pid 2>/dev/null || true
    
    # Stop server
    echo "Stopping server..."
    kill $server_pid 2>/dev/null || true
    sleep 5
    # Force kill if still running
    kill -9 $server_pid 2>/dev/null || true
    
    # Clear PIDs
    SERVER_PID=""
    COLLECTOR_PID=""
    
    # Generate plots
    echo "Generating plots..."
    python plot_rollback_stats.py \
        --input "$output_dir/stats.json" \
        --output "$output_dir"
    
    echo "✓ Completed: step_size=$step_size"
}

# Run experiments for each step size
for step_size in "${STEP_SIZES[@]}"; do
    run_experiment "$step_size"
done

# Generate comparison plot
echo ""
echo "=========================================="
echo "Generating comparison plots..."
echo "=========================================="

stats_files=()
labels=()
for step_size in "${STEP_SIZES[@]}"; do
    stats_files+=("$RESULTS_DIR/step_${step_size}/stats.json")
    labels+=("step=${step_size}")
done

python plot_rollback_stats.py \
    --compare "${stats_files[@]}" \
    --labels "${labels[@]}" \
    --output "$RESULTS_DIR/comparison"

echo ""
echo "=========================================="
echo "✓ Sweep complete!"
echo "=========================================="
echo "Results in: $RESULTS_DIR/"
echo ""
ls -la "$RESULTS_DIR"
