#!/bin/bash

# Component-based SGLang Deterministic Mode Testing with Etalon
# This script tests deterministic modes with mixed temperature distributions
#
# Test Matrix:
# 1. Baseline (Non-deterministic) - all components non-deterministic
# 2. Mode 66 - batch-invariant with vllm-rmsnorm + cutlass matmul
# 3. Mode 257 - batch-invariant with native-rmsnorm + thinking-machine matmul
# 4. Mode 578 - temperature-based switching with configurable temp=0 percentage

set -e

# Default configuration
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
HOST="0.0.0.0"
PORT=30000
TP_SIZE=1
ATTENTION_BACKEND="flashinfer"
OUTPUT_DIR="mixed_temp_results"
QPS=1.0
MAX_REQUESTS=256
TIMEOUT=600
NUM_CLIENTS=1
CONCURRENT=256
TRACE_FILE="../etalon/data/processed_traces/arxiv_summarization_filtered_stats_llama2_tokenizer.csv"
MAX_TOKENS=8192
WARMUP_TIME=30  # Time to wait for server to warmup

# Mixed temperature configuration
TEMP0_PCT=10  # Default: 10% of requests have temperature=0
ASSIGNMENT_MODE="random"  # Default: random assignment with seed
SEED=42  # Default seed for random assignment

# Modes to test
# Mode 66 = batch-invariant:: vllm-rmsnorm + cutlass matmul
# Mode 257 = batch-invariant:: native-rmsnorm + thinking-machine matmul
# Mode 578 = temperature-based switching + cutlass matmul + vllm-rmsnorm
MODES=("baseline" "66" "257" "578")
MODE_DESCRIPTIONS=(
    "Baseline (Non-deterministic)"
    "Mode 66 (batch-invariant: vllm-rmsnorm + cutlass)"
    "Mode 257 (batch-invariant: native-rmsnorm + TM)"
    "Mode 578 (temp-based with mixed temperatures)"
)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --temp0-pct)
            TEMP0_PCT="$2"
            shift 2
            ;;
        --assignment-mode)
            ASSIGNMENT_MODE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --qps)
            QPS="$2"
            shift 2
            ;;
        --max-requests)
            MAX_REQUESTS="$2"
            shift 2
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --warmup-time)
            WARMUP_TIME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Mixed temperature deterministic testing script"
            echo "Tests different deterministic modes with configurable temperature distribution"
            echo ""
            echo "Test modes:"
            echo "  1. Baseline - Non-deterministic (all components)"
            echo "  2. Mode 66 - batch-invariant (vllm-rmsnorm + cutlass matmul)"
            echo "  3. Mode 257 - batch-invariant (native-rmsnorm + thinking-machine matmul)"
            echo "  4. Mode 578 - temperature-based switching with mixed temperatures"
            echo ""
            echo "Options:"
            echo "  --model MODEL             Model path (default: $MODEL)"
            echo "  --port PORT               Server port (default: $PORT)"
            echo "  --temp0-pct PCT           Percentage of requests with temp=0 (default: $TEMP0_PCT)"
            echo "  --assignment-mode MODE    'random' or 'fixed' (default: $ASSIGNMENT_MODE)"
            echo "  --seed SEED               Random seed for 'random' mode (default: $SEED)"
            echo "  --qps QPS                 Queries per second (default: $QPS)"
            echo "  --max-requests N          Max requests (default: $MAX_REQUESTS)"
            echo "  --timeout SECONDS         Timeout (default: $TIMEOUT)"
            echo "  --warmup-time SECONDS     Server warmup time (default: $WARMUP_TIME)"
            echo "  --help                    Show this help"
            echo ""
            echo "Examples:"
            echo "  # 10% random with seed 42:"
            echo "  $0 --temp0-pct 10 --assignment-mode random --seed 42"
            echo ""
            echo "  # 5% fixed (every 20th request):"
            echo "  $0 --temp0-pct 5 --assignment-mode fixed"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Determine Python command
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: Python not found."
    exit 1
fi

# Set output directory based on temp0 percentage and assignment mode
OUTPUT_DIR="pct_${TEMP0_PCT}_${ASSIGNMENT_MODE}"

echo "================================================"
echo "Mixed Temperature Deterministic Testing"
echo "================================================"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Temperature Configuration:"
echo "  - ${TEMP0_PCT}% of requests with temperature=0"
echo "  - $((100 - TEMP0_PCT))% of requests with temperature=1"
echo "  - Assignment mode: $ASSIGNMENT_MODE"
if [ "$ASSIGNMENT_MODE" = "random" ]; then
    echo "  - Random seed: $SEED"
fi
echo "Output Directory: $OUTPUT_DIR"
echo ""
echo "Test Matrix:"
for i in "${!MODES[@]}"; do
    echo "  $((i+1)). ${MODE_DESCRIPTIONS[$i]}"
done
echo "================================================"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Function to kill any existing server
kill_server() {
    echo "Stopping any existing SGLang server..."
    pkill -f "sglang.launch_server" || true
    sleep 5
}

# Function to wait for server to be ready
wait_for_server() {
    local max_attempts=60
    local attempt=0
    
    echo "Waiting for server to be ready..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -s -f "http://localhost:${PORT}/v1/models" > /dev/null 2>&1; then
            echo "✓ Server is ready!"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    
    echo "Error: Server failed to start within timeout"
    return 1
}

# Function to launch server
launch_server() {
    local mode=$1
    local mode_name=$2
    local mode_desc=$3
    
    echo ""
    echo "================================================"
    echo "Launching server: $mode_name"
    echo "Description: $mode_desc"
    echo "================================================"
    
    # Kill existing server
    kill_server
    
    # Launch new server
    if [ "$mode" = "baseline" ]; then
        echo "Starting non-deterministic server..."
        $PYTHON_CMD -m sglang.launch_server \
            --model-path $MODEL \
            --host $HOST \
            --port $PORT \
            --tp-size $TP_SIZE \
            --attention-backend $ATTENTION_BACKEND \
            --disable-radix-cache \
            > "${OUTPUT_DIR}/${mode_name}_server.log" 2>&1 &
    else
        echo "Starting server with deterministic mode $mode..."
        $PYTHON_CMD -m sglang.launch_server \
            --model-path $MODEL \
            --host $HOST \
            --port $PORT \
            --tp-size $TP_SIZE \
            --attention-backend $ATTENTION_BACKEND \
            --disable-radix-cache \
            --enable-deterministic-inference $mode \
            > "${OUTPUT_DIR}/${mode_name}_server.log" 2>&1 &
    fi
    
    SERVER_PID=$!
    echo "Server PID: $SERVER_PID"
    echo "Server log: ${OUTPUT_DIR}/${mode_name}_server.log"
    
    # Wait for server to be ready
    if ! wait_for_server; then
        echo "Error: Failed to start server. Check log: ${OUTPUT_DIR}/${mode_name}_server.log"
        return 1
    fi
    
    # Additional warmup time
    echo "Warming up for ${WARMUP_TIME} seconds..."
    sleep $WARMUP_TIME
    
    return 0
}

# Function to run benchmark
run_benchmark() {
    local mode=$1
    local mode_name=$2
    local run_output_dir="${OUTPUT_DIR}/${mode_name}"
    
    echo ""
    echo "================================================"
    echo "Running benchmark: $mode_name"
    echo "================================================"
    
    # Create output directory
    mkdir -p "$run_output_dir"
    
    # Set API configuration for etalon
    export OPENAI_API_KEY="EMPTY"
    export OPENAI_API_BASE="http://localhost:${PORT}/v1"
    export WANDB_MODE=disabled
    
    echo "Etalon will connect to: $OPENAI_API_BASE"
    
    # For mode 578, use the mixed temperature benchmark wrapper
    # For baseline, 66, 257: run all with the same mixed temperature distribution for fair comparison
    echo "Using mixed temperature benchmark wrapper"
    echo "  Temperature 0 percentage: ${TEMP0_PCT}%"
    echo "  Assignment mode: ${ASSIGNMENT_MODE}"
    if [ "$ASSIGNMENT_MODE" = "random" ]; then
        echo "  Random seed: ${SEED}"
    fi
    
    local wrapper_args=(
        --temp0-pct "$TEMP0_PCT"
        --assignment-mode "$ASSIGNMENT_MODE"
        --model "$MODEL"
        --max-requests $MAX_REQUESTS
        --timeout $TIMEOUT
        --num-clients $NUM_CLIENTS
        --concurrent $CONCURRENT
        --output-dir "$run_output_dir"
        --qps $QPS
        --trace-file "$TRACE_FILE"
        --max-tokens $MAX_TOKENS
    )
    
    if [ "$ASSIGNMENT_MODE" = "random" ]; then
        wrapper_args+=(--seed "$SEED")
    fi
    
    if $PYTHON_CMD "$(dirname "$0")/run_mixed_temperature_benchmark.py" "${wrapper_args[@]}" \
        2>&1 | tee "${run_output_dir}/benchmark.log"; then
        
        echo "✓ Benchmark completed: $mode_name"
        unset OPENAI_API_KEY
        unset OPENAI_API_BASE
        unset WANDB_MODE
        return 0
    else
        echo "⚠ Benchmark had issues: $mode_name"
        unset OPENAI_API_KEY
        unset OPENAI_API_BASE
        unset WANDB_MODE
        return 1
    fi
}

# Main test loop
echo "Starting mixed temperature testing..."
echo ""

for i in "${!MODES[@]}"; do
    mode="${MODES[$i]}"
    mode_desc="${MODE_DESCRIPTIONS[$i]}"
    
    # Generate mode name based on mode number
    if [ "$mode" = "baseline" ]; then
        mode_name="baseline_nondet"
    else
        mode_name="det_mode_${mode}"
    fi
    
    echo "###############################################"
    echo "# Test $((i+1))/${#MODES[@]}: $mode_name"
    echo "###############################################"
    
    # Launch server
    if ! launch_server "$mode" "$mode_name" "$mode_desc"; then
        echo "Failed to launch server for $mode_name"
        continue
    fi
    
    # Run benchmark
    run_benchmark "$mode" "$mode_name"
    
    echo ""
done

# Kill server after all tests
echo ""
echo "================================================"
echo "All tests completed!"
echo "================================================"
kill_server

echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""

echo "To plot results, run:"
echo "  ./plot_results.sh --input-dir $OUTPUT_DIR"
echo ""