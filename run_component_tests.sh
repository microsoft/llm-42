#!/bin/bash

# Component-based SGLang Deterministic Mode Testing with Etalon
# This script tests individual deterministic components with both TM and Cutlass implementations
#
# Test Matrix:
# 1. Baseline (Non-deterministic) - all components non-deterministic
# 2. Matmul Deterministic Only (TM) - mode 449 (1+64+128+256)
# 3. Matmul Deterministic Only (Cutlass) - mode 450 (2+64+128+256)
# 4. Attention Deterministic Only - mode 352 (32+64+256)
# 5. RMSNorm Deterministic Only - mode 416 (32+128+256)
# 6. RMSNorm Deterministic Ours - mode 224 (32+64+128)

set -e

# Default configuration
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
HOST="0.0.0.0"
PORT=30000
TP_SIZE=1
ATTENTION_BACKEND="flashinfer"
OUTPUT_DIR="etalon_results_component_tests"
QPS=1.0
MAX_REQUESTS=256
TIMEOUT=600
NUM_CLIENTS=1
CONCURRENT=256
TRACE_FILE="./etalon/data/processed_traces/arxiv_summarization_filtered_stats_llama2_tokenizer.csv"
MAX_TOKENS=8192
WARMUP_TIME=30  # Time to wait for server to warmup

# Corrected modes based on bit flags:
# Bit 1 (1): Enable deterministic base (uses Triton/ThinkingMachine kernel for matmul)
# Bit 2 (2): Use CUTLASS kernel matmul (only if bit 1 is NOT set - they are mutually exclusive!)
# Bit 6 (32): Use non-det matmul
# Bit 7 (64): Use non-det rmsnorm  
# Bit 8 (128): Use non-det attention
#
# IMPORTANT: The code uses elif, so bit 1 and bit 2 are mutually exclusive!
# - If bit 1 is set: uses Triton/TM kernel
# - If bit 1 is NOT set but bit 2 is set: uses CUTLASS kernel
# - If neither is set: uses standard non-det matmul
#
# To get "only X deterministic", we set the appropriate bit and mark others as non-det:
MODES=("baseline" "449" "450" "352" "416" "224")
MODE_NAMES=("baseline_nondet" "matmul_det_tm" "matmul_det_cutlass" "attention_det_only" "rmsnorm_det_only" "rmsnorm_det_ours")
MODE_DESCRIPTIONS=(
    "Baseline - Non-deterministic (all components non-det)"
    "Matmul Deterministic Only - TM (mode 449: 1+64+128+256, Triton/ThinkingMachine)"
    "Matmul Deterministic Only - Cutlass (mode 450: 2+64+128+256, CUTLASS kernel)"
    "Attention Deterministic Only (mode 352: 32+64+256)"
    "RMSNorm Deterministic Only (mode 416: 32+128+256)"
    "RMSNorm Deterministic Ours (mode 224: 32+64+128)"
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
        --output-dir)
            OUTPUT_DIR="$2"
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
            echo "Component-based deterministic testing script"
            echo "Tests individual components (Matmul, Attention, RMSNorm) in isolation"
            echo ""
            echo "Test modes:"
            echo "  1. Baseline - Non-deterministic (all components)"
            echo "  2. Matmul Deterministic Only - TM (Thinking Machine)"
            echo "  3. Matmul Deterministic Only - Cutlass kernel"
            echo "  4. Attention Deterministic Only"
            echo "  5. RMSNorm Deterministic Only"
            echo "  6. RMSNorm Deterministic Ours (mode 224)"
            echo ""
            echo "Options:"
            echo "  --model MODEL           Model path (default: $MODEL)"
            echo "  --port PORT             Server port (default: $PORT)"
            echo "  --output-dir DIR        Output directory (default: $OUTPUT_DIR)"
            echo "  --qps QPS               Queries per second (default: $QPS)"
            echo "  --max-requests N        Max requests (default: $MAX_REQUESTS)"
            echo "  --timeout SECONDS       Timeout (default: $TIMEOUT)"
            echo "  --warmup-time SECONDS   Server warmup time (default: $WARMUP_TIME)"
            echo "  --help                  Show this help"
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

echo "================================================"
echo "Component-based Deterministic Testing"
echo "================================================"
echo "Model: $MODEL"
echo "Port: $PORT"
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
    
    # Run etalon benchmark directly
    if python3 -m etalon.run_benchmark \
        --client_config_model "$MODEL" \
        --max_completed_requests $MAX_REQUESTS \
        --timeout $TIMEOUT \
        --client_config_num_clients $NUM_CLIENTS \
        --client_config_num_concurrent_requests_per_client $CONCURRENT \
        --metrics_config_output_dir "$run_output_dir" \
        --metrics_config_should_write_metrics \
        --request_interval_generator_config_type "poisson" \
        --poisson_request_interval_generator_config_qps $QPS \
        --request_length_generator_config_type "trace" \
        --trace_request_length_generator_config_trace_file "$TRACE_FILE" \
        --trace_request_length_generator_config_max_tokens $MAX_TOKENS \
        --deadline_config_ttft_deadline 0.3 \
        --deadline_config_tbt_deadline 0.03 \
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
echo "Starting component-based testing..."
echo ""

for i in "${!MODES[@]}"; do
    mode="${MODES[$i]}"
    mode_name="${MODE_NAMES[$i]}"
    mode_desc="${MODE_DESCRIPTIONS[$i]}"
    
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
echo "All component tests completed!"
echo "================================================"
kill_server

echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""

echo "To plot results, run:"
echo "  ./plot_component_results.sh --input-dir $OUTPUT_DIR"
echo ""
echo "Bit flag explanation:"
echo "  Mode 449 = 1 + 64 + 128 + 256 (TM + Non-det RMSNorm + Non-det Attention + bit 256) = Matmul Det TM only"
echo "  Mode 450 = 2 + 64 + 128 + 256 (CUTLASS + Non-det RMSNorm + Non-det Attention + bit 256) = Matmul Det CUTLASS only"
echo "  Mode 352 = 32 + 64 + 256 (Non-det Matmul + Non-det RMSNorm + bit 256) = Attention Det only"
echo "  Mode 416 = 32 + 128 + 256 (Non-det Matmul + Non-det Attention + bit 256) = RMSNorm Det only"
echo "  Mode 224 = 32 + 64 + 128 (Non-det Matmul + Non-det RMSNorm + Non-det Attention) = RMSNorm Det Ours"
echo ""
echo "Note: Bit 1 (Triton/TM) and Bit 2 (CUTLASS) are mutually exclusive (code uses elif)!"
echo "Note: Bit 256 appears to be a new flag added to the deterministic modes"
