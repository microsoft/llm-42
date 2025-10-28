#!/bin/bash

# Quick comparison test between static deterministic and temperature-based all-deterministic
# This script compares:
# 1. Mode 66 - Static deterministic (batch-invariant with vllm-rmsnorm + cutlass matmul)
# 2. Mode 578 - Temperature-based with 100% temp=0 (should perform the same as mode 66)

set -e

# Default configuration
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
HOST="0.0.0.0"
PORT=30000
TP_SIZE=1
ATTENTION_BACKEND="flashinfer"
OUTPUT_DIR="fix_comparison_results"
QPS=1.0
MAX_REQUESTS=256
TIMEOUT=600
NUM_CLIENTS=1
CONCURRENT=256
TRACE_FILE="../etalon/data/processed_traces/arxiv_summarization_filtered_stats_llama2_tokenizer.csv"
MAX_TOKENS=8192
WARMUP_TIME=30  # Time to wait for server to warmup

# Modes to test
MODES=("66" "578")
MODE_NAMES=("det_mode_66" "det_mode_578_temp0_100pct")
TEMP0_PERCENTAGES=("0" "100")
MODE_DESCRIPTIONS=(
    "Mode 66 (Static deterministic: vllm-rmsnorm + cutlass)"
    "Mode 578 (Temperature-based: 100% temp=0, 0% temp=1)"
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
            echo "Quick comparison test between static and temperature-based deterministic modes"
            echo ""
            echo "Test modes:"
            echo "  1. Mode 66 - Static deterministic (baseline)"
            echo "  2. Mode 578 - Temperature-based with 100% temp=0 (should match mode 66)"
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
echo "Deterministic Mode Comparison Test"
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
            # --cuda-graph-max-bs 32 \
        # --mem-fraction-static 0.7 \
    
    # Launch new server
    echo "Starting server with deterministic mode $mode..."
    $PYTHON_CMD -m sglang.launch_server \
        --model-path $MODEL \
        --host $HOST \
        --port $PORT \
        --tp-size $TP_SIZE \
        --attention-backend $ATTENTION_BACKEND \
        --disable-radix-cache \
        --disable-cuda-graph \
        --enable-deterministic-inference $mode \
        > "${OUTPUT_DIR}/${mode_name}_server.log" 2>&1 &
    
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
    local temp0_pct=$3
    local run_output_dir="${OUTPUT_DIR}/${mode_name}"
    
    echo ""
    echo "================================================"
    echo "Running benchmark: $mode_name"
    if [ "$temp0_pct" != "0" ]; then
        echo "Temperature 0 percentage: ${temp0_pct}%"
    fi
    echo "================================================"
    
    # Create output directory
    mkdir -p "$run_output_dir"
    
    # Set API configuration for etalon
    export OPENAI_API_KEY="EMPTY"
    export OPENAI_API_BASE="http://localhost:${PORT}/v1"
    export WANDB_MODE=disabled
    
    echo "Etalon will connect to: $OPENAI_API_BASE"
    
    # For mode 578 with temperature distribution, use the wrapper script
    if [ "$temp0_pct" != "0" ]; then
        echo "Using mixed temperature benchmark wrapper"
        if $PYTHON_CMD "$(dirname "$0")/run_mixed_temperature_benchmark.py" \
            --temp0-pct "$temp0_pct" \
            --model "$MODEL" \
            --max-requests $MAX_REQUESTS \
            --timeout $TIMEOUT \
            --num-clients $NUM_CLIENTS \
            --concurrent $CONCURRENT \
            --output-dir "$run_output_dir" \
            --qps $QPS \
            --trace-file "$TRACE_FILE" \
            --max-tokens $MAX_TOKENS \
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
    else
        # Standard etalon benchmark for mode 66
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
    fi
}

# Main test loop
echo "Starting comparison test..."
echo ""

for i in "${!MODES[@]}"; do
    mode="${MODES[$i]}"
    mode_name="${MODE_NAMES[$i]}"
    mode_desc="${MODE_DESCRIPTIONS[$i]}"
    temp0_pct="${TEMP0_PERCENTAGES[$i]}"
    
    echo "###############################################"
    echo "# Test $((i+1))/${#MODES[@]}: $mode_name"
    echo "###############################################"
    
    # Launch server
    if ! launch_server "$mode" "$mode_name" "$mode_desc"; then
        echo "Failed to launch server for $mode_name"
        continue
    fi
    
    # Run benchmark
    run_benchmark "$mode" "$mode_name" "$temp0_pct"
    
    echo ""
done

# Kill server after all tests
echo ""
echo "================================================"
echo "Comparison test completed!"
echo "================================================"
kill_server

echo ""
echo "Results saved to: $OUTPUT_DIR"
echo ""

# Quick comparison of results
echo "================================================"
echo "Quick Results Comparison"
echo "================================================"
echo ""

for i in "${!MODE_NAMES[@]}"; do
    mode_name="${MODE_NAMES[$i]}"
    ttft_file="${OUTPUT_DIR}/${mode_name}/ttft.csv"
    
    if [ -f "$ttft_file" ]; then
        echo "### $mode_name ###"
        echo "Description: ${MODE_DESCRIPTIONS[$i]}"
        
        # Extract p50 and p99 TTFT
        p50=$(awk -F',' 'NR==51 {print $3}' "$ttft_file")
        p99=$(awk -F',' 'NR==100 {print $3}' "$ttft_file")
        
        echo "  TTFT p50: $p50 seconds"
        echo "  TTFT p99: $p99 seconds"
        echo ""
    fi
done

echo "For detailed analysis, check the CSV files in: $OUTPUT_DIR"
echo ""
