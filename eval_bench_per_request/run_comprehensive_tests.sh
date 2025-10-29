#!/bin/bash

# Comprehensive Testing Script for SGLang Deterministic Mode
# Tests multiple configurations:
# - Baseline (non-deterministic)
# - Mode 66 (batch-invariant: vllm-rmsnorm + cutlass)
# - Mode 257 (batch-invariant: native-rmsnorm + TM)
# - Mode 578 with varying temperature percentages: 0%, 1%, 2%, 5%, 10%, 50%, 100%

set -e

# Default configuration
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
HOST="0.0.0.0"
PORT=30000
TP_SIZE=1
ATTENTION_BACKEND="flashinfer"
BASE_OUTPUT_DIR="comprehensive_results"
QPS=1.0
MAX_REQUESTS=256
TIMEOUT=600
WARMUP_TIME=30
ASSIGNMENT_MODE="random"
SEED=42
TRACE_TYPE="arxiv"  # Options: "arxiv", "sharegpt", or "lmsys"

# Test configurations
# Format: "mode:temp0_pct:description"
TEST_CONFIGS=(
    "baseline:100:Baseline (Non-deterministic, 100% temp=0)"
    "66:100:Mode 66 (batch-invariant vllm-rmsnorm + cutlass, 100% temp=0)"
    "257:100:Mode 257 (batch-invariant native-rmsnorm + TM, 100% temp=0)"
    "578:0:Mode 578 (temp-based, 0% temp=0 - all temp=1)"
    "578:1:Mode 578 (temp-based, 1% temp=0)"
    "578:2:Mode 578 (temp-based, 2% temp=0)"
    "578:5:Mode 578 (temp-based, 5% temp=0)"
    "578:10:Mode 578 (temp-based, 10% temp=0)"
    "578:50:Mode 578 (temp-based, 50% temp=0)"
    "578:100:Mode 578 (temp-based, 100% temp=0 - all deterministic)"
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
        --base-output-dir)
            BASE_OUTPUT_DIR="$2"
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
        --trace-type)
            TRACE_TYPE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Comprehensive deterministic testing script"
            echo "Tests baseline, modes 66, 257, and mode 578 with varying temperature percentages"
            echo ""
            echo "Test matrix (${#TEST_CONFIGS[@]} total tests):"
            echo "  1. Baseline (non-det, 100% temp=0)"
            echo "  2. Mode 66 (batch-invariant vllm+cutlass, 100% temp=0)"
            echo "  3. Mode 257 (batch-invariant native+TM, 100% temp=0)"
            echo "  4. Mode 578 with 0% temp=0 (all temp=1)"
            echo "  5. Mode 578 with 1% temp=0"
            echo "  6. Mode 578 with 2% temp=0"
            echo "  7. Mode 578 with 5% temp=0"
            echo "  8. Mode 578 with 10% temp=0"
            echo "  9. Mode 578 with 50% temp=0"
            echo "  10. Mode 578 with 100% temp=0 (all deterministic)"
            echo ""
            echo "Options:"
            echo "  --model MODEL             Model path (default: $MODEL)"
            echo "  --port PORT               Server port (default: $PORT)"
            echo "  --base-output-dir DIR     Base output directory (default: $BASE_OUTPUT_DIR)"
            echo "  --assignment-mode MODE    'random' or 'fixed' (default: $ASSIGNMENT_MODE)"
            echo "  --seed SEED               Random seed (default: $SEED)"
            echo "  --qps QPS                 Queries per second (default: $QPS)"
            echo "  --max-requests N          Max requests (default: $MAX_REQUESTS)"
            echo "  --timeout SECONDS         Timeout (default: $TIMEOUT)"
            echo "  --warmup-time SECONDS     Server warmup time (default: $WARMUP_TIME)"
            echo "  --trace-type TYPE         Trace file type: 'arxiv', 'sharegpt', or 'lmsys' (default: $TRACE_TYPE)"
            echo "  --help                    Show this help"
            echo ""
            echo "Examples:"
            echo "  # Run all tests with default settings (arxiv):"
            echo "  $0"
            echo ""
            echo "  # Run with sharegpt trace:"
            echo "  $0 --trace-type sharegpt"
            echo ""
            echo "  # Run with lmsys trace:"
            echo "  $0 --trace-type lmsys"
            echo ""
            echo "  # Run with custom output directory:"
            echo "  $0 --base-output-dir my_comprehensive_results"
            echo ""
            echo "  # Run with fixed assignment mode:"
            echo "  $0 --assignment-mode fixed"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
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

# Set trace file based on trace type
case $TRACE_TYPE in
    arxiv)
        TRACE_FILE="../etalon/data/processed_traces/arxiv_summarization_filtered_stats_llama2_tokenizer.csv"
        ;;
    sharegpt)
        TRACE_FILE="../etalon/data/processed_traces/sharegpt_8k_filtered_stats_llama2_tokenizer.csv"
        ;;
    lmsys)
        TRACE_FILE="../etalon/data/processed_traces/lmsys_chat_1m_conversation_stats_llama2_tokenizer.csv"
        ;;
    *)
        echo "Error: Invalid trace type '$TRACE_TYPE'. Must be 'arxiv', 'sharegpt', or 'lmsys'."
        exit 1
        ;;
esac

# Verify trace file exists
if [ ! -f "$TRACE_FILE" ]; then
    echo "Error: Trace file not found: $TRACE_FILE"
    echo "Please ensure the trace file exists or choose a different trace type."
    exit 1
fi

# Extract model name from path (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct -> Meta-Llama-3.1-8B-Instruct)
MODEL_NAME=$(basename "$MODEL")

# Create base output directory with timestamp and structured naming
# Format: {trace-name}_{model_name}_tp{size}_{timestamp}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${TRACE_TYPE}_${MODEL_NAME}_tp${TP_SIZE}_${TIMESTAMP}"
mkdir -p "$OUTPUT_DIR"

# Save configuration
CONFIG_FILE="${OUTPUT_DIR}/test_config.txt"
cat > "$CONFIG_FILE" << EOF
Comprehensive Deterministic Testing Configuration
==================================================
Timestamp: $(date)
Model: $MODEL
Port: $PORT
Trace Type: $TRACE_TYPE
Trace File: $TRACE_FILE
Assignment Mode: $ASSIGNMENT_MODE
Seed: $SEED
QPS: $QPS
Max Requests: $MAX_REQUESTS
Timeout: $TIMEOUT
Warmup Time: $WARMUP_TIME

Test Matrix (${#TEST_CONFIGS[@]} configurations):
EOF

for i in "${!TEST_CONFIGS[@]}"; do
    IFS=':' read -r mode temp_pct description <<< "${TEST_CONFIGS[$i]}"
    echo "  $((i+1)). $description" >> "$CONFIG_FILE"
done

echo ""
echo "================================================"
echo "Comprehensive Deterministic Testing"
echo "================================================"
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Trace Type: $TRACE_TYPE"
echo "Trace File: $TRACE_FILE"
echo "Assignment Mode: $ASSIGNMENT_MODE"
if [ "$ASSIGNMENT_MODE" = "random" ]; then
    echo "Random Seed: $SEED"
fi
echo "Output Directory: $OUTPUT_DIR"
echo ""
echo "Test Matrix (${#TEST_CONFIGS[@]} configurations):"
for i in "${!TEST_CONFIGS[@]}"; do
    IFS=':' read -r mode temp_pct description <<< "${TEST_CONFIGS[$i]}"
    echo "  $((i+1)). $description"
done
echo "================================================"
echo ""

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
    local run_name=$2
    local description=$3
    
    echo ""
    echo "================================================"
    echo "Launching server: $run_name"
    echo "Description: $description"
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
            > "${OUTPUT_DIR}/${run_name}_server.log" 2>&1 &
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
            > "${OUTPUT_DIR}/${run_name}_server.log" 2>&1 &
    fi
    
    SERVER_PID=$!
    echo "Server PID: $SERVER_PID"
    echo "Server log: ${OUTPUT_DIR}/${run_name}_server.log"
    
    # Wait for server to be ready
    if ! wait_for_server; then
        echo "Error: Failed to start server. Check log: ${OUTPUT_DIR}/${run_name}_server.log"
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
    local temp_pct=$2
    local run_name=$3
    local run_output_dir="${OUTPUT_DIR}/${run_name}"
    
    echo ""
    echo "================================================"
    echo "Running benchmark: $run_name"
    echo "Mode: $mode, Temp=0 Percentage: ${temp_pct}%"
    echo "================================================"
    
    # Create output directory
    mkdir -p "$run_output_dir"
    
    # Set API configuration for etalon
    export OPENAI_API_KEY="EMPTY"
    export OPENAI_API_BASE="http://localhost:${PORT}/v1"
    export WANDB_MODE=disabled
    
    echo "Etalon will connect to: $OPENAI_API_BASE"
    echo "Temperature 0 percentage: ${temp_pct}%"
    echo "Assignment mode: ${ASSIGNMENT_MODE}"
    if [ "$ASSIGNMENT_MODE" = "random" ]; then
        echo "Random seed: ${SEED}"
    fi
    
    # Build wrapper arguments
    local wrapper_args=(
        --temp0-pct "$temp_pct"
        --assignment-mode "$ASSIGNMENT_MODE"
        --model "$MODEL"
        --max-requests $MAX_REQUESTS
        --timeout $TIMEOUT
        --num-clients 1
        --concurrent $MAX_REQUESTS
        --output-dir "$run_output_dir"
        --qps $QPS
        --trace-file "$TRACE_FILE"
        --max-tokens 8192
    )
    
    if [ "$ASSIGNMENT_MODE" = "random" ]; then
        wrapper_args+=(--seed "$SEED")
    fi
    
    if $PYTHON_CMD "$(dirname "$0")/run_mixed_temperature_benchmark.py" "${wrapper_args[@]}" \
        2>&1 | tee "${run_output_dir}/benchmark.log"; then
        
        echo "✓ Benchmark completed: $run_name"
        unset OPENAI_API_KEY
        unset OPENAI_API_BASE
        unset WANDB_MODE
        return 0
    else
        echo "⚠ Benchmark had issues: $run_name"
        unset OPENAI_API_KEY
        unset OPENAI_API_BASE
        unset WANDB_MODE
        return 1
    fi
}

# Track test results
declare -a SUCCESSFUL_TESTS
declare -a FAILED_TESTS

# Main test loop
echo "Starting comprehensive testing..."
echo "Total tests: ${#TEST_CONFIGS[@]}"
echo ""

for i in "${!TEST_CONFIGS[@]}"; do
    IFS=':' read -r mode temp_pct description <<< "${TEST_CONFIGS[$i]}"
    
    # Generate run name
    if [ "$mode" = "baseline" ]; then
        run_name="baseline_${temp_pct}pct"
    else
        run_name="mode${mode}_${temp_pct}pct"
    fi
    
    echo ""
    echo "###############################################"
    echo "# Test $((i+1))/${#TEST_CONFIGS[@]}: $run_name"
    echo "# $description"
    echo "###############################################"
    
    # Launch server
    if ! launch_server "$mode" "$run_name" "$description"; then
        echo "Failed to launch server for $run_name"
        FAILED_TESTS+=("$run_name: Server launch failed")
        continue
    fi
    
    # Run benchmark
    if run_benchmark "$mode" "$temp_pct" "$run_name"; then
        SUCCESSFUL_TESTS+=("$run_name")
    else
        FAILED_TESTS+=("$run_name: Benchmark failed")
    fi
    
    echo ""
done

# Kill server after all tests
echo ""
echo "================================================"
echo "All tests completed!"
echo "================================================"
kill_server

# Print summary
echo ""
echo "Test Summary:"
echo "=============="
echo "Total tests: ${#TEST_CONFIGS[@]}"
echo "Successful: ${#SUCCESSFUL_TESTS[@]}"
echo "Failed: ${#FAILED_TESTS[@]}"
echo ""

if [ ${#SUCCESSFUL_TESTS[@]} -gt 0 ]; then
    echo "Successful tests:"
    for test in "${SUCCESSFUL_TESTS[@]}"; do
        echo "  ✓ $test"
    done
    echo ""
fi

if [ ${#FAILED_TESTS[@]} -gt 0 ]; then
    echo "Failed tests:"
    for test in "${FAILED_TESTS[@]}"; do
        echo "  ✗ $test"
    done
    echo ""
fi

echo "Results saved to: $OUTPUT_DIR"
echo "Configuration saved to: $CONFIG_FILE"
echo ""

# Automatically generate plots
echo "================================================"
echo "Generating plots..."
echo "================================================"
echo ""

PLOT_SCRIPT="$(dirname "$0")/plot_comprehensive_results.sh"
if [ -f "$PLOT_SCRIPT" ]; then
    if bash "$PLOT_SCRIPT" --input-dir "$OUTPUT_DIR"; then
        echo ""
        echo "✓ Plots generated successfully!"
        echo "  Check $OUTPUT_DIR for plot images"
    else
        echo ""
        echo "⚠ Plot generation had issues. You can try running manually:"
        echo "  ./plot_comprehensive_results.sh --input-dir $OUTPUT_DIR"
    fi
else
    echo "⚠ Plot script not found: $PLOT_SCRIPT"
    echo "  To plot results manually, run:"
    echo "  ./plot_comprehensive_results.sh --input-dir $OUTPUT_DIR"
fi
echo ""

# Final summary with plot information
if [ ${#FAILED_TESTS[@]} -eq 0 ]; then
    echo "================================================"
    echo "All tests and plots completed successfully!"
    echo "================================================"
    echo "Results directory: $OUTPUT_DIR"
    echo ""
    echo "Generated plots:"
    echo "  - mode_578_progression.png"
    echo "  - mode_comparison_100pct.png"
    echo "  - all_modes_comparison.png"
else
    echo "================================================"
    echo "Tests completed with some failures"
    echo "================================================"
    echo "Results directory: $OUTPUT_DIR"
    echo ""
    echo "Review logs in $OUTPUT_DIR before analyzing plots."
fi
echo ""
