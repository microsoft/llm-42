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
MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
HOST="0.0.0.0"
PORT=30000
TP_SIZE=1
ATTENTION_BACKEND="flashinfer"
OUTPUT_DIR=""  # Will be set based on model, dataset, and percentages
QPS=1
MAX_REQUESTS=512
TIMEOUT=600
NUM_CLIENTS=1
CONCURRENT=512
TRACE_TYPE="arxiv"  # Default trace type: arxiv, sharegpt, or lmsys
TRACE_FILE=""  # Will be set based on TRACE_TYPE or can be overridden
TRACE_NAME=""  # Will be extracted from TRACE_FILE or set via command line
MAX_TOKENS=8192
WARMUP_TIME=30  # Time to wait for server to warmup

# Mixed temperature configuration
TEMP0_PCTS="10"  # Default: 10% of requests have temperature=0 (comma-separated list)
ASSIGNMENT_MODE="random"  # Default: random assignment with seed
SEED=42  # Default seed for random assignment

# Modes to test
# Mode 66 = batch-invariant:: vllm-rmsnorm + cutlass matmul
# Mode 257 = batch-invariant:: native-rmsnorm + thinking-machine matmul
# Mode 578 = temperature-based switching + cutlass matmul + vllm-rmsnorm
MODES=("baseline" "257" "578")
MODE_DESCRIPTIONS=(
    "Baseline (Non-deterministic)"
    #"Mode 66 (batch-invariant: vllm-rmsnorm + cutlass)"
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
        --trace-type)
            TRACE_TYPE="$2"
            shift 2
            ;;
        --trace-file)
            TRACE_FILE="$2"
            shift 2
            ;;
        --trace-name)
            TRACE_NAME="$2"
            shift 2
            ;;
        --tp)
            TP_SIZE="$2"
            shift 2
            ;;
        --temp0-pcts)
            TEMP0_PCTS="$2"
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
            echo "  --trace-type TYPE         Predefined trace type: 'arxiv', 'sharegpt', or 'lmsys'"
            echo "                            (default: $TRACE_TYPE)"
            echo "  --trace-file FILE         Custom trace file path (overrides --trace-type)"
            echo "  --trace-name NAME         Name of dataset/trace (default: extracted from trace file)"
            echo "  --temp0-pcts PCTS         Comma-separated percentages for mode 578 (default: $TEMP0_PCTS)"
            echo "                            Example: '0,1,5,10'"
            echo "                            Note: Baseline/66/257 always run at 100% (once)"
            echo "                                  Mode 578 runs at each specified percentage"
            echo "  --assignment-mode MODE    'random' or 'fixed' (default: $ASSIGNMENT_MODE)"
            echo "  --seed SEED               Random seed for 'random' mode (default: $SEED)"
            echo "  --qps QPS                 Queries per second (default: $QPS)"
            echo "  --max-requests N          Max requests (default: $MAX_REQUESTS)"
            echo "  --timeout SECONDS         Timeout (default: $TIMEOUT)"
            echo "  --warmup-time SECONDS     Server warmup time (default: $WARMUP_TIME)"
            echo "  --help                    Show this help"
            echo ""
            echo "Examples:"
            echo "  # Test with multiple percentages (runs 3 + 4 = 7 tests):"
            echo "  $0 --trace-type arxiv --temp0-pcts 0,1,5,10"
            echo "    → baseline/66/257 @ 100%, mode 578 @ 0/1/5/10"
            echo ""
            echo "  # Include 100% for mode 578 comparison (runs 3 + 5 = 8 tests):"
            echo "  $0 --trace-type sharegpt --temp0-pcts 0,1,5,10,100"
            echo "    → baseline/66/257 @ 100%, mode 578 @ 0/1/5/10/100"
            echo ""
            echo "  # Only 100% comparison (runs 4 tests):"
            echo "  $0 --trace-type lmsys --temp0-pcts 100"
            echo "    → all modes @ 100%"
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

# Set trace file based on trace type (if not explicitly provided)
if [ -z "$TRACE_FILE" ]; then
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
fi

# Verify trace file exists
if [ ! -f "$TRACE_FILE" ]; then
    echo "Error: Trace file not found: $TRACE_FILE"
    echo "Please ensure the trace file exists or choose a different trace type."
    exit 1
fi

# Extract trace name if not provided
if [ -z "$TRACE_NAME" ]; then
    # Extract filename without path and extension
    TRACE_NAME=$(basename "$TRACE_FILE" .csv)
    # Remove common suffixes like _filtered_stats_llama2_tokenizer
    TRACE_NAME=$(echo "$TRACE_NAME" | sed 's/_filtered_stats_llama2_tokenizer//' | sed 's/_filtered//')
fi

# Extract model name from path (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct -> Meta-Llama-3.1-8B-Instruct)
MODEL_NAME=$(basename "$MODEL")
# Also create a short name for directory naming (e.g., llama-3.1-8b)
MODEL_SHORT=$(echo "$MODEL" | sed 's/.*\///' | tr '[:upper:]' '[:lower:]' | sed 's/meta-llama-/llama-/' | sed 's/-instruct//')

# Parse percentages into array
IFS=',' read -ra PCT_ARRAY <<< "$TEMP0_PCTS"

# Create directory name with percentages
PCT_STR=$(echo "$TEMP0_PCTS" | tr ',' '_')
OUTPUT_DIR="${MODEL_SHORT}_${TRACE_NAME}_pct_${PCT_STR}"

echo "================================================"
echo "Mixed Temperature Deterministic Testing"
echo "================================================"
echo "Model: $MODEL"
echo "Model Name: $MODEL_NAME"
echo "Port: $PORT"
echo "Trace Type: $TRACE_TYPE"
echo "Trace File: $TRACE_FILE"
echo "Trace Name: $TRACE_NAME"
echo "Temperature Configuration:"
echo "  - Mode 578 percentages: ${TEMP0_PCTS}%"
echo "  - Baseline/66/257: Always 100% (all deterministic)"
echo "  - Assignment mode: ${ASSIGNMENT_MODE}"
if [ "$ASSIGNMENT_MODE" = "random" ]; then
    echo "  - Random seed: $SEED"
fi
echo "Output Directory: $OUTPUT_DIR"
echo ""
echo "Test Strategy:"
echo "  1. Baseline (Non-deterministic) - 100% deterministic requests"
echo "  2. Mode 66 (batch-invariant: vllm+cutlass) - 100% deterministic requests"
echo "  3. Mode 257 (batch-invariant: native+TM) - 100% deterministic requests"
echo "  4. Mode 578 (temp-based) - ${#PCT_ARRAY[@]} runs with percentages: ${TEMP0_PCTS}%"
echo ""
echo "Total tests: 3 + ${#PCT_ARRAY[@]} = $((3 + ${#PCT_ARRAY[@]}))"
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
    local temp0_pct=$3
    local run_output_dir="${OUTPUT_DIR}/pct_${temp0_pct}/${mode_name}"
    
    echo ""
    echo "================================================"
    echo "Running benchmark: $mode_name (${temp0_pct}% temp=0)"
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
    echo "  Temperature 0 percentage: ${temp0_pct}%"
    echo "  Assignment mode: ${ASSIGNMENT_MODE}"
    if [ "$ASSIGNMENT_MODE" = "random" ]; then
        echo "  Random seed: ${SEED}"
    fi
    
    local wrapper_args=(
        --temp0-pct "$temp0_pct"
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
        
        echo "✓ Benchmark completed: $mode_name (${temp0_pct}%)"
        unset OPENAI_API_KEY
        unset OPENAI_API_BASE
        unset WANDB_MODE
        return 0
    else
        echo "⚠ Benchmark had issues: $mode_name (${temp0_pct}%)"
        unset OPENAI_API_KEY
        unset OPENAI_API_BASE
        unset WANDB_MODE
        return 1
    fi
}

# Main test loop
echo "Starting mixed temperature testing..."
echo ""

# Create main output directory
mkdir -p "$OUTPUT_DIR"

# Save configuration for reference
cat > "$OUTPUT_DIR/test_config.txt" << EOF
Test Configuration
==================
Model: $MODEL
Model Name: $MODEL_NAME
Model Short: $MODEL_SHORT
Trace Type: $TRACE_TYPE
Trace File: $TRACE_FILE
Trace Name: $TRACE_NAME
Percentages for Mode 578: ${TEMP0_PCTS}
Assignment Mode: $ASSIGNMENT_MODE
Seed: $SEED
Max Requests: $MAX_REQUESTS
QPS: $QPS
Timeout: $TIMEOUT
Warmup Time: $WARMUP_TIME

Test Strategy:
- Baseline, Mode 66, Mode 257: Run once with 100% deterministic (pct_100/)
- Mode 578 (temperature-based): Run with varying percentages (${TEMP0_PCTS})

Date: $(date)
EOF

test_counter=0
# Baseline, 66, 257 run once (at 100%), mode 578 runs for each percentage
total_tests=$((3 + ${#PCT_ARRAY[@]}))

echo ""
echo "###############################################"
echo "# Running baseline, mode 66, and mode 257 with 100% deterministic"
echo "###############################################"
echo ""

# Run baseline, mode 66, mode 257 once with 100% (all deterministic)
FIXED_MODES=("baseline" "66" "257")
FIXED_MODE_DESCRIPTIONS=(
    "Baseline (Non-deterministic)"
    "Mode 66 (batch-invariant: vllm-rmsnorm + cutlass)"
    "Mode 257 (batch-invariant: native-rmsnorm + TM)"
)

for i in "${!FIXED_MODES[@]}"; do
    mode="${FIXED_MODES[$i]}"
    mode_desc="${FIXED_MODE_DESCRIPTIONS[$i]}"
    test_counter=$((test_counter + 1))
    
    # Generate mode name based on mode number
    if [ "$mode" = "baseline" ]; then
        mode_name="baseline_nondet"
    else
        mode_name="det_mode_${mode}"
    fi
    
    echo ""
    echo "###############################################"
    echo "# Test ${test_counter}/${total_tests}: $mode_name @ 100% (all deterministic)"
    echo "###############################################"
    
    # Launch server
    if ! launch_server "$mode" "$mode_name" "$mode_desc"; then
        echo "Failed to launch server for $mode_name"
        continue
    fi
    
    # Run benchmark with 100% (all requests deterministic)
    run_benchmark "$mode" "$mode_name" 100
    
    echo ""
done

echo ""
echo "###############################################"
echo "# Running mode 578 (temperature-based) with varying percentages"
echo "###############################################"
echo ""

# Loop through each percentage for mode 578 only
for pct in "${PCT_ARRAY[@]}"; do
    mode="578"
    mode_desc="Mode 578 (temp-based with mixed temperatures)"
    mode_name="det_mode_578"
    test_counter=$((test_counter + 1))
    
    echo ""
    echo "###############################################"
    echo "# Test ${test_counter}/${total_tests}: $mode_name @ ${pct}% temperature=0"
    echo "###############################################"
    
    # Launch server
    if ! launch_server "$mode" "$mode_name" "$mode_desc"; then
        echo "Failed to launch server for $mode_name"
        continue
    fi
    
    # Run benchmark with specified percentage
    run_benchmark "$mode" "$mode_name" "$pct"
    
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
echo "Directory structure:"
echo "  $OUTPUT_DIR/pct_100/"
echo "    ├── baseline_nondet/     (100% deterministic)"
echo "    ├── det_mode_66/         (100% deterministic)"
echo "    └── det_mode_257/        (100% deterministic)"
for pct in "${PCT_ARRAY[@]}"; do
    echo "  $OUTPUT_DIR/pct_${pct}/"
    echo "    └── det_mode_578/      (${pct}% temperature=0)"
done
echo ""

echo "Note: Baseline, mode 66, and mode 257 always run with 100% deterministic."
echo "      Mode 578 (temperature-based) runs with varying percentages: ${TEMP0_PCTS}"
echo ""

echo "To plot results, run:"
echo "  ./plot_results.sh --input-dir $OUTPUT_DIR"
echo ""