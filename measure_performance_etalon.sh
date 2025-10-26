#!/bin/bash

# Etalon Performance Measurement Script for SGLang
# This script benchmarks SGLang server using etalon for different deterministic modes
#
# Usage: ./measure_performance_etalon.sh [OPTIONS]
#
# Options:
#   --model MODEL               Model path (default: meta-llama/Meta-Llama-3.1-8B-Instruct)
#   --host HOST                 Server host (default: localhost)
#   --port PORT                 Server port (default: 30000)
#   --output-dir DIR            Output directory (default: etalon_results)
#   --qps QPS                   Queries per second (default: 1.0)
#   --max-requests N            Max completed requests (default: 150)
#   --timeout SECONDS           Timeout in seconds (default: 600)
#   --num-clients N             Number of clients (default: 2)
#   --concurrent N              Concurrent requests per client (default: 5)
#   --trace-file FILE           Trace file for request lengths
#   --max-tokens N              Max tokens (default: 8192)
#   --modes MODES               Comma-separated deterministic modes to test (default: "baseline,1,2,194")
#   --wandb-project PROJECT     Wandb project name (optional)
#   --wandb-group GROUP         Wandb group name (optional)
#   --skip-chrome-check         Skip Chrome/Kaleido dependency check
#   --install-chrome            Install Chrome for plot generation
#   --help                      Show this help message

set -e

# Function to check and install Chrome for kaleido
check_chrome_dependency() {
    echo "Checking Chrome/Kaleido dependencies for plot generation..."
    
    # Try to import kaleido and check if Chrome is available
    if python3 -c "import kaleido; kaleido.get_chrome_sync()" 2>/dev/null; then
        echo "✓ Chrome/Kaleido is properly configured"
        return 0
    else
        echo "⚠ Chrome is not installed or kaleido is not configured"
        echo ""
        echo "Etalon uses kaleido to generate plots, which requires Chrome."
        echo ""
        echo "Options:"
        echo "  1. Install Chrome automatically: $0 --install-chrome"
        echo "  2. Install manually: plotly_get_chrome (or kaleido_get_chrome)"
        echo "  3. Skip plots and continue: Use --skip-chrome-check flag"
        echo ""
        return 1
    fi
}

# Function to install Chrome for kaleido
install_chrome() {
    echo "Installing Chrome for kaleido..."
    if python3 -c "import kaleido; kaleido.get_chrome_sync()"; then
        echo "✓ Chrome installed successfully"
    else
        echo "Failed to install Chrome automatically"
        echo "Please try: plotly_get_chrome or kaleido_get_chrome"
        exit 1
    fi
}

# Default configuration
MODEL="meta-llama/Meta-Llama-3.1-8B-Instruct"
HOST="localhost"
PORT=30000
OUTPUT_DIR="etalon_results"
QPS=1.0
MAX_REQUESTS=150
TIMEOUT=600
NUM_CLIENTS=1
CONCURRENT=150
TRACE_FILE="./etalon/data/processed_traces/arxiv_summarization_filtered_stats_llama2_tokenizer.csv"
MAX_TOKENS=8192
MODES="baseline,1,2,194"
WANDB_PROJECT=""
WANDB_GROUP=""
TTFT_DEADLINE=0.3
TBT_DEADLINE=0.03
SKIP_CHROME_CHECK=false
INSTALL_CHROME=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --host)
            HOST="$2"
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
        --num-clients)
            NUM_CLIENTS="$2"
            shift 2
            ;;
        --concurrent)
            CONCURRENT="$2"
            shift 2
            ;;
        --trace-file)
            TRACE_FILE="$2"
            shift 2
            ;;
        --max-tokens)
            MAX_TOKENS="$2"
            shift 2
            ;;
        --modes)
            MODES="$2"
            shift 2
            ;;
        --wandb-project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --wandb-group)
            WANDB_GROUP="$2"
            shift 2
            ;;
        --ttft-deadline)
            TTFT_DEADLINE="$2"
            shift 2
            ;;
        --tbt-deadline)
            TBT_DEADLINE="$2"
            shift 2
            ;;
        --skip-chrome-check)
            SKIP_CHROME_CHECK=true
            shift
            ;;
        --install-chrome)
            INSTALL_CHROME=true
            shift
            ;;
        --help)
            head -n 30 "$0" | grep "^#" | sed 's/^# //'
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Handle Chrome installation if requested
if [ "$INSTALL_CHROME" = true ]; then
    install_chrome
    exit 0
fi

# Check Chrome dependency unless skipped
if [ "$SKIP_CHROME_CHECK" = false ]; then
    if ! check_chrome_dependency; then
        echo ""
        read -p "Continue anyway? Results will be saved but plots may not be generated. (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Exiting. Please install Chrome or use --skip-chrome-check to bypass."
            exit 1
        fi
        echo "Continuing without Chrome (plots may fail)..."
    fi
fi

# Setup API key and base URL
export OPENAI_API_KEY="EMPTY"
export OPENAI_API_BASE="http://${HOST}:${PORT}/v1"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Convert comma-separated modes to array
IFS=',' read -ra MODE_ARRAY <<< "$MODES"

echo "================================================"
echo "Etalon Performance Measurement for SGLang"
echo "================================================"
echo "Model: $MODEL"
echo "Server: $OPENAI_API_BASE"
echo "Output Directory: $OUTPUT_DIR"
echo "QPS: $QPS"
echo "Max Requests: $MAX_REQUESTS"
echo "Timeout: ${TIMEOUT}s"
echo "Clients: $NUM_CLIENTS"
echo "Concurrent Requests per Client: $CONCURRENT"
echo "Trace File: $TRACE_FILE"
echo "Max Tokens: $MAX_TOKENS"
echo "Modes to Test: ${MODE_ARRAY[*]}"
echo "================================================"
echo ""

# Function to run benchmark
run_benchmark() {
    local mode=$1
    local mode_name=$2
    local run_output_dir="${OUTPUT_DIR}/${mode_name}"
    
    # Create output directory before running
    mkdir -p "$run_output_dir"
    
    echo "================================================"
    echo "Running benchmark: $mode_name"
    echo "Output: $run_output_dir"
    echo "================================================"
    
    # Build wandb arguments if provided, otherwise disable wandb
    WANDB_ARGS=""
    if [ -n "$WANDB_PROJECT" ]; then
        WANDB_ARGS="--metrics_config_wandb_project $WANDB_PROJECT"
        
        if [ -n "$WANDB_GROUP" ]; then
            WANDB_ARGS="$WANDB_ARGS --metrics_config_wandb_group $WANDB_GROUP"
        fi
        
        WANDB_ARGS="$WANDB_ARGS --metrics_config_wandb_run_name ${mode_name}"
    else
        # Disable wandb if no project specified
        export WANDB_MODE=disabled
    fi
    
    # Run etalon benchmark with metrics writing always enabled
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
        --deadline_config_ttft_deadline $TTFT_DEADLINE \
        --deadline_config_tbt_deadline $TBT_DEADLINE \
        $WANDB_ARGS 2>&1 | tee "${run_output_dir}/benchmark.log"; then
        
        echo ""
        echo "✓ Benchmark completed successfully: $mode_name"
        echo "Results saved to: $run_output_dir"
        echo ""
        
        # Unset WANDB_MODE for next run
        unset WANDB_MODE
        return 0
    else
        echo ""
        echo "⚠ Benchmark completed with errors: $mode_name"
        echo "Check log: ${run_output_dir}/benchmark.log"
        echo ""
        
        # If Chrome error, provide helpful message
        if grep -q "ChromeNotFoundError\|Kaleido requires" "${run_output_dir}/benchmark.log" 2>/dev/null; then
            echo "Note: Plot generation failed due to missing Chrome."
            echo "      Metrics data may still be available in JSON format."
            echo "      To fix: Run './measure_performance_etalon.sh --install-chrome'"
        fi
        
        # If WandB error, provide helpful message
        if grep -q "wandb.errors\|api_key not configured" "${run_output_dir}/benchmark.log" 2>/dev/null; then
            echo "Note: WandB error detected."
            echo "      This should be disabled when no --wandb-project is specified."
            echo "      Metrics should still be saved locally."
        fi
        
        # Unset WANDB_MODE for next run
        unset WANDB_MODE
        return 1
    fi
}

# Check if server is running
echo "Checking if SGLang server is running at $OPENAI_API_BASE..."
if ! curl -s -f "${OPENAI_API_BASE}/models" > /dev/null 2>&1; then
    echo "Error: Cannot connect to SGLang server at $OPENAI_API_BASE"
    echo "Please start the server first using:"
    echo "  ./launch_server.sh deterministic 1"
    echo "  or"
    echo "  python3 -m sglang.launch_server --model-path $MODEL --port $PORT ..."
    exit 1
fi
echo "Server is running!"
echo ""

# Run benchmarks for each mode
for mode in "${MODE_ARRAY[@]}"; do
    mode=$(echo "$mode" | xargs)  # Trim whitespace
    
    if [ "$mode" = "baseline" ] || [ "$mode" = "non-deterministic" ] || [ "$mode" = "0" ]; then
        # Baseline: non-deterministic mode
        echo "Note: Testing baseline (non-deterministic) mode"
        echo "      Make sure server is running in non-deterministic mode:"
        echo "      ./launch_server.sh non-deterministic"
        echo ""
        read -p "Press Enter to continue or Ctrl+C to abort..."
        run_benchmark "baseline" "baseline_nondet"
    else
        # Deterministic mode
        echo "Note: Testing deterministic mode $mode"
        echo "      Make sure server is running with mode $mode:"
        echo "      ./launch_server.sh det-$mode"
        echo ""
        read -p "Press Enter to continue or Ctrl+C to abort..."
        run_benchmark "$mode" "det_mode_${mode}"
    fi
done

echo "================================================"
echo "All benchmarks completed!"
echo "================================================"
echo "Results directory: $OUTPUT_DIR"
echo ""
echo "Summary of results:"
for mode in "${MODE_ARRAY[@]}"; do
    mode=$(echo "$mode" | xargs)
    if [ "$mode" = "baseline" ] || [ "$mode" = "non-deterministic" ] || [ "$mode" = "0" ]; then
        mode_name="baseline_nondet"
    else
        mode_name="det_mode_${mode}"
    fi
    
    result_file="${OUTPUT_DIR}/${mode_name}/summary.json"
    if [ -f "$result_file" ]; then
        echo "  - ${mode_name}: $result_file"
    fi
done
echo ""
echo "To compare results, check the individual summary.json files"
echo "or view them in WandB if configured."
