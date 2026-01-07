#!/bin/bash

# QPS Comparison Benchmark (Non-Deterministic)
# Runs servers in parallel and tests different QPS values
# Datasets: ShareGPT and ArXiv
#
# Metrics collected:
#   - CDF of TTFT (Time to First Token)
#   - CDF of E2E latency

set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Configuration
NUM_GPUS="${NUM_GPUS:-4}"
MODEL_PATH="${SGLANG_TEST_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
HOST="${SGLANG_HOST:-0.0.0.0}"
BASE_PORT="${SGLANG_BASE_PORT:-30005}"
TP_SIZE="${SGLANG_TP_SIZE:-1}"
ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-fa3}"

# Benchmark parameters
NUM_PROMPTS="${NUM_PROMPTS:-4096}"
SHAREGPT_CONTEXT_LEN="${SHAREGPT_CONTEXT_LEN:-16384}"

# QPS values to test per dataset
SHAREGPT_QPS_VALUES=(10 12 14 16)
ARXIV_QPS_VALUES=(0.6 0.8 1.0 1.2)

# Output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results_n${NUM_PROMPTS}_${TIMESTAMP}}"
LOG_DIR="${OUTPUT_DIR}/server_logs"
RESULTS_FILE="${OUTPUT_DIR}/benchmark_results.jsonl"

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p "$LOG_DIR"

# Determine Python command
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: Python not found"
    exit 1
fi

echo "=============================================="
echo "QPS Comparison Benchmark (Non-Deterministic)"
echo "==============================================" 
echo "Model: $MODEL_PATH"
echo "Datasets: ShareGPT, ArXiv"
echo "Num Prompts: $NUM_PROMPTS"
echo "ShareGPT QPS Values: ${SHAREGPT_QPS_VALUES[*]}"
echo "ArXiv QPS Values: ${ARXIV_QPS_VALUES[*]}"
echo "=============================================="
echo ""

# Function to wait for server to be ready
wait_for_server() {
    local url="$1"
    local max_attempts="${2:-120}"
    local attempt=0
    
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "${url}/v1/models" 2>&1 | grep -q '"object":"list"'; then
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    return 1
}

# Function to kill servers
kill_servers() {
    local pids=("$@")
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    # Wait a bit for cleanup
    sleep 3
}

# Function to run benchmark for a single server
run_benchmark() {
    local url="$1"
    local qps="$2"
    local dataset="$3"
    local config_name="${dataset}_qps_${qps}"
    local temp_result="${OUTPUT_DIR}/temp_${config_name}.jsonl"
    
    echo "[${config_name}] Running benchmark with QPS=${qps} on ${dataset}..."
    
    # Build dataset-specific arguments
    local dataset_args=""
    if [ "$dataset" = "sharegpt" ]; then
        dataset_args="--dataset-name sharegpt --sharegpt-context-len $SHAREGPT_CONTEXT_LEN"
    elif [ "$dataset" = "arxiv" ]; then
        dataset_args="--dataset-name arxiv --sharegpt-context-len $SHAREGPT_CONTEXT_LEN"
    fi
    
    $PYTHON_CMD -m sglang.bench_serving \
        --backend sglang \
        --base-url "$url" \
        --model "$MODEL_PATH" \
        $dataset_args \
        --num-prompts "$NUM_PROMPTS" \
        --request-rate "$qps" \
        --deterministic-ratio 1.0 \
        --deterministic-seed 42 \
        --extra-request-body '{"ignore_eos": true, "temperature": 0}' \
        --output-file "$temp_result" \
        --output-details \
        2>&1 | tee "${OUTPUT_DIR}/log_${config_name}.log"
    
    # Extract metrics and append to results
    if [ -f "$temp_result" ]; then
        $PYTHON_CMD -c "
import json

with open('$temp_result', 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                result = json.loads(line)
                result['config_name'] = '$config_name'
                result['qps'] = $qps
                result['dataset_name'] = '$dataset'
                result['server_url'] = '$url'
                
                # Remove verbose fields to keep results file manageable
                for key in ['meta_info', 'generated_texts', 'output_ids', 'itls', 'errors']:
                    result.pop(key, None)
                
                print(json.dumps(result))
            except json.JSONDecodeError:
                pass
" >> "$RESULTS_FILE"
        rm -f "$temp_result"
    fi
    
    echo "[${config_name}] Completed"
}

# Function to run a batch of benchmarks for a dataset
run_dataset_benchmarks() {
    local dataset="$1"
    shift
    local qps_values=("$@")
    local num_qps=${#qps_values[@]}
    
    echo ""
    echo "=========================================="
    echo "Running ${dataset} benchmarks"
    echo "QPS values: ${qps_values[*]}"
    echo "=========================================="
    
    # Check if all QPS values for this dataset already have results
    local all_done=true
    for qps in "${qps_values[@]}"; do
        config_name="${dataset}_qps_${qps}"
        if ! grep -q "\"config_name\": \"$config_name\"" "$RESULTS_FILE" 2>/dev/null; then
            all_done=false
            break
        fi
    done
    
    if [ "$all_done" = true ]; then
        echo "All ${dataset} configurations already have results. Skipping..."
        return 0
    fi
    
    # Launch servers (one per GPU, non-deterministic)
    declare -a SERVER_PIDS=()
    declare -a SERVER_URLS=()
    
    for ((i=0; i<num_qps && i<NUM_GPUS; i++)); do
        GPU_ID=$i
        PORT=$((BASE_PORT + i))
        SERVER_LOG="${LOG_DIR}/server_${dataset}_gpu${GPU_ID}.log"
        
        echo "Starting server on GPU $GPU_ID, port $PORT..."
        
        CUDA_VISIBLE_DEVICES=$GPU_ID $PYTHON_CMD -m sglang.launch_server \
            --model-path "$MODEL_PATH" \
            --host "$HOST" \
            --port "$PORT" \
            --tp "$TP_SIZE" \
            --attention-backend "$ATTENTION_BACKEND" \
            --disable-radix-cache \
            --disable-chunked-prefix-cache \
            --disable-overlap-schedule \
            --enable-metrics \
            --random-seed 42 \
            --chunked-prefill-size -1 \
            > "$SERVER_LOG" 2>&1 &
        
        SERVER_PIDS+=($!)
        SERVER_URLS+=("http://127.0.0.1:$PORT")
        
        sleep 2
    done
    
    # Wait for all servers to be ready
    echo "Waiting for servers to be ready..."
    local all_ready=true
    for ((i=0; i<${#SERVER_URLS[@]}; i++)); do
        url="${SERVER_URLS[$i]}"
        echo -n "  Waiting for server at $url..."
        if wait_for_server "$url" 180; then
            echo " ✓"
        else
            echo " ✗ (timeout)"
            all_ready=false
        fi
    done
    
    if [ "$all_ready" = false ]; then
        echo "ERROR: Some servers failed to start. Cleaning up..."
        kill_servers "${SERVER_PIDS[@]}"
        return 1
    fi
    
    echo ""
    echo "All servers ready. Running ${dataset} benchmarks in parallel..."
    
    # Run benchmarks in parallel (each QPS on a different server)
    declare -a BENCH_PIDS=()
    for ((i=0; i<num_qps && i<${#SERVER_URLS[@]}; i++)); do
        url="${SERVER_URLS[$i]}"
        qps="${qps_values[$i]}"
        config_name="${dataset}_qps_${qps}"
        
        # Skip if results already exist
        if grep -q "\"config_name\": \"$config_name\"" "$RESULTS_FILE" 2>/dev/null; then
            echo "[${config_name}] Results already exist. Skipping..."
            continue
        fi
        
        run_benchmark "$url" "$qps" "$dataset" &
        BENCH_PIDS+=($!)
    done
    
    # Wait for benchmarks to complete
    for pid in "${BENCH_PIDS[@]}"; do
        wait "$pid" || true
    done
    
    echo ""
    echo "${dataset} benchmarks complete. Stopping servers..."
    
    # Kill servers
    kill_servers "${SERVER_PIDS[@]}"
    
    # Extra cleanup time between batches
    sleep 5
}

# Run ShareGPT benchmarks
run_dataset_benchmarks "sharegpt" "${SHAREGPT_QPS_VALUES[@]}"

# Run ArXiv benchmarks
run_dataset_benchmarks "arxiv" "${ARXIV_QPS_VALUES[@]}"

echo ""
echo "=============================================="
echo "QPS Benchmark Complete!"
echo "=============================================="
echo "Results saved to: $RESULTS_FILE"
echo ""
echo "To generate CDF plots:"
echo "  python plot.py --results-file $RESULTS_FILE"
echo "=============================================="
