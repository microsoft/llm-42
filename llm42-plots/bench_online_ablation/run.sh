#!/bin/bash

# Matrix Ablation Experiment for LLM42 Window Size vs Batch Size
# Runs a 6x6 grid of configurations (36 total) in batches of 4 configs per batch.
# Uses ShareGPT dataset with 100% deterministic ratio.
#
# Window sizes: 16, 32, 64, 128, 256, 512
# Batch sizes: 1, 2, 4, 8, 16, 32
#
# Metrics collected:
#   - P99 E2E latency (ms)
#   - Recompute ratio (total_tokens_rolled_back / total_output_tokens)

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
DETERMINISTIC_SEED="${DETERMINISTIC_SEED:-42}"
REQUEST_RATE="${REQUEST_RATE:-12}"

# Output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results_n${NUM_PROMPTS}_qps_${REQUEST_RATE}_seed${DETERMINISTIC_SEED}_${TIMESTAMP}}"
LOG_DIR="${OUTPUT_DIR}/server_logs"
RESULTS_FILE="${OUTPUT_DIR}/benchmark_results.jsonl"

# Window sizes and batch sizes for the matrix
WINDOW_SIZES=(16 32 64 128 256 512)
BATCH_SIZES=(1 2 4 8 16 32)

# Generate all config names (only where window_size * batch_size <= 512)
ALL_CONFIGS=()
for ws in "${WINDOW_SIZES[@]}"; do
    for bs in "${BATCH_SIZES[@]}"; do
        if [ $((ws * bs)) -le 512 ]; then
            ALL_CONFIGS+=("llm42_ws_${ws}_bs_${bs}")
        fi
    done
done

TOTAL_CONFIGS=${#ALL_CONFIGS[@]}
BATCH_SIZE=$NUM_GPUS
NUM_BATCHES=$(( (TOTAL_CONFIGS + BATCH_SIZE - 1) / BATCH_SIZE ))

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
echo "Matrix Ablation Experiment: Window Size x Batch Size"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Dataset: ShareGPT (deterministic_ratio=1.0)"
echo "Num Prompts: $NUM_PROMPTS"
echo "Total Configs: $TOTAL_CONFIGS"
echo "Batch Size: $BATCH_SIZE (using $NUM_GPUS GPUs)"
echo "Num Batches: $NUM_BATCHES"
echo "Output Dir: $OUTPUT_DIR"
echo ""
echo "Window Sizes: ${WINDOW_SIZES[*]}"
echo "Batch Sizes: ${BATCH_SIZES[*]}"
echo "=============================================="
echo ""

# Function to get config-specific arguments (same as in launch_servers_parallel.sh)
get_config_args() {
    local config_name="$1"
    # Dynamic parsing: llm42_ws_<window_size>_bs_<batch_size>
    local ws=$(echo "$config_name" | sed -E 's/llm42_ws_([0-9]+)_bs_([0-9]+)/\1/')
    local bs=$(echo "$config_name" | sed -E 's/llm42_ws_([0-9]+)_bs_([0-9]+)/\2/')
    echo "--llm-42-window-size $ws --enable-llm-42 3 --llm-42-verify-batch-size $bs"
}

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
    local config_name="$2"
    local temp_result="${OUTPUT_DIR}/temp_${config_name}.jsonl"
    
    echo "[${config_name}] Running benchmark..."
    
    $PYTHON_CMD -m sglang.bench_serving \
        --backend sglang \
        --base-url "$url" \
        --model "$MODEL_PATH" \
        --dataset-name sharegpt \
        --sharegpt-context-len "$SHAREGPT_CONTEXT_LEN" \
        --num-prompts "$NUM_PROMPTS" \
        --request-rate "$REQUEST_RATE" \
        --deterministic-ratio 1.0 \
        --deterministic-seed "$DETERMINISTIC_SEED" \
        --extra-request-body '{"ignore_eos": true, "temperature": 0}' \
        --output-file "$temp_result" \
        --output-details \
        2>&1 | tee "${OUTPUT_DIR}/log_${config_name}.log"
    
    # Extract metrics and append to results
    if [ -f "$temp_result" ]; then
        # Parse ws and bs from config name
        local ws=$(echo "$config_name" | sed -E 's/llm42_ws_([0-9]+)_bs_([0-9]+)/\1/')
        local bs=$(echo "$config_name" | sed -E 's/llm42_ws_([0-9]+)_bs_([0-9]+)/\2/')
        
        $PYTHON_CMD -c "
import json

with open('$temp_result', 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                result = json.loads(line)
                result['config_name'] = '$config_name'
                result['window_size'] = $ws
                result['batch_size'] = $bs
                result['dataset_name'] = 'sharegpt'
                result['deterministic_ratio'] = 1.0
                result['server_url'] = '$url'
                
                # Extract rollback stats from meta_info
                meta_info_list = result.get('meta_info', [])
                output_lens = result.get('output_lens', [])
                if meta_info_list:
                    det_num_rollbacks = [m.get('llm_42_num_rollbacks', 0) for m in meta_info_list if m]
                    det_tokens_rolled_back = [m.get('llm_42_tokens_rolled_back', 0) for m in meta_info_list if m]
                    det_num_verification_windows = [m.get('llm_42_num_verification_windows', 0) for m in meta_info_list if m]
                    
                    num_requests = len(det_num_rollbacks)
                    total_output_tokens = sum(output_lens) if output_lens else result.get('total_output_tokens', 0)
                    if num_requests > 0:
                        result['rollback_stats'] = {
                            'total_rollbacks': sum(det_num_rollbacks),
                            'total_tokens_rolled_back': sum(det_tokens_rolled_back),
                            'total_verification_windows': sum(det_num_verification_windows),
                            'total_output_tokens': total_output_tokens,
                            'avg_rollbacks_per_request': sum(det_num_rollbacks) / num_requests,
                            'avg_tokens_rolled_back_per_request': sum(det_tokens_rolled_back) / num_requests,
                            'max_rollbacks_per_request': max(det_num_rollbacks) if det_num_rollbacks else 0,
                            'max_tokens_rolled_back_per_request': max(det_tokens_rolled_back) if det_tokens_rolled_back else 0,
                            'requests_with_rollbacks': sum(1 for x in det_num_rollbacks if x > 0),
                            'num_requests': num_requests,
                        }
                        # Save per-request rollback data for CDF plots
                        result['per_request_rollbacks'] = det_num_rollbacks
                        result['per_request_tokens_rolled_back'] = det_tokens_rolled_back
                        result['per_request_verification_windows'] = det_num_verification_windows
                
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

# Main loop: process configs in batches
for ((batch_idx=0; batch_idx<NUM_BATCHES; batch_idx++)); do
    start_idx=$((batch_idx * BATCH_SIZE))
    end_idx=$((start_idx + BATCH_SIZE))
    if [ $end_idx -gt $TOTAL_CONFIGS ]; then
        end_idx=$TOTAL_CONFIGS
    fi
    
    # Get configs for this batch
    BATCH_CONFIGS=("${ALL_CONFIGS[@]:$start_idx:$((end_idx - start_idx))}")
    NUM_IN_BATCH=${#BATCH_CONFIGS[@]}
    
    echo ""
    echo "=========================================="
    echo "Batch $((batch_idx + 1)) / $NUM_BATCHES"
    echo "Configs: ${BATCH_CONFIGS[*]}"
    echo "=========================================="
    
    # Check if all configs in this batch already have results (for resume capability)
    ALL_DONE=true
    for config in "${BATCH_CONFIGS[@]}"; do
        if ! grep -q "\"config_name\": \"$config\"" "$RESULTS_FILE" 2>/dev/null; then
            ALL_DONE=false
            break
        fi
    done
    
    if [ "$ALL_DONE" = true ]; then
        echo "All configs in this batch already have results. Skipping..."
        continue
    fi
    
    # Launch servers for this batch
    declare -a SERVER_PIDS=()
    declare -a SERVER_URLS=()
    
    for ((i=0; i<NUM_IN_BATCH; i++)); do
        GPU_ID=$i
        PORT=$((BASE_PORT + i))
        CONFIG_NAME="${BATCH_CONFIGS[$i]}"
        CONFIG_ARGS=$(get_config_args "$CONFIG_NAME")
        SERVER_LOG="${LOG_DIR}/server_${CONFIG_NAME}.log"
        
        echo "Starting server on GPU $GPU_ID, port $PORT for $CONFIG_NAME..."
        
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
            $CONFIG_ARGS \
            > "$SERVER_LOG" 2>&1 &
        
        SERVER_PIDS+=($!)
        SERVER_URLS+=("http://127.0.0.1:$PORT")
        
        sleep 2
    done
    
    # Wait for all servers to be ready
    echo "Waiting for servers to be ready..."
    ALL_READY=true
    for ((i=0; i<NUM_IN_BATCH; i++)); do
        url="${SERVER_URLS[$i]}"
        config="${BATCH_CONFIGS[$i]}"
        echo -n "  Waiting for $config at $url..."
        if wait_for_server "$url" 180; then
            echo " ✓"
        else
            echo " ✗ (timeout)"
            ALL_READY=false
        fi
    done
    
    if [ "$ALL_READY" = false ]; then
        echo "ERROR: Some servers failed to start. Cleaning up..."
        kill_servers "${SERVER_PIDS[@]}"
        exit 1
    fi
    
    echo ""
    echo "All servers ready. Running benchmarks in parallel..."
    
    # Run benchmarks in parallel
    declare -a BENCH_PIDS=()
    for ((i=0; i<NUM_IN_BATCH; i++)); do
        url="${SERVER_URLS[$i]}"
        config="${BATCH_CONFIGS[$i]}"
        
        # Skip if results already exist
        if grep -q "\"config_name\": \"$config\"" "$RESULTS_FILE" 2>/dev/null; then
            echo "[$config] Results already exist. Skipping..."
            continue
        fi
        
        run_benchmark "$url" "$config" &
        BENCH_PIDS+=($!)
    done
    
    # Wait for benchmarks to complete
    for pid in "${BENCH_PIDS[@]}"; do
        wait "$pid" || true
    done
    
    echo ""
    echo "Batch $((batch_idx + 1)) complete. Stopping servers..."
    
    # Kill servers
    kill_servers "${SERVER_PIDS[@]}"
    
    # Extra cleanup time between batches
    sleep 5
done

echo ""
echo "=============================================="
echo "Matrix Ablation Complete!"
echo "=============================================="
echo "Results saved to: $RESULTS_FILE"
echo ""
echo "To generate heatmap plots:"
echo "  python plot_matrix_heatmap.py --results-file $RESULTS_FILE"
echo "=============================================="
