#!/bin/bash

# Online QPS Benchmark
# Compares different server configurations across QPS values
# 
# Server configs:
#   - default: no deterministic inference
#   - global: --enable-deterministic-inference 2
#   - llm42_ws64_bs8: --enable-llm-42 3 --llm-42-window-size 64 --llm-42-verify-batch-size 8
#   - llm42_ws32_bs16: --enable-llm-42 3 --llm-42-window-size 32 --llm-42-verify-batch-size 16
#
# For each QPS:
#   - default, global: deterministic-ratio 1.0
#   - llm42: deterministic-ratio [0.02, 0.05, 0.1, 0.2, 0.5, 1.0]

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
DETERMINISTIC_SEED="${DETERMINISTIC_SEED:-142}"

# QPS values per dataset
SHAREGPT_QPS_VALUES=(12 14 16 18)
ARXIV_QPS_VALUES=()

# Deterministic ratios for llm42 configs
LLM42_RATIOS=(0.02 0.05 0.1 0.2 0.5 1.0)

# Server configurations
SERVER_CONFIGS=("default" "global" "llm42_ws64_bs8")

# Output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/results_n${NUM_PROMPTS}_seed${DETERMINISTIC_SEED}_${TIMESTAMP}}"
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
echo "Online QPS Benchmark"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "Datasets: ShareGPT, ArXiv"
echo "Num Prompts: $NUM_PROMPTS"
echo "ShareGPT QPS Values: ${SHAREGPT_QPS_VALUES[*]}"
echo "ArXiv QPS Values: ${ARXIV_QPS_VALUES[*]}"
echo "Server Configs: ${SERVER_CONFIGS[*]}"
echo "LLM42 Ratios: ${LLM42_RATIOS[*]}"
echo "Output Dir: $OUTPUT_DIR"
echo "=============================================="
echo ""

# Function to get server-specific arguments
get_server_args() {
    local config="$1"
    case "$config" in
        "default")
            echo ""
            ;;
        "global")
            echo "--enable-deterministic-inference 2"
            ;;
        "llm42_ws64_bs8")
            echo "--enable-llm-42 3 --llm-42-window-size 64 --llm-42-verify-batch-size 8"
            ;;
        "llm42_ws32_bs16")
            echo "--enable-llm-42 3 --llm-42-window-size 32 --llm-42-verify-batch-size 16"
            ;;
        *)
            echo ""
            ;;
    esac
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
    sleep 3
}

# Function to run benchmark
run_benchmark() {
    local url="$1"
    local qps="$2"
    local dataset="$3"
    local server_config="$4"
    local det_ratio="$5"
    local config_name="${dataset}_qps${qps}_${server_config}_ratio${det_ratio}"
    local temp_result="${OUTPUT_DIR}/temp_${config_name}.jsonl"
    
    echo "[${config_name}] Running benchmark..."
    
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
        --deterministic-ratio "$det_ratio" \
        --deterministic-seed "$DETERMINISTIC_SEED" \
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
                result['server_config'] = '$server_config'
                result['deterministic_ratio'] = $det_ratio
                result['server_url'] = '$url'
                
                # Remove verbose fields
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

# Function to run benchmarks for a single QPS value
run_qps_benchmarks() {
    local dataset="$1"
    local qps="$2"
    
    echo ""
    echo "=========================================="
    echo "Running ${dataset} benchmarks at QPS=${qps}"
    echo "=========================================="
    
    # Launch 4 servers: default, global, and 2x llm42_ws64_bs8
    declare -a SERVER_PIDS=()
    declare -a SERVER_URLS=()
    declare -a SERVER_CONFIGS_RUNNING=()
    
    # GPU 0: default, GPU 1: global, GPU 2: llm42_ws64_bs8, GPU 3: llm42_ws64_bs8
    local configs_to_launch=("default" "global" "llm42_ws64_bs8" "llm42_ws64_bs8")
    
    for ((i=0; i<4 && i<NUM_GPUS; i++)); do
        GPU_ID=$i
        PORT=$((BASE_PORT + i))
        config="${configs_to_launch[$i]}"
        config_args=$(get_server_args "$config")
        SERVER_LOG="${LOG_DIR}/server_${dataset}_qps${qps}_${config}_gpu${i}.log"
        
        echo "Starting server on GPU $GPU_ID, port $PORT (${config})..."
        
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
            $config_args \
            > "$SERVER_LOG" 2>&1 &
        
        SERVER_PIDS+=($!)
        SERVER_URLS+=("http://127.0.0.1:$PORT")
        SERVER_CONFIGS_RUNNING+=("$config")
        
        sleep 2
    done
    
    # Wait for all servers to be ready
    echo "Waiting for servers to be ready..."
    local all_ready=true
    for ((i=0; i<${#SERVER_URLS[@]}; i++)); do
        url="${SERVER_URLS[$i]}"
        config="${SERVER_CONFIGS_RUNNING[$i]}"
        echo -n "  Waiting for ${config} (GPU $i) at $url..."
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
    echo "All servers ready. Running benchmarks..."
    
    # Phase 1: Run ratio=1.0 on default, global, and one llm42 server
    # Use the second llm42 server (GPU 3) to start on ratio 0.5
    echo ""
    echo "=== Phase 1: Running ratio=1.0 on default/global/llm42 + ratio=0.5 on llm42 ==="
    declare -a BENCH_PIDS=()
    
    # default (GPU 0) - ratio 1.0
    config_name="${dataset}_qps${qps}_default_ratio1.0"
    if grep -q "\"config_name\": \"$config_name\"" "$RESULTS_FILE" 2>/dev/null; then
        echo "[${config_name}] Results already exist. Skipping..."
    else
        run_benchmark "${SERVER_URLS[0]}" "$qps" "$dataset" "default" "1.0" &
        BENCH_PIDS+=($!)
    fi
    
    # global (GPU 1) - ratio 1.0
    config_name="${dataset}_qps${qps}_global_ratio1.0"
    if grep -q "\"config_name\": \"$config_name\"" "$RESULTS_FILE" 2>/dev/null; then
        echo "[${config_name}] Results already exist. Skipping..."
    else
        run_benchmark "${SERVER_URLS[1]}" "$qps" "$dataset" "global" "1.0" &
        BENCH_PIDS+=($!)
    fi
    
    # llm42 (GPU 2) - ratio 1.0
    config_name="${dataset}_qps${qps}_llm42_ws64_bs8_ratio1.0"
    if grep -q "\"config_name\": \"$config_name\"" "$RESULTS_FILE" 2>/dev/null; then
        echo "[${config_name}] Results already exist. Skipping..."
    else
        run_benchmark "${SERVER_URLS[2]}" "$qps" "$dataset" "llm42_ws64_bs8" "1.0" &
        BENCH_PIDS+=($!)
    fi
    
    # llm42 (GPU 3) - ratio 0.5
    config_name="${dataset}_qps${qps}_llm42_ws64_bs8_ratio0.5"
    if grep -q "\"config_name\": \"$config_name\"" "$RESULTS_FILE" 2>/dev/null; then
        echo "[${config_name}] Results already exist. Skipping..."
    else
        run_benchmark "${SERVER_URLS[3]}" "$qps" "$dataset" "llm42_ws64_bs8" "0.5" &
        BENCH_PIDS+=($!)
    fi
    
    # Wait for phase 1 to complete
    for pid in "${BENCH_PIDS[@]}"; do
        wait "$pid"
    done
    
    echo ""
    echo "=== Phase 2: Stopping default/global, launching 2 more llm42 servers ==="
    
    # Kill default and global servers (GPU 0 and 1)
    echo "Stopping default and global servers..."
    kill "${SERVER_PIDS[0]}" 2>/dev/null || true
    kill "${SERVER_PIDS[1]}" 2>/dev/null || true
    sleep 3
    
    # Launch additional llm42 servers on GPU 0 and 1
    declare -a LLM42_PIDS=()
    declare -a LLM42_URLS=()
    
    # Keep original llm42 servers (GPU 2 and 3)
    LLM42_PIDS+=("${SERVER_PIDS[2]}")
    LLM42_PIDS+=("${SERVER_PIDS[3]}")
    LLM42_URLS+=("${SERVER_URLS[2]}")
    LLM42_URLS+=("${SERVER_URLS[3]}")
    
    # Launch new llm42 servers on GPU 0 and 1
    for i in 0 1; do
        GPU_ID=$i
        PORT=$((BASE_PORT + i))
        config="llm42_ws64_bs8"
        config_args=$(get_server_args "$config")
        SERVER_LOG="${LOG_DIR}/server_${dataset}_qps${qps}_${config}_gpu${i}_phase2.log"
        
        echo "Starting ${config} on GPU $GPU_ID, port $PORT..."
        
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
            $config_args \
            > "$SERVER_LOG" 2>&1 &
        
        LLM42_PIDS+=($!)
        LLM42_URLS+=("http://127.0.0.1:$PORT")
        
        sleep 2
    done
    
    # Wait for new servers to be ready
    echo "Waiting for new llm42 servers..."
    for i in 2 3; do
        url="${LLM42_URLS[$i]}"
        echo -n "  Waiting for llm42_ws64_bs8 (GPU $((i-2))) at $url..."
        if wait_for_server "$url" 180; then
            echo " ✓"
        else
            echo " ✗ (timeout)"
        fi
    done
    
    echo ""
    echo "=== Phase 3: Running remaining ratios with 4 llm42 servers ==="
    
    # Remaining ratios: 0.02, 0.05, 0.1, 0.2
    # With 4 servers, we can run 4 ratios in parallel per wave
    # Wave 1: 0.02, 0.05, 0.1, 0.2 (all 4 in parallel)
    REMAINING_RATIOS=(0.02 0.05 0.1 0.2)
    
    BENCH_PIDS=()
    for ((i=0; i<${#REMAINING_RATIOS[@]} && i<${#LLM42_URLS[@]}; i++)); do
        ratio="${REMAINING_RATIOS[$i]}"
        url="${LLM42_URLS[$i]}"
        config_name="${dataset}_qps${qps}_llm42_ws64_bs8_ratio${ratio}"
        
        if grep -q "\"config_name\": \"$config_name\"" "$RESULTS_FILE" 2>/dev/null; then
            echo "[${config_name}] Results already exist. Skipping..."
        else
            run_benchmark "$url" "$qps" "$dataset" "llm42_ws64_bs8" "$ratio" &
            BENCH_PIDS+=($!)
        fi
    done
    
    # Wait for all benchmarks to complete
    for pid in "${BENCH_PIDS[@]}"; do
        wait "$pid"
    done
    
    echo ""
    echo "QPS=${qps} benchmarks complete. Stopping servers..."
    kill_servers "${LLM42_PIDS[@]}"
    sleep 5
}

# Run benchmarks for each dataset and QPS
for qps in "${SHAREGPT_QPS_VALUES[@]}"; do
    run_qps_benchmarks "sharegpt" "$qps"
done

for qps in "${ARXIV_QPS_VALUES[@]}"; do
    run_qps_benchmarks "arxiv" "$qps"
done

echo ""
echo "=============================================="
echo "Online QPS Benchmark Complete!"
echo "=============================================="
echo "Results saved to: $RESULTS_FILE"
echo ""
echo "To generate plots:"
echo "  python plot.py --results-file $RESULTS_FILE"
echo "=============================================="
