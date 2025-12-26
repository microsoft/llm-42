#!/bin/bash
# Online Serving Benchmarks - Measures TTFT, TPOT, E2E latency
# Usage: NUM_GPUS=4 ./run_online_benchmarks.sh --start-servers

set -e

# ============================================
# Configuration
# ============================================
MODEL="${SGLANG_TEST_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
HOST="${SGLANG_HOST:-0.0.0.0}"
ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-flashinfer}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
NUM_GPUS="${NUM_GPUS:-8}"
PORT_BASE=30006

OUTPUT_DIR="$(dirname "$0")/final_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="${OUTPUT_DIR}/results_${TIMESTAMP}.jsonl"
ROLLBACK_FILE="${OUTPUT_DIR}/rollback_metrics_${TIMESTAMP}.jsonl"
DATASET_FILE="${OUTPUT_DIR}/arxiv_dataset_${NUM_PROMPTS}.json"

# Benchmark parameters
SHAREGPT_RATES=(5 5.5 6 6.5 7)
ARXIV_RATES=(0.8 0.9 1 1.1 1.2)
DET_RATIOS=(1.0 0.10 0.05 0.01 0.0)

# All configurations - separate arrays for names and args
CONFIG_NAMES=(
    "baseline"
    "global_det"
    "det_infer_step128"
    "det_infer_step256"
    "det_infer_step64"
    "det_infer_step32"
    "det_infer_step16"
    "det_infer_step512"
)

CONFIG_ARGS=(
    ""
    "--enable-deterministic-inference 2"
    "--enable-det-infer 3 --det-infer-window-size 128 --det-infer-verify-batch-size 1"
    "--enable-det-infer 3 --det-infer-window-size 256 --det-infer-verify-batch-size 1"
    "--enable-det-infer 3 --det-infer-window-size 64 --det-infer-verify-batch-size 1"
    "--enable-det-infer 3 --det-infer-window-size 32 --det-infer-verify-batch-size 1"
    "--enable-det-infer 3 --det-infer-window-size 16 --det-infer-verify-batch-size 1"
    "--enable-det-infer 3 --det-infer-window-size 512 --det-infer-verify-batch-size 1"
)

PYTHON_CMD=$(command -v python || command -v python3) || { echo "Error: Python not found"; exit 1; }
mkdir -p "$OUTPUT_DIR" "${OUTPUT_DIR}/latencies"

# ============================================
# Dataset Preparation
# ============================================
prepare_dataset() {
    if [ ! -f "$DATASET_FILE" ]; then
        echo "=============================================="
        echo "Preparing dataset (first time only)..."
        echo "=============================================="
        $PYTHON_CMD "$(dirname "$0")/run_arxiv_benchmark.py" \
            --model "$MODEL" \
            --num-prompts "$NUM_PROMPTS" \
            --context-len 16384 \
            --dataset-file "$DATASET_FILE" \
            --save-dataset-only
        echo "Dataset saved to $DATASET_FILE"
    else
        echo "Using existing dataset: $DATASET_FILE"
    fi
}

# ============================================
# Server Functions
# ============================================
launch_server() {
    local name=$1
    local port=$2
    local gpu=$3
    shift 3
    local extra_args="$*"
    
    echo "Launching $name on GPU $gpu, port $port..."
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON_CMD -m sglang.launch_server \
        --model-path "$MODEL" --host "$HOST" --port "$port" --tp 1 \
        --attention-backend "$ATTENTION_BACKEND" \
        --disable-radix-cache --disable-chunked-prefix-cache \
        --chunked-prefill-size -1 \
        --disable-overlap-schedule --enable-metrics $extra_args \
        > "${OUTPUT_DIR}/server_${name}.log" 2>&1 &
    echo $! > "${OUTPUT_DIR}/server_${name}.pid"
}

wait_for_server() {
    local url=$1
    local name=$2
    local waited=0
    echo "Waiting for $name..."
    while [ $waited -lt 300 ]; do
        if curl -s "${url}/health" > /dev/null 2>&1; then
            echo "  $name ready"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
    done
    echo "ERROR: $name failed to start"
    return 1
}

stop_server() {
    local name=$1
    local pidfile="${OUTPUT_DIR}/server_${name}.pid"
    if [ -f "$pidfile" ]; then
        local pid=$(cat "$pidfile")
        echo "Stopping $name (PID $pid)..."
        kill "$pid" 2>/dev/null || true
        local waited=0
        # Increase timeout to 120 seconds to allow requests to drain properly
        while kill -0 "$pid" 2>/dev/null && [ $waited -lt 120 ]; do
            sleep 1
            waited=$((waited + 1))
        done
        if kill -0 "$pid" 2>/dev/null; then
            echo "WARNING: $name did not stop gracefully after ${waited}s, force killing..."
        fi
        kill -9 "$pid" 2>/dev/null || true
        rm -f "$pidfile"
    fi
}

stop_all() {
    echo "Stopping all servers..."
    for f in "${OUTPUT_DIR}"/server_*.pid; do
        if [ -f "$f" ]; then
            local pid=$(cat "$f")
            kill "$pid" 2>/dev/null || true
            kill -9 "$pid" 2>/dev/null || true
            rm -f "$f"
        fi
    done
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    sleep 3
}
trap 'stop_all' EXIT

wait_for_gpu_cleanup() {
    echo "Waiting for GPU memory to be released..."
    local waited=0
    while [ $waited -lt 30 ]; do
        local gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
        if [ "$gpu_procs" -eq 0 ]; then
            echo "  GPU memory released"
            return 0
        fi
        sleep 2
        waited=$((waited + 2))
    done
    echo "  WARNING: GPU processes may still be running after 30s"
    sleep 5
}

# ============================================
# Benchmark Functions
# ============================================
get_rollback_metrics() {
    local url=$1
    local metrics=$(curl -s "${url}/metrics" 2>/dev/null || echo "")
    # Use awk to sum all values for the metric (handles multiple label variants)
    local rollbacks=$(echo "$metrics" | grep 'sglang:num_rollbacks_total' | grep -v '^#' | awk '{sum += $2} END {printf "%d", sum}')
    local tokens_rolled=$(echo "$metrics" | grep 'sglang:tokens_rolled_back_total' | grep -v '^#' | awk '{sum += $2} END {printf "%d", sum}')
    # Debug: log raw metrics for investigation
    if [[ -n "${DEBUG_ROLLBACKS:-}" ]]; then
        echo "[DEBUG] Raw rollback lines:" >&2
        echo "$metrics" | grep -E 'sglang:num_rollbacks_total|sglang:tokens_rolled_back_total' >&2
        echo "[DEBUG] Parsed: rollbacks=${rollbacks:-0}, tokens=${tokens_rolled:-0}" >&2
    fi
    echo "${rollbacks:-0} ${tokens_rolled:-0}"
}

run_bench() {
    local config=$1
    local url=$2
    local dataset=$3
    local rate=$4
    local det=$5
    local latency_file="${OUTPUT_DIR}/latencies/${config}_${dataset}_rate${rate}_det${det}.jsonl"
    local tmp="${OUTPUT_DIR}/.tmp_${config}_${RANDOM}.json"
    
    echo ">>> $config | $dataset | rate=$rate | det=$det"
    
    # Get metrics BEFORE benchmark
    local before=($(get_rollback_metrics "$url"))
    local rollbacks_before=${before[0]:-0}
    local tokens_before=${before[1]:-0}
    
    $PYTHON_CMD -m sglang.bench_serving --backend sglang --base-url "$url" --model "$MODEL" \
        --dataset-name "$dataset" --num-prompts "$NUM_PROMPTS" --request-rate "$rate" \
        --deterministic-ratio "$det" --warmup-requests 0 --sharegpt-context-len 16384 \
        --output-file "$tmp" --output-latencies "$latency_file" 2>&1 | tail -3
    
    # Get metrics AFTER benchmark and compute delta
    local after=($(get_rollback_metrics "$url"))
    local rollbacks_after=${after[0]:-0}
    local tokens_after=${after[1]:-0}
    local rollbacks_delta=$((rollbacks_after - rollbacks_before))
    local tokens_delta=$((tokens_after - tokens_before))
    
    echo "{\"config\":\"$config\",\"dataset\":\"$dataset\",\"rate\":$rate,\"det_ratio\":$det,\"num_rollbacks\":$rollbacks_delta,\"tokens_recomputed\":$tokens_delta}" >> "$ROLLBACK_FILE"
    
    if [ -f "$tmp" ]; then
        $PYTHON_CMD -c "
import json
with open('$tmp') as f: r = json.load(f)
r.update({'config':'$config','dataset':'$dataset','rate':$rate,'det_ratio':$det})
print(json.dumps(r))" >> "$RESULTS_FILE"
        rm -f "$tmp"
    fi
}

run_arxiv() {
    local config=$1
    local url=$2
    local rate=$3
    local det=$4
    local latency_file="${OUTPUT_DIR}/latencies/${config}_arxiv_rate${rate}_det${det}.jsonl"
    local tmp="${OUTPUT_DIR}/.tmp_arxiv_${config}_${RANDOM}.jsonl"
    
    echo ">>> $config | arxiv | rate=$rate | det=$det"
    
    # Get metrics BEFORE benchmark
    local before=($(get_rollback_metrics "$url"))
    local rollbacks_before=${before[0]:-0}
    local tokens_before=${before[1]:-0}
    
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting arxiv benchmark: $config"
    $PYTHON_CMD "$(dirname "$0")/run_arxiv_benchmark.py" --base-url "$url" --model "$MODEL" \
        --num-prompts "$NUM_PROMPTS" --request-rate "$rate" --deterministic-ratio "$det" \
        --dataset-file "$DATASET_FILE" \
        --output-file "$tmp" --output-latencies "$latency_file" 2>&1 | tee "${OUTPUT_DIR}/.arxiv_${config}_rate${rate}_det${det}.log" | tail -3
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Finished arxiv benchmark: $config"
    
    # Get metrics AFTER benchmark and compute delta
    local after=($(get_rollback_metrics "$url"))
    local rollbacks_after=${after[0]:-0}
    local tokens_after=${after[1]:-0}
    local rollbacks_delta=$((rollbacks_after - rollbacks_before))
    local tokens_delta=$((tokens_after - tokens_before))
    
    echo "{\"config\":\"$config\",\"dataset\":\"arxiv\",\"rate\":$rate,\"det_ratio\":$det,\"num_rollbacks\":$rollbacks_delta,\"tokens_recomputed\":$tokens_delta}" >> "$ROLLBACK_FILE"
    
    if [ -f "$tmp" ]; then
        $PYTHON_CMD -c "
import json
with open('$tmp') as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            r['config'] = '$config'
            print(json.dumps(r))" >> "$RESULTS_FILE"
        rm -f "$tmp"
    fi
}

run_benchmarks_for_config() {
    local config=$1
    local url=$2
    
    # Determine which det_ratios to use
    local det_ratios_to_use
    if [[ "$config" == det_infer_* ]]; then
        det_ratios_to_use=("${DET_RATIOS[@]}")
    else
        det_ratios_to_use=("1.0")
    fi
    
    # Run arXiv benchmarks first
    for rate in "${ARXIV_RATES[@]}"; do 
        for det in "${det_ratios_to_use[@]}"; do
            run_arxiv "$config" "$url" "$rate" "$det"
            sleep 2  # Allow metrics to stabilize between runs
        done
    done
    
    # Run ShareGPT benchmarks
    for rate in "${SHAREGPT_RATES[@]}"; do 
        for det in "${det_ratios_to_use[@]}"; do
            run_bench "$config" "$url" sharegpt "$rate" "$det"
            sleep 2  # Allow metrics to stabilize between runs
        done
    done
}

# ============================================
# Run All Configurations in Batches
# ============================================
run_all_configs() {
    echo -e "\n===== Running All Configurations in Batches of $NUM_GPUS GPUs ====="
    
    local total_configs=${#CONFIG_NAMES[@]}
    
    echo "Total configurations: $total_configs"
    echo "Configurations: ${CONFIG_NAMES[*]}"
    
    # Process configs in batches of NUM_GPUS
    for ((i=0; i<total_configs; i+=NUM_GPUS)); do
        local batch_num=$(((i/NUM_GPUS) + 1))
        local batch_end=$((i+NUM_GPUS < total_configs ? i+NUM_GPUS : total_configs))
        
        echo -e "\n=============================================="
        echo "Batch $batch_num: Configs $((i+1))-${batch_end} of $total_configs"
        echo "=============================================="
        
        # Track what we launch in this batch
        local batch_configs=()
        local batch_urls=()
        local batch_gpus=()
        
        # Launch servers for this batch
        for ((j=0; j<NUM_GPUS; j++)); do
            local idx=$((i + j))
            if [ $idx -ge $total_configs ]; then
                break
            fi
            
            local gpu=$j
            local config="${CONFIG_NAMES[$idx]}"
            local args="${CONFIG_ARGS[$idx]}"
            local port=$((PORT_BASE + gpu))
            local url="http://localhost:$port"
            local server_name="${config}_g${gpu}"
            
            echo "GPU $gpu -> $config (port $port)"
            launch_server "$server_name" "$port" "$gpu" $args
            
            batch_configs+=("$config")
            batch_urls+=("$url")
            batch_gpus+=("$gpu")
        done
        
        # Wait for all servers to be ready
        for ((j=0; j<${#batch_configs[@]}; j++)); do
            local config="${batch_configs[$j]}"
            local url="${batch_urls[$j]}"
            local gpu="${batch_gpus[$j]}"
            wait_for_server "$url" "${config}_g${gpu}"
        done
        
        # Run benchmarks in parallel across GPUs
        local pids=()
        for ((j=0; j<${#batch_configs[@]}; j++)); do
            local config="${batch_configs[$j]}"
            local url="${batch_urls[$j]}"
            
            run_benchmarks_for_config "$config" "$url" &
            pids+=($!)
        done
        
        # Wait for all benchmarks to complete
        echo "Running benchmarks in parallel on ${#batch_configs[@]} GPUs..."
        for pid in "${pids[@]}"; do
            wait $pid
        done
        echo "Batch $batch_num benchmarks complete."
        
        # Stop all servers
        for ((j=0; j<${#batch_configs[@]}; j++)); do
            local config="${batch_configs[$j]}"
            local gpu="${batch_gpus[$j]}"
            stop_server "${config}_g${gpu}"
        done
        
        # Wait for GPU cleanup before next batch
        wait_for_gpu_cleanup
        echo "Waiting additional 20 seconds for complete cleanup..."
        sleep 20
    done
}

# ============================================
# Main
# ============================================
echo "=============================================="
echo "Online Benchmark Suite"
echo "Model: $MODEL | GPUs: $NUM_GPUS | Prompts: $NUM_PROMPTS"
echo "=============================================="

[[ "$1" != "--start-servers" ]] && { echo "Usage: NUM_GPUS=4 $0 --start-servers"; exit 1; }

prepare_dataset

run_all_configs

echo -e "\n=============================================="
echo "Complete! Results: $RESULTS_FILE"
echo "=============================================="
