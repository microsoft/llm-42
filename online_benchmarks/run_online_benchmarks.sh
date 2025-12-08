#!/bin/bash
# Online Serving Benchmarks - Measures TTFT, TPOT, E2E latency
# Datasets: sharegpt, arxiv (ccdv/arxiv-summarization from HuggingFace)
# Tests different server configurations: baseline, global_det, det_infer with various step sizes
#
# Uses 4 GPUs (0, 1, 2, 3) to run servers and benchmarks in parallel with TP=1
#
# Usage:
#   ./run_online_benchmarks.sh                  # Run benchmarks only (assumes servers running)
#   ./run_online_benchmarks.sh --start-servers  # Start servers then run benchmarks
#   ./run_online_benchmarks.sh --servers-only   # Only start servers (no benchmarks)

set -e

# ============================================
# Configuration
# ============================================

MODEL="${SGLANG_TEST_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
HOST="${SGLANG_HOST:-0.0.0.0}"
TP_SIZE=1
ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-flashinfer}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
OUTPUT_DIR="$(dirname "$0")/qps_6_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="${OUTPUT_DIR}/results_${TIMESTAMP}.jsonl"
ROLLBACK_FILE="${OUTPUT_DIR}/rollback_metrics_${TIMESTAMP}.jsonl"

# GPU assignments
GPU_0=0
GPU_1=1
GPU_2=2
GPU_3=3

# Server port base
PORT_BASE=30006

# Benchmark parameters
REQUEST_RATES=(6)
DET_RATIOS=(1.0 0.10 0.05 0.01)
STEP_SIZES=(128 256 64 32 16 512)

# Determine Python command
PYTHON_CMD=$(command -v python || command -v python3) || { echo "Error: Python not found"; exit 1; }

mkdir -p "$OUTPUT_DIR"

# ============================================
# Server Management Functions
# ============================================

launch_server() {
    local config_name=$1 port=$2 gpu_id=$3
    shift 3
    local extra_args="$@"
    
    echo "Launching $config_name on GPU $gpu_id, port $port..."
    CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON_CMD -m sglang.launch_server \
        --model-path "$MODEL" \
        --host "$HOST" \
        --port "$port" \
        --tp "$TP_SIZE" \
        --attention-backend "$ATTENTION_BACKEND" \
        --disable-radix-cache \
        --disable-chunked-prefix-cache \
        --disable-overlap-schedule \
        --enable-metrics \
        $extra_args \
        > "${OUTPUT_DIR}/server_${config_name}.log" 2>&1 &
    echo $! > "${OUTPUT_DIR}/server_${config_name}.pid"
    echo "  PID: $(cat ${OUTPUT_DIR}/server_${config_name}.pid)"
}

wait_for_server() {
    local url=$1 name=$2 max_wait=300 waited=0
    echo "Waiting for $name..."
    while [ $waited -lt $max_wait ]; do
        curl -s "${url}/health" > /dev/null 2>&1 && echo "  $name is ready!" && return 0
        sleep 5
        waited=$((waited + 5))
    done
    echo "  ERROR: $name failed to start within ${max_wait}s"
    return 1
}

stop_server() {
    local config_name=$1
    local pidfile="${OUTPUT_DIR}/server_${config_name}.pid"
    if [ -f "$pidfile" ]; then
        local pid=$(cat "$pidfile")
        kill -0 "$pid" 2>/dev/null && echo "  Stopping $config_name (PID $pid)" && kill "$pid" 2>/dev/null || true
        rm -f "$pidfile"
    fi
    sleep 2
}

stop_all_servers() {
    echo "Stopping all servers..."
    for pidfile in "${OUTPUT_DIR}"/server_*.pid; do
        [ -f "$pidfile" ] || continue
        local pid=$(cat "$pidfile")
        kill -0 "$pid" 2>/dev/null && kill "$pid" 2>/dev/null || true
        rm -f "$pidfile"
    done
    sleep 2
}

check_server() {
    local url=$1 name=$2
    for i in {1..3}; do
        curl -s "${url}/health" > /dev/null 2>&1 && echo "  $name: OK" && return 0
        sleep 1
    done
    echo "  $name: NOT AVAILABLE"
    return 1
}

# Cleanup on exit
trap 'stop_all_servers' EXIT

# ============================================
# Rollback Metrics Functions
# ============================================

get_rollback_metrics() {
    local base_url=$1
    # Returns: num_rollbacks,tokens_rolled_back
    $PYTHON_CMD -c "
import requests, re, sys
try:
    r = requests.get('${base_url}/metrics', timeout=5)
    if r.status_code == 200:
        rollbacks = tokens = 0
        for name, var in [('sglang:num_rollbacks_total', 'rollbacks'), ('sglang:tokens_rolled_back_total', 'tokens')]:
            m = re.search(rf'{re.escape(name)}\\{{[^}}]*\\}}\\s+([\\d.]+)', r.text)
            if m: exec(f'{var} = {m.group(1)}')
        print(f'{int(rollbacks)},{int(tokens)}')
    else:
        print('0,0')
except:
    print('0,0')
" 2>/dev/null
}

# ============================================
# Benchmark Functions
# ============================================

run_bench() {
    local config_name=$1 base_url=$2 dataset=$3 rate=$4 det=$5 extra_args=$6
    echo ">>> $config_name | $dataset | rate=$rate | det_ratio=$det"
    
    local tmp="${OUTPUT_DIR}/.tmp_${config_name}_${RANDOM}.json"
    local latency_file="${OUTPUT_DIR}/latencies/${config_name}_${dataset}_rate${rate}_det${det}.jsonl"
    
    mkdir -p "${OUTPUT_DIR}/latencies"
    
    # Get rollback metrics before benchmark
    local metrics_before=$(get_rollback_metrics "$base_url")
    local rollbacks_before=$(echo "$metrics_before" | cut -d',' -f1)
    local tokens_before=$(echo "$metrics_before" | cut -d',' -f2)
    
    $PYTHON_CMD -m sglang.bench_serving \
        --backend sglang --base-url "$base_url" --model "$MODEL" \
        --dataset-name "$dataset" --num-prompts "$NUM_PROMPTS" \
        --request-rate "$rate" --deterministic-ratio "$det" \
        --output-file "$tmp" --output-latencies "$latency_file" $extra_args 2>&1 | tail -5
    
    # Get rollback metrics after benchmark
    local metrics_after=$(get_rollback_metrics "$base_url")
    local rollbacks_after=$(echo "$metrics_after" | cut -d',' -f1)
    local tokens_after=$(echo "$metrics_after" | cut -d',' -f2)
    
    # Calculate deltas
    local rollbacks_delta=$((rollbacks_after - rollbacks_before))
    local tokens_delta=$((tokens_after - tokens_before))
    
    # Save rollback metrics
    echo "{\"config_name\":\"$config_name\",\"dataset\":\"$dataset\",\"rate\":$rate,\"det_ratio\":$det,\"rollbacks\":$rollbacks_delta,\"tokens_recomputed\":$tokens_delta}" >> "$ROLLBACK_FILE"
    
    [[ -f "$tmp" ]] && $PYTHON_CMD -c "
import json
with open('$tmp') as f: r = json.load(f)
r.update({'config_name':'$config_name','dataset':'$dataset','rate':'$rate','det_ratio':$det})
print(json.dumps(r))
" >> "$RESULTS_FILE" && rm -f "$tmp"
}

run_arxiv_bench() {
    local config_name=$1 base_url=$2 rate=$3 det=$4
    echo ">>> $config_name | arxiv | rate=$rate | det_ratio=$det"
    
    local tmp="${OUTPUT_DIR}/.tmp_arxiv_${config_name}_${RANDOM}.jsonl"
    local latency_file="${OUTPUT_DIR}/latencies/${config_name}_arxiv_rate${rate}_det${det}.jsonl"
    
    mkdir -p "${OUTPUT_DIR}/latencies"
    
    # Get rollback metrics before benchmark
    local metrics_before=$(get_rollback_metrics "$base_url")
    local rollbacks_before=$(echo "$metrics_before" | cut -d',' -f1)
    local tokens_before=$(echo "$metrics_before" | cut -d',' -f2)
    
    $PYTHON_CMD "$(dirname "$0")/run_arxiv_benchmark.py" \
        --base-url "$base_url" --model "$MODEL" --num-prompts "$NUM_PROMPTS" \
        --request-rate "$rate" --deterministic-ratio "$det" \
        --output-file "$tmp" --output-latencies "$latency_file" 2>&1 | tail -5
    
    # Get rollback metrics after benchmark
    local metrics_after=$(get_rollback_metrics "$base_url")
    local rollbacks_after=$(echo "$metrics_after" | cut -d',' -f1)
    local tokens_after=$(echo "$metrics_after" | cut -d',' -f2)
    
    # Calculate deltas
    local rollbacks_delta=$((rollbacks_after - rollbacks_before))
    local tokens_delta=$((tokens_after - tokens_before))
    
    # Save rollback metrics
    echo "{\"config_name\":\"$config_name\",\"dataset\":\"arxiv\",\"rate\":$rate,\"det_ratio\":$det,\"rollbacks\":$rollbacks_delta,\"tokens_recomputed\":$tokens_delta}" >> "$ROLLBACK_FILE"
    
    [[ -f "$tmp" ]] && $PYTHON_CMD -c "
import json
with open('$tmp') as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            r['config_name'] = '$config_name'
            print(json.dumps(r))
" >> "$RESULTS_FILE" && rm -f "$tmp"
}

run_benchmarks_for_config() {
    local config_name=$1 base_url=$2
    
    echo ""
    echo "========== Configuration: $config_name =========="
    echo "URL: $base_url"
    
    echo "--- ShareGPT Dataset ---"
    for rate in "${REQUEST_RATES[@]}"; do
        for det in "${DET_RATIOS[@]}"; do
            run_bench "$config_name" "$base_url" sharegpt "$rate" "$det" ""
        done
    done
    
    echo "--- Arxiv Dataset ---"
    for rate in "${REQUEST_RATES[@]}"; do
        for det in "${DET_RATIOS[@]}"; do
            run_arxiv_bench "$config_name" "$base_url" "$rate" "$det"
        done
    done
}

# ============================================
# Argument Parsing
# ============================================

START_SERVERS=false
SERVERS_ONLY=false
for arg in "$@"; do
    case $arg in
        --start-servers) START_SERVERS=true ;;
        --servers-only)  START_SERVERS=true; SERVERS_ONLY=true ;;
    esac
done

# ============================================
# Main
# ============================================

echo "=============================================="
echo "Online Benchmark Suite"
echo "=============================================="
echo "Model: $MODEL"
echo "Num Prompts: $NUM_PROMPTS"
echo "Results: $RESULTS_FILE"
echo "GPUs: $GPU_0, $GPU_1, $GPU_2, $GPU_3"
echo "Configurations: baseline, global_det, det_infer (step sizes: ${STEP_SIZES[*]})"
echo "=============================================="

if [ "$START_SERVERS" = true ]; then
    # Build list of all configurations
    ALL_CONFIGS=(
        "baseline::"
        "global_det::--enable-deterministic-inference 2"
    )
    for step_size in "${STEP_SIZES[@]}"; do
        ALL_CONFIGS+=("det_infer_step${step_size}::--enable-det-infer 3 --min-det-step-size $step_size --max-det-verify-batch-size 1")
    done
    
    echo ""
    echo "Total configurations: ${#ALL_CONFIGS[@]}"
    echo "Running in batches of 4 (parallel on 4 GPUs)"
    
    # Process configs in batches of 4
    for ((i=0; i<${#ALL_CONFIGS[@]}; i+=4)); do
        batch_num=$((i/4 + 1))
        batch_end=$((i+4 < ${#ALL_CONFIGS[@]} ? i+4 : ${#ALL_CONFIGS[@]}))
        echo ""
        echo "=============================================="
        echo "Batch $batch_num: Configs $((i+1))-${batch_end} of ${#ALL_CONFIGS[@]}"
        echo "=============================================="
        
        # Launch servers
        ACTIVE_CONFIGS=()
        for j in 0 1 2 3; do
            idx=$((i + j))
            [ $idx -ge ${#ALL_CONFIGS[@]} ] && break
            
            config_entry="${ALL_CONFIGS[$idx]}"
            config_name="${config_entry%%::*}"
            extra_args="${config_entry#*::}"
            port=$((PORT_BASE + j))
            gpu_id=$((j))
            url="http://localhost:${port}"
            
            launch_server "$config_name" "$port" "$gpu_id" "$extra_args"
            ACTIVE_CONFIGS+=("$config_name:$url")
        done
        
        # Wait for servers
        for entry in "${ACTIVE_CONFIGS[@]}"; do
            wait_for_server "${entry#*:}" "${entry%%:*}"
        done
        
        if [ "$SERVERS_ONLY" = true ]; then
            echo ""
            echo "Servers running. Press Ctrl+C to stop."
            wait
            exit 0
        fi
        
        # Run benchmarks in parallel
        echo "Running benchmarks in parallel on ${#ACTIVE_CONFIGS[@]} GPUs..."
        BENCH_PIDS=()
        for entry in "${ACTIVE_CONFIGS[@]}"; do
            run_benchmarks_for_config "${entry%%:*}" "${entry#*:}" &
            BENCH_PIDS+=($!)
        done
        
        for pid in "${BENCH_PIDS[@]}"; do
            wait $pid
        done
        echo "Batch $batch_num complete."
        
        # Stop servers
        for entry in "${ACTIVE_CONFIGS[@]}"; do
            stop_server "${entry%%:*}"
        done
    done

else
    # Manual mode: discover running servers
    echo "Checking for running servers..."
    AVAILABLE_CONFIGS=()
    
    for port in $(seq $PORT_BASE $((PORT_BASE + 10))); do
        url="http://localhost:${port}"
        if curl -s "${url}/health" > /dev/null 2>&1; then
            echo "  Found server on port $port"
            AVAILABLE_CONFIGS+=("server_port${port}:$url")
        fi
    done
    
    if [ ${#AVAILABLE_CONFIGS[@]} -eq 0 ]; then
        echo "ERROR: No servers available!"
        echo "Use --start-servers to launch servers automatically"
        exit 1
    fi
    
    for entry in "${AVAILABLE_CONFIGS[@]}"; do
        run_benchmarks_for_config "${entry%%:*}" "${entry#*:}"
    done
fi

echo ""
echo "=============================================="
echo "Benchmarking Complete!"
echo "Results saved to: $RESULTS_FILE"
echo "=============================================="

echo ""
echo "Generating plots..."
$PYTHON_CMD "$(dirname "$0")/plot_results.py" "$RESULTS_FILE" --output-dir "${OUTPUT_DIR}/plots_${TIMESTAMP}"

echo ""
echo "Generating CDF plots from latency data..."
$PYTHON_CMD "$(dirname "$0")/plot_cdf.py" "${OUTPUT_DIR}/latencies" --output-dir "${OUTPUT_DIR}/plots_${TIMESTAMP}"

echo ""
echo "Generating rollback plots..."
$PYTHON_CMD "$(dirname "$0")/plot_rollbacks.py" "$ROLLBACK_FILE" --output-dir "${OUTPUT_DIR}/plots_${TIMESTAMP}"

echo "Done!"
