#!/bin/bash
# Online Serving Benchmarks - Measures TTFT, TPOT, E2E latency
# Datasets: random, sharegpt, arxiv (ccdv/arxiv-summarization from HuggingFace)
# Tests different server configurations: baseline, global_det, det_infer with various step sizes
#
# Uses 3 GPUs (0, 1, 2) to run servers in parallel with TP=1
#
# Usage:
#   ./run_online_benchmarks.sh              # Run benchmarks only (assumes servers running)
#   ./run_online_benchmarks.sh --start-servers  # Start servers then run benchmarks
#   ./run_online_benchmarks.sh --servers-only   # Only start servers (no benchmarks)

set -e

# Configuration
MODEL="${SGLANG_TEST_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
HOST="${SGLANG_HOST:-0.0.0.0}"
TP_SIZE=1  # Fixed to 1 for single GPU per server
ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-flashinfer}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
OUTPUT_DIR="$(dirname "$0")/results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="${OUTPUT_DIR}/results_${TIMESTAMP}.jsonl"

# GPU assignments (3 GPUs for parallel execution)
GPU_0=0
GPU_1=1
GPU_2=2

# Server port base
PORT_BASE=30006

# Benchmark parameters
REQUEST_RATES=(1 4 8 16 32)
DET_RATIOS=(0.02 0.05 0.1 1.0)
STEP_SIZES=(16 32 64 128 256 512)  # Sweep over min-det-step-size

# Determine Python command
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: Python not found"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ============================================
# Server Launch Functions
# ============================================

# Generic server launcher
# Args: config_name, port, gpu_id, extra_args...
launch_server() {
    local config_name=$1
    local port=$2
    local gpu_id=$3
    shift 3
    local extra_args="$@"
    
    echo "Launching $config_name server on GPU $gpu_id, port $port..."
    CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON_CMD -m sglang.launch_server \
        --model-path "$MODEL" \
        --host "$HOST" \
        --port "$port" \
        --tp "$TP_SIZE" \
        --attention-backend "$ATTENTION_BACKEND" \
        --disable-radix-cache \
        --disable-chunked-prefix-cache \
        --disable-overlap-schedule \
        $extra_args \
        > "${OUTPUT_DIR}/server_${config_name}.log" 2>&1 &
    echo $! > "${OUTPUT_DIR}/server_${config_name}.pid"
    echo "  PID: $(cat ${OUTPUT_DIR}/server_${config_name}.pid), GPU: $gpu_id"
}

wait_for_server() {
    local url=$1
    local name=$2
    local max_wait=300  # 5 minutes
    local waited=0
    echo "Waiting for $name to be ready..."
    while [ $waited -lt $max_wait ]; do
        if curl -s "${url}/health" > /dev/null 2>&1; then
            echo "  $name is ready!"
            return 0
        fi
        sleep 5
        waited=$((waited + 5))
        echo "  Waiting... ${waited}s"
    done
    echo "  ERROR: $name failed to start within ${max_wait}s"
    return 1
}

stop_servers() {
    echo "Stopping servers..."
    for pidfile in "${OUTPUT_DIR}"/server_*.pid; do
        if [ -f "$pidfile" ]; then
            pid=$(cat "$pidfile")
            if kill -0 "$pid" 2>/dev/null; then
                echo "  Stopping PID $pid"
                kill "$pid" 2>/dev/null || true
            fi
            rm -f "$pidfile"
        fi
    done
    sleep 2
}

stop_server() {
    local config_name=$1
    local pidfile="${OUTPUT_DIR}/server_${config_name}.pid"
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Stopping $config_name (PID $pid)"
            kill "$pid" 2>/dev/null || true
        fi
        rm -f "$pidfile"
    fi
    sleep 2
}

# Handle cleanup on exit
cleanup() {
    if [ "$START_SERVERS" = true ]; then
        stop_servers
    fi
}
trap cleanup EXIT

# Parse arguments
START_SERVERS=false
SERVERS_ONLY=false
for arg in "$@"; do
    case $arg in
        --start-servers)
            START_SERVERS=true
            ;;
        --servers-only)
            START_SERVERS=true
            SERVERS_ONLY=true
            ;;
    esac
done

echo "=============================================="
echo "Online Benchmark Suite"
echo "=============================================="
echo "Model: $MODEL"
echo "Num Prompts: $NUM_PROMPTS"
echo "Results: $RESULTS_FILE"
echo "GPUs: $GPU_0, $GPU_1, $GPU_2"
echo "Step Sizes: ${STEP_SIZES[*]}"
echo "=============================================="

# Check server health
check_server() {
    local url=$1
    local name=$2
    for i in {1..3}; do
        curl -s "${url}/health" > /dev/null 2>&1 && echo "  $name: OK" && return 0
        sleep 1
    done
    echo "  $name: NOT AVAILABLE"
    return 1
}

run_bench() {
    local config_name=$1 base_url=$2 dataset=$3 rate=$4 det=$5 extra_args=$6
    echo ">>> $config_name | $dataset | rate=$rate | det_ratio=$det"
    
    local tmp="${OUTPUT_DIR}/.tmp_${config_name}_${RANDOM}.json"
    
    python -m sglang.bench_serving \
        --backend sglang --base-url "$base_url" --model "$MODEL" \
        --dataset-name "$dataset" --num-prompts "$NUM_PROMPTS" \
        --request-rate "$rate" --deterministic-ratio "$det" \
        --output-file "$tmp" $extra_args 2>&1 | tail -5
    
    # Append with metadata
    [[ -f "$tmp" ]] && python -c "
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
    python "$(dirname "$0")/run_arxiv_benchmark.py" \
        --base-url "$base_url" --model "$MODEL" --num-prompts "$NUM_PROMPTS" \
        --request-rate "$rate" --deterministic-ratio "$det" \
        --output-file "$tmp" 2>&1 | tail -5
    
    # Add config_name and append to main results
    if [[ -f "$tmp" ]]; then
        python -c "
import json
with open('$tmp') as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            r['config_name'] = '$config_name'
            print(json.dumps(r))
" >> "$RESULTS_FILE"
        rm -f "$tmp"
    fi
}

run_benchmarks_for_config() {
    local config_name=$1
    local base_url=$2
    
    echo ""
    echo "========== Configuration: $config_name =========="
    echo "URL: $base_url"
    echo ""
    
    # Random dataset benchmarks
    echo "--- Random Dataset ---"
    for rate in "${REQUEST_RATES[@]}"; do
        for det in "${DET_RATIOS[@]}"; do
            run_bench "$config_name" "$base_url" random "$rate" "$det" "--random-input-len 1024 --random-output-len 256"
        done
    done
    
    # ShareGPT benchmarks
    echo "--- ShareGPT Dataset ---"
    for rate in "${REQUEST_RATES[@]}"; do
        for det in "${DET_RATIOS[@]}"; do
            run_bench "$config_name" "$base_url" sharegpt "$rate" "$det" ""
        done
    done
    
    # Arxiv benchmarks
    echo "--- Arxiv Dataset ---"
    for rate in "${REQUEST_RATES[@]}"; do
        for det in "${DET_RATIOS[@]}"; do
            run_arxiv_bench "$config_name" "$base_url" "$rate" "$det"
        done
    done
}

# ============================================
# Main Execution
# ============================================

if [ "$START_SERVERS" = true ]; then
    echo ""
    echo "=============================================="
    echo "Phase 1: Static Configurations (baseline, global_det)"
    echo "=============================================="
    
    # Launch baseline and global_det in parallel on GPU 0 and 1
    BASELINE_PORT=$((PORT_BASE + 0))
    GLOBAL_DET_PORT=$((PORT_BASE + 1))
    BASELINE_URL="http://localhost:${BASELINE_PORT}"
    GLOBAL_DET_URL="http://localhost:${GLOBAL_DET_PORT}"
    
    launch_server "baseline" "$BASELINE_PORT" "$GPU_0" ""
    launch_server "global_det" "$GLOBAL_DET_PORT" "$GPU_1" "--enable-deterministic-inference 2"
    
    wait_for_server "$BASELINE_URL" "baseline"
    wait_for_server "$GLOBAL_DET_URL" "global_det"
    
    if [ "$SERVERS_ONLY" = true ]; then
        echo ""
        echo "Servers running. Press Ctrl+C to stop."
        wait
        exit 0
    fi
    
    # Run benchmarks for baseline and global_det
    run_benchmarks_for_config "baseline" "$BASELINE_URL"
    run_benchmarks_for_config "global_det" "$GLOBAL_DET_URL"
    
    # Stop these servers
    stop_server "baseline"
    stop_server "global_det"
    
    echo ""
    echo "=============================================="
    echo "Phase 2: Det-Infer with Step Size Sweep"
    echo "=============================================="
    
    # Now sweep through step sizes, using all 3 GPUs in parallel
    # Process step sizes in batches of 3
    for ((i=0; i<${#STEP_SIZES[@]}; i+=3)); do
        echo ""
        echo "--- Step Size Batch: ${STEP_SIZES[$i]}, ${STEP_SIZES[$((i+1))]:-}, ${STEP_SIZES[$((i+2))]:-} ---"
        
        # Launch up to 3 servers in parallel
        ACTIVE_CONFIGS=()
        
        for j in 0 1 2; do
            idx=$((i + j))
            if [ $idx -lt ${#STEP_SIZES[@]} ]; then
                step_size=${STEP_SIZES[$idx]}
                config_name="det_infer_step${step_size}"
                port=$((PORT_BASE + j))
                gpu_var="GPU_$j"
                gpu_id=${!gpu_var}
                url="http://localhost:${port}"
                
                launch_server "$config_name" "$port" "$gpu_id" "--enable-det-infer 1 --min-det-step-size $step_size"
                ACTIVE_CONFIGS+=("$config_name:$url")
            fi
        done
        
        # Wait for all launched servers
        for config_entry in "${ACTIVE_CONFIGS[@]}"; do
            config_name="${config_entry%%:*}"
            url="${config_entry#*:}"
            wait_for_server "$url" "$config_name"
        done
        
        # Run benchmarks for each config
        for config_entry in "${ACTIVE_CONFIGS[@]}"; do
            config_name="${config_entry%%:*}"
            url="${config_entry#*:}"
            run_benchmarks_for_config "$config_name" "$url"
        done
        
        # Stop servers for this batch
        for config_entry in "${ACTIVE_CONFIGS[@]}"; do
            config_name="${config_entry%%:*}"
            stop_server "$config_name"
        done
    done

else
    # Manual mode: check what servers are available
    echo "Checking servers..."
    AVAILABLE_CONFIGS=()
    
    # Check standard ports for baseline and global_det
    BASELINE_URL="http://localhost:$((PORT_BASE + 0))"
    GLOBAL_DET_URL="http://localhost:$((PORT_BASE + 1))"
    check_server "$BASELINE_URL" "baseline" && AVAILABLE_CONFIGS+=("baseline:$BASELINE_URL")
    check_server "$GLOBAL_DET_URL" "global_det" && AVAILABLE_CONFIGS+=("global_det:$GLOBAL_DET_URL")
    
    # Check for det_infer servers on ports 30002+
    for port in $(seq $((PORT_BASE + 2)) $((PORT_BASE + 10))); do
        url="http://localhost:${port}"
        if curl -s "${url}/health" > /dev/null 2>&1; then
            echo "  Found server on port $port"
            AVAILABLE_CONFIGS+=("det_infer_port${port}:$url")
        fi
    done
    
    if [ ${#AVAILABLE_CONFIGS[@]} -eq 0 ]; then
        echo "ERROR: No servers available!"
        echo "Use --start-servers to launch servers automatically"
        exit 1
    fi
    echo ""
    
    # Run benchmarks for available configs
    for config_entry in "${AVAILABLE_CONFIGS[@]}"; do
        config_name="${config_entry%%:*}"
        base_url="${config_entry#*:}"
        run_benchmarks_for_config "$config_name" "$base_url"
    done
fi

echo ""
echo "=============================================="
echo "Benchmarking Complete!"
echo "Results saved to: $RESULTS_FILE"
echo "=============================================="

# Generate plots
echo ""
echo "Generating plots..."
python "$(dirname "$0")/plot_results.py" "$RESULTS_FILE" --output-dir "${OUTPUT_DIR}/plots_${TIMESTAMP}"

echo "Done!"
