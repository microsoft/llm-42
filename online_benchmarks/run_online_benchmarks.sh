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
NUM_GPUS="${NUM_GPUS:-4}"
PORT_BASE=30006

OUTPUT_DIR="$(dirname "$0")/online_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="${OUTPUT_DIR}/results_${TIMESTAMP}.jsonl"

# Benchmark parameters
SHAREGPT_RATES=(4 6 8 10)
ARXIV_RATES=(1 2 3 4)
DET_RATIOS=(0.0 1.0 0.10 0.05 0.01)
STEP_SIZES=(128 256 64 32 16 512)

PYTHON_CMD=$(command -v python || command -v python3) || { echo "Error: Python not found"; exit 1; }
mkdir -p "$OUTPUT_DIR" "${OUTPUT_DIR}/latencies"

# ============================================
# Server Functions
# ============================================
launch_server() {
    local name=$1 port=$2 gpu=$3; shift 3
    echo "Launching $name on GPU $gpu, port $port..."
    CUDA_VISIBLE_DEVICES=$gpu $PYTHON_CMD -m sglang.launch_server \
        --model-path "$MODEL" --host "$HOST" --port "$port" --tp 1 \
        --attention-backend "$ATTENTION_BACKEND" \
        --disable-radix-cache --disable-chunked-prefix-cache \
        --disable-overlap-schedule --enable-metrics $@ \
        > "${OUTPUT_DIR}/server_${name}.log" 2>&1 &
    echo $! > "${OUTPUT_DIR}/server_${name}.pid"
}

wait_for_server() {
    local url=$1 name=$2 waited=0
    while [ $waited -lt 300 ]; do
        curl -s "${url}/health" > /dev/null 2>&1 && echo "  $name ready" && return 0
        sleep 5; waited=$((waited + 5))
    done
    echo "ERROR: $name failed to start"; return 1
}

stop_server() {
    local name=$1 pidfile="${OUTPUT_DIR}/server_${name}.pid"
    if [ -f "$pidfile" ]; then
        local pid=$(cat "$pidfile")
        # First try graceful shutdown
        kill "$pid" 2>/dev/null || true
        # Wait up to 10 seconds for process to terminate
        local waited=0
        while kill -0 "$pid" 2>/dev/null && [ $waited -lt 10 ]; do
            sleep 1; waited=$((waited + 1))
        done
        # Force kill if still running
        kill -9 "$pid" 2>/dev/null || true
        rm -f "$pidfile"
    fi
}

stop_all() {
    # Kill all tracked servers
    for f in "${OUTPUT_DIR}"/server_*.pid; do
        [ -f "$f" ] && { kill $(cat "$f") 2>/dev/null || true; kill -9 $(cat "$f") 2>/dev/null || true; rm -f "$f"; }
    done
    # Also kill any stray sglang processes on our ports
    pkill -9 -f "sglang.launch_server" 2>/dev/null || true
    # Wait for GPU memory to be released (check every 2 seconds, max 30 seconds)
    echo "Waiting for GPU memory to be released..."
    local waited=0
    while [ $waited -lt 30 ]; do
        local gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
        [ "$gpu_procs" -eq 0 ] && echo "  GPU memory released" && break
        sleep 2; waited=$((waited + 2))
    done
    [ $waited -ge 30 ] && echo "  WARNING: GPU processes may still be running"
    sleep 3  # Extra buffer for CUDA cleanup
}
trap 'stop_all' EXIT

wait_for_gpu_cleanup() {
    # Wait for GPU memory to be released (check every 2 seconds, max 30 seconds)
    echo "Waiting for GPU memory to be released..."
    local waited=0
    while [ $waited -lt 30 ]; do
        local gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
        [ "$gpu_procs" -eq 0 ] && echo "  GPU memory released" && return 0
        sleep 2; waited=$((waited + 2))
    done
    echo "  WARNING: GPU processes may still be running after 30s"
    sleep 5  # Extra buffer
}

# ============================================
# Benchmark Functions
# ============================================
get_rollback_metrics() {
    local url=$1
    local metrics=$(curl -s "${url}/metrics" 2>/dev/null || echo "")
    local rollbacks=$(echo "$metrics" | grep 'sglang:num_rollbacks_total' | grep -v '^#' | awk '{print $2}' | tail -1)
    local tokens_rolled=$(echo "$metrics" | grep 'sglang:tokens_rolled_back_total' | grep -v '^#' | awk '{print $2}' | tail -1)
    echo "${rollbacks:-0} ${tokens_rolled:-0}"
}

run_bench() {
    local config=$1 url=$2 dataset=$3 rate=$4 det=$5
    local latency_file="${OUTPUT_DIR}/latencies/${config}_${dataset}_rate${rate}_det${det}.jsonl"
    local tmp="${OUTPUT_DIR}/.tmp_${RANDOM}.json"
    
    echo ">>> $config | $dataset | rate=$rate | det=$det"
    
    # Get metrics BEFORE benchmark
    local before=($(get_rollback_metrics "$url"))
    local rollbacks_before=${before[0]}
    local tokens_before=${before[1]}
    
    $PYTHON_CMD -m sglang.bench_serving --backend sglang --base-url "$url" --model "$MODEL" \
        --dataset-name "$dataset" --num-prompts "$NUM_PROMPTS" --request-rate "$rate" \
        --deterministic-ratio "$det" --warmup-requests 0 \
        --output-file "$tmp" --output-latencies "$latency_file" 2>&1 | tail -3
    
    # Get metrics AFTER benchmark and compute delta
    local after=($(get_rollback_metrics "$url"))
    local rollbacks_delta=$((${after[0]} - rollbacks_before))
    local tokens_delta=$((${after[1]} - tokens_before))
    
    mkdir -p "${OUTPUT_DIR}/metrics"
    echo "{\"config\":\"$config\",\"dataset\":\"$dataset\",\"rate\":$rate,\"det_ratio\":$det,\"num_rollbacks\":$rollbacks_delta,\"tokens_recomputed\":$tokens_delta}" >> "${OUTPUT_DIR}/rollback_metrics.jsonl"
    
    [ -f "$tmp" ] && $PYTHON_CMD -c "
import json
with open('$tmp') as f: r = json.load(f)
r.update({'config':'$config','dataset':'$dataset','rate':$rate,'det_ratio':$det})
print(json.dumps(r))" >> "$RESULTS_FILE" && rm -f "$tmp"
}

run_arxiv() {
    local config=$1 url=$2 rate=$3 det=$4
    local latency_file="${OUTPUT_DIR}/latencies/${config}_arxiv_rate${rate}_det${det}.jsonl"
    local tmp="${OUTPUT_DIR}/.tmp_${RANDOM}.jsonl"
    
    echo ">>> $config | arxiv | rate=$rate | det=$det"
    
    # Get metrics BEFORE benchmark
    local before=($(get_rollback_metrics "$url"))
    local rollbacks_before=${before[0]}
    local tokens_before=${before[1]}
    
    $PYTHON_CMD "$(dirname "$0")/run_arxiv_benchmark.py" --base-url "$url" --model "$MODEL" \
        --num-prompts "$NUM_PROMPTS" --request-rate "$rate" --deterministic-ratio "$det" \
        --output-file "$tmp" --output-latencies "$latency_file" 2>&1 | tail -3
    
    # Get metrics AFTER benchmark and compute delta
    local after=($(get_rollback_metrics "$url"))
    local rollbacks_delta=$((${after[0]} - rollbacks_before))
    local tokens_delta=$((${after[1]} - tokens_before))
    
    mkdir -p "${OUTPUT_DIR}/metrics"
    echo "{\"config\":\"$config\",\"dataset\":\"arxiv\",\"rate\":$rate,\"det_ratio\":$det,\"num_rollbacks\":$rollbacks_delta,\"tokens_recomputed\":$tokens_delta}" >> "${OUTPUT_DIR}/rollback_metrics.jsonl"
    
    [ -f "$tmp" ] && $PYTHON_CMD -c "
import json
with open('$tmp') as f:
    for line in f:
        if line.strip():
            r = json.loads(line); r['config'] = '$config'
            print(json.dumps(r))" >> "$RESULTS_FILE" && rm -f "$tmp"
}

# ============================================
# Rate Distribution (one-to-one across GPUs)
# ============================================
distribute_rates() {
    local -n sg=$1 ax=$2; local n=$3
    for ((g=0; g<n; g++)); do 
        sg[$g]=""
        ax[$g]=""
        # Assign one ShareGPT rate per GPU
        if [ $g -lt ${#SHAREGPT_RATES[@]} ]; then
            sg[$g]="${SHAREGPT_RATES[$g]}"
        fi
        # Assign one arXiv rate per GPU
        if [ $g -lt ${#ARXIV_RATES[@]} ]; then
            ax[$g]="${ARXIV_RATES[$g]}"
        fi
    done
}

# ============================================
# Run Phase (baseline/global_det)
# ============================================
run_phase() {
    local name=$1 config=$2 args=$3
    echo -e "\n===== Phase: $name ($NUM_GPUS GPUs) ====="
    
    declare -a SG AX; distribute_rates SG AX $NUM_GPUS
    
    for ((g=0; g<NUM_GPUS; g++)); do launch_server "${config}_g$g" $((PORT_BASE+g)) $g $args; done
    for ((g=0; g<NUM_GPUS; g++)); do wait_for_server "http://localhost:$((PORT_BASE+g))" "${config}_g$g"; done
    
    # Run all benchmarks (each GPU runs its sharegpt + arxiv rates independently in parallel)
    PIDS=()
    for ((g=0; g<NUM_GPUS; g++)); do
        url="http://localhost:$((PORT_BASE+g))"
        ( 
            for rate in ${SG[$g]}; do run_bench "$config" "$url" sharegpt "$rate" "0.0"; done
            for rate in ${AX[$g]}; do run_arxiv "$config" "$url" "$rate" "0.0"; done
        ) & PIDS+=($!)
    done
    for p in "${PIDS[@]}"; do wait $p; done
    
    for ((g=0; g<NUM_GPUS; g++)); do stop_server "${config}_g$g"; done
    wait_for_gpu_cleanup
}

# ============================================
# Run DetInfer Batch
# ============================================
run_detinfer_batch() {
    local -a steps=("$@"); local n=${#steps[@]}
    echo -e "\n===== DetInfer: ${steps[*]} ====="
    
    declare -a SG AX; distribute_rates SG AX $NUM_GPUS
    declare -a CONFIGS=(); local g=0
    
    # Launch servers - distribute GPUs evenly across configs
    for ((j=0; j<n; j++)); do
        local step=${steps[$j]} gpus=$(( (NUM_GPUS - g) / (n - j) ))
        for ((r=0; r<gpus; r++)); do
            launch_server "det${step}_g$g" $((PORT_BASE+g)) $g \
                "--enable-det-infer 3 --min-det-step-size $step --max-det-verify-batch-size 1"
            CONFIGS+=("det_infer_step${step}:$g"); g=$((g+1))
        done
    done
    
    for ((g=0; g<NUM_GPUS; g++)); do wait_for_server "http://localhost:$((PORT_BASE+g))" "gpu$g"; done
    
    # Run benchmarks (each GPU independently processes all its work in parallel)
    PIDS=()
    for entry in "${CONFIGS[@]}"; do
        config="${entry%:*}"; g="${entry#*:}"
        url="http://localhost:$((PORT_BASE+g))"
        ( 
          for rate in ${SG[$g]}; do for det in "${DET_RATIOS[@]}"; do run_bench "$config" "$url" sharegpt "$rate" "$det"; done; done
          for rate in ${AX[$g]}; do for det in "${DET_RATIOS[@]}"; do run_arxiv "$config" "$url" "$rate" "$det"; done; done
        ) & PIDS+=($!)
    done
    for p in "${PIDS[@]}"; do wait $p; done
    
    # Stop servers
    g=0
    for ((j=0; j<n; j++)); do
        local step=${steps[$j]} gpus=$(( (NUM_GPUS - g) / (n - j) ))
        for ((r=0; r<gpus; r++)); do stop_server "det${step}_g$g"; g=$((g+1)); done
    done
    wait_for_gpu_cleanup
}

# ============================================
# Main
# ============================================
echo "=============================================="
echo "Online Benchmark Suite"
echo "Model: $MODEL | GPUs: $NUM_GPUS | Prompts: $NUM_PROMPTS"
echo "=============================================="

[[ "$1" != "--start-servers" ]] && { echo "Usage: NUM_GPUS=4 $0 --start-servers"; exit 1; }

run_phase "Baseline" "baseline" ""
run_phase "GlobalDet" "global_det" "--enable-deterministic-inference 2"

for ((i=0; i<${#STEP_SIZES[@]}; i+=NUM_GPUS)); do
    batch=("${STEP_SIZES[@]:$i:$NUM_GPUS}")
    run_detinfer_batch "${batch[@]}"
done

echo -e "\n=============================================="
echo "Complete! Results: $RESULTS_FILE"
echo "=============================================="
