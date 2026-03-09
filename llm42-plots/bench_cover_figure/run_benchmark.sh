#!/bin/bash
set -euo pipefail

# Run batch size experiment: batch sizes 10, 11 with input 1024, output 512
# Compare non-deterministic vs global-deterministic

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Server configuration
NON_DET_PORT=${NON_DET_PORT:-30005}
GLOBAL_DET_PORT=${GLOBAL_DET_PORT:-30006}
LLM42_PORT=${LLM42_PORT:-30007}
NON_DET_URL="http://127.0.0.1:${NON_DET_PORT}"
GLOBAL_DET_URL="http://127.0.0.1:${GLOBAL_DET_PORT}"
LLM42_URL="http://127.0.0.1:${LLM42_PORT}"

# GPU assignment (use different GPUs for each server)
NON_DET_GPU=${NON_DET_GPU:-0}
GLOBAL_DET_GPU=${GLOBAL_DET_GPU:-1}
LLM42_GPU=${LLM42_GPU:-2}

# Model and server parameters
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
TP_SIZE=${TP_SIZE:-1}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-fa3}

# Benchmark parameters
INPUT_LEN=512
OUTPUT_LEN=256
DETERMINISTIC_SEED=42
BACKEND=sglang

# Batch sizes to test
BATCH_SIZES="11"

# Output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${ROOT}/results_in${INPUT_LEN}_out${OUTPUT_LEN}_${TIMESTAMP}"
RESULTS_FILE="${OUTPUT_DIR}/benchmark_results.jsonl"

mkdir -p "$OUTPUT_DIR"

# PIDs for cleanup
NON_DET_PID=""
GLOBAL_DET_PID=""
LLM42_PID=""

# Function to check server health
check_server_health() {
    local url="$1"
    local max_retries="${2:-120}"
    local retry_interval="${3:-5}"
    
    for ((i=1; i<=max_retries; i++)); do
        if curl -s "${url}/v1/models" 2>/dev/null | grep -q '"object":"list"'; then
            return 0
        fi
        sleep "$retry_interval"
    done
    return 1
}

# Function to launch a server
launch_server() {
    local port="$1"
    local config_name="$2"
    local config_args="$3"
    local gpu_id="$4"
    local log_file="${OUTPUT_DIR}/server_${config_name}.log"
    
    echo "Launching $config_name server on port $port (GPU $gpu_id)..."
    echo "  Config args: $config_args"
    echo "  Log file: $log_file"
    
    python -m sglang.launch_server \
        --model-path "$MODEL" \
        --host 0.0.0.0 \
        --port "$port" \
        --tp "$TP_SIZE" \
        --base-gpu-id "$gpu_id" \
        --attention-backend "$ATTENTION_BACKEND" \
        --disable-radix-cache \
        --disable-chunked-prefix-cache \
        --disable-overlap-schedule \
        --enable-metrics \
        --random-seed 42 \
        --chunked-prefill-size -1 \
        --max-running-requests 64 \
        $config_args \
        > "$log_file" 2>&1 &
    
    echo $!
}

# Flag to prevent double cleanup
CLEANUP_DONE=false

# Function to stop servers
stop_servers() {
    if [ "$CLEANUP_DONE" = true ]; then
        return
    fi
    CLEANUP_DONE=true
    
    echo ""
    echo "Stopping servers..."
    for pid in $NON_DET_PID $GLOBAL_DET_PID $LLM42_PID; do
        if [ -n "$pid" ]; then
            echo "  Killing PID $pid and children..."
            pkill -P "$pid" 2>/dev/null || true
            kill -9 "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    # Also kill any remaining sglang processes on our ports
    fuser -k ${NON_DET_PORT}/tcp 2>/dev/null || true
    fuser -k ${GLOBAL_DET_PORT}/tcp 2>/dev/null || true
    fuser -k ${LLM42_PORT}/tcp 2>/dev/null || true
    sleep 2
    echo "Servers stopped."
}

# Handle Ctrl+C gracefully
handle_sigint() {
    echo ""
    echo "Caught Ctrl+C, cleaning up..."
    stop_servers
    exit 130
}

# Cleanup on exit
trap handle_sigint INT
trap stop_servers EXIT TERM

echo "=============================================="
echo "Batch Size Experiment"
echo "=============================================="
echo "Model: $MODEL"
echo "TP Size: $TP_SIZE"
echo "Input Length: $INPUT_LEN"
echo "Output Length: $OUTPUT_LEN"
echo "Batch Sizes: $BATCH_SIZES"
echo "Non-Det Server: $NON_DET_URL (GPU $NON_DET_GPU)"
echo "Global-Det Server: $GLOBAL_DET_URL (GPU $GLOBAL_DET_GPU)"
echo "LLM42 Server: $LLM42_URL (GPU $LLM42_GPU)"
echo "Output Dir: $OUTPUT_DIR"
echo "=============================================="
echo ""

# Launch servers
echo "========== Launching Servers =========="

# Non-deterministic server (no special args) on GPU 0
NON_DET_PID=$(launch_server "$NON_DET_PORT" "non_det" "" "$NON_DET_GPU")
echo "  Non-Det PID: $NON_DET_PID"

# Global-deterministic server on GPU 1
GLOBAL_DET_PID=$(launch_server "$GLOBAL_DET_PORT" "global_det" "--enable-deterministic-inference 2" "$GLOBAL_DET_GPU")
echo "  Global-Det PID: $GLOBAL_DET_PID"

# LLM42 server on GPU 2 (ws=64, bs=8)
LLM42_PID=$(launch_server "$LLM42_PORT" "llm42" "--enable-llm42 3 --llm42-window-size 64 --llm42-verify-batch-size 8" "$LLM42_GPU")
echo "  LLM42 PID: $LLM42_PID"

# Wait for servers to be ready
echo ""
echo "Waiting for servers to be ready..."

echo -n "  Checking Non-Det server... "
if check_server_health "$NON_DET_URL" 120 5; then
    echo "✓"
else
    echo "✗ FAILED"
    echo "ERROR: Non-Det server failed to start. Check log: ${OUTPUT_DIR}/server_non_det.log"
    exit 1
fi

echo -n "  Checking Global-Det server... "
if check_server_health "$GLOBAL_DET_URL" 120 5; then
    echo "✓"
else
    echo "✗ FAILED"
    echo "ERROR: Global-Det server failed to start. Check log: ${OUTPUT_DIR}/server_global_det.log"
    exit 1
fi

echo -n "  Checking LLM42 server... "
if check_server_health "$LLM42_URL" 120 5; then
    echo "✓"
else
    echo "✗ FAILED"
    echo "ERROR: LLM42 server failed to start. Check log: ${OUTPUT_DIR}/server_llm42.log"
    exit 1
fi

echo "All servers ready!"
echo ""

# Function to run single benchmark
run_benchmark() {
    local url="$1"
    local config_name="$2"
    local batch_size="$3"
    local det_ratio="${4:-1.0}"  # Default to 1.0 for global-det
    
    local temp_result="${OUTPUT_DIR}/temp_${config_name}_bs${batch_size}.jsonl"
    
    echo "[${config_name}] Running batch_size=$batch_size, det_ratio=$det_ratio..."
    
    python -m sglang.bench_serving \
        --backend "$BACKEND" \
        --base-url "$url" \
        --model "$MODEL" \
        --dataset-name random \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --random-range-ratio 1.0 \
        --num-prompts "$batch_size" \
        --request-rate inf \
        --deterministic-ratio "$det_ratio" \
        --deterministic-seed "$DETERMINISTIC_SEED" \
        --extra-request-body '{"ignore_eos": true, "temperature": 0}' \
        --output-file "$temp_result" \
        --output-details \
        2>&1 | tee "${OUTPUT_DIR}/log_${config_name}_bs${batch_size}.log"
    
    # Extract metrics and append to results
    if [ -f "$temp_result" ]; then
        python -c "
import json
with open('$temp_result', 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                result = json.loads(line)
                result['config_name'] = '$config_name'
                result['batch_size'] = $batch_size
                result['input_len'] = $INPUT_LEN
                result['output_len'] = $OUTPUT_LEN
                
                # Remove verbose fields
                for key in ['meta_info', 'generated_texts', 'output_ids', 'itls', 'errors']:
                    result.pop(key, None)
                
                print(json.dumps(result))
            except json.JSONDecodeError:
                pass
" >> "$RESULTS_FILE"
        rm -f "$temp_result"
    fi
    
    echo "[${config_name}] Completed batch_size=$batch_size"
}

# Run benchmarks
echo ""
echo "========== Batch Size: 10 (Non-Det only) =========="
run_benchmark "$NON_DET_URL" "non_det" "10" "1.0"

echo ""
echo "========== Batch Size: 11 (All configs) =========="
# Run all 3 configs in parallel
# Non-det: det_ratio doesn't matter (no deterministic processing)
# Global-det: det_ratio=1.0 (all deterministic)
# LLM42: det_ratio=0.09 (1 out of 11 requests deterministic)
run_benchmark "$NON_DET_URL" "non_det" "11" "1.0" &
pid1=$!
run_benchmark "$GLOBAL_DET_URL" "global_det" "11" "1.0" &
pid2=$!
run_benchmark "$LLM42_URL" "llm42" "11" "0.09" &
pid3=$!

wait $pid1
wait $pid2
wait $pid3

echo ""
echo "=============================================="
echo "Experiment Complete!"
echo "Results saved to: $RESULTS_FILE"
echo "=============================================="

# Generate plot
echo ""
echo "Generating plot..."
python "${ROOT}/plot_batchsize_throughput.py" \
    --input "$RESULTS_FILE" \
    --output "${OUTPUT_DIR}/throughput_vs_batchsize.pdf"

echo "Plot saved to: ${OUTPUT_DIR}/throughput_vs_batchsize.pdf"
