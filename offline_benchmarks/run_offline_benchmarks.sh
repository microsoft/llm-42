#!/bin/bash

# Offline Throughput Benchmarking Script
# Runs multiple configurations and saves results for plotting
# Supports parallel execution across multiple GPUs

set -e

# Configuration
MODEL_PATH="${SGLANG_TEST_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
TP_SIZE="${SGLANG_TP_SIZE:-1}"
ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-flashinfer}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"
NUM_GPUS="${NUM_GPUS:-4}"  # Number of GPUs to use in parallel

# Output directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/new_results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="${OUTPUT_DIR}/benchmark_results_${TIMESTAMP}.jsonl"

# Benchmark parameters
INPUT_LENS=(256 512 1024 2048)
OUTPUT_LENS=(128 256 512 1024)
DET_INFER_WINDOW_SIZES=(16 64 128 256 512)
DETERMINISTIC_RATIOS=(0.01 0.02 0.05 0.1 0.2 0.5 1.0)

# Determine Python command
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: Python not found"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Lock file for thread-safe writes to results file
LOCK_FILE="${OUTPUT_DIR}/.results_lock"

# Track background jobs for each GPU
declare -A GPU_PIDS
for ((i=0; i<NUM_GPUS; i++)); do
    GPU_PIDS[$i]=""
done

# Function to get an available GPU
get_available_gpu() {
    while true; do
        for ((i=0; i<NUM_GPUS; i++)); do
            if [ -z "${GPU_PIDS[$i]}" ] || ! kill -0 "${GPU_PIDS[$i]}" 2>/dev/null; then
                GPU_PIDS[$i]=""
                echo $i
                return
            fi
        done
        # All GPUs busy, wait a bit
        sleep 5
    done
}

# Function to wait for all GPU jobs to complete
wait_all_gpus() {
    for ((i=0; i<NUM_GPUS; i++)); do
        if [ -n "${GPU_PIDS[$i]}" ]; then
            wait "${GPU_PIDS[$i]}" 2>/dev/null || true
            GPU_PIDS[$i]=""
        fi
    done
}

echo "=============================================="
echo "Offline Throughput Benchmark Suite"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "TP Size: $TP_SIZE"
echo "Attention Backend: $ATTENTION_BACKEND"
echo "Num Prompts: $NUM_PROMPTS"
echo "Num GPUs (parallel): $NUM_GPUS"
echo "Results File: $RESULTS_FILE"
echo "=============================================="
echo ""

# Common server args (without tp-size, will be added per-GPU)
BASE_ARGS="--attention-backend $ATTENTION_BACKEND \
    --disable-radix-cache \
    --disable-chunked-prefix-cache \
    --disable-overlap-schedule"

# Function to run a single benchmark on a specific GPU
run_benchmark_on_gpu() {
    local gpu_id="$1"
    local config_name="$2"
    local server_args="$3"
    local input_len="$4"
    local output_len="$5"
    local det_ratio="$6"
    local extra_info="$7"
    
    local temp_result="${OUTPUT_DIR}/temp_result_gpu${gpu_id}_$$.json"
    local log_file="${OUTPUT_DIR}/log_gpu${gpu_id}_$$.log"
    
    echo "[GPU $gpu_id] Starting: $config_name, input=$input_len, output=$output_len, det_ratio=$det_ratio"
    
    # Run the benchmark with deterministic ratio on specific GPU
    CUDA_VISIBLE_DEVICES=$gpu_id $PYTHON_CMD -m sglang.bench_offline_throughput \
        --model-path "$MODEL_PATH" \
        --tp-size $TP_SIZE \
        $server_args \
        --dataset-name random \
        --random-input-len "$input_len" \
        --random-output-len "$output_len" \
        --num-prompts "$NUM_PROMPTS" \
        --result-filename "$temp_result" \
        --deterministic-ratio "$det_ratio" \
        --extra-request-body '{"ignore_eos": true}' \
        > "$log_file" 2>&1
    
    # Add metadata and append to results file (with lock for thread safety)
    if [ -f "$temp_result" ]; then
        (
            flock -x 200
            tail -1 "$temp_result" | $PYTHON_CMD -c "
import sys
import json

line = sys.stdin.read().strip()
if line:
    result = json.loads(line)
    result['config_name'] = '$config_name'
    result['input_len'] = $input_len
    result['output_len'] = $output_len
    result['deterministic_ratio'] = $det_ratio
    result['extra_info'] = '$extra_info'
    result['model_path'] = '$MODEL_PATH'
    result['tp_size'] = $TP_SIZE
    result['attention_backend'] = '$ATTENTION_BACKEND'
    result['gpu_id'] = $gpu_id
    print(json.dumps(result))
" >> "$RESULTS_FILE"
        ) 200>"$LOCK_FILE"
        rm -f "$temp_result"
    fi
    
    rm -f "$log_file"
    echo "[GPU $gpu_id] Completed: $config_name, input=$input_len, output=$output_len"
}

# Function to schedule a benchmark on the next available GPU
schedule_benchmark() {
    local config_name="$1"
    local server_args="$2"
    local input_len="$3"
    local output_len="$4"
    local det_ratio="$5"
    local extra_info="$6"
    
    local gpu_id=$(get_available_gpu)
    
    run_benchmark_on_gpu "$gpu_id" "$config_name" "$server_args" "$input_len" "$output_len" "$det_ratio" "$extra_info" &
    GPU_PIDS[$gpu_id]=$!
}

# ============================================
# Configuration 1: Default (baseline)
# ============================================
echo "========== Configuration 1: Default (Baseline) =========="
for det_ratio in "${DETERMINISTIC_RATIOS[@]}"; do
    echo "--- Deterministic Ratio: $det_ratio ---"
    for input_len in "${INPUT_LENS[@]}"; do
        for output_len in "${OUTPUT_LENS[@]}"; do
            schedule_benchmark \
                "default" \
                "$BASE_ARGS" \
                "$input_len" \
                "$output_len" \
                "$det_ratio" \
                "baseline"
        done
    done
done

# Wait for all baseline benchmarks to complete before next config
wait_all_gpus

# ============================================
# Configuration 2: enable-deterministic-inference 2
# ============================================
echo "========== Configuration 2: enable-deterministic-inference 2 =========="
for det_ratio in "${DETERMINISTIC_RATIOS[@]}"; do
    echo "--- Deterministic Ratio: $det_ratio ---"
    for input_len in "${INPUT_LENS[@]}"; do
        for output_len in "${OUTPUT_LENS[@]}"; do
            schedule_benchmark \
                "det_inference_2" \
                "$BASE_ARGS --enable-deterministic-inference 2" \
                "$input_len" \
                "$output_len" \
                "$det_ratio" \
                "global_deterministic"
        done
    done
done

# Wait for all det_inference_2 benchmarks to complete before next config
wait_all_gpus

# ============================================
# Configuration 3: enable-det-infer 3 with varying det-infer-window-size
# ============================================
echo "========== Configuration 3: enable-det-infer 3 =========="
for det_ratio in "${DETERMINISTIC_RATIOS[@]}"; do
    echo "--- Deterministic Ratio: $det_ratio ---"
    for det_step in "${DET_INFER_WINDOW_SIZES[@]}"; do
        echo "--- det-infer-window-size: $det_step ---"
        for input_len in "${INPUT_LENS[@]}"; do
            for output_len in "${OUTPUT_LENS[@]}"; do
                schedule_benchmark \
                    "det_infer_3_step${det_step}" \
                    "$BASE_ARGS --enable-det-infer 3 --det-infer-verify-batch-size 1 --det-infer-window-size $det_step" \
                    "$input_len" \
                    "$output_len" \
                    "$det_ratio" \
                    "det_infer_window_size=$det_step"
            done
        done
    done
done

# Wait for all remaining benchmarks to complete
wait_all_gpus

# Cleanup lock file
rm -f "$LOCK_FILE"

echo "=============================================="
echo "Benchmarking Complete!"
echo "Results saved to: $RESULTS_FILE"
echo "=============================================="

# Generate plots
echo ""
echo "Generating plots..."
$PYTHON_CMD "${SCRIPT_DIR}/plot_results.py" "$RESULTS_FILE" --output-dir "${OUTPUT_DIR}/plots_${TIMESTAMP}"

echo "Done!"
