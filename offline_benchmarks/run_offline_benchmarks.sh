#!/bin/bash

# Offline Throughput Benchmarking Script
# Runs multiple configurations and saves results for plotting

set -e

# Configuration
MODEL_PATH="${SGLANG_TEST_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
TP_SIZE="${SGLANG_TP_SIZE:-1}"
ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-flashinfer}"
NUM_PROMPTS="${NUM_PROMPTS:-1000}"

# Output directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_DIR="${SCRIPT_DIR}/results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="${OUTPUT_DIR}/benchmark_results_${TIMESTAMP}.jsonl"

# Benchmark parameters
INPUT_LENS=(512 1024 2048)
OUTPUT_LENS=(128 256 512 1024)
MIN_DET_STEP_SIZES=(16 64 128)

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

echo "=============================================="
echo "Offline Throughput Benchmark Suite"
echo "=============================================="
echo "Model: $MODEL_PATH"
echo "TP Size: $TP_SIZE"
echo "Attention Backend: $ATTENTION_BACKEND"
echo "Num Prompts: $NUM_PROMPTS"
echo "Results File: $RESULTS_FILE"
echo "=============================================="
echo ""

# Common server args
COMMON_ARGS="--tp-size $TP_SIZE \
    --attention-backend $ATTENTION_BACKEND \
    --disable-radix-cache \
    --disable-chunked-prefix-cache \
    --disable-overlap-schedule "

# Function to run a single benchmark
run_benchmark() {
    local config_name="$1"
    local server_args="$2"
    local input_len="$3"
    local output_len="$4"
    local extra_info="$5"
    
    echo "----------------------------------------------"
    echo "Config: $config_name"
    echo "Input Len: $input_len, Output Len: $output_len"
    echo "Extra: $extra_info"
    echo "----------------------------------------------"
    
    local temp_result="${OUTPUT_DIR}/temp_result.json"
    
    # Run the benchmark with is_deterministic=True in extra request body
    $PYTHON_CMD -m sglang.bench_offline_throughput \
        --model-path "$MODEL_PATH" \
        $server_args \
        --dataset-name random \
        --random-input-len "$input_len" \
        --random-output-len "$output_len" \
        --num-prompts "$NUM_PROMPTS" \
        --result-filename "$temp_result" \
        --extra-request-body '{"is_deterministic": true}'
    
    # Add metadata and append to results file
    if [ -f "$temp_result" ]; then
        # Read the last line (most recent result) and add metadata
        tail -1 "$temp_result" | $PYTHON_CMD -c "
import sys
import json

line = sys.stdin.read().strip()
if line:
    result = json.loads(line)
    result['config_name'] = '$config_name'
    result['input_len'] = $input_len
    result['output_len'] = $output_len
    result['extra_info'] = '$extra_info'
    result['model_path'] = '$MODEL_PATH'
    result['tp_size'] = $TP_SIZE
    result['attention_backend'] = '$ATTENTION_BACKEND'
    print(json.dumps(result))
" >> "$RESULTS_FILE"
        rm -f "$temp_result"
    fi
    
    echo ""
}

# ============================================
# Configuration 1: Default (baseline)
# ============================================
echo "========== Configuration 1: Default (Baseline) =========="
for input_len in "${INPUT_LENS[@]}"; do
    for output_len in "${OUTPUT_LENS[@]}"; do
        run_benchmark \
            "default" \
            "$COMMON_ARGS" \
            "$input_len" \
            "$output_len" \
            "baseline"
    done
done

# ============================================
# Configuration 2: enable-deterministic-inference 2
# ============================================
echo "========== Configuration 2: enable-deterministic-inference 2 =========="
for input_len in "${INPUT_LENS[@]}"; do
    for output_len in "${OUTPUT_LENS[@]}"; do
        run_benchmark \
            "det_inference_2" \
            "$COMMON_ARGS --enable-deterministic-inference 2" \
            "$input_len" \
            "$output_len" \
            "global_deterministic"
    done
done

# ============================================
# Configuration 3: enable-det-infer 3 with varying min-det-step-size
# ============================================
echo "========== Configuration 3: enable-det-infer 3 =========="
for min_det_step in "${MIN_DET_STEP_SIZES[@]}"; do
    echo "--- min-det-step-size: $min_det_step ---"
    for input_len in "${INPUT_LENS[@]}"; do
        for output_len in "${OUTPUT_LENS[@]}"; do
            run_benchmark \
                "det_infer_3_step${min_det_step}" \
                "$COMMON_ARGS --enable-det-infer 3 --max-det-verify-batch-size 1 --min-det-step-size $min_det_step" \
                "$input_len" \
                "$output_len" \
                "min_det_step_size=$min_det_step"
        done
    done
done

echo "=============================================="
echo "Benchmarking Complete!"
echo "Results saved to: $RESULTS_FILE"
echo "=============================================="

# Generate plots
echo ""
echo "Generating plots..."
$PYTHON_CMD "${SCRIPT_DIR}/plot_results.py" "$RESULTS_FILE" --output-dir "${OUTPUT_DIR}/plots_${TIMESTAMP}"

echo "Done!"
