#!/bin/bash
set -euo pipefail

# Run batch size experiment: vary num_prompts (batch size) with fixed input/output lengths
# Compare global-deterministic vs detinfer (LLM-42)

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Server URLs - 3 servers: non-det, global-det, and detinfer
NON_DET_URL=${NON_DET_URL:-"http://127.0.0.1:30005"}
GLOBAL_DET_URL=${GLOBAL_DET_URL:-"http://127.0.0.1:30006"}
DETINFER_URL=${DETINFER_URL:-"http://127.0.0.1:30007"}

# Benchmark parameters
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
TOKENIZER=${TOKENIZER:-}
INPUT_LEN=${INPUT_LEN:-1024}
OUTPUT_LEN=${OUTPUT_LEN:-512}
DETERMINISTIC_SEED=${DETERMINISTIC_SEED:-42}
BACKEND=${BACKEND:-sglang}

# Batch sizes to test (powers of 2)
BATCH_SIZES=${BATCH_SIZES:-"2 4 8 16 32 64 128 256 512"}

# Output directory
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${ROOT}/results_batchsize_in${INPUT_LEN}_out${OUTPUT_LEN}_${TIMESTAMP}"
RESULTS_FILE="${OUTPUT_DIR}/benchmark_results.jsonl"

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Batch Size Experiment"
echo "=============================================="
echo "Model: $MODEL"
echo "Input Length: $INPUT_LEN"
echo "Output Length: $OUTPUT_LEN"
echo "Batch Sizes: $BATCH_SIZES"
echo "Non-Det Server: $NON_DET_URL"
echo "Global-Det Server: $GLOBAL_DET_URL"
echo "DetInfer Server: $DETINFER_URL"
echo "Output Dir: $OUTPUT_DIR"
echo "=============================================="
echo ""

# Check server health
echo "Checking server health..."
for URL in "$NON_DET_URL" "$GLOBAL_DET_URL" "$DETINFER_URL"; do
    echo -n "  Checking $URL ... "
    RESPONSE=$(timeout 5 curl -s "${URL}/v1/models" 2>&1)
    if echo "$RESPONSE" | grep -q '"object":"list"'; then
        echo "✓"
    else
        echo "✗ (not responding)"
        echo "ERROR: Server not healthy. Please launch servers first."
        exit 1
    fi
done
echo ""

# Build tokenizer arg if provided
TOKENIZER_ARG=""
if [[ -n "${TOKENIZER}" ]]; then
    TOKENIZER_ARG="--tokenizer ${TOKENIZER}"
fi

# Function to run single benchmark
run_benchmark() {
    local url="$1"
    local config_name="$2"
    local batch_size="$3"
    
    local temp_result="${OUTPUT_DIR}/temp_${config_name}_bs${batch_size}.jsonl"
    
    echo "[${config_name}] Running batch_size=$batch_size..."
    
    python -m sglang.bench_serving \
        --backend "$BACKEND" \
        --base-url "$url" \
        --model "$MODEL" \
        $TOKENIZER_ARG \
        --dataset-name random \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --num-prompts "$batch_size" \
        --request-rate inf \
        --deterministic-ratio 1.0 \
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

# Run benchmarks for each batch size
for batch_size in $BATCH_SIZES; do
    echo ""
    echo "========== Batch Size: $batch_size =========="
    
    # Run all three configs in parallel for same batch size
    run_benchmark "$NON_DET_URL" "non_det" "$batch_size" &
    pid1=$!
    run_benchmark "$GLOBAL_DET_URL" "global_det" "$batch_size" &
    pid2=$!
    run_benchmark "$DETINFER_URL" "detinfer" "$batch_size" &
    pid3=$!
    
    wait $pid1
    wait $pid2
    wait $pid3
done

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
