#!/bin/bash
set -euo pipefail

# Run offline throughput benchmarks against pre-launched servers
# For non-det and global-det: run with det_ratio=1.0
# For detinfer configs: run with multiple det_ratios (0.02, 0.05, 0.1, 0.2, 0.5, 1.0)

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Server URLs (comma-separated)
BASE_URLS=${BASE_URLS:-"http://127.0.0.1:30005,http://127.0.0.1:30006,http://127.0.0.1:30007,http://127.0.0.1:30008"}
CONFIG_NAMES=${CONFIG_NAMES:-"sglang_non_deterministic,sglang_global_deterministic,detinfer_ws_32_bs_16,detinfer_ws_64_bs_8"}

# Benchmark parameters
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
TOKENIZER=${TOKENIZER:-}
NUM_PROMPTS=${NUM_PROMPTS:-1024}
DATASET_NAME=${DATASET_NAME:-sharegpt}  # sharegpt, random, or arxiv
DATASET_PATH=${DATASET_PATH:-}
SHAREGPT_CONTEXT_LEN=${SHAREGPT_CONTEXT_LEN:-16384}
RANDOM_INPUT_LEN=${RANDOM_INPUT_LEN:-1024}
RANDOM_OUTPUT_LEN=${RANDOM_OUTPUT_LEN:-128}
DETERMINISTIC_SEED=${DETERMINISTIC_SEED:-42}
BACKEND=${BACKEND:-sglang}

# Deterministic ratios for different config types
BASELINE_RATIOS="1.0"
DETINFER_RATIOS="0.02 0.05 0.1 0.2 0.5 1.0"

# Output directory structure: results/<config_name>/
RESULTS_ROOT="${ROOT}/results"
if [ "$DATASET_NAME" = "random" ]; then
    CONFIG_DIR_NAME="random_in${RANDOM_INPUT_LEN}_out${RANDOM_OUTPUT_LEN}"
else
    CONFIG_DIR_NAME="${DATASET_NAME}"
fi
OUTPUT_DIR="${RESULTS_ROOT}/${CONFIG_DIR_NAME}"
RESULTS_FILE="${OUTPUT_DIR}/benchmark_results.jsonl"

# Convert comma-separated strings to arrays
IFS=',' read -ra URLS_ARRAY <<< "$BASE_URLS"
IFS=',' read -ra CONFIG_ARRAY <<< "$CONFIG_NAMES"

NUM_SERVERS=${#URLS_ARRAY[@]}
NUM_CONFIGS=${#CONFIG_ARRAY[@]}
NUM_RUNS=$((NUM_SERVERS < NUM_CONFIGS ? NUM_SERVERS : NUM_CONFIGS))

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Offline Throughput Benchmark (v2)"
echo "=============================================="
echo "Model: $MODEL"
echo "Num Prompts: $NUM_PROMPTS"
echo "Dataset: $DATASET_NAME"
if [ "$DATASET_NAME" = "random" ]; then
    echo "  Input Length: $RANDOM_INPUT_LEN"
    echo "  Output Length: $RANDOM_OUTPUT_LEN"
else
    echo "  Context Length: $SHAREGPT_CONTEXT_LEN"
    echo "  Output Length: (from dataset)"
fi
echo "Baseline Ratios: $BASELINE_RATIOS"
echo "DetInfer Ratios: $DETINFER_RATIOS"
echo "Output Dir: $OUTPUT_DIR"
echo ""
echo "Config to Server Mapping:"
for ((i=0; i<NUM_RUNS; i++)); do
    echo "  ${CONFIG_ARRAY[$i]} -> ${URLS_ARRAY[$i]}"
done
echo "=============================================="
echo ""

# Check server health before starting
echo "Checking server health..."
ALL_HEALTHY=true
for ((i=0; i<NUM_RUNS; i++)); do
    URL="${URLS_ARRAY[$i]}"
    echo -n "  Checking $URL ... "
    
    RESPONSE=$(timeout 5 curl -s "${URL}/v1/models" 2>&1)
    if echo "$RESPONSE" | grep -q '"object":"list"'; then
        echo "✓"
    else
        echo "✗ (not responding)"
        ALL_HEALTHY=false
    fi
done

if [ "$ALL_HEALTHY" = false ]; then
    echo ""
    echo "ERROR: Some servers are not healthy. Please launch servers first:"
    echo "  ./launch_servers_parallel.sh"
    exit 1
fi

echo ""
echo "All servers healthy. Running benchmarks..."
echo ""

# Function to run benchmark for a single server
run_benchmark() {
    local url="$1"
    local config_name="$2"
    local det_ratio="$3"
    
    local temp_result="${OUTPUT_DIR}/temp_${config_name}_det${det_ratio}.jsonl"
    
    echo "[${config_name}] Running benchmark with det_ratio=$det_ratio..."
    
    # Build tokenizer arg if provided
    TOKENIZER_ARG=""
    if [[ -n "${TOKENIZER}" ]]; then
        TOKENIZER_ARG="--tokenizer ${TOKENIZER}"
    fi
    
    # Build dataset args based on dataset type
    if [ "$DATASET_NAME" = "random" ]; then
        DATASET_ARGS="--dataset-name random --random-input-len $RANDOM_INPUT_LEN --random-output-len $RANDOM_OUTPUT_LEN --random-range-ratio 1.0"
        INPUT_LEN_FOR_RESULT=$RANDOM_INPUT_LEN
        OUTPUT_LEN_FOR_RESULT=$RANDOM_OUTPUT_LEN
        EXTRA_BODY='{"ignore_eos": true, "temperature": 0}'
    else
        # sharegpt, arxiv, and other datasets
        DATASET_ARGS="--dataset-name ${DATASET_NAME} --sharegpt-context-len $SHAREGPT_CONTEXT_LEN"
        if [[ -n "${DATASET_PATH}" ]]; then
            DATASET_ARGS="$DATASET_ARGS --dataset-path ${DATASET_PATH}"
        fi
        INPUT_LEN_FOR_RESULT=0
        OUTPUT_LEN_FOR_RESULT=0
        EXTRA_BODY='{"ignore_eos": true, "temperature": 0}'
    fi
    
    python -m sglang.bench_serving \
        --backend "$BACKEND" \
        --base-url "$url" \
        --model "$MODEL" \
        $TOKENIZER_ARG \
        $DATASET_ARGS \
        --num-prompts "$NUM_PROMPTS" \
        --request-rate inf \
        --disable-stream \
        --deterministic-ratio "$det_ratio" \
        --deterministic-seed "$DETERMINISTIC_SEED" \
        --extra-request-body "$EXTRA_BODY" \
        --output-file "$temp_result" \
        --output-details \
        2>&1 | tee "${OUTPUT_DIR}/log_${config_name}_det${det_ratio}.log"
    
    # Extract metrics and append to results
    if [ -f "$temp_result" ]; then
        # Parse the JSONL output and add metadata including rollback stats
        python -c "
import json
import sys

with open('$temp_result', 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                result = json.loads(line)
                result['config_name'] = '$config_name'
                result['dataset_name'] = '$DATASET_NAME'
                result['input_len'] = $INPUT_LEN_FOR_RESULT
                result['output_len'] = $OUTPUT_LEN_FOR_RESULT
                result['deterministic_ratio'] = $det_ratio
                result['server_url'] = '$url'
                
                # Extract rollback stats from meta_info
                meta_info_list = result.get('meta_info', [])
                output_lens = result.get('output_lens', [])
                if meta_info_list:
                    det_num_rollbacks = [m.get('det_infer_num_rollbacks', 0) for m in meta_info_list if m]
                    det_tokens_rolled_back = [m.get('det_infer_tokens_rolled_back', 0) for m in meta_info_list if m]
                    
                    num_requests = len(det_num_rollbacks)
                    total_output_tokens = sum(output_lens) if output_lens else result.get('total_output_tokens', 0)
                    if num_requests > 0:
                        result['rollback_stats'] = {
                            'total_rollbacks': sum(det_num_rollbacks),
                            'total_tokens_rolled_back': sum(det_tokens_rolled_back),
                            'total_output_tokens': total_output_tokens,
                            'avg_rollbacks_per_request': sum(det_num_rollbacks) / num_requests,
                            'avg_tokens_rolled_back_per_request': sum(det_tokens_rolled_back) / num_requests,
                            'max_rollbacks_per_request': max(det_num_rollbacks) if det_num_rollbacks else 0,
                            'max_tokens_rolled_back_per_request': max(det_tokens_rolled_back) if det_tokens_rolled_back else 0,
                            'requests_with_rollbacks': sum(1 for x in det_num_rollbacks if x > 0),
                            'num_requests': num_requests,
                        }
                
                # Remove verbose fields to keep results file manageable
                for key in ['meta_info', 'generated_texts', 'output_ids', 'itls', 'errors']:
                    result.pop(key, None)
                
                print(json.dumps(result))
            except json.JSONDecodeError:
                pass
" >> "$RESULTS_FILE"
        rm -f "$temp_result"
    fi
    
    echo "[${config_name}] Completed det_ratio=$det_ratio"
}

# Function to run all benchmarks for a single server
run_server_benchmarks() {
    local url="$1"
    local config_name="$2"
    
    if [[ "$config_name" == *"detinfer"* ]]; then
        # DetInfer: run all ratios sequentially
        for ratio in $DETINFER_RATIOS; do
            run_benchmark "$url" "$config_name" "$ratio"
        done
    else
        # Baseline: run only with ratio 1.0
        run_benchmark "$url" "$config_name" "1.0"
    fi
}

# Run all servers in parallel - each server runs its own workload
echo "========== Running All Servers in Parallel =========="
echo "Baseline configs: det_ratio=1.0"
echo "DetInfer configs: det_ratios=$DETINFER_RATIOS"
echo ""

pids=()
for ((i=0; i<NUM_RUNS; i++)); do
    url="${URLS_ARRAY[$i]}"
    config_name="${CONFIG_ARRAY[$i]}"
    
    echo "Starting benchmarks for $config_name on $url..."
    run_server_benchmarks "$url" "$config_name" &
    pids+=($!)
done

# Wait for all servers to complete
echo ""
echo "Waiting for all servers to complete..."
for pid in "${pids[@]}"; do
    wait "$pid"
done

echo ""
echo "=============================================="
echo "Benchmarking Complete!"
echo "Results saved to: $RESULTS_FILE"
echo "=============================================="
# Export per-request data to CSV
CSV_FILE="${OUTPUT_DIR}/per_request_data.csv"
echo ""
echo "Exporting per-request data to CSV..."
python "${ROOT}/export_per_request_csv.py" \
    --input "$RESULTS_FILE" \
    --output "$CSV_FILE"

echo ""
echo "Per-request CSV: $CSV_FILE"
echo "Fields: config_name, dataset_name, det_ratio, request_idx, prompt_hash,"
echo "        is_deterministic, input_len, output_len, rollbacks, tokens_rolled_back,"
echo "        ttft_s, latency_s, error"