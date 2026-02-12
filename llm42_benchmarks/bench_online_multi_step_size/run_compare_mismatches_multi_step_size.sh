#!/usr/bin/env bash
set -euo pipefail

# Multi-Step-Size Mismatch Comparison Script
# Runs the same QPS (12) across servers with different step sizes and compares the mismatches.
#
# Environment overrides:
#   BASE_URLS (comma-separated list, default: http://127.0.0.1:30005,http://127.0.0.1:30006,http://127.0.0.1:30007,http://127.0.0.1:30008)
#   STEP_SIZES (comma-separated list, default: 32,64,128,256)
#   QPS (default: 12)
#   MODEL (default: meta-llama/Llama-3.1-8B-Instruct)
#   TOKENIZER (default: empty = same as model)
#   DATASET_PATH (optional local ShareGPT JSON)
#   NUM_PROMPTS (default: 49152)
#   SEED (default: 42)
#   SEQ_CONCURRENCY (default: 1)
#   EXTRA_REQUEST_BODY (default: '{"temperature":0}')
#   BACKEND (default: sglang)
#   OUTPUT_DIR (default: $ROOT/multi_step_size_out)

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${PYTHONPATH:-}:${ROOT}/python"

# Parse configuration
BASE_URLS=${BASE_URLS:-"http://127.0.0.1:30005,http://127.0.0.1:30006,http://127.0.0.1:30007,http://127.0.0.1:30008"}
STEP_SIZES=${STEP_SIZES:-"32,64,128,256"}
QPS=${QPS:-12}
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
TOKENIZER=${TOKENIZER:-}
DATASET_PATH=${DATASET_PATH:-}
NUM_PROMPTS=${NUM_PROMPTS:-16384}
SEED=${SEED:-42}
SEQ_CONCURRENCY=${SEQ_CONCURRENCY:-1}
SHAREGPT_CONTEXT_LEN=${SHAREGPT_CONTEXT_LEN:-16384}
EXTRA_REQUEST_BODY=${EXTRA_REQUEST_BODY:-'{"temperature":0}'}
BACKEND=${BACKEND:-sglang}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=${OUTPUT_DIR:-"${ROOT}/reqs${NUM_PROMPTS}_qps${QPS}_multi_step_size_${TIMESTAMP}"}

# Convert comma-separated strings to arrays
IFS=',' read -ra URLS_ARRAY <<< "$BASE_URLS"
IFS=',' read -ra STEP_SIZES_ARRAY <<< "$STEP_SIZES"

# Validate that we have at least 2 servers and step sizes
if [ ${#URLS_ARRAY[@]} -lt 2 ]; then
    echo "Error: Need at least 2 server URLs. Got: ${#URLS_ARRAY[@]}"
    exit 1
fi

if [ ${#STEP_SIZES_ARRAY[@]} -lt 2 ]; then
    echo "Error: Need at least 2 step sizes. Got: ${#STEP_SIZES_ARRAY[@]}"
    exit 1
fi

# Use the minimum of the two array sizes
NUM_SERVERS=${#URLS_ARRAY[@]}
NUM_STEP_SIZES=${#STEP_SIZES_ARRAY[@]}
NUM_RUNS=$((NUM_SERVERS < NUM_STEP_SIZES ? NUM_SERVERS : NUM_STEP_SIZES))

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Multi-Step-Size Mismatch Comparison"
echo "=============================================="
echo "Configuration:"
echo "  Model:           $MODEL"
echo "  Dataset:         ${DATASET_PATH:-ShareGPT (default)}"
echo "  Num Prompts:     $NUM_PROMPTS"
echo "  QPS:             $QPS (same for all servers)"
echo "  Seed:            $SEED"
echo "  Num Servers:     $NUM_RUNS"
echo "  Output Dir:      $OUTPUT_DIR"
echo ""
echo "Step Size to Server Mapping:"
for ((i=0; i<NUM_RUNS; i++)); do
    echo "  step_size=${STEP_SIZES_ARRAY[$i]} -> ${URLS_ARRAY[$i]}"
done
echo "=============================================="
echo ""

# Store PIDs for background jobs
declare -a PIDS=()
declare -a RUN_DIRS=()

# Function to cleanup on exit
cleanup() {
    if [ ${#PIDS[@]} -gt 0 ]; then
        echo ""
        echo "Cleaning up background processes..."
        for pid in "${PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill "$pid" 2>/dev/null || true
            fi
        done
        wait
    fi
}

trap cleanup EXIT INT TERM

# Check server health before starting
echo "Checking server health..."
ALL_HEALTHY=true
for ((i=0; i<NUM_RUNS; i++)); do
    URL="${URLS_ARRAY[$i]}"
    echo -n "  Checking $URL ... "
    
    # Try /v1/models endpoint with timeout and check if it returns JSON
    RESPONSE=$(timeout 5 curl -s "${URL}/v1/models" 2>&1)
    if echo "$RESPONSE" | grep -q '"object":"list"'; then
        echo "✓"
    else
        echo "✗ (not responding or not ready)"
        ALL_HEALTHY=false
    fi
done

if [ "$ALL_HEALTHY" = false ]; then
    echo ""
    echo "WARNING: Some servers are not healthy. Waiting 10 seconds for servers to initialize..."
    sleep 10
    
    echo "Rechecking server health..."
    ALL_HEALTHY=true
    for ((i=0; i<NUM_RUNS; i++)); do
        URL="${URLS_ARRAY[$i]}"
        echo -n "  Checking $URL ... "
        
        RESPONSE=$(timeout 5 curl -s "${URL}/v1/models" 2>&1)
        if echo "$RESPONSE" | grep -q '"object":"list"'; then
            echo "✓"
        else
            echo "✗ (still not responding)"
            ALL_HEALTHY=false
        fi
    done
    
    if [ "$ALL_HEALTHY" = false ]; then
        echo ""
        echo "ERROR: Some servers are still not healthy. Please check server logs."
        exit 1
    fi
fi

echo ""
echo "All servers healthy. Running multi-step-size comparison..."
echo ""

# Build command for direct step size comparison
cmd=(
    python "${ROOT}/compare_multi_step_size_outputs.py"
    --backend "${BACKEND}"
    --base-urls "${BASE_URLS}"
    --step-sizes "${STEP_SIZES}"
    --qps "${QPS}"
    --model "${MODEL}"
    --num-prompts "${NUM_PROMPTS}"
    --seed "${SEED}"
    --deterministic-ratio 1.0
    --output-dir "${OUTPUT_DIR}"
    --extra-request-body "${EXTRA_REQUEST_BODY}"
)

if [[ -n "${TOKENIZER}" ]]; then
    cmd+=(--tokenizer "${TOKENIZER}")
fi
if [[ -n "${DATASET_PATH}" ]]; then
    cmd+=(--dataset-path "${DATASET_PATH}")
fi
if [[ -n "${SHAREGPT_CONTEXT_LEN}" ]]; then
    cmd+=(--sharegpt-context-len "${SHAREGPT_CONTEXT_LEN}")
fi

# Run the comparison
echo "Command: ${cmd[*]}"
echo ""

"${cmd[@]}"
RESULT=$?

if [ $RESULT -eq 0 ]; then
    echo ""
    echo "=============================================="
    echo "Comparison completed successfully!"
    echo "=============================================="
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    
    # Display summary if available
    SUMMARY_FILE="$OUTPUT_DIR/summary.json"
    if [ -f "$SUMMARY_FILE" ] && command -v jq &> /dev/null; then
        echo "Pairwise Mismatch Summary:"
        jq -r '.pairwise_comparisons[] | "  step_size \(.step_size_1) vs step_size \(.step_size_2): \(.mismatch_fraction * 100 | round / 100)% mismatch"' "$SUMMARY_FILE"
        echo ""
        echo "Heatmap plot: $OUTPUT_DIR/mismatch_heatmap.pdf"
    fi
else
    echo ""
    echo "=============================================="
    echo "ERROR: Comparison failed with exit code $RESULT"
    echo "=============================================="
fi

exit $RESULT
