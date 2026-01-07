#!/usr/bin/env bash
set -euo pipefail

# Multi-Config Comparison Script
# Runs multiple (QPS, order_seed, arrival_seed) configurations and compares outputs by prompt hash.
# Uses the same prompts (via select_seed) but different arrival orders (via order_seed).
#
# Environment overrides:
#   BASE_URLS (comma-separated list, default: http://127.0.0.1:30005,http://127.0.0.1:30006,http://127.0.0.1:30007,http://127.0.0.1:30008)
#   CONFIGS (semicolon-separated, e.g., "qps=6,order=40;qps=6,order=242;qps=12,order=34;qps=12,order=123")
#   SELECT_SEED (default: 42) - Same for all configs to get same prompts
#   MODEL (default: meta-llama/Llama-3.1-8B-Instruct)
#   TOKENIZER (default: empty = same as model)
#   DATASET_PATH (optional local ShareGPT JSON)
#   NUM_PROMPTS (default: 100)
#   EXTRA_REQUEST_BODY (default: '{"temperature":0}')
#   BACKEND (default: sglang)
#   OUTPUT_DIR (default: $ROOT/multi_config_compare_out_<timestamp>)

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${PYTHONPATH:-}:${ROOT}/python"

# Parse configuration
BASE_URLS=${BASE_URLS:-"http://127.0.0.1:30005,http://127.0.0.1:30006,http://127.0.0.1:30007,http://127.0.0.1:30008"}

# Generate CONFIGS if not provided
# Configurable parameters for auto-generation:
#   NUM_CONFIGS_TO_GENERATE (default: 30)
#   QPS_START (default: 4.0) - Starting QPS value
#   QPS_STEP (default: 0.5) - QPS increment per config
#   ORDER_SEED_START (default: 130) - Starting seed for order
#   ARRIVAL_SEED_START (default: 10) - Starting seed for arrival
if [[ -z "${CONFIGS:-}" ]]; then
    NUM_CONFIGS_TO_GENERATE=${NUM_CONFIGS_TO_GENERATE:-16}
    QPS_START=${QPS_START:-5.0}
    QPS_STEP=${QPS_STEP:-1.0}
    ORDER_SEED_START=${ORDER_SEED_START:-130}
    ARRIVAL_SEED_START=${ARRIVAL_SEED_START:-10}
    
    CONFIGS=""
    for ((i=0; i<NUM_CONFIGS_TO_GENERATE; i++)); do
        # Calculate QPS using awk for floating point arithmetic
        qps=$(awk "BEGIN {printf \"%.1f\", $QPS_START + $i * $QPS_STEP}" | sed 's/\.0$//')
        order_seed=$((ORDER_SEED_START + i))
        arrival_seed=$((ARRIVAL_SEED_START + i))
        if [ -n "$CONFIGS" ]; then
            CONFIGS="${CONFIGS};"
        fi
        CONFIGS="${CONFIGS}qps=${qps},order=${order_seed},arrival=${arrival_seed}"
    done
    echo "Auto-generated $NUM_CONFIGS_TO_GENERATE configs with QPS from $QPS_START (step $QPS_STEP)"
fi

SELECT_SEED=${SELECT_SEED:-42}
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
TOKENIZER=${TOKENIZER:-}
DATASET_PATH=${DATASET_PATH:-}
NUM_PROMPTS_ARRAY=${NUM_PROMPTS_ARRAY:-"9999,92812"}
SHAREGPT_CONTEXT_LEN=${SHAREGPT_CONTEXT_LEN:-16384}
EXTRA_REQUEST_BODY=${EXTRA_REQUEST_BODY:-'{"temperature":0}'}
BACKEND=${BACKEND:-sglang}
DETERMINISTIC_RATIO=${DETERMINISTIC_RATIO:-1.0}
WARMUP_REQUESTS=${WARMUP_REQUESTS:-0}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT_DIR=${OUTPUT_DIR:-"${ROOT}/fi_stream_di3_s64_bs8_t0_${TIMESTAMP}"}

# Convert comma-separated strings to arrays for display
IFS=',' read -ra URLS_ARRAY <<< "$BASE_URLS"
IFS=';' read -ra CONFIGS_ARRAY <<< "$CONFIGS"
IFS=',' read -ra NUM_PROMPTS_ARRAY <<< "$NUM_PROMPTS_ARRAY"

NUM_SERVERS=${#URLS_ARRAY[@]}
NUM_CONFIGS=${#CONFIGS_ARRAY[@]}

mkdir -p "$BASE_OUTPUT_DIR"

echo "=============================================="
echo "Multi-Config Comparison"
echo "=============================================="
echo "Configuration:"
echo "  Model:           $MODEL"
echo "  Dataset:         ${DATASET_PATH:-ShareGPT (default)}"
echo "  Num Prompts:     ${NUM_PROMPTS_ARRAY[*]}"
echo "  Select Seed:     $SELECT_SEED (same prompts for all configs)"
echo "  Num Servers:     $NUM_SERVERS"
echo "  Num Configs:     $NUM_CONFIGS"
echo "  Output Dir:      $BASE_OUTPUT_DIR"
echo ""
echo "Configs to run:"
for config in "${CONFIGS_ARRAY[@]}"; do
    echo "  - $config"
done
echo "=============================================="
echo ""

# Check server health before starting
echo "Checking server health..."
ALL_HEALTHY=true
for ((i=0; i<NUM_SERVERS; i++)); do
    URL="${URLS_ARRAY[$i]}"
    echo -n "  Checking $URL ... "
    
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
    for ((i=0; i<NUM_SERVERS; i++)); do
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
echo "All servers healthy. Running multi-config comparison..."
echo ""

# Loop through each NUM_PROMPTS value
for NUM_PROMPTS in "${NUM_PROMPTS_ARRAY[@]}"; do
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/reqs_${NUM_PROMPTS}"
    mkdir -p "$OUTPUT_DIR"

    echo "=============================================="
    echo "Running with NUM_PROMPTS=$NUM_PROMPTS"
    echo "Output: $OUTPUT_DIR"
    echo "=============================================="
    echo "Running $NUM_CONFIGS configs across $NUM_SERVERS servers"
    echo "(Python script will batch automatically if needed)"
    echo ""

    # Build command - Python script handles batching internally
    cmd=(
        python "${ROOT}/compare_multi_config_outputs.py"
        --backend "${BACKEND}"
        --base-urls "${BASE_URLS}"
        --configs "${CONFIGS}"
        --select-seed "${SELECT_SEED}"
        --model "${MODEL}"
        --num-prompts "${NUM_PROMPTS}"
        --deterministic-ratio "${DETERMINISTIC_RATIO}"
        --output-dir "${OUTPUT_DIR}"
        --extra-request-body "${EXTRA_REQUEST_BODY}"
        --warmup-requests "${WARMUP_REQUESTS}"
        --ignore-eos
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
        echo "Multi-config comparison completed successfully!"
        echo "=============================================="
        echo "Results saved to: $OUTPUT_DIR"
        echo ""
        echo "Output files:"
        echo "  - comparison_summary.txt (human-readable summary)"
        echo "  - comparison_detailed.json (full JSON results with all $NUM_CONFIGS configs)"
        echo "  - config_*_summary.log (per-config summary logs)"
        echo "  - config_*_detailed.json (per-config detailed JSON)"
    else
        echo ""
        echo "=============================================="
        echo "ERROR: Multi-config comparison failed with exit code $RESULT"
        echo "=============================================="
        exit $RESULT
    fi
done

exit 0
