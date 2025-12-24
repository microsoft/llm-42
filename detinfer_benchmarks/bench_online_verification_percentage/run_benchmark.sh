#!/usr/bin/env bash
set -euo pipefail

# Verification Percentage Benchmark - Run Single Dataset Config
# Compares different determinism modes and mismatch percentages.
#
# Environment overrides:
#   BASE_URLS (comma-separated list, default: http://127.0.0.1:30010,http://127.0.0.1:30011,http://127.0.0.1:30012,http://127.0.0.1:30013)
#   CONFIG_NAMES (comma-separated list, default: default,global,detinfer_512_0pct,detinfer_512_5pct)
#   QPS (default: 12)
#   MODEL (default: meta-llama/Llama-3.1-8B-Instruct)
#   TOKENIZER (default: empty = same as model)
#   DATASET_NAME (default: random)
#   NUM_PROMPTS (default: 4096)
#   RANDOM_INPUT_LEN (default: 512)
#   RANDOM_OUTPUT_LEN (default: 1024)
#   SEED (default: 42)
#   EXTRA_REQUEST_BODY (default: '{"temperature":0}')
#   BACKEND (default: sglang)
#   OUTPUT_DIR (default: auto-generated)

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
WORKSPACE_ROOT=$(dirname "$(dirname "$ROOT")")
export PYTHONPATH="${PYTHONPATH:-}:${WORKSPACE_ROOT}/python"

# Parse configuration
BASE_URLS=${BASE_URLS:-"http://127.0.0.1:30010,http://127.0.0.1:30011,http://127.0.0.1:30012,http://127.0.0.1:30013"}
CONFIG_NAMES=${CONFIG_NAMES:-"default,global,detinfer_512_0pct,detinfer_512_5pct"}
QPS=${QPS:-12}
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
TOKENIZER=${TOKENIZER:-}
DATASET_NAME=${DATASET_NAME:-random}
NUM_PROMPTS=${NUM_PROMPTS:-4096}
RANDOM_INPUT_LEN=${RANDOM_INPUT_LEN:-512}
RANDOM_OUTPUT_LEN=${RANDOM_OUTPUT_LEN:-1024}
SEED=${SEED:-42}
SHAREGPT_CONTEXT_LEN=${SHAREGPT_CONTEXT_LEN:-16384}
EXTRA_REQUEST_BODY=${EXTRA_REQUEST_BODY:-'{"temperature":0}'}
BACKEND=${BACKEND:-sglang}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Set default output directory based on dataset type
OUTPUT_DIR=${OUTPUT_DIR:-"${ROOT}/results_in${RANDOM_INPUT_LEN}_out${RANDOM_OUTPUT_LEN}_qps${QPS}_n${NUM_PROMPTS}"}

# Convert comma-separated strings to arrays
IFS=',' read -ra URLS_ARRAY <<< "$BASE_URLS"
IFS=',' read -ra CONFIG_ARRAY <<< "$CONFIG_NAMES"

# Validate that we have at least 2 servers and configs
if [ ${#URLS_ARRAY[@]} -lt 2 ]; then
    echo "Error: Need at least 2 server URLs. Got: ${#URLS_ARRAY[@]}"
    exit 1
fi

if [ ${#CONFIG_ARRAY[@]} -lt 2 ]; then
    echo "Error: Need at least 2 config names. Got: ${#CONFIG_ARRAY[@]}"
    exit 1
fi

# Use the minimum of the two array sizes
NUM_SERVERS=${#URLS_ARRAY[@]}
NUM_CONFIGS=${#CONFIG_ARRAY[@]}
NUM_RUNS=$((NUM_SERVERS < NUM_CONFIGS ? NUM_SERVERS : NUM_CONFIGS))

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Verification Percentage Benchmark"
echo "=============================================="
echo "Configuration:"
echo "  Model:           $MODEL"
echo "  Dataset:         random (synthetic)"
echo "  Input Length:    $RANDOM_INPUT_LEN"
echo "  Output Length:   $RANDOM_OUTPUT_LEN"
echo "  Num Prompts:     $NUM_PROMPTS"
echo "  QPS:             $QPS (same for all servers)"
echo "  Seed:            $SEED"
echo "  Num Servers:     $NUM_RUNS"
echo "  Output Dir:      $OUTPUT_DIR"
echo ""
echo "Configurations being compared:"
echo "  - default: Non-deterministic baseline"
echo "  - global: Global deterministic mode"
echo "  - detinfer_512_0pct: DetInfer with 0% mismatches"
echo "  - detinfer_512_5pct: DetInfer with 5% mismatches"
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
echo "All servers healthy. Running verification percentage benchmark..."
echo ""

# Use the local compare script
COMPARE_SCRIPT="${ROOT}/compare_verification_percentage.py"

if [ ! -f "$COMPARE_SCRIPT" ]; then
    echo "Error: compare_verification_percentage.py not found at $COMPARE_SCRIPT"
    exit 1
fi

# Build command
cmd=(
    python "${COMPARE_SCRIPT}"
    --backend "${BACKEND}"
    --base-urls "${BASE_URLS}"
    --config-names "${CONFIG_NAMES}"
    --qps "${QPS}"
    --model "${MODEL}"
    --dataset-name "${DATASET_NAME}"
    --num-prompts "${NUM_PROMPTS}"
    --random-input-len "${RANDOM_INPUT_LEN}"
    --random-output-len "${RANDOM_OUTPUT_LEN}"
    --seed "${SEED}"
    --deterministic-ratio 1.0
    --output-dir "${OUTPUT_DIR}"
    --extra-request-body "${EXTRA_REQUEST_BODY}"
    --ignore-eos
)

if [[ -n "${TOKENIZER}" ]]; then
    cmd+=(--tokenizer "${TOKENIZER}")
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
    echo "Benchmark completed successfully!"
    echo "=============================================="
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    
    # Display summary if available
    SUMMARY_FILE="$OUTPUT_DIR/summary.json"
    if [ -f "$SUMMARY_FILE" ] && command -v jq &> /dev/null; then
        echo "Latency Summary by Configuration:"
        jq -r '.config_stats[] | "  \(.config_name): TTFT=\(.mean_ttft_ms // "N/A" | tostring | .[0:6])ms, TPOT=\(.mean_tpot_ms // "N/A" | tostring | .[0:5])ms, E2E=\(.mean_e2e_ms // "N/A" | tostring | .[0:8])ms"' "$SUMMARY_FILE" 2>/dev/null || true
        echo ""
    fi
    
    # Generate plots
    echo "Generating latency plots..."
    python "${ROOT}/plot_latency.py" --results-dir "${OUTPUT_DIR}" || true
else
    echo ""
    echo "=============================================="
    echo "ERROR: Benchmark failed with exit code $RESULT"
    echo "=============================================="
fi

exit $RESULT
