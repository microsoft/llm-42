#!/usr/bin/env bash
set -euo pipefail

# Multi-QPS Mismatch Comparison Script
# Runs multiple QPS values sequentially on a single server and compares the mismatches.
# Designed for MoE models requiring TP=4 where only 1 server can run on 4 GPUs.
#
# Environment overrides:
#   BASE_URLS (comma-separated list, default: http://127.0.0.1:30005 - single server for MoE TP=4)
#   QPS_VALUES (comma-separated list, default: 8,11,16,21,...)
#   MODEL (default: Qwen/Qwen3-30B-A3B)
#   TOKENIZER (default: empty = same as model)
#   DATASET_PATH (optional local ShareGPT JSON)
#   NUM_PROMPTS (default: 800)
#   SEED (default: 42)
#   SEQ_CONCURRENCY (default: 1)
#   EXTRA_REQUEST_BODY (default: '{"temperature":0}')
#   BACKEND (default: sglang)
#   OUTPUT_DIR (default: $ROOT/multi_qps_compare_out)

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${PYTHONPATH:-}:${ROOT}/python"

# Parse configuration
# Single server setup for MoE model with TP=4
BASE_URLS=${BASE_URLS:-"http://127.0.0.1:30005"}
QPS_VALUES=${QPS_VALUES:-"8,11,16,21,12,18,25,30,32,36,40,42,6,9,10,14,20,28,34,38"}  # Comma-separated list of QPS values (run sequentially)
MODEL=${MODEL:-Qwen/Qwen3-30B-A3B}
TOKENIZER=${TOKENIZER:-}
DATASET_PATH=${DATASET_PATH:-}
NUM_PROMPTS_LIST=${NUM_PROMPTS_LIST:-"60000"}  # Comma-separated list of num_prompts values
SEED=${SEED:-42}
SEQ_CONCURRENCY=${SEQ_CONCURRENCY:-1}
SHAREGPT_CONTEXT_LEN=${SHAREGPT_CONTEXT_LEN:-16384}
EXTRA_REQUEST_BODY=${EXTRA_REQUEST_BODY:-'{"temperature":0}'}
BACKEND=${BACKEND:-sglang}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BASE_OUTPUT_DIR=${BASE_OUTPUT_DIR:-"${ROOT}/no_stream_temp0_di3_s43_bs17_fa3_${TIMESTAMP}"}

# Convert comma-separated strings to arrays
IFS=',' read -ra URLS_ARRAY <<< "$BASE_URLS"
IFS=',' read -ra QPS_ARRAY <<< "$QPS_VALUES"
IFS=',' read -ra NUM_PROMPTS_ARRAY <<< "$NUM_PROMPTS_LIST"

# Validate that we have at least 1 server and 2 QPS values
if [ ${#URLS_ARRAY[@]} -lt 1 ]; then
    echo "Error: Need at least 1 server URL. Got: ${#URLS_ARRAY[@]}"
    exit 1
fi

if [ ${#QPS_ARRAY[@]} -lt 2 ]; then
    echo "Error: Need at least 2 QPS values. Got: ${#QPS_ARRAY[@]}"
    exit 1
fi

# Note: With single server, QPS values will be run sequentially
if [ ${#URLS_ARRAY[@]} -eq 1 ]; then
    echo "Note: Running with single server - QPS values will be tested sequentially"
fi

# Calculate batching for QPS values
NUM_SERVERS=${#URLS_ARRAY[@]}
NUM_QPS=${#QPS_ARRAY[@]}
NUM_BATCHES=$(( (NUM_QPS + NUM_SERVERS - 1) / NUM_SERVERS ))  # Ceiling division

mkdir -p "$BASE_OUTPUT_DIR"

echo "=============================================="
echo "Multi-QPS Mismatch Comparison"
echo "=============================================="
echo "Configuration:"
echo "  Model:           $MODEL"
echo "  Dataset:         ${DATASET_PATH:-ShareGPT (default)}"
echo "  Num Prompts:     ${NUM_PROMPTS_LIST} (${#NUM_PROMPTS_ARRAY[@]} runs)"
echo "  Seed:            $SEED"
echo "  Num Servers:     $NUM_SERVERS"
echo "  Num QPS Values:  $NUM_QPS"
echo "  Num Batches:     $NUM_BATCHES"
echo "  Base Output Dir: $BASE_OUTPUT_DIR"
echo ""
echo "QPS Batching:"
for ((batch=0; batch<NUM_BATCHES; batch++)); do
    start_idx=$((batch * NUM_SERVERS))
    end_idx=$((start_idx + NUM_SERVERS))
    if [ $end_idx -gt $NUM_QPS ]; then
        end_idx=$NUM_QPS
    fi
    batch_qps=""
    for ((i=start_idx; i<end_idx; i++)); do
        if [ -n "$batch_qps" ]; then
            batch_qps+=","
        fi
        batch_qps+="${QPS_ARRAY[$i]}"
    done
    echo "  Batch $((batch+1)): QPS values [$batch_qps]"
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
for ((i=0; i<NUM_SERVERS; i++)); do
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
echo "All servers healthy. Running multi-QPS comparison..."
echo ""

# Track overall success
OVERALL_RESULT=0

# Loop through each NUM_PROMPTS value
for NUM_PROMPTS in "${NUM_PROMPTS_ARRAY[@]}"; do
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/reqs_${NUM_PROMPTS}"
    mkdir -p "$OUTPUT_DIR"
    
    echo "=============================================="
    echo "Running with NUM_PROMPTS=$NUM_PROMPTS"
    echo "Output: $OUTPUT_DIR"
    echo "=============================================="
    
    # Loop through batches of QPS values
    for ((batch=0; batch<NUM_BATCHES; batch++)); do
        # Calculate which QPS values are in this batch
        start_idx=$((batch * NUM_SERVERS))
        end_idx=$((start_idx + NUM_SERVERS))
        if [ $end_idx -gt $NUM_QPS ]; then
            end_idx=$NUM_QPS
        fi
        
        # Build comma-separated QPS values for this batch
        batch_qps=""
        batch_urls=""
        batch_size=$((end_idx - start_idx))
        for ((i=start_idx; i<end_idx; i++)); do
            if [ -n "$batch_qps" ]; then
                batch_qps+=","
                batch_urls+=","
            fi
            batch_qps+="${QPS_ARRAY[$i]}"
            batch_urls+="${URLS_ARRAY[$((i - start_idx))]}"
        done
        
        BATCH_OUTPUT_DIR="${OUTPUT_DIR}/batch_$((batch+1))"
        mkdir -p "$BATCH_OUTPUT_DIR"
        
        echo ""
        echo "--- Batch $((batch+1))/$NUM_BATCHES: QPS values [$batch_qps] ---"
        
        # Build command for direct QPS comparison 
        cmd=(
            python "${ROOT}/compare_multi_qps_outputs.py"
            --backend "${BACKEND}"
            --base-urls "${batch_urls}"
            --qps-values "${batch_qps}"
            --model "${MODEL}"
            --num-prompts "${NUM_PROMPTS}"
            --seed "${SEED}"
            --deterministic-ratio 1.0
            --output-dir "${BATCH_OUTPUT_DIR}"
            --extra-request-body "${EXTRA_REQUEST_BODY}"
            --warmup-requests 0
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
            echo "--- Batch $((batch+1))/$NUM_BATCHES completed successfully! ---"
            
            # Display summary if available
            SUMMARY_FILE="$BATCH_OUTPUT_DIR/summary.json"
            if [ -f "$SUMMARY_FILE" ] && command -v jq &> /dev/null; then
                echo "Pairwise Mismatch Summary (re-tokenized):"
                jq -r '.pairwise_comparisons[] | "  QPS \(.qps_1) vs QPS \(.qps_2): \(.num_mismatches) mismatches"' "$SUMMARY_FILE"
                echo ""
            fi
        else
            echo ""
            echo "--- ERROR: Batch $((batch+1))/$NUM_BATCHES failed with exit code $RESULT ---"
            OVERALL_RESULT=$RESULT
        fi
    done  # End of batch loop
    
    echo ""
    echo "=============================================="
    echo "NUM_PROMPTS=$NUM_PROMPTS: All batches completed"
    echo "Results saved to: $OUTPUT_DIR"
    echo "=============================================="
    echo ""
done

echo "=============================================="
echo "All runs completed!"
echo "=============================================="
echo "Results saved to: $BASE_OUTPUT_DIR"

exit $OVERALL_RESULT
