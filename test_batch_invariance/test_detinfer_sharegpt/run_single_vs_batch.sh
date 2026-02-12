#!/usr/bin/env bash
set -euo pipefail

# Single vs Batch Prompt Comparison Script
# Runs a single prompt once as baseline, then runs the same prompt N times
# at a configurable QPS and checks if all outputs match the baseline.
#
# Environment overrides:
#   BASE_URL (default: http://127.0.0.1:30000)
#   MODEL (default: meta-llama/Llama-3.1-8B-Instruct)
#   TOKENIZER (default: empty = same as model)
#   PROMPT (default: "What is the capital of France?")
#   NUM_REPEATS (default: 10) - number of times to run the same prompt
#   QPS (default: 4) - requests per second for the batch run
#   SEED (default: 42)
#   MAX_TOKENS (default: 256)
#   EXTRA_REQUEST_BODY (default: '{"temperature":0}')
#   OUTPUT_DIR (default: $ROOT/single_vs_batch_out_<timestamp>)

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${PYTHONPATH:-}:${ROOT}/../python"

# Parse configuration
BASE_URL=${BASE_URL:-"http://127.0.0.1:30005"}
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
TOKENIZER=${TOKENIZER:-}
PROMPT=${PROMPT:-"You are a helpful assistant. Please provide a detailed explanation of the theory of relativity, including both special and general relativity. Explain the key concepts, the mathematical foundations, and how these theories have been experimentally verified. Also discuss some practical applications of relativity in modern technology."}
NUM_REPEATS=${NUM_REPEATS:-12244}
QPS=${QPS:-10}
SEED=${SEED:-42}
MAX_TOKENS=${MAX_TOKENS:-4096}
EXTRA_REQUEST_BODY=${EXTRA_REQUEST_BODY:-'{"temperature":0}'}
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR=${OUTPUT_DIR:-"${ROOT}/single_vs_batch_out_${TIMESTAMP}"}

mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Single vs Batch Prompt Comparison"
echo "=============================================="
echo "Configuration:"
echo "  Server URL:      $BASE_URL"
echo "  Model:           $MODEL"
echo "  Prompt:          ${PROMPT:0:50}..."
echo "  Num Repeats:     $NUM_REPEATS"
echo "  QPS:             $QPS"
echo "  Max Tokens:      $MAX_TOKENS"
echo "  Seed:            $SEED"
echo "  Output Dir:      $OUTPUT_DIR"
echo "=============================================="
echo ""

# Check server health before starting
echo "Checking server health..."
echo -n "  Checking $BASE_URL ... "

RESPONSE=$(timeout 5 curl -s "${BASE_URL}/v1/models" 2>&1)
if echo "$RESPONSE" | grep -q '"object":"list"'; then
    echo "✓"
else
    echo "✗ (not responding or not ready)"
    echo ""
    echo "ERROR: Server is not healthy. Please check server logs."
    exit 1
fi

echo ""
echo "Server healthy. Running comparison..."
echo ""

# Build command
cmd=(
    python "${ROOT}/compare_single_vs_batch.py"
    --base-url "${BASE_URL}"
    --model "${MODEL}"
    --prompt "${PROMPT}"
    --num-repeats "${NUM_REPEATS}"
    --qps "${QPS}"
    --max-tokens "${MAX_TOKENS}"
    --seed "${SEED}"
    --output-dir "${OUTPUT_DIR}"
    --extra-request-body "${EXTRA_REQUEST_BODY}"
)

if [[ -n "${TOKENIZER}" ]]; then
    cmd+=(--tokenizer "${TOKENIZER}")
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
else
    echo ""
    echo "=============================================="
    echo "ERROR: Comparison failed with exit code $RESULT"
    echo "=============================================="
fi

exit $RESULT
