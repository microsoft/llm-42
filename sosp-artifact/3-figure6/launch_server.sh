#!/bin/bash

# Launch SGLang server for batch invariance testing (standalone).
# Prefer run_compare_mismatches.sh which handles server lifecycle automatically.

set -euo pipefail

# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../gpu_utils.sh"

# Model and server configuration
MODEL_PATH="${SGLANG_TEST_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
HOST="${SGLANG_HOST:-0.0.0.0}"
PORT="${SGLANG_PORT:-30000}"
TP_SIZE="${SGLANG_TP_SIZE:-1}"

ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-$ATTENTION_BACKEND}"
EXTRA_SERVER_ARGS=""
if [ "${DISABLE_CUSTOM_ALL_REDUCE:-0}" = "1" ]; then
    EXTRA_SERVER_ARGS="$EXTRA_SERVER_ARGS --disable-custom-all-reduce"
fi
if [ "${ENABLE_TORCH_SYMM_MEM:-0}" = "1" ]; then
    EXTRA_SERVER_ARGS="$EXTRA_SERVER_ARGS --enable-torch-symm-mem"
fi

echo "=============================================="
echo "Starting SGLang Server for Batch Invariance Testing"
echo "=============================================="
echo "GPU: $GPU_SHORT_NAME"
echo "Model: $MODEL_PATH"
echo "Host: $HOST"
echo "Port: $PORT"
echo "TP Size: $TP_SIZE"
echo "Attention Backend: $ATTENTION_BACKEND"
echo "Extra Args: ${EXTRA_SERVER_ARGS:-<none>}"
echo "=============================================="
echo ""

python -m sglang.launch_server \
    --model-path "$MODEL_PATH" \
    --host "$HOST" \
    --port "$PORT" \
    --tp "$TP_SIZE" \
    --attention-backend "$ATTENTION_BACKEND" \
    --disable-radix-cache \
    --disable-chunked-prefix-cache \
    --disable-overlap-schedule \
    --enable-metrics \
    --chunked-prefill-size -1 \
    --max-running-requests 64 \
    --random-seed 42 \
    $EXTRA_SERVER_ARGS