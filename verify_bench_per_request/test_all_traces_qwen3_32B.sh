#!/bin/bash

# Test all three predefined trace types with multiple percentages
# This script runs comprehensive tests across arxiv, sharegpt, and lmsys datasets

set -e

# Array of trace types
TRACES=("arxiv" "sharegpt")
TEMP0_PCTS="0,5,10"
MAX_REQUESTS=256
QPS=1
WARMUP_TIME=30

echo "================================================"
echo "Running comprehensive multi-trace tests"
echo "================================================"
echo "Percentages: $TEMP0_PCTS"
echo "Max requests: $MAX_REQUESTS"
echo "QPS: $QPS"
echo ""
echo "This will run w traces x (3 percentages + 3 modes) = 12 total tests"
echo "================================================"
echo ""

for trace in "${TRACES[@]}"; do
    echo ""
    echo "###############################################"
    echo "# Testing with $trace trace"
    echo "###############################################"
    echo ""
    
    ./run_tests.sh \
        --trace-type "$trace" \
        --temp0-pcts "$TEMP0_PCTS" \
        --assignment-mode random \
        --seed 42 \
        --max-requests $MAX_REQUESTS \
        --qps $QPS \
        --warmup-time $WARMUP_TIME \
        --model "Qwen/Qwen3-4B-Instruct-2507" \
        --tp 1
    
    echo ""
    echo "✓ Completed tests for $trace"
    echo ""
done

echo ""
echo "================================================"
echo "All trace tests completed!"
echo "================================================"
echo ""
echo "Results directories:"
for trace in "${TRACES[@]}"; do
    # The actual directory name will depend on the trace file name
    case $trace in
        arxiv)
            echo "  - qwen-3-4b_arxiv_summarization_pct_${TEMP0_PCTS//,/_}/"
            ;;
        sharegpt)
            echo "  - qwen-3-4b_sharegpt_8k_pct_${TEMP0_PCTS//,/_}/"
            ;;
        lmsys)
            echo "  - qwen-3-4b_lmsys_chat_1m_conversation_pct_${TEMP0_PCTS//,/_}/"
            ;;
    esac
done
echo ""
echo "To plot results for each trace:"
echo "  ./plot_results.sh --input-dir <directory_name>"
echo ""
