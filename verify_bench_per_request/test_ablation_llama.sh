#!/bin/bash

# Test all three predefined trace types with multiple percentages
# This script runs comprehensive tests across arxiv, sharegpt, and lmsys datasets

set -e

# Array of trace types
TRACES=("arxiv" "sharegpt" "lmsys")
TEMP0_PCTS=""
MAX_REQUESTS=512
QPS=1
WARMUP_TIME=30

echo "================================================"
echo "Running comprehensive multi-trace tests"
echo "================================================"
echo "Percentages: $TEMP0_PCTS"
echo "Max requests: $MAX_REQUESTS"
echo "QPS: $QPS"
echo ""
echo "This will run 3 traces x (6 percentages + 3 modes) = 27 total tests"
echo "================================================"
echo ""

for trace in "${TRACES[@]}"; do
    echo ""
    echo "###############################################"
    echo "# Testing with $trace trace"
    echo "###############################################"
    echo ""
    
    ./run_tests_ablation.sh \
        --trace-type "$trace" \
        --temp0-pcts "$TEMP0_PCTS" \
        --assignment-mode random \
        --seed 42 \
        --max-requests $MAX_REQUESTS \
        --qps $QPS \
        --warmup-time $WARMUP_TIME \
        --model "meta-llama/Meta-Llama-3-8B-Instruct"
    
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
            echo "  - llama-3-8b_arxiv_summarization_pct_${TEMP0_PCTS//,/_}/"
            ;;
        sharegpt)
            echo "  - llama-3-8b_sharegpt_8k_pct_${TEMP0_PCTS//,/_}/"
            ;;
        lmsys)
            echo "  - llama-3-8b_lmsys_chat_1m_conversation_pct_${TEMP0_PCTS//,/_}/"
            ;;
    esac
done
echo ""
echo "To plot results for each trace:"
echo "  ./plot_results.sh --input-dir <directory_name>"
echo ""
