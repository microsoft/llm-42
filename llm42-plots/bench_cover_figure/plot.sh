#!/bin/bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Find the most recent results directory or use provided path
if [ $# -ge 1 ]; then
    RESULTS_FILE="$1"
else
    # Find most recent results directory
    LATEST_DIR=$(ls -td "${ROOT}"/results_* 2>/dev/null | head -1)
    if [ -z "$LATEST_DIR" ]; then
        echo "Error: No results directory found. Please run run_benchmark.sh first or provide results file path."
        exit 1
    fi
    RESULTS_FILE="${LATEST_DIR}/benchmark_results.jsonl"
fi

if [ ! -f "$RESULTS_FILE" ]; then
    echo "Error: Results file not found: $RESULTS_FILE"
    exit 1
fi

OUTPUT_DIR=$(dirname "$RESULTS_FILE")
OUTPUT_FILE="${OUTPUT_DIR}/throughput_vs_batchsize.pdf"

echo "Generating plot from: $RESULTS_FILE"
python "${ROOT}/plot_batchsize_throughput.py" \
    --input "$RESULTS_FILE" \
    --output "$OUTPUT_FILE"

echo "Plot saved to: $OUTPUT_FILE"
