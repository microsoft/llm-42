#!/bin/bash

# Benchmark forward pass cost for Llama 3.1-8B Instruct
# Measures latency for token counts: 16, 32, 64, 128, 256, 512

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
RESULT_FILE="forward_cost_results.csv"
PLOT_FILE="forward_cost_plot.pdf"

echo "========================================"
echo "Forward Pass Cost Benchmark"
echo "========================================"
echo "Model: $MODEL_PATH"
echo "Output: $RESULT_FILE"
echo "========================================"
echo ""

# Run benchmark
python bench_forward_cost.py \
    --model-path "$MODEL_PATH" \
    --disable-cuda-graph \
    --input-lens 16 32 64 128 256 512 \
    --warmup-iters 10 \
    --bench-iters 50 \
    --result-file "$RESULT_FILE"

echo ""
echo "========================================"
echo "Generating plot..."
echo "========================================"

# Generate plot
python plot_results.py \
    --input "$RESULT_FILE" \
    --output "$PLOT_FILE"

echo ""
echo "========================================"
echo "Benchmark complete!"
echo "Results: $RESULT_FILE"
echo "Plot: $PLOT_FILE"
echo "========================================"
