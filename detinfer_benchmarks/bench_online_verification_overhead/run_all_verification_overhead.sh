#!/bin/bash
set -euo pipefail

# Run verification overhead benchmark for multiple dataset configurations
# This measures the pure verification overhead (no rollback) for different step sizes
#
# Dataset configurations:
#   - 512 input, 1024 output
#   - 1024 input, 1024 output
#   - 512 input, 2048 output
#   - 1024 input, 2048 output
#
# Step sizes tested:
#   - 32, 128, 512, 1024

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Common settings
export QPS=${QPS:-12}
export NUM_PROMPTS=${NUM_PROMPTS:-16384}

echo "=============================================="
echo "Verification Overhead Benchmark - All Dataset Configs"
echo "=============================================="
echo "QPS: $QPS"
echo "NUM_PROMPTS: $NUM_PROMPTS"
echo ""
echo "Step sizes being tested: 32, 128, 512, 1024"
echo "All servers have skip_mismatch enabled (no rollback)"
echo ""

# Dataset configurations: input_len output_len
declare -a DATASET_CONFIGS=(
    "512 1025"
    "1024 1025"
    "512 2049"
    "1024 2049"
)

# Run each dataset configuration
for config in "${DATASET_CONFIGS[@]}"; do
    read -r INPUT_LEN OUTPUT_LEN <<< "$config"
    
    echo ""
    echo "=============================================="
    echo "Running: input=$INPUT_LEN, output=$OUTPUT_LEN"
    echo "=============================================="
    
    export RANDOM_INPUT_LEN=$INPUT_LEN
    export RANDOM_OUTPUT_LEN=$OUTPUT_LEN
    export OUTPUT_DIR="${ROOT}/results_in${INPUT_LEN}_out${OUTPUT_LEN}_qps${QPS}_n${NUM_PROMPTS}"
    
    echo "Dataset: random (synthetic)"
    echo "Input Length: $RANDOM_INPUT_LEN"
    echo "Output Length: $RANDOM_OUTPUT_LEN"
    echo "Output Dir: $OUTPUT_DIR"
    echo ""
    
    # Run the benchmark
    "${ROOT}/run_benchmark.sh"
    
    echo ""
    echo "Completed: input=$INPUT_LEN, output=$OUTPUT_LEN"
    echo ""
done

echo ""
echo "=============================================="
echo "All dataset configurations completed!"
echo "=============================================="
echo ""
echo "Results saved in:"
for config in "${DATASET_CONFIGS[@]}"; do
    read -r INPUT_LEN OUTPUT_LEN <<< "$config"
    echo "  - ${ROOT}/results_in${INPUT_LEN}_out${OUTPUT_LEN}_qps${QPS}_n${NUM_PROMPTS}"
done
echo ""
echo "To plot latency comparison for a specific result:"
echo "  python plot_latency.py --results-dir results_in512_out1024_qps${QPS}_n${NUM_PROMPTS}"
echo ""
echo "Plot files generated in each results directory:"
echo "  - ttft_bars.pdf, tpot_bars.pdf, e2e_bars.pdf"
echo "  - latency_combined.pdf"
echo "  - latency_mean_comparison.pdf"
echo "  - ttft_cdf.pdf, tpot_cdf.pdf, e2e_cdf.pdf"
