#!/bin/bash
set -euo pipefail

# Run verification percentage benchmark for multiple dataset configurations
# This compares different determinism modes and mismatch percentages
#
# Dataset configurations:
#   - 512 input, 1024 output
#   - 1024 input, 1024 output
#   - 512 input, 2048 output
#   - 1024 input, 2048 output
#
# Configurations tested:
#   - default: Non-deterministic baseline
#   - global: Global deterministic mode
#   - detinfer_512_0pct: DetInfer with 0% mismatches
#   - detinfer_512_5pct: DetInfer with 5% mismatches

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Common settings
export QPS=${QPS:-12}
export NUM_PROMPTS=${NUM_PROMPTS:-16384}

echo "=============================================="
echo "Verification Percentage Benchmark - All Dataset Configs"
echo "=============================================="
echo "QPS: $QPS"
echo "NUM_PROMPTS: $NUM_PROMPTS"
echo ""
echo "Configurations being tested:"
echo "  - default: Non-deterministic baseline"
echo "  - global: Global deterministic mode (--enable-deterministic-inference 2)"
echo "  - detinfer_512_0pct: DetInfer step=512, 0% mismatches (no rollback)"
echo "  - detinfer_512_5pct: DetInfer step=512, 5% mismatches (forced rollback)"
echo ""

# Dataset configurations: input_len output_len
declare -a DATASET_CONFIGS=(
    "512 513"
    "1024 513"
    "2048 513"
    "4096 513"
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
echo "  python plot_latency.py --results-dir results_in512_out1025_qps${QPS}_n${NUM_PROMPTS}"
echo ""
echo "Plot files generated in each results directory:"
echo "  - ttft_bars.pdf, tpot_bars.pdf, e2e_bars.pdf"
echo "  - latency_combined.pdf"
echo "  - latency_mean_comparison.pdf"
echo "  - ttft_cdf.pdf, tpot_cdf.pdf, e2e_cdf.pdf"
