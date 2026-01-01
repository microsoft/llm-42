#!/bin/bash
set -euo pipefail

# Run offline throughput benchmarks for multiple dataset configurations
# This script runs 4 dataset configs and saves results for throughput analysis

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Common settings
export NUM_PROMPTS=${NUM_PROMPTS:-16384}
export DETERMINISTIC_RATIOS="1.0"

echo "=============================================="
echo "Running Offline Throughput for Multiple Datasets"
echo "=============================================="
echo "NUM_PROMPTS: $NUM_PROMPTS"
echo ""

# Dataset configurations
declare -a DATASET_CONFIGS=(
    "sharegpt"
    "random_in1024_out1"
    "random_in1024_out257"
    "random_in1024_out513"
    "random_in1024_out1025"
    "random_in512_out257"
    "random_in2048_out257"
    "random_in4096_out257"
)

# Run each dataset configuration
for config in "${DATASET_CONFIGS[@]}"; do
    echo ""
    echo "=============================================="
    echo "Running dataset config: $config"
    echo "=============================================="
    
    case "$config" in
        "sharegpt")
            export DATASET_NAME=sharegpt
            unset RANDOM_INPUT_LEN RANDOM_OUTPUT_LEN 2>/dev/null || true
            ;;
        "random_in1024_out1")
            export DATASET_NAME=random
            export RANDOM_INPUT_LEN=1024
            export RANDOM_OUTPUT_LEN=1
            ;;
        "random_in1024_out257")
            export DATASET_NAME=random
            export RANDOM_INPUT_LEN=1024
            export RANDOM_OUTPUT_LEN=257
            ;;
        "random_in1024_out513")
            export DATASET_NAME=random
            export RANDOM_INPUT_LEN=1024
            export RANDOM_OUTPUT_LEN=513
            ;;
        "random_in1024_out1025")
            export DATASET_NAME=random
            export RANDOM_INPUT_LEN=1024
            export RANDOM_OUTPUT_LEN=1025
            ;;
        "random_in512_out257")
            export DATASET_NAME=random
            export RANDOM_INPUT_LEN=512
            export RANDOM_OUTPUT_LEN=257
            ;;
        "random_in2048_out257")
            export DATASET_NAME=random
            export RANDOM_INPUT_LEN=2048
            export RANDOM_OUTPUT_LEN=257
            ;;
        "random_in4096_out257")
            export DATASET_NAME=random
            export RANDOM_INPUT_LEN=4096
            export RANDOM_OUTPUT_LEN=257
            ;;
    esac
    
    echo "Dataset: $DATASET_NAME"
    if [ "$DATASET_NAME" = "random" ]; then
        echo "Input Length: $RANDOM_INPUT_LEN"
        echo "Output Length: $RANDOM_OUTPUT_LEN"
    fi
    echo ""
    
    # Run the benchmark
    "${ROOT}/run_offline_benchmark.sh"
    
    echo ""
    echo "Completed: $config"
    echo ""
done

echo ""
echo "=============================================="
echo "All dataset configurations completed!"
echo "=============================================="
echo ""
echo "Results saved in:"
for config in "${DATASET_CONFIGS[@]}"; do
    echo "  - ${ROOT}/results_${config}_*"
done
echo ""
echo "Generating throughput comparison plots..."
python "${ROOT}/plot_throughput_comparison.py" \
    --results-dirs "${ROOT}"/results_*_n${NUM_PROMPTS}_* \
    --output "${ROOT}/throughput_comparison" \
    --det-ratio 1.0

echo ""
echo "Generated:"
echo "  - throughput_comparison_output.pdf (decode throughput)"
echo "  - throughput_comparison_total.pdf (total throughput)"
echo "  - throughput_comparison.csv"
echo ""
echo "To regenerate plots:"
echo "  python plot_throughput_comparison.py --results-dirs results_*_n${NUM_PROMPTS}_* --output throughput_comparison"
