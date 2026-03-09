#!/bin/bash
set -euo pipefail

# Run offline throughput benchmarks for multiple dataset configurations
# - Non-det and Global-det: run with det_ratio=1.0
# - LLM42: run with det_ratios 0.02, 0.05, 0.1, 0.2, 0.5, 1.0

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Common settings
export NUM_PROMPTS=${NUM_PROMPTS:-4096}

echo "=============================================="
echo "Running Offline Throughput for Multiple Datasets"
echo "=============================================="
echo "NUM_PROMPTS: $NUM_PROMPTS"
echo ""
echo "Configurations:"
echo "  - Non-Det, Global-Det: det_ratio=1.0"
echo "  - LLM42: det_ratios=0.02, 0.05, 0.1, 0.2, 0.5, 1.0"
echo ""

# Dataset configurations
declare -a DATASET_CONFIGS=(
    "sharegpt"
    "random_in512_out256"
    "random_in1024_out256"
    "random_in1024_out512"
    "random_in2048_out256"
    "random_in2048_out512"
    "random_in4096_out512"
    "arxiv"
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
        "arxiv")
            export DATASET_NAME=arxiv
            unset RANDOM_INPUT_LEN RANDOM_OUTPUT_LEN 2>/dev/null || true
            ;;
        "random_in1024_out1")
            export DATASET_NAME=random
            export RANDOM_INPUT_LEN=1024
            export RANDOM_OUTPUT_LEN=1
            ;;
        "random_in1024_out256")
            export DATASET_NAME=random
            export RANDOM_INPUT_LEN=1024
            export RANDOM_OUTPUT_LEN=256
            ;;
        "random_in1024_out512")
            export DATASET_NAME=random
            export RANDOM_INPUT_LEN=1024
            export RANDOM_OUTPUT_LEN=512
            ;;
        "random_in1024_out1024")
            export DATASET_NAME=random
            export RANDOM_INPUT_LEN=1024
            export RANDOM_OUTPUT_LEN=1024
            ;;
        "random_in512_out256")
            export DATASET_NAME=random
            export RANDOM_INPUT_LEN=512
            export RANDOM_OUTPUT_LEN=256
            ;;
        "random_in2048_out256")
            export DATASET_NAME=random
            export RANDOM_INPUT_LEN=2048
            export RANDOM_OUTPUT_LEN=256
            ;;
        "random_in2048_out512")
            export DATASET_NAME=random
            export RANDOM_INPUT_LEN=2048
            export RANDOM_OUTPUT_LEN=512
            ;;
        "random_in4096_out256")
            export DATASET_NAME=random
            export RANDOM_INPUT_LEN=4096
            export RANDOM_OUTPUT_LEN=256
            ;;
        "random_in4096_out512")
            export DATASET_NAME=random
            export RANDOM_INPUT_LEN=4096
            export RANDOM_OUTPUT_LEN=512
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
echo "Results saved in: ${ROOT}/results/"
for config in "${DATASET_CONFIGS[@]}"; do
    echo "  - ${ROOT}/results/${config}/"
done
echo ""

# Create plots directory
PLOTS_DIR="${ROOT}/plots"
mkdir -p "$PLOTS_DIR"

echo "Generating throughput comparison plots..."
python "${ROOT}/plot_throughput_comparison.py" \
    --results-dirs "${ROOT}"/results/* \
    --output "${PLOTS_DIR}/throughput_comparison"

# Create tables directory
TABLES_DIR="${ROOT}/tables"
mkdir -p "$TABLES_DIR"

echo ""
echo "Generating LaTeX tables..."
python "${ROOT}/generate_latex_tables.py" \
    --results-dirs "${ROOT}"/results/* \
    --output-dir "$TABLES_DIR"

echo ""
echo "Generated in ${PLOTS_DIR}/:"
echo "  - throughput_comparison_ws32bs16.pdf (LLM42 ws=32, bs=16)"
echo "  - throughput_comparison_ws64bs8.pdf (LLM42 ws=64, bs=8)"
echo "  - throughput_comparison.csv"
echo ""
echo "Generated in ${TABLES_DIR}/:"
echo "  - rollback_ws_32_bs_16.tex"
echo "  - rollback_ws_64_bs_8.tex"
echo ""
echo "To regenerate:"
echo "  python plot_throughput_comparison.py --results-dirs results/* --output plots/throughput_comparison"
echo "  python generate_latex_tables.py --results-dirs results/* --output-dir tables"
