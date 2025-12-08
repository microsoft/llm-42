#!/bin/bash
# Generate rollback metric plots from benchmark results
#
# Usage:
#   ./plot_rollbacks.sh <input_jsonl> [output_dir] [--cross-dataset] [--sharegpt-qps N] [--arxiv-qps N]
#
# Examples:
#   ./plot_rollbacks.sh qps_6_results/rollback_metrics.jsonl
#   ./plot_rollbacks.sh qps_6_results/rollback_metrics.jsonl ./plots --cross-dataset
#   ./plot_rollbacks.sh data.jsonl ./plots --cross-dataset --sharegpt-qps 6 --arxiv-qps 4

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SCRIPT_DIR}/../.venv/bin/python"

if [ ! -f "$PYTHON" ]; then
    echo "Error: Python not found at $PYTHON"
    echo "Please ensure the virtual environment exists."
    exit 1
fi

if [ $# -lt 1 ]; then
    echo "Usage: $0 <input_jsonl> [output_dir] [options]"
    echo ""
    echo "Options:"
    echo "  --cross-dataset       Create cross-dataset plots (ShareGPT + Arxiv combined)"
    echo "  --sharegpt-qps N      QPS rate for ShareGPT in cross-dataset plots"
    echo "  --arxiv-qps N         QPS rate for Arxiv in cross-dataset plots"
    exit 1
fi

INPUT_FILE="$1"
shift

# Default output directory
OUTPUT_DIR="./rollback_plots"

# Parse remaining arguments
EXTRA_ARGS=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --cross-dataset)
            EXTRA_ARGS="$EXTRA_ARGS --cross-dataset"
            shift
            ;;
        --sharegpt-qps)
            EXTRA_ARGS="$EXTRA_ARGS --sharegpt-qps $2"
            shift 2
            ;;
        --arxiv-qps)
            EXTRA_ARGS="$EXTRA_ARGS --arxiv-qps $2"
            shift 2
            ;;
        *)
            # First non-option argument is the output directory
            if [[ ! "$1" == --* ]]; then
                OUTPUT_DIR="$1"
            fi
            shift
            ;;
    esac
done

echo "Input file: $INPUT_FILE"
echo "Output directory: $OUTPUT_DIR"

$PYTHON "${SCRIPT_DIR}/plot_rollbacks_publication.py" "$INPUT_FILE" --output-dir "$OUTPUT_DIR" $EXTRA_ARGS
