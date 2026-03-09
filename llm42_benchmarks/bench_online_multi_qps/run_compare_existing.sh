#!/usr/bin/env bash
set -euo pipefail

# Compare existing benchmark output files
# Usage:
#   ./run_compare_existing.sh /path/to/results/reqs_92812
#   INPUT_DIR=/path/to/results ./run_compare_existing.sh

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

INPUT_DIR=${1:-${INPUT_DIR:-""}}
OUTPUT_DIR=${OUTPUT_DIR:-""}
PATTERN=${PATTERN:-"config_qps*.jsonl"}
WRITE_PER_CONFIG=${WRITE_PER_CONFIG:-false}

if [[ -z "$INPUT_DIR" ]]; then
    echo "Usage: $0 <input_dir>"
    echo ""
    echo "Environment variables:"
    echo "  INPUT_DIR       - Directory containing config_*.jsonl files"
    echo "  OUTPUT_DIR      - Output directory (default: same as INPUT_DIR)"
    echo "  PATTERN         - Glob pattern for finding config files (default: config_qps*.jsonl)"
    echo "  WRITE_PER_CONFIG - Write per-config summary logs (default: false)"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/results/reqs_92812"
    echo "  INPUT_DIR=/path/to/results OUTPUT_DIR=/tmp/comparison $0"
    exit 1
fi

echo "=============================================="
echo "Compare Existing Benchmark Outputs"
echo "=============================================="
echo "Input:  $INPUT_DIR"
echo "Output: ${OUTPUT_DIR:-$INPUT_DIR}"
echo "Pattern: $PATTERN"
echo "=============================================="
echo ""

# Build command
cmd=(
    python "${ROOT}/compare_existing_outputs.py"
    --input-dir "$INPUT_DIR"
    --pattern "$PATTERN"
)

if [[ -n "$OUTPUT_DIR" ]]; then
    cmd+=(--output-dir "$OUTPUT_DIR")
fi

if [[ "$WRITE_PER_CONFIG" == "true" ]]; then
    cmd+=(--write-per-config)
fi

# Run comparison
echo "Running: ${cmd[*]}"
echo ""

"${cmd[@]}"
