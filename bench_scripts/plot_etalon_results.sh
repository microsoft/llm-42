#!/bin/bash

# Wrapper script to plot etalon results
# This script calls the Python plotting script

# Default input directory
INPUT_DIR="etalon_results_automated"
OUTPUT_DIR=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --input-dir DIR    Input directory with results (default: etalon_results_automated)"
            echo "  --output-dir DIR   Output directory for plots (default: same as input-dir)"
            echo "  --help             Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Determine Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found."
    exit 1
fi

# Build command
CMD="$PYTHON_CMD plot_etalon_results.py --input-dir \"$INPUT_DIR\""
if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output-dir \"$OUTPUT_DIR\""
fi

# Run plotting script
eval $CMD
