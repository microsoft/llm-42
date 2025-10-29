#!/bin/bash

# Shell wrapper for plot_results.py
# Makes it easier to run the plotting script for mixed temperature tests

# Default configuration
INPUT_DIR=""
OUTPUT_DIR=""

# Parse command line arguments
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
            echo "Plot mixed temperature benchmark results"
            echo "Supports both single-percentage (legacy) and multi-percentage formats"
            echo ""
            echo "Options:"
            echo "  --input-dir DIR     Input directory with results (REQUIRED)"
            echo "                      Examples:"
            echo "                        Single percentage: pct_10_random, pct_5_fixed"
            echo "                        Multi percentage: llama-3.1-8b_arxiv_pct_0_1_5_10"
            echo "  --output-dir DIR    Output directory for plots (default: same as input-dir)"
            echo "  --help              Show this help"
            echo ""
            echo "Examples:"
            echo "  # Plot results from 10% random test (legacy format):"
            echo "  $0 --input-dir pct_10_random"
            echo ""
            echo "  # Plot results from multi-percentage test (new format):"
            echo "  $0 --input-dir llama-3.1-8b_arxiv_pct_0_1_5_10"
            echo ""
            echo "  # Custom output directory:"
            echo "  $0 --input-dir llama-3.1-8b_arxiv_pct_0_1_5_10 --output-dir my_plots"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if input directory is provided
if [ -z "$INPUT_DIR" ]; then
    echo "Error: --input-dir is required"
    echo "Use --help for usage information"
    exit 1
fi

# Determine Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found."
    exit 1
fi

echo "================================================"
echo "Mixed Temperature Results Plotter"
echo "================================================"
echo "Input directory: $INPUT_DIR"
if [ -n "$OUTPUT_DIR" ]; then
    echo "Output directory: $OUTPUT_DIR"
else
    echo "Output directory: $INPUT_DIR (same as input)"
fi

# Detect format
if ls "$INPUT_DIR"/pct_* > /dev/null 2>&1; then
    echo "Format: Multi-percentage (new)"
    echo "Found percentage subdirectories:"
    ls -d "$INPUT_DIR"/pct_* 2>/dev/null | xargs -n1 basename | sort -V
else
    echo "Format: Single-percentage (legacy)"
fi

echo "================================================"
echo ""

# Run the Python plotting script
if [ -n "$OUTPUT_DIR" ]; then
    $PYTHON_CMD plot_results.py --input-dir "$INPUT_DIR" --output-dir "$OUTPUT_DIR"
else
    $PYTHON_CMD plot_results.py --input-dir "$INPUT_DIR"
fi

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "✓ Plotting completed successfully!"
    echo "================================================"
else
    echo ""
    echo "================================================"
    echo "✗ Plotting failed with exit code: $exit_code"
    echo "================================================"
fi

exit $exit_code
