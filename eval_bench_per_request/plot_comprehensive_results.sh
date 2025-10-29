#!/bin/bash

# Shell wrapper for plot_comprehensive_results.py
# Makes it easier to plot comprehensive test results

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
            echo "Plot comprehensive benchmark results"
            echo ""
            echo "Generates three types of plots:"
            echo "  1. Ours-per-request progression: Shows how Ours-per-request performs as temp=0 % varies"
            echo "  2. Mode comparison at 100%: Compares all modes at 100% temp=0"
            echo "  3. All modes comparison: Scatter plot showing all modes across temp percentages"
            echo ""
            echo "Options:"
            echo "  --input-dir DIR     Input directory with results (REQUIRED)"
            echo "                      Example: comprehensive_results_20250101_120000"
            echo "  --output-dir DIR    Output directory for plots (default: same as input-dir)"
            echo "  --help              Show this help"
            echo ""
            echo "Examples:"
            echo "  # Plot results from comprehensive test run:"
            echo "  $0 --input-dir comprehensive_results_20250101_120000"
            echo ""
            echo "  # Plot results with custom output directory:"
            echo "  $0 --input-dir comprehensive_results_20250101_120000 --output-dir my_plots"
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

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Determine Python command
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: Python not found."
    exit 1
fi

# Build command
CMD="$PYTHON_CMD $(dirname "$0")/plot_comprehensive_results.py --input-dir \"$INPUT_DIR\""

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output-dir \"$OUTPUT_DIR\""
fi

# Run plotting script
echo "Plotting comprehensive results..."
echo "Input: $INPUT_DIR"
if [ -n "$OUTPUT_DIR" ]; then
    echo "Output: $OUTPUT_DIR"
else
    echo "Output: $INPUT_DIR (same as input)"
fi
echo ""

eval $CMD

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Plotting completed successfully!"
else
    echo ""
    echo "✗ Plotting failed. Check error messages above."
    exit 1
fi
