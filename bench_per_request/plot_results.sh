#!/bin/bash

# Shell wrapper for plot_component_results.py
# Makes it easier to run the plotting script

# Default configuration
INPUT_DIR="etalon_results"
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
            echo "Plot component test benchmark results"
            echo ""
            echo "Options:"
            echo "  --input-dir DIR     Input directory with results (default: $INPUT_DIR)"
            echo "  --output-dir DIR    Output directory for plots (default: same as input-dir)"
            echo "  --help              Show this help"
            echo ""
            echo "Example:"
            echo "  $0 --input-dir etalon_results_component_tests"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
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

echo "================================================"
echo "Component Test Results Plotter"
echo "================================================"
echo "Input directory: $INPUT_DIR"
if [ -n "$OUTPUT_DIR" ]; then
    echo "Output directory: $OUTPUT_DIR"
else
    echo "Output directory: $INPUT_DIR (same as input)"
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
