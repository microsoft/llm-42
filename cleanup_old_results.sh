#!/bin/bash

# Clean up old/incorrect test results

RESULTS_DIR=${1:-"etalon_results_automated"}

echo "================================================"
echo "Cleaning up old test results"
echo "================================================"

if [ ! -d "$RESULTS_DIR" ]; then
    echo "Directory $RESULTS_DIR does not exist. Nothing to clean."
    exit 0
fi

echo "Removing incorrectly named result directories..."

# Remove directories with wrong names
rm -rf "$RESULTS_DIR"

echo "✓ Cleanup complete"
echo ""
echo "You can now run:"
echo "  ./run_automated_tests.sh"
