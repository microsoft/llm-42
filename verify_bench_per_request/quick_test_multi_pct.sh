#!/bin/bash

# Quick Test - Run with 0,1,5,10 percentages
# This is a quick example showing the new multi-percentage feature

# Default trace type
TRACE_TYPE=${1:-arxiv}

echo "Running quick test with multiple percentages: 0%, 1%, 5%, 10%"
echo "Trace type: $TRACE_TYPE"
echo ""

./run_tests.sh \
  --trace-type "$TRACE_TYPE" \
  --temp0-pcts 0,1,5,10 \
  --assignment-mode random \
  --seed 42 \
  --max-requests 256 \
  --qps 1 \
  --warmup-time 30

echo ""
echo "Test complete!"
echo ""
echo "To plot results, check the output directory shown above and run:"
echo "  ./plot_results.sh --input-dir <output_directory>"
echo ""
echo "Usage: $0 [trace_type]"
echo "  trace_type: arxiv (default), sharegpt, or lmsys"
echo ""
