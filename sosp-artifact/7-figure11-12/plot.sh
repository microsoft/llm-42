#!/bin/bash
set -euo pipefail

# Auto-discover all online benchmark runs and regenerate plots + summaries.
#
# Usage:
#   ./plot.sh                          # process all runs, all ratios
#   ./plot.sh --ratios 0.05 0.1        # only plot specific LLM-42 ratios
#   ./plot.sh runs/h100_...            # process a specific run directory
#   ./plot.sh --ratios 0.05 runs/...   # combine both

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if command -v python &> /dev/null; then
    PY=python
elif command -v python3 &> /dev/null; then
    PY=python3
else
    echo "Error: Python not found"
    exit 1
fi

# Parse arguments: extract --ratios and remaining positional args
RATIOS_ARGS=()
RUN_ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --ratios|-r)
            shift
            while [ $# -gt 0 ] && [[ "$1" != -* ]] && [[ "$1" != runs/* ]] && [[ "$1" != /* ]]; do
                RATIOS_ARGS+=("$1")
                shift
            done
            ;;
        *)
            RUN_ARGS+=("$1")
            shift
            ;;
    esac
done

# Build --ratios flag for Python script
PLOT_EXTRA_ARGS=()
if [ ${#RATIOS_ARGS[@]} -gt 0 ]; then
    PLOT_EXTRA_ARGS+=("--ratios" "${RATIOS_ARGS[@]}")
    echo "LLM-42 ratios filter: ${RATIOS_ARGS[*]}"
fi

# Determine which run directories to process
if [ ${#RUN_ARGS[@]} -gt 0 ]; then
    RUN_DIRS=("${RUN_ARGS[@]}")
else
    RUN_DIRS=()
    for d in "$ROOT"/runs/*_online; do
        [ -d "$d" ] && [ -f "$d/results/benchmark_results.jsonl" ] && RUN_DIRS+=("$d")
    done
fi

if [ ${#RUN_DIRS[@]} -eq 0 ]; then
    echo "No online run directories found under $ROOT/runs/"
    exit 1
fi

echo "Found ${#RUN_DIRS[@]} run(s) to process"
echo ""

for RUN_DIR in "${RUN_DIRS[@]}"; do
    RUN_DIR="${RUN_DIR%/}"
    echo "=============================================="
    echo "Processing: $(basename "$RUN_DIR")"
    echo "=============================================="

    RESULTS_DIR="$RUN_DIR/results"
    JSONL="$RESULTS_DIR/benchmark_results.jsonl"
    if [ ! -f "$JSONL" ]; then
        echo "  No results/benchmark_results.jsonl, skipping"
        continue
    fi

    echo "  Generating summary CSV..."
    $PY "$ROOT/summarize_online_csv.py" \
        --input "$JSONL" \
        --output "$RESULTS_DIR/summary.csv"

    echo "  Generating per-request CSV..."
    $PY "$ROOT/export_per_request_csv.py" \
        --input "$JSONL" \
        --output "$RESULTS_DIR/per_request_data.csv"

    # Plots — output into the run directory (plots/ subdir created by Python)
    echo ""
    echo "  Generating plots..."
    $PY "$ROOT/plot.py" \
        --results-dirs "$RESULTS_DIR" \
        --output-dir "$RUN_DIR" \
        "${PLOT_EXTRA_ARGS[@]+"${PLOT_EXTRA_ARGS[@]}"}" || echo "  WARNING: plot generation failed"

    echo ""
    echo "  Done: $RUN_DIR"
done

echo ""
echo "All runs processed."
