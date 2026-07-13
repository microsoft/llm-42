#!/usr/bin/env bash
set -euo pipefail

# Run the TP-aware matmul benchmarks for all models.
#
# Usage:
#   ./run.sh                                          # defaults (8b+70b, TP=1,2,4,8)
#   MODELS=llama3-8b TP_SIZES=1,2 ./run.sh            # single model
#   MODELS="llama3-8b,llama3-70b" ITERS=200 ./run.sh   # custom iters
#   ./run.sh --plot-only                               # re-plot from existing CSVs
#   ./run.sh --force                                   # re-run even if results exist
#
# By default a model whose results.csv already exists is skipped (resume);
# pass --force to re-run and overwrite it.

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

#MODELS="${MODELS:-llama3-8b,llama3-70b}"
MODELS="${MODELS:-llama3-70b}"
TP_SIZES="${TP_SIZES:-1,2,4,8}"
BATCH_SIZES="${BATCH_SIZES:-1,2,4,8,16,32,64,128,256,512,1024,2048}"
ITERS="${ITERS:-50}"
PLOT_ONLY=false
FORCE=0

for arg in "$@"; do
    case "$arg" in
        --plot-only) PLOT_ONLY=true ;;
        --force)     FORCE=1 ;;
        *) echo "Unknown argument: $arg" >&2
           echo "Usage: $0 [--plot-only] [--force]" >&2
           exit 1 ;;
    esac
done

if command -v python &>/dev/null; then PY=python; else PY=python3; fi

IFS="," read -ra MODEL_LIST <<< "$MODELS"

for MODEL in "${MODEL_LIST[@]}"; do
    echo "========================================"

    if $PLOT_ONLY; then
        echo "Re-plotting from existing CSV..."
        echo "  Model:  $MODEL"
        echo "========================================"
        # Find the CSV — look in runs/*/$MODEL/results.csv
        CSV=$(find "$ROOT/runs" -path "*/$MODEL/results.csv" 2>/dev/null | head -1)
        if [[ -z "$CSV" ]]; then
            echo "  ERROR: No results.csv found for $MODEL"
            continue
        fi
        $PY "$ROOT/plot.py" "$CSV" --model "$MODEL"
    else
        # Resume: skip a model whose results.csv already exists (unless --force).
        EXISTING_CSV=$(find "$ROOT/runs" -path "*/$MODEL/results.csv" 2>/dev/null | head -1)
        if [[ -n "$EXISTING_CSV" && "$FORCE" -ne 1 ]]; then
            echo "Skipping benchmark (already done): $MODEL -> $EXISTING_CSV"
            echo "  Regenerating plot from existing data (use --force to re-run)."
            $PY "$ROOT/plot.py" "$EXISTING_CSV" --model "$MODEL"
            echo "========================================"
            echo ""
            continue
        fi
        echo "Running TP-aware matmul benchmark..."
        echo "  Model:  $MODEL"
        echo "  TP:     $TP_SIZES"
        echo "  Iters:  $ITERS"
        echo "========================================"
        echo ""

        $PY "$ROOT/bench_matmul_tp.py" \
            --model "$MODEL" \
            --tp-sizes "$TP_SIZES" \
            --batch-sizes "$BATCH_SIZES" \
            --iters "$ITERS"
    fi

    echo ""
done

echo "Done! Check runs/ for outputs."
