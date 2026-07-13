#!/bin/bash

# Benchmark forward pass cost for multiple models.
# Profiles both Llama-8B (tp=1) and Llama-70B (tp=8) by default,
# then generates a combined plot.
#
# Llama-70B is skipped when fewer than 4 GPUs are available. With >=4 GPUs its
# TP size is auto-reduced to fit the GPU count, e.g. Llama-70B runs with TP-4
# instead of TP-8 on a 4-GPU node. Override the detected count with NUM_GPUS.
#
# Usage:
#   ./run.sh                       # defaults (8b + 70b)
#   ./run.sh --plot-only           # re-plot from existing CSVs
#   ./run.sh --force               # re-run even if results exist
#
# By default a model whose results CSV already exists is skipped (resume);
# pass --force to re-run and overwrite it.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

source "$SCRIPT_DIR/../gpu_utils.sh"

# Number of visible GPUs (override via NUM_GPUS env var).
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"

# Largest power of two <= n. TP size must be a power of two that divides
# the model's head count, so we round the GPU count down to one.
largest_pow2_le() {
    local n=$1 p=1
    while (( p * 2 <= n )); do p=$((p * 2)); done
    echo "$p"
}

INPUT_LENS="${INPUT_LENS:-16 32 64 128 256 512 1024}"
WARMUP_ITERS="${WARMUP_ITERS:-10}"
BENCH_ITERS="${BENCH_ITERS:-50}"
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

RUNS_DIR="${SCRIPT_DIR}/runs"
mkdir -p "$RUNS_DIR"

# Define models: "model_path tp_size label"
# (label gets a "(TP-N)" suffix appended with the actual TP size used)
MODELS=(
    "meta-llama/Meta-Llama-3.1-8B-Instruct 1 Llama-3-8B"
    "meta-llama/Llama-3.3-70B-Instruct 8 Llama-3-70B"
)

CSV_FILES=()

for MODEL_ENTRY in "${MODELS[@]}"; do
    read -r MODEL_PATH TP_SIZE LABEL <<< "$MODEL_ENTRY"

    # Skip the 70B model when the node has fewer than 4 GPUs.
    if [[ "${MODEL_PATH,,}" == *70b* ]] && (( NUM_GPUS < 4 )); then
        echo "NOTE: skipping ${LABEL} (70B) -- requires >=4 GPUs but only ${NUM_GPUS} visible."
        continue
    fi

    # Auto-reduce TP when the machine has fewer GPUs than requested
    # (e.g. TP-8 -> TP-4 on a 4-GPU node).
    if (( NUM_GPUS >= 1 && TP_SIZE > NUM_GPUS )); then
        NEW_TP=$(largest_pow2_le "$NUM_GPUS")
        echo "NOTE: $LABEL requested TP-${TP_SIZE} but only ${NUM_GPUS} GPU(s) available; using TP-${NEW_TP}"
        TP_SIZE=$NEW_TP
    fi

    LABEL="${LABEL} (TP-${TP_SIZE})"

    _MODEL_TAG=$(basename "$MODEL_PATH" | tr [:upper:] [:lower:])
    _PREFIX="${GPU_SHORT_NAME}_${_MODEL_TAG}_tp${TP_SIZE}_${ATTENTION_BACKEND}"
    RESULT_FILE="${RUNS_DIR}/${_PREFIX}_results.csv"
    CSV_FILES+=("$RESULT_FILE:$LABEL")

    if $PLOT_ONLY; then
        if [[ ! -f "$RESULT_FILE" ]]; then
            echo "WARNING: $RESULT_FILE not found, skipping $LABEL"
            continue
        fi
        echo "Using existing: $RESULT_FILE"
        continue
    fi

    # Resume: skip a model whose result CSV already exists (unless --force).
    # It was already added to CSV_FILES above, so the combined plot still uses it.
    if [[ -f "$RESULT_FILE" && "$FORCE" -ne 1 ]]; then
        echo "Skipping (already done): $LABEL -> $RESULT_FILE (use --force to re-run)"
        continue
    fi

    echo "========================================"
    echo "Forward Pass Cost Benchmark"
    echo "========================================"
    echo "Model:      $MODEL_PATH"
    echo "Label:      $LABEL"
    echo "TP Size:    $TP_SIZE"
    echo "Attention:  $ATTENTION_BACKEND"
    echo "GPU Type:   $GPU_SHORT_NAME"
    echo "Input Lens: $INPUT_LENS"
    echo "Results:    $RESULT_FILE"
    echo "========================================"
    echo ""

    python bench_forward_cost.py \
        --model-path "$MODEL_PATH" \
        --tp "$TP_SIZE" \
        --attention-backend "$ATTENTION_BACKEND" \
        --disable-cuda-graph \
        --input-lens $INPUT_LENS \
        --warmup-iters "$WARMUP_ITERS" \
        --bench-iters "$BENCH_ITERS" \
        --result-file "$RESULT_FILE"

    echo ""
done

# Build plot args: --input file1.csv --label label1 --input file2.csv --label label2
PLOT_ARGS=()
for entry in "${CSV_FILES[@]}"; do
    IFS=":" read -r csv label <<< "$entry"
    if [[ -f "$csv" ]]; then
        PLOT_ARGS+=(--input "$csv" --label "$label")
    fi
done

COMBINED_PLOT="${RUNS_DIR}/${GPU_SHORT_NAME}_${ATTENTION_BACKEND}_verification_cost_combined.pdf"

echo "========================================"
echo "Generating combined plot..."
echo "========================================"

python plot.py "${PLOT_ARGS[@]}" --output "$COMBINED_PLOT"

echo ""
echo "========================================"
echo "Benchmark complete!"
echo "Combined plot: $COMBINED_PLOT"
echo "========================================"
