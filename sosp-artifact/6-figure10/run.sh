#!/bin/bash
set -euo pipefail

# Run GPU-balanced offline throughput benchmarks across multiple dataset
# configurations, then generate PDF comparison plots and summary CSVs.
#
# Each dataset is run with ALL server configs (non-det, global-det, llm42
# variants) using the balanced job-queue approach of run_balanced_benchmark.sh.
#
# Which models to run (--models): comma-separated list of 8b and/or 70b.
#   8b       Llama-3.1-8B-Instruct  (TP-1)  [default]
#   70b      Llama-3.3-70B-Instruct (TP-8)
#   8b,70b   both
#
# Run duration (--duration): dataset list + workload.
#   full    (default)  All 6 datasets, 2048 prompts.
#   quick              sharegpt + arxiv datasets, 256 prompts.
#
# The 70B model is skipped entirely when fewer than 4 GPUs are visible. With
# >=4 GPUs it defaults to TP-8 and is auto-reduced to the largest power of two
# <= NUM_GPUS (e.g. TP-4 on a 4-GPU node). Override detection with NUM_GPUS=.
#
# Usage:
#   ./run.sh                                  # full datasets, 8B model
#   ./run.sh --models 8b --duration quick     # quick smoke test
#   ./run.sh --models 8b,70b --force          # both models, force re-run
#   NUM_GPUS=8 NUM_PROMPTS=4096 ./run.sh
#
# Env overrides (optional): MODEL + TP_SIZE run a single explicit model instead
# of --models; DATASET_CONFIGS (space-separated) replaces the duration's dataset
# list; NUM_PROMPTS replaces the duration's prompt count.

# ---- Parse flags ----
FORCE_FLAG=""
DURATION="full"
MODELS_ARG=""
while [ $# -gt 0 ]; do
    case "$1" in
        --force)      FORCE_FLAG="--force" ;;
        --duration)   if [ $# -lt 2 ]; then echo "Error: --duration requires a value (quick|full)" >&2; exit 1; fi
                      DURATION="$2"; shift ;;
        --duration=*) DURATION="${1#*=}" ;;
        --models)     if [ $# -lt 2 ]; then echo "Error: --models requires a value (8b|70b|8b,70b)" >&2; exit 1; fi
                      MODELS_ARG="$2"; shift ;;
        --models=*)   MODELS_ARG="${1#*=}" ;;
        *) echo "Unknown argument: $1" >&2
           echo "Usage: $0 [--duration quick|full] [--models 8b|70b|8b,70b] [--force]" >&2
           exit 1 ;;
    esac
    shift
done

case "$DURATION" in
    quick|full) ;;
    *) echo "Error: --duration must be 'quick' or 'full' (got '$DURATION')" >&2; exit 1 ;;
esac

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "${ROOT}/../gpu_utils.sh"
_GPU_TYPE="$GPU_SHORT_NAME"

# ---- GPU / TP helpers ----
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
export NUM_GPUS
# Largest power of two <= n. TP size must be a power of two that divides the
# available GPU count, so this picks the biggest TP that fits (8 -> 4 on 4 GPUs).
largest_pow2_le() { local n=$1 p=1; while (( p*2 <= n )); do p=$((p*2)); done; echo "$p"; }

# ---- Common settings (exported so run_balanced_benchmark.sh picks them up) ----
export HOST="${HOST:-0.0.0.0}"
export BASE_PORT="${BASE_PORT:-30005}"

# Forward optional config overrides
[ -n "${CONFIG_NAMES:-}" ]      && export CONFIG_NAMES
[ -n "${LLM42_RATIOS:-}" ]      && export LLM42_RATIOS
[ -n "${TOKENIZER:-}" ]         && export TOKENIZER
[ -n "${DATASET_PATH:-}" ]      && export DATASET_PATH
[ -n "${BACKEND:-}" ]           && export BACKEND
[ -n "${DETERMINISTIC_SEED:-}" ] && export DETERMINISTIC_SEED
[ -n "${SHAREGPT_CONTEXT_LEN:-}" ] && export SHAREGPT_CONTEXT_LEN
[ -n "${SERVER_STARTUP_TIMEOUT:-}" ] && export SERVER_STARTUP_TIMEOUT
[ -n "${DISABLE_CUSTOM_ALL_REDUCE:-}" ] && export DISABLE_CUSTOM_ALL_REDUCE
[ -n "${ENABLE_TORCH_SYMM_MEM:-}" ]   && export ENABLE_TORCH_SYMM_MEM

# ---- Model selection (--models, or an explicit MODEL env override) ----
# MODELS entries have the form "<hf_model_path> <default_tp>". A MODEL env var
# runs that single explicit model (honouring TP_SIZE) and overrides --models.
model_spec() {
    case "$1" in
        8b)  echo "meta-llama/Llama-3.1-8B-Instruct 1" ;;
        70b) echo "meta-llama/Llama-3.3-70B-Instruct 8" ;;
        *)   return 1 ;;
    esac
}
declare -a MODELS=()
if [ -n "${MODEL:-}" ]; then
    MODELS=("${MODEL} ${TP_SIZE:-1}")
else
    MODELS_ARG="${MODELS_ARG:-8b}"
    IFS=',' read -ra _MODEL_KEYS <<< "$MODELS_ARG"
    for _key in "${_MODEL_KEYS[@]}"; do
        _key="$(echo "$_key" | tr '[:upper:]' '[:lower:]' | tr -d '[:space:]')"
        [ -z "$_key" ] && continue
        if ! _spec="$(model_spec "$_key")"; then
            echo "Error: --models must be 8b, 70b, or 8b,70b (got '$_key')" >&2
            exit 1
        fi
        MODELS+=("$_spec")
    done
fi
if [ "${#MODELS[@]}" -eq 0 ]; then
    echo "Error: no models selected (use --models 8b|70b|8b,70b)" >&2
    exit 1
fi

# ---- Duration-specific configuration (--duration) ----
# quick: sharegpt + arxiv datasets, 256 prompts.
# full:  all 6 datasets, 2048 prompts.
if [ "$DURATION" = "quick" ]; then
    DEFAULT_DATASETS=(
        "sharegpt"
        "arxiv"
        "random_in1024_out256"
    )
    DEFAULT_NUM_PROMPTS=256
else
    DEFAULT_DATASETS=(
        "sharegpt"
        "arxiv"
        "random_in1024_out256"
        "random_in1024_out1024"
        "random_in4096_out1024"
        "random_in4096_out4096"
    )
    DEFAULT_NUM_PROMPTS=2048
fi

# NUM_PROMPTS: an explicit env value overrides the mode default.
export NUM_PROMPTS="${NUM_PROMPTS:-$DEFAULT_NUM_PROMPTS}"

# Datasets: DATASET_CONFIGS env (space-separated) overrides the mode default.
declare -a DATASET_ARRAY
if [ -n "${DATASET_CONFIGS:-}" ]; then
    IFS=' ' read -ra DATASET_ARRAY <<< "$DATASET_CONFIGS"
else
    DATASET_ARRAY=("${DEFAULT_DATASETS[@]}")
fi
export DATASET_CONFIGS_LIST="${DATASET_ARRAY[*]}"

# Python
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: Python not found"
    exit 1
fi

echo "=============================================="
echo "Multi-Dataset Offline Throughput Benchmarks"
echo "=============================================="
echo "Duration: $DURATION"
echo "GPU:      $_GPU_TYPE  (NUM_GPUS=$NUM_GPUS)"
echo "Models:   ${#MODELS[@]} (${MODEL:-$MODELS_ARG})"
echo "Prompts:  $NUM_PROMPTS per dataset"
echo "Datasets: ${DATASET_ARRAY[*]}"
echo "=============================================="
echo ""

# ---- Run each model ----
for MODEL_ENTRY in "${MODELS[@]}"; do
    read -r MODEL_PATH REQ_TP <<< "$MODEL_ENTRY"
    TP_SIZE=$REQ_TP

    # Skip the 70B model when the node has fewer than 4 GPUs.
    if [[ "${MODEL_PATH,,}" == *70b* ]] && (( NUM_GPUS < 4 )); then
        echo "NOTE: skipping $(basename "$MODEL_PATH") (70B) -- requires >=4 GPUs but only ${NUM_GPUS} visible."
        continue
    fi

    # Auto-reduce TP when the machine has fewer GPUs than requested
    # (e.g. TP-8 -> TP-4 on a 4-GPU node).
    if (( NUM_GPUS >= 1 && TP_SIZE > NUM_GPUS )); then
        NEW_TP=$(largest_pow2_le "$NUM_GPUS")
        echo "NOTE: $(basename "$MODEL_PATH") requested TP-${TP_SIZE} but only ${NUM_GPUS} GPU(s) available; using TP-${NEW_TP}"
        TP_SIZE=$NEW_TP
    fi

    export MODEL="$MODEL_PATH"
    export TP_SIZE

    # ---- Per-model run directory ----
    _MODEL_TAG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' | sed 's/^meta-//')
    RUN_DIR="${ROOT}/runs/${_GPU_TYPE}_${_MODEL_TAG}-tp${TP_SIZE}_${ATTENTION_BACKEND}_n${NUM_PROMPTS}"
    export RESULTS_ROOT="${RUN_DIR}/results"
    export SERVER_LOG_DIR="${RUN_DIR}/server_logs"
    mkdir -p "$RESULTS_ROOT" "$SERVER_LOG_DIR"

    echo "=============================================="
    echo "Model:    $MODEL  (TP=$TP_SIZE)"
    echo "Run Dir:  $RUN_DIR"
    echo "=============================================="
    echo ""

    # Run all datasets for this model in a single balanced benchmark pass.
    echo "Launching balanced benchmark across all datasets..."
    echo ""
    "${ROOT}/run_balanced_benchmark.sh" $FORCE_FLAG

    echo ""
    echo "Generating throughput comparison plots..."
    $PYTHON_CMD "${ROOT}/plot.py" \
        --results-dirs "$RESULTS_ROOT" \
        --output-dir "$RUN_DIR"

    SUMMARY_CSV="${RUN_DIR}/summary.csv"
    echo ""
    echo "Generating cross-dataset summary CSV..."
    $PYTHON_CMD "${ROOT}/summarize_results_csv.py" \
        --input-dirs "$RESULTS_ROOT"/*/ \
        --output "$SUMMARY_CSV"

    echo ""
    echo "Model done: $MODEL"
    echo "  Run directory:  $RUN_DIR"
    echo "  Summary CSV:    $SUMMARY_CSV"
    echo ""
done

echo "=============================================="
echo "All Done!  (duration=$DURATION)"
echo "=============================================="
