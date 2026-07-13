#!/bin/bash
set -euo pipefail

# Run GPU-balanced online (QPS-driven) benchmarks for the selected model(s),
# then generate CDF/TTFT comparison plots and summary CSVs.
#
# Each (dataset, qps) combination is run with ALL server configs (non-det,
# deterministic, llm42 variants) using the balanced job-queue approach of
# run_balanced_online_benchmark.sh.
#
# On the first run, the mixed_workload dataset is built via prepare_mixed_workload.py
# and cached at ~/.cache/llm42_bench/; subsequent runs reuse it without rebuilding.
#
# Which models to run (--models): comma-separated list of 8b and/or 70b.
#   8b       Llama-3.1-8B-Instruct  (TP-1)  [default]
#   70b      Llama-3.3-70B-Instruct (TP-8)
#   8b,70b   both
#
# Run duration (--duration): request count + QPS sweep.
#   full    (default)  1024 requests, QPS 4,4.5,5,5.5, selected --models.
#   quick              128 requests, QPS 4,4.5,5,5.5, 8B model only.
#   (both durations sweep all LLM-42 ratios.)
#
# The 70B model is skipped entirely when fewer than 4 GPUs are visible. With
# >=4 GPUs it defaults to TP-8 and is auto-reduced to the largest power of two
# <= NUM_GPUS (e.g. TP-4 on a 4-GPU node). Override detection with NUM_GPUS=.
#
# Usage:
#   ./run.sh                               # full sweep, 8B model
#   ./run.sh --models 8b --duration quick  # quick smoke test
#   ./run.sh --models 8b,70b --force       # both models, force re-run
#   NUM_GPUS=8 QPS_VALUES=3,4 ./run.sh
#
# Env overrides (optional): MODEL + TP_SIZE run a single explicit model instead
# of --models; NUM_REQUESTS replaces the duration's request count; QPS_VALUES
# replaces the duration's QPS sweep; DATASET_CONFIGS (space-separated) replaces
# the default mixed_workload dataset list; MIXED_WORKLOAD_TOTAL sizes the
# mixed_workload prompt pool built on the first run (default 2048).

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

# ---- Common settings (exported so run_balanced_online_benchmark.sh picks them up) ----
export HOST="${HOST:-0.0.0.0}"
export BASE_PORT="${BASE_PORT:-30005}"

# Forward optional config overrides
[ -n "${CONFIG_NAMES:-}" ]              && export CONFIG_NAMES
[ -n "${LLM42_RATIOS:-}" ]              && export LLM42_RATIOS
[ -n "${TOKENIZER:-}" ]                 && export TOKENIZER
[ -n "${DATASET_PATH:-}" ]              && export DATASET_PATH
[ -n "${BACKEND:-}" ]                   && export BACKEND
[ -n "${DETERMINISTIC_SEED:-}" ]        && export DETERMINISTIC_SEED
[ -n "${SHAREGPT_CONTEXT_LEN:-}" ]      && export SHAREGPT_CONTEXT_LEN
[ -n "${SERVER_STARTUP_TIMEOUT:-}" ]    && export SERVER_STARTUP_TIMEOUT
[ -n "${DISABLE_CUSTOM_ALL_REDUCE:-}" ] && export DISABLE_CUSTOM_ALL_REDUCE
[ -n "${ENABLE_TORCH_SYMM_MEM:-}" ]     && export ENABLE_TORCH_SYMM_MEM

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

# ---- Quick mode runs only the 8B model ----
# --duration quick is a fast smoke test: drop 70B so the QPS sweep stays quick.
# Full mode honours the --models / MODEL selection as given.
if [ "$DURATION" = "quick" ]; then
    _kept=()
    for _e in "${MODELS[@]}"; do
        [[ "${_e,,}" == *70b* ]] && continue
        _kept+=("$_e")
    done
    if [ "${#_kept[@]}" -ne "${#MODELS[@]}" ]; then
        echo "NOTE: quick mode (--duration quick) runs only the 8B model; skipping 70B."
    fi
    [ "${#_kept[@]}" -eq 0 ] && _kept=("$(model_spec 8b)")
    MODELS=("${_kept[@]}")
    MODELS_ARG="8b"
    unset MODEL
fi

# ---- Duration-specific configuration (--duration) ----
# quick: 128 requests, full QPS sweep (4,4.5,5,5.5), 8B model only.
# full:  1024 requests, QPS 4,4.5,5,5.5, selected --models.
# Both durations sweep all LLM-42 ratios (set in run_balanced_online_benchmark.sh).
if [ "$DURATION" = "quick" ]; then
    DEFAULT_NUM_REQUESTS=128
    DEFAULT_QPS_VALUES="4,4.5,5,5.5"
else
    DEFAULT_NUM_REQUESTS=1024
    DEFAULT_QPS_VALUES="4,4.5,5,5.5"
fi

# NUM_REQUESTS / QPS_VALUES: explicit env values override the duration defaults.
export NUM_REQUESTS="${NUM_REQUESTS:-$DEFAULT_NUM_REQUESTS}"
export QPS_VALUES="${QPS_VALUES:-$DEFAULT_QPS_VALUES}"

# ---- Dataset configurations ----
# DATASET_CONFIGS env (space-separated) overrides the default mixed_workload.
declare -a DATASET_ARRAY
if [ -n "${DATASET_CONFIGS:-}" ]; then
    IFS=' ' read -ra DATASET_ARRAY <<< "$DATASET_CONFIGS"
else
    DATASET_ARRAY=(
        "mixed_workload"
    )
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
echo "Multi-Model Online (QPS) Benchmarks"
echo "=============================================="
echo "Duration: $DURATION"
echo "GPU:      $_GPU_TYPE  (NUM_GPUS=$NUM_GPUS)"
echo "Models:   ${#MODELS[@]} (${MODEL:-$MODELS_ARG})"
echo "Requests: $NUM_REQUESTS per (dataset, qps)"
echo "QPS:      $QPS_VALUES"
echo "Datasets: ${DATASET_ARRAY[*]}"
echo "=============================================="
echo ""

# ---- Prepare the mixed_workload dataset (first run only) ----
# prepare_mixed_workload.py writes ~/.cache/llm42_bench/mixed_workload_sglang.jsonl
# (auto-detected by run_balanced_online_benchmark.sh) and self-skips when the file
# already exists, so only the first run builds it; later runs reuse it as-is.
# MIXED_WORKLOAD_TOTAL sizes the prompt pool (must be >= the largest NUM_REQUESTS).
for _ds in "${DATASET_ARRAY[@]}"; do
    if [ "$_ds" = "mixed_workload" ]; then
        echo "Preparing mixed_workload dataset (first run only)..."
        $PYTHON_CMD "${ROOT}/prepare_mixed_workload.py" --total "${MIXED_WORKLOAD_TOTAL:-2048}"
        echo ""
        break
    fi
done

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

    # ---- Per-model run directory base (matches run_balanced_online_benchmark.sh) ----
    _MODEL_TAG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')
    _RUN_DIR_BASE="${ROOT}/runs/${_GPU_TYPE}_${_MODEL_TAG}-tp${TP_SIZE}_${ATTENTION_BACKEND}_n${NUM_REQUESTS}"

    echo "=============================================="
    echo "Model:    $MODEL  (TP=$TP_SIZE)"
    echo "Run Base: ${_RUN_DIR_BASE}_*_online"
    echo "=============================================="
    echo ""

    # Run all datasets for this model in a single balanced online benchmark pass.
    export DATASET_CONFIGS_LIST="${DATASET_ARRAY[*]}"
    echo "Launching balanced online benchmark across all datasets..."
    echo ""
    "${ROOT}/run_balanced_online_benchmark.sh" $FORCE_FLAG

    echo ""
    echo "Datasets completed.  Generating outputs..."

    # Generate plots and summaries per dataset run directory.
    for ds in "${DATASET_ARRAY[@]}"; do
        _ds_run_dir="${_RUN_DIR_BASE}_${ds}_online"
        _ds_results_dir="${_ds_run_dir}/results"
        if [ ! -f "${_ds_results_dir}/benchmark_results.jsonl" ]; then
            echo "  WARNING: no results for $ds, skipping post-processing"
            continue
        fi

        echo ""
        echo "--- $ds ---"

        echo "  Generating summary CSV..."
        $PYTHON_CMD "${ROOT}/summarize_online_csv.py" \
            --input "${_ds_results_dir}/benchmark_results.jsonl" \
            --output "${_ds_results_dir}/summary.csv"

        echo "  Generating per-request CSV..."
        $PYTHON_CMD "${ROOT}/export_per_request_csv.py" \
            --input "${_ds_results_dir}/benchmark_results.jsonl" \
            --output "${_ds_results_dir}/per_request_data.csv"

        echo "  Generating plots..."
        $PYTHON_CMD "${ROOT}/plot.py" \
            --results-dirs "${_ds_results_dir}" \
            --output-dir "${_ds_run_dir}"
    done

    echo ""
    echo "Model done: $MODEL"
    echo ""
done

echo "=============================================="
echo "All Done!  (duration=$DURATION)"
echo "=============================================="
