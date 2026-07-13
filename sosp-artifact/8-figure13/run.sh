#!/usr/bin/env bash
set -euo pipefail

# Profile Rollback Statistics Across LLM42 Configurations (paper Figure 13).
# For each model it launches SGLang servers with different
# (window_size, verify_batch_size) combos, runs the same workload against each,
# collects rollback stats, and renders the recompute-cost / throughput heatmaps.
#
# Which models to run (--models): comma-separated list of 8b and/or 70b.
#   8b       Llama-3.1-8B-Instruct  (TP-1)  [default]
#   70b      Llama-3.3-70B-Instruct (TP-8)
#   8b,70b   both
#
# Run duration (--duration): grid size + workload.
#   full    (default)  All 25 (window_size, verify_batch_size) configs, 1024 prompts.
#   quick              Reduced 10-config triangular grid, 128 prompts.
#
# The 70B model is skipped entirely when fewer than 4 GPUs are visible. With
# >=4 GPUs it defaults to TP-8 and is auto-reduced to the largest power of two
# <= NUM_GPUS (e.g. TP-4 on a 4-GPU node). Override detection with NUM_GPUS=.
#
# Usage:
#   ./run.sh                                 # full grid, 8B model
#   ./run.sh --models 8b --duration quick    # quick smoke test (8B, 10 configs)
#   ./run.sh --models 8b,70b                 # both models, full grid
#   WINDOW_SIZES="32,64,128" VERIFY_BATCH_SIZES="4,8,16" ./run.sh
#   MODEL=meta-llama/Llama-3.1-70B-Instruct TP_SIZE=4 ./run.sh
#
# Env overrides (optional): MODEL + TP_SIZE run a single explicit model instead
# of --models; NUM_PROMPTS replaces the duration's prompt count.
#
# By default a model whose summary.json already exists is skipped (resume);
# pass --force to re-run and overwrite it.

# ---- Parse flags ----
DURATION="full"
MODELS_ARG=""
FORCE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --duration)   if [ $# -lt 2 ]; then echo "Error: --duration requires a value (quick|full)" >&2; exit 1; fi
                      DURATION="$2"; shift ;;
        --duration=*) DURATION="${1#*=}" ;;
        --models)     if [ $# -lt 2 ]; then echo "Error: --models requires a value (8b|70b|8b,70b)" >&2; exit 1; fi
                      MODELS_ARG="$2"; shift ;;
        --models=*)   MODELS_ARG="${1#*=}" ;;
        --force)      FORCE=1 ;;
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
export PYTHONPATH="${PYTHONPATH:-}:${ROOT}/../0-test-determinism/python"

# ---- Server Configuration ----
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "Error: No GPUs detected. Set NUM_GPUS manually."
    exit 1
fi
HOST="${HOST:-0.0.0.0}"
BASE_PORT="${BASE_PORT:-30005}"
ENABLE_SGLANG_DETERMINISM="${ENABLE_SGLANG_DETERMINISM:-0}"
ENABLE_LLM42="${ENABLE_LLM42:-3}"
SERVER_STARTUP_TIMEOUT=${SERVER_STARTUP_TIMEOUT:-300}

# largest power of two <= n (tensor-parallel fallback)
largest_pow2_le() { local n=$1 p=1; while (( p*2 <= n )); do p=$((p*2)); done; echo "$p"; }

# ---- Profile Grid ----
# Grid axes (WINDOW_SIZES, VERIFY_BATCH_SIZES) and the ws*bs product cap
# (MAX_PRODUCT) are duration-specific; their defaults are set in the DURATION block
# below and may be overridden via the matching environment variables.

# ---- Client Configuration ----
QPS="${QPS:-8}"
ORDER_SEED="${ORDER_SEED:-132}"
ARRIVAL_SEED="${ARRIVAL_SEED:-16}"
SELECT_SEED=${SELECT_SEED:-42}
TOKENIZER=${TOKENIZER:-}
DATASET_PATH=${DATASET_PATH:-}
SHAREGPT_CONTEXT_LEN=${SHAREGPT_CONTEXT_LEN:-16384}
EXTRA_REQUEST_BODY=${EXTRA_REQUEST_BODY:-'{"temperature":0}'}
BACKEND=${BACKEND:-sglang}
DETERMINISTIC_RATIO=${DETERMINISTIC_RATIO:-1.0}
WARMUP_REQUESTS=${WARMUP_REQUESTS:-0}

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

# ---- Grid + workload size (--duration) ----
# quick: reduced triangular grid (window <=128, batch <=16, ws*bs <=128), 128 prompts.
# full:  the complete 25-config grid, 1024 prompts.
if [ "$DURATION" = "quick" ]; then
    DEFAULT_NUM_PROMPTS=128
    DEFAULT_WINDOW_SIZES="16,32,64,128"
    DEFAULT_VERIFY_BATCH_SIZES="1,2,4,8,16"
    DEFAULT_MAX_PRODUCT=128
else
    DEFAULT_NUM_PROMPTS=1024
    DEFAULT_WINDOW_SIZES="16,32,64,128,256"
    DEFAULT_VERIFY_BATCH_SIZES="1,2,4,8,16,32,64"
    DEFAULT_MAX_PRODUCT=1024
fi
# NUM_PROMPTS: an explicit env value overrides the mode default.
NUM_PROMPTS="${NUM_PROMPTS:-$DEFAULT_NUM_PROMPTS}"
# Profile grid: explicit env values override the mode defaults.
WINDOW_SIZES="${WINDOW_SIZES:-$DEFAULT_WINDOW_SIZES}"
VERIFY_BATCH_SIZES="${VERIFY_BATCH_SIZES:-$DEFAULT_VERIFY_BATCH_SIZES}"
MAX_PRODUCT="${MAX_PRODUCT:-$DEFAULT_MAX_PRODUCT}"

# ---- GPU utils (sets GPU_SHORT_NAME, ATTENTION_BACKEND) ----
source "${ROOT}/../gpu_utils.sh"
_GPU_TYPE="$GPU_SHORT_NAME"

# ---- Python ----
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: Python not found"; exit 1
fi

# ---- Build Config Grid (shared across models) ----
IFS=',' read -ra WS_ARRAY <<< "$WINDOW_SIZES"
IFS=',' read -ra BS_ARRAY <<< "$VERIFY_BATCH_SIZES"
declare -a PROFILE_CONFIGS=()
for ws in "${WS_ARRAY[@]}"; do
    for bs in "${BS_ARRAY[@]}"; do
        if (( ws * bs <= MAX_PRODUCT )); then
            PROFILE_CONFIGS+=("ws=${ws},bs=${bs}")
        fi
    done
done
NUM_CONFIGS=${#PROFILE_CONFIGS[@]}

# ---- Display ----
echo "=============================================="
echo "Rollback Profiler (Figure 13)"
echo "=============================================="
echo "Duration:         $DURATION"
echo "Models:           ${#MODELS[@]} (${MODEL:-$MODELS_ARG})"
echo "GPUs:             $NUM_GPUS"
echo "Attention:        $ATTENTION_BACKEND"
echo "LLM42:            enable=$ENABLE_LLM42"
echo "Window Sizes:     $WINDOW_SIZES"
echo "Verify Batch:     $VERIFY_BATCH_SIZES"
echo "Max ws*bs:        $MAX_PRODUCT"
echo "Configs:          $NUM_CONFIGS per model"
echo "QPS:              $QPS"
echo "Num Prompts:      $NUM_PROMPTS"
echo "Det Ratio:        $DETERMINISTIC_RATIO"
echo "=============================================="
echo ""

# ---- Cleanup (SERVER_PIDS stays global for the trap) ----
declare -a SERVER_PIDS=()
cleanup() {
    echo ""
    echo "Shutting down servers..."
    for pid in "${SERVER_PIDS[@]:-}"; do
        if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
            kill_server "$pid"
        fi
    done
    wait 2>/dev/null || true
    echo "All servers stopped."
}
trap cleanup EXIT INT TERM

# ---- Helper Functions ----
get_slot_devices() {
    local slot_id=$1
    local first_gpu=$((slot_id * TP_SIZE))
    local devices=""
    for ((g=first_gpu; g<first_gpu+TP_SIZE; g++)); do
        devices="${devices:+${devices},}${g}"
    done
    echo "$devices"
}

wait_for_server() {
    local url="$1"
    local timeout_secs="$2"
    local start=$SECONDS
    while (( SECONDS - start < timeout_secs )); do
        if timeout 5 curl -s "${url}/v1/models" 2>/dev/null | grep -q '"object":"list"'; then
            return 0
        fi
        sleep 3
    done
    return 1
}

parse_profile_config() {
    local cfg_str="$1"
    local key="$2"
    echo "$cfg_str" | tr ',' '\n' | grep "^${key}=" | cut -d= -f2
}

# ---- Run one model across the full profile grid ----
run_model() {
    local MODEL="$1"
    local TP_SIZE="$2"

    local NUM_SLOTS=$((NUM_GPUS / TP_SIZE))
    if [ "$NUM_SLOTS" -eq 0 ]; then
        echo "ERROR: TP_SIZE ($TP_SIZE) exceeds NUM_GPUS ($NUM_GPUS) for $MODEL; skipping."
        return 1
    fi

    local _MODEL_TAG
    _MODEL_TAG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' | sed 's/^meta-//')
    local SERVER_LOG_DIR="${ROOT}/server_logs/${_GPU_TYPE}_${_MODEL_TAG}-tp${TP_SIZE}_${ATTENTION_BACKEND}"
    local RUN_DIR="${ROOT}/runs/${_GPU_TYPE}_${_MODEL_TAG}-tp${TP_SIZE}_${ATTENTION_BACKEND}"

    # Resume: skip the (expensive) benchmark when summary.json already exists,
    # but still regenerate the heatmaps/paper figures from it (unless --force).
    if [[ -f "${RUN_DIR}/summary.json" && "$FORCE" -ne 1 ]]; then
        echo "Skipping benchmark (already done): $MODEL -> ${RUN_DIR}"
        echo "  Regenerating heatmaps from existing data (use --force to re-run the benchmark)."
        $PYTHON_CMD "${ROOT}/plot.py" --run-dir "${RUN_DIR}"
        return 0
    fi

    # (Re)running: clear any partial output and recreate the directories.
    rm -rf "$SERVER_LOG_DIR" "$RUN_DIR"
    mkdir -p "$SERVER_LOG_DIR" "$RUN_DIR"

    echo "=============================================="
    echo "Model:    $MODEL  (TP=$TP_SIZE, $NUM_SLOTS slots)"
    echo "Run Dir:  $RUN_DIR"
    echo "=============================================="
    for cfg in "${PROFILE_CONFIGS[@]}"; do echo "  - $cfg"; done
    echo ""

    local NUM_BATCHES=$(( (NUM_CONFIGS + NUM_SLOTS - 1) / NUM_SLOTS ))
    echo "Running $NUM_CONFIGS configs in $NUM_BATCHES batch(es) ($NUM_SLOTS slots per batch)"
    echo ""

    local batch START_IDX END_IDX BATCH_SIZE
    for ((batch=0; batch<NUM_BATCHES; batch++)); do
        START_IDX=$((batch * NUM_SLOTS))
        END_IDX=$((START_IDX + NUM_SLOTS))
        if [ $END_IDX -gt $NUM_CONFIGS ]; then END_IDX=$NUM_CONFIGS; fi
        BATCH_SIZE=$((END_IDX - START_IDX))

        echo "=============================================="
        echo "Batch $((batch+1))/$NUM_BATCHES (configs $((START_IDX+1))-$END_IDX)"
        echo "=============================================="

        # Reset server PIDs for this batch
        SERVER_PIDS=()
        local BASE_URLS=""

        # Launch one server per config in this batch
        echo "Launching $BATCH_SIZE servers..."
        local i CFG_IDX CFG WS BS PORT DEVICES LOG_FILE URL
        for ((i=0; i<BATCH_SIZE; i++)); do
            CFG_IDX=$((START_IDX + i))
            CFG="${PROFILE_CONFIGS[$CFG_IDX]}"
            WS=$(parse_profile_config "$CFG" "ws")
            BS=$(parse_profile_config "$CFG" "bs")
            PORT=$((BASE_PORT + i))
            DEVICES=$(get_slot_devices $i)
            LOG_FILE="$SERVER_LOG_DIR/server_ws${WS}_bs${BS}_port${PORT}.log"

            CUDA_VISIBLE_DEVICES=$DEVICES $PYTHON_CMD -m sglang.launch_server \
                --model-path "$MODEL" \
                --host "$HOST" \
                --port "$PORT" \
                --tp "$TP_SIZE" \
                --attention-backend "$ATTENTION_BACKEND" \
                --disable-radix-cache \
                --disable-chunked-prefix-cache \
                --disable-overlap-schedule \
                --enable-metrics \
                --random-seed 42 \
                --enable-deterministic-inference $ENABLE_SGLANG_DETERMINISM \
                --chunked-prefill-size -1 \
                --enable-llm42 "$ENABLE_LLM42" \
                --llm42-window-size "$WS" \
                --llm42-verify-batch-size "$BS" \
                $([ "${DISABLE_CUSTOM_ALL_REDUCE:-0}" = "1" ] && echo "--disable-custom-all-reduce" || true) \
                $([ "${ENABLE_TORCH_SYMM_MEM:-0}" = "1" ] && echo "--enable-torch-symm-mem" || true) \
                > "$LOG_FILE" 2>&1 &

            SERVER_PIDS+=($!)
            BASE_URLS="${BASE_URLS:+${BASE_URLS},}http://127.0.0.1:${PORT}"
            echo "  Slot $i [GPUs $DEVICES] ws=$WS bs=$BS port=$PORT (PID $!) → $LOG_FILE"
            sleep 2
        done

        echo ""
        echo "Waiting for servers to be ready (timeout ${SERVER_STARTUP_TIMEOUT}s)..."
        for ((i=0; i<BATCH_SIZE; i++)); do
            PORT=$((BASE_PORT + i))
            URL="http://127.0.0.1:${PORT}"
            echo -n "  Slot $i ($URL) ... "
            if wait_for_server "$URL" "$SERVER_STARTUP_TIMEOUT"; then
                echo "✓"
            else
                echo "✗ (FAILED)"
                echo "ERROR: Server on slot $i failed to start. Check $SERVER_LOG_DIR"
                return 1
            fi
        done
        echo ""
        echo "All servers ready!"
        echo ""

        # Build profile-configs string for the Python script (semicolon-separated)
        local BATCH_PROFILE_CONFIGS=""
        for ((i=0; i<BATCH_SIZE; i++)); do
            CFG_IDX=$((START_IDX + i))
            if [ -n "$BATCH_PROFILE_CONFIGS" ]; then BATCH_PROFILE_CONFIGS="${BATCH_PROFILE_CONFIGS};"; fi
            BATCH_PROFILE_CONFIGS="${BATCH_PROFILE_CONFIGS}${PROFILE_CONFIGS[$CFG_IDX]}"
        done

        # Run profiling
        local cmd=(
            $PYTHON_CMD "${ROOT}/run_profile.py"
            --backend "${BACKEND}"
            --base-urls "${BASE_URLS}"
            --profile-configs "${BATCH_PROFILE_CONFIGS}"
            --model "${MODEL}"
            --num-prompts "${NUM_PROMPTS}"
            --select-seed "${SELECT_SEED}"
            --qps "${QPS}"
            --order-seed "${ORDER_SEED}"
            --arrival-seed "${ARRIVAL_SEED}"
            --deterministic-ratio "${DETERMINISTIC_RATIO}"
            --extra-request-body "${EXTRA_REQUEST_BODY}"
            --warmup-requests "${WARMUP_REQUESTS}"
            --ignore-eos
            --output-dir "${RUN_DIR}"
        )
        if [[ -n "${TOKENIZER}" ]]; then cmd+=(--tokenizer "${TOKENIZER}"); fi
        if [[ -n "${DATASET_PATH}" ]]; then cmd+=(--dataset-path "${DATASET_PATH}"); fi
        if [[ -n "${SHAREGPT_CONTEXT_LEN}" ]]; then cmd+=(--sharegpt-context-len "${SHAREGPT_CONTEXT_LEN}"); fi

        echo "Command: ${cmd[*]}"
        echo ""
        "${cmd[@]}"

        echo ""
        echo "Batch $((batch+1)) completed."

        # Kill servers before next batch
        echo "Stopping servers for this batch..."
        for pid in "${SERVER_PIDS[@]}"; do
            if kill -0 "$pid" 2>/dev/null; then
                kill_server "$pid"
            fi
        done
        wait 2>/dev/null || true
        SERVER_PIDS=()
        echo "Servers stopped."
        echo ""
    done

    # ---- Summarize + heatmaps for this model ----
    echo "=============================================="
    echo "Generating summary for $MODEL..."
    echo "=============================================="
    $PYTHON_CMD "${ROOT}/summarize_profiles.py" --run-dir "${RUN_DIR}"

    echo ""
    echo "=============================================="
    echo "Generating heatmaps for $MODEL..."
    echo "=============================================="
    $PYTHON_CMD "${ROOT}/plot.py" --run-dir "${RUN_DIR}"

    echo ""
    echo "Model done: $MODEL  ->  $RUN_DIR"
    echo ""
}

# ---- Run each model ----
for MODEL_ENTRY in "${MODELS[@]}"; do
    read -r MODEL_PATH REQ_TP <<< "$MODEL_ENTRY"
    TP_SIZE_RUN=$REQ_TP

    # Skip the 70B model when the node has fewer than 4 GPUs.
    if [[ "${MODEL_PATH,,}" == *70b* ]] && (( NUM_GPUS < 4 )); then
        echo "NOTE: skipping $(basename "$MODEL_PATH") (70B) -- requires >=4 GPUs but only ${NUM_GPUS} visible."
        continue
    fi

    # Tensor-parallel fallback: cap to the largest power of two <= NUM_GPUS.
    if (( NUM_GPUS >= 1 && TP_SIZE_RUN > NUM_GPUS )); then
        NEW_TP=$(largest_pow2_le "$NUM_GPUS")
        echo "NOTE: $(basename "$MODEL_PATH") requested TP-${TP_SIZE_RUN} but only ${NUM_GPUS} GPU(s) available; using TP-${NEW_TP}"
        TP_SIZE_RUN=$NEW_TP
    fi

    run_model "$MODEL_PATH" "$TP_SIZE_RUN"
done

echo "=============================================="
echo "All profiling completed!"
echo "=============================================="

exit 0
