#!/usr/bin/env bash
set -euo pipefail

# Profile rollback (recompute) statistics for the paper's Figure 9b.
#
# For each model, launches SGLang servers across the (window_size,
# verify_batch_size) grid, runs the same workload against each, collects
# rollback stats, and finally plots a combined recompute-cost bar chart
# (8B vs 70B).
#
# The 70B model is skipped entirely when fewer than 4 GPUs are visible. With
# >=4 GPUs, TP size defaults per model but auto-falls back to the largest power
# of two that fits the available GPUs (e.g. 70B: TP-8 -> TP-4 on a 4-GPU node)
# -- same logic as 4-figure9a.
#
# Usage:
#   ./run.sh                                   # 8B + 70B (auto TP)
#   ./run.sh --force                           # re-run even if results exist
#   WINDOW_SIZES="32,64,128" VERIFY_BATCH_SIZES="1" ./run.sh
#   NUM_GPUS=4 ./run.sh                        # force GPU count
#   MODEL=meta-llama/Llama-3.3-70B-Instruct TP_SIZE=8 ./run.sh  # single model

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ---- Parse flags ----
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
        *) echo "Unknown argument: $arg" >&2
           echo "Usage: $0 [--force]" >&2
           exit 1 ;;
    esac
done

# ---- GPU count ----
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "Error: No GPUs detected. Set NUM_GPUS manually."
    exit 1
fi

# Largest power of two <= n. TP must be a power of two that divides the head
# count, so we round the available GPU count down to one.
largest_pow2_le() {
    local n=$1 p=1
    while (( p * 2 <= n )); do p=$((p * 2)); done
    echo "$p"
}

# ---- Server Configuration ----
HOST="${HOST:-0.0.0.0}"
BASE_PORT="${BASE_PORT:-30005}"
ENABLE_SGLANG_DETERMINISM="${ENABLE_SGLANG_DETERMINISM:-0}"
ENABLE_LLM42="${ENABLE_LLM42:-3}"
SERVER_STARTUP_TIMEOUT=${SERVER_STARTUP_TIMEOUT:-300}

# ---- Models to profile: "model_path  requested_tp  label" ----
# 70B defaults to TP-8 and auto-reduces to fit the available GPUs.
# Override the set with a single model via: MODEL=... [TP_SIZE=...] ./run.sh
if [[ -n "${MODEL:-}" ]]; then
    MODELS=("${MODEL} ${TP_SIZE:-1} $(basename "${MODEL}")")
else
    MODELS=(
        "meta-llama/Llama-3.1-8B-Instruct 1 Llama-3-8B"
        "meta-llama/Llama-3.3-70B-Instruct 8 Llama-3-70B"
    )
fi

# ---- Profile Grid ----
# Figure 9b uses verify batch size 1 and window sizes from 16 to 1024 tokens.
WINDOW_SIZES="${WINDOW_SIZES:-16,32,64,128,256,512,1024}"
VERIFY_BATCH_SIZES="${VERIFY_BATCH_SIZES:-1}"

# ---- Client Configuration ----
QPS="${QPS:-8}"
ORDER_SEED="${ORDER_SEED:-132}"
ARRIVAL_SEED="${ARRIVAL_SEED:-16}"
SELECT_SEED=${SELECT_SEED:-42}
TOKENIZER=${TOKENIZER:-}
DATASET_PATH=${DATASET_PATH:-}
NUM_PROMPTS=${NUM_PROMPTS:-128}
SHAREGPT_CONTEXT_LEN=${SHAREGPT_CONTEXT_LEN:-16384}
EXTRA_REQUEST_BODY=${EXTRA_REQUEST_BODY:-'{"temperature":0}'}
BACKEND=${BACKEND:-sglang}
DETERMINISTIC_RATIO=${DETERMINISTIC_RATIO:-1.0}
WARMUP_REQUESTS=${WARMUP_REQUESTS:-0}

# ---- GPU-aware defaults + shared server helpers (kill_server, ...) ----
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../gpu_utils.sh"
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
        if (( ws * bs < 1025 )); then
            PROFILE_CONFIGS+=("ws=${ws},bs=${bs}")
        fi
    done
done
NUM_CONFIGS=${#PROFILE_CONFIGS[@]}

# ---- Cleanup (SERVER_PIDS is global so the EXIT trap can see it) ----
declare -a SERVER_PIDS=()

cleanup() {
    echo ""
    echo "Shutting down servers..."
    for pid in "${SERVER_PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
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

# ---- Per-model profiling ----
declare -a RUN_DIRS=()
declare -a RUN_LABELS=()

# profile_model MODEL REQUESTED_TP LABEL_BASE
# TP_SIZE / NUM_SLOTS are declared local here but stay visible to the helper
# functions (get_slot_devices) via bash dynamic scoping.
profile_model() {
    local MODEL="$1" REQ_TP="$2" LABEL_BASE="$3"

    # Auto-reduce TP to fit the available GPUs (see 4-figure9a).
    local TP_SIZE=$REQ_TP
    if (( REQ_TP > NUM_GPUS )); then
        TP_SIZE=$(largest_pow2_le "$NUM_GPUS")
        echo "NOTE: ${LABEL_BASE} requested TP-${REQ_TP} but only ${NUM_GPUS} GPU(s) available; using TP-${TP_SIZE}"
    fi

    local NUM_SLOTS=$((NUM_GPUS / TP_SIZE))
    if (( NUM_SLOTS == 0 || NUM_SLOTS * TP_SIZE != NUM_GPUS )); then
        echo "Error: NUM_GPUS ($NUM_GPUS) must be a positive multiple of TP-${TP_SIZE} for ${LABEL_BASE}."
        exit 1
    fi

    local LABEL="${LABEL_BASE} (TP-${TP_SIZE})"
    local _MODEL_TAG
    _MODEL_TAG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' | sed 's/^meta-//')
    local SERVER_LOG_DIR="${ROOT}/server_logs/${_GPU_TYPE}_${_MODEL_TAG}-tp${TP_SIZE}_${ATTENTION_BACKEND}"
    local RUN_DIR="${ROOT}/runs/${_GPU_TYPE}_${_MODEL_TAG}-tp${TP_SIZE}_${ATTENTION_BACKEND}"

    # Resume: skip a model whose summary.json already exists (unless --force).
    # Still record it so the combined bar chart includes this model.
    if [[ -f "${RUN_DIR}/summary.json" && "$FORCE" -ne 1 ]]; then
        echo "Skipping (already done): ${LABEL} -> ${RUN_DIR} (use --force to re-run)"
        RUN_DIRS+=("$RUN_DIR")
        RUN_LABELS+=("$LABEL")
        return 0
    fi

    # (Re)running: clear any partial output and recreate the directories.
    rm -rf "$SERVER_LOG_DIR" "$RUN_DIR"
    mkdir -p "$SERVER_LOG_DIR" "$RUN_DIR"

    # ---- Display ----
    echo "=============================================="
    echo "Rollback Profiler -- ${LABEL}"
    echo "=============================================="
    echo "Model:            $MODEL"
    echo "GPUs:             $NUM_GPUS (TP=$TP_SIZE, $NUM_SLOTS slots)"
    echo "Attention:        $ATTENTION_BACKEND"
    echo "LLM42:            enable=$ENABLE_LLM42"
    echo "Window Sizes:     $WINDOW_SIZES"
    echo "Verify Batch:     $VERIFY_BATCH_SIZES"
    echo "Configs:          $NUM_CONFIGS total"
    echo "QPS:              $QPS"
    echo "Num Prompts:      $NUM_PROMPTS"
    echo "Select Seed:      $SELECT_SEED"
    echo "Det Ratio:        $DETERMINISTIC_RATIO"
    echo "Server Logs:      $SERVER_LOG_DIR"
    echo "Run Dir:          $RUN_DIR"
    echo "=============================================="
    echo ""

    # ---- Run Batches ----
    local NUM_BATCHES=$(( (NUM_CONFIGS + NUM_SLOTS - 1) / NUM_SLOTS ))
    echo "Running $NUM_CONFIGS configs in $NUM_BATCHES batch(es) ($NUM_SLOTS slots per batch)"
    echo ""

    local batch START_IDX END_IDX BATCH_SIZE i CFG_IDX CFG WS BS PORT DEVICES LOG_FILE
    local BASE_URLS BATCH_PROFILE_CONFIGS URL RESULT pid
    for ((batch=0; batch<NUM_BATCHES; batch++)); do
        START_IDX=$((batch * NUM_SLOTS))
        END_IDX=$((START_IDX + NUM_SLOTS))
        if [ $END_IDX -gt $NUM_CONFIGS ]; then END_IDX=$NUM_CONFIGS; fi
        BATCH_SIZE=$((END_IDX - START_IDX))

        echo "=============================================="
        echo "[$LABEL] Batch $((batch+1))/$NUM_BATCHES (configs $((START_IDX+1))-$END_IDX)"
        echo "=============================================="

        # Reset server PIDs for this batch
        SERVER_PIDS=()
        BASE_URLS=""

        # Launch one server per config in this batch
        echo "Launching $BATCH_SIZE servers..."
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
            echo "  Slot $i [GPUs $DEVICES] ws=$WS bs=$BS port=$PORT (PID $!) -> $LOG_FILE"
            sleep 2
        done

        echo ""
        echo "Waiting for servers to be ready (timeout ${SERVER_STARTUP_TIMEOUT}s)..."
        for ((i=0; i<BATCH_SIZE; i++)); do
            PORT=$((BASE_PORT + i))
            URL="http://127.0.0.1:${PORT}"
            echo -n "  Slot $i ($URL) ... "
            if wait_for_server "$URL" "$SERVER_STARTUP_TIMEOUT"; then
                echo "ready"
            else
                echo "FAILED"
                echo "ERROR: Server on slot $i failed to start. Check $SERVER_LOG_DIR"
                exit 1
            fi
        done
        echo ""
        echo "All servers ready!"
        echo ""

        # Build profile-configs string for the Python script (semicolon-separated)
        BATCH_PROFILE_CONFIGS=""
        for ((i=0; i<BATCH_SIZE; i++)); do
            CFG_IDX=$((START_IDX + i))
            if [ -n "$BATCH_PROFILE_CONFIGS" ]; then BATCH_PROFILE_CONFIGS="${BATCH_PROFILE_CONFIGS};"; fi
            BATCH_PROFILE_CONFIGS="${BATCH_PROFILE_CONFIGS}${PROFILE_CONFIGS[$CFG_IDX]}"
        done

        # Run profiling
        local -a cmd=(
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
        RESULT=$?

        if [ $RESULT -ne 0 ]; then
            echo "ERROR: Profiling batch $((batch+1)) failed (exit $RESULT)"
            exit $RESULT
        fi

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

    # ---- Summarize (writes summary.json consumed by the plot) ----
    echo "=============================================="
    echo "[$LABEL] Generating summary..."
    echo "=============================================="
    $PYTHON_CMD "${ROOT}/summarize_profiles.py" --run-dir "${RUN_DIR}"
    echo ""

    RUN_DIRS+=("$RUN_DIR")
    RUN_LABELS+=("$LABEL")
}

# ---- Profile each model ----
for MODEL_ENTRY in "${MODELS[@]}"; do
    read -r _M _T _L <<< "$MODEL_ENTRY"
    if [[ "${_M,,}" == *70b* ]] && (( NUM_GPUS < 4 )); then
        echo "NOTE: skipping ${_L} (70B) -- requires >=4 GPUs but only ${NUM_GPUS} visible."
        continue
    fi
    profile_model "$_M" "$_T" "$_L"
done

# ---- Combined recompute-cost bar chart (e.g. 8B vs 70B) ----
echo "=============================================="
echo "Generating combined recompute cost bar chart..."
echo "=============================================="

PLOT_ARGS=()
for i in "${!RUN_DIRS[@]}"; do
    PLOT_ARGS+=(--run-dir "${RUN_DIRS[$i]}" --label "${RUN_LABELS[$i]}")
done

COMBINED_PLOT="${ROOT}/runs/${_GPU_TYPE}_${ATTENTION_BACKEND}_recompute_cost_combined.pdf"
$PYTHON_CMD "${ROOT}/plot.py" "${PLOT_ARGS[@]}" --output "$COMBINED_PLOT"

echo ""
echo "=============================================="
echo "All profiling completed!"
echo "Run dirs:"
for d in "${RUN_DIRS[@]}"; do echo "  - $d"; done
echo "Combined plot: $COMBINED_PLOT"
echo "=============================================="

exit 0
