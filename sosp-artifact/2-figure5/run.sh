#!/bin/bash
set -euo pipefail

# Run batch size experiment: batch sizes 10, 11 with input 1024, output 512
# Compare non-deterministic vs global-deterministic
#
# By default configs already recorded in benchmark_results.jsonl are skipped
# (per-config resume); pass --force to wipe prior results and re-run everything.

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

# ---------------------------------------------------------------------------
# GPU detection — derive a short name like a100, h100, b200
# ---------------------------------------------------------------------------
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../gpu_utils.sh"

# Server configuration
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
BASE_PORT="${BASE_PORT:-30005}"
TP_SIZE="${TP_SIZE:-1}"
NUM_SLOTS=$((NUM_GPUS / TP_SIZE))
SERVER_STARTUP_TIMEOUT=${SERVER_STARTUP_TIMEOUT:-300}

if [ "$NUM_SLOTS" -eq 0 ] || [ $((NUM_SLOTS * TP_SIZE)) -ne "$NUM_GPUS" ]; then
    echo "Error: NUM_GPUS ($NUM_GPUS) must be a positive multiple of TP_SIZE ($TP_SIZE)"
    exit 1
fi

# Model and server parameters
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}

# Benchmark parameters
INPUT_LEN=${INPUT_LEN:-512}
OUTPUT_LEN=${OUTPUT_LEN:-512}
DETERMINISTIC_SEED=42
BACKEND=sglang

# Batch size to test
BATCH_SIZE="${BATCH_SIZE:-9}"

OUTPUT_DIR="${ROOT}/runs/${GPU_SHORT_NAME}_cover_in${INPUT_LEN}_out${OUTPUT_LEN}_bs${BATCH_SIZE}"
RESULTS_FILE="${OUTPUT_DIR}/benchmark_results.jsonl"

# Reuse a stable directory; with --force wipe prior results, otherwise keep
# RESULTS_FILE and resume only the configs that haven't completed yet.
mkdir -p "$OUTPUT_DIR"
if [ "$FORCE" -eq 1 ]; then
    rm -f "$RESULTS_FILE"
fi

# ---- Build Benchmark Configs ----
# Each entry: "config_name|server_args|batch_size|det_ratio"
BS_MINUS1=$((BATCH_SIZE - 1))
det_ratio=$(python3 -c "print(round(1.0/$BATCH_SIZE, 4))")

declare -a BENCH_CONFIGS=(
    "non_det||${BS_MINUS1}|1.0"
    "non_det||${BATCH_SIZE}|1.0"
    "global_det|--enable-deterministic-inference 2|${BATCH_SIZE}|1.0"
    "llm42|--enable-llm42 3 --llm42-window-size 64 --llm42-verify-batch-size 1|${BATCH_SIZE}|${det_ratio}"
)
NUM_CONFIGS=${#BENCH_CONFIGS[@]}

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

check_server_health() {
    local url="$1"
    local max_retries="${2:-120}"
    local retry_interval="${3:-5}"
    for ((i=1; i<=max_retries; i++)); do
        if curl -s "${url}/v1/models" 2>/dev/null | grep -q '"object":"list"'; then
            return 0
        fi
        sleep "$retry_interval"
    done
    return 1
}

parse_bench_config() {
    local cfg="$1"
    local field="$2"
    echo "$cfg" | cut -d'|' -f"$field"
}

# Extra server args for B200
EXTRA_SERVER_ARGS=""
if [ "${DISABLE_CUSTOM_ALL_REDUCE:-0}" = "1" ]; then
    EXTRA_SERVER_ARGS="$EXTRA_SERVER_ARGS --disable-custom-all-reduce"
fi
if [ "${ENABLE_TORCH_SYMM_MEM:-0}" = "1" ]; then
    EXTRA_SERVER_ARGS="$EXTRA_SERVER_ARGS --enable-torch-symm-mem"
fi

launch_server() {
    local port="$1"
    local config_name="$2"
    local config_args="$3"
    local gpu_devices="$4"
    local log_file="${OUTPUT_DIR}/server_${config_name}_port${port}.log"

    echo "  Launching $config_name on port $port [GPUs $gpu_devices]..." >&2
    echo "  Command: CUDA_VISIBLE_DEVICES=$gpu_devices python -m sglang.launch_server --model-path $MODEL --host 0.0.0.0 --port $port --tp $TP_SIZE --attention-backend $ATTENTION_BACKEND --disable-radix-cache --disable-chunked-prefix-cache --disable-overlap-schedule --enable-metrics --random-seed 42 --chunked-prefill-size -1 --max-running-requests 64 $EXTRA_SERVER_ARGS $config_args" >&2
    echo "  Log file: $log_file" >&2

    CUDA_VISIBLE_DEVICES=$gpu_devices python -m sglang.launch_server \
        --model-path "$MODEL" \
        --host 0.0.0.0 \
        --port "$port" \
        --tp "$TP_SIZE" \
        --attention-backend "$ATTENTION_BACKEND" \
        --disable-radix-cache \
        --disable-chunked-prefix-cache \
        --disable-overlap-schedule \
        --enable-metrics \
        --random-seed 42 \
        --chunked-prefill-size -1 \
        --max-running-requests 64 \
        $EXTRA_SERVER_ARGS \
        $config_args \
        > "$log_file" 2>&1 &

    echo $!
}

run_benchmark() {
    local url="$1"
    local config_name="$2"
    local batch_size="$3"
    local det_ratio="${4:-1.0}"

    local temp_result="${OUTPUT_DIR}/temp_${config_name}_bs${batch_size}.jsonl"

    echo "[${config_name}] Running batch_size=$batch_size, det_ratio=$det_ratio..."

    curl -s "${url}/reset_llm42_stats" -X POST > /dev/null 2>&1 || true
    python -m sglang.bench_serving \
        --backend "$BACKEND" \
        --base-url "$url" \
        --model "$MODEL" \
        --dataset-name random \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --random-range-ratio 1.0 \
        --num-prompts "$batch_size" \
        --request-rate inf \
        --deterministic-ratio "$det_ratio" \
        --deterministic-seed "$DETERMINISTIC_SEED" \
        --extra-request-body '{"ignore_eos": true, "temperature": 0}' \
        --output-file "$temp_result" \
        --output-details \
        2>&1 | tee "${OUTPUT_DIR}/log_${config_name}_bs${batch_size}.log"

    if [ -f "$temp_result" ]; then
        python -c "
import json
with open('$temp_result', 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                result = json.loads(line)
                result['config_name'] = '$config_name'
                result['batch_size'] = $batch_size
                result['input_len'] = $INPUT_LEN
                result['output_len'] = $OUTPUT_LEN

                # Remove verbose fields
                for key in ['meta_info', 'generated_texts', 'output_ids', 'itls', 'errors']:
                    result.pop(key, None)

                print(json.dumps(result))
            except json.JSONDecodeError:
                pass
" >> "$RESULTS_FILE"
        rm -f "$temp_result"
    fi

    echo "[${config_name}] Completed batch_size=$batch_size"
}

# ---- Cleanup ----
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

# ---- Resume: drop (config_name, batch_size) pairs already in RESULTS_FILE ----
# With --force this file was cleared above, so nothing is skipped and every
# config runs. Otherwise only configs missing from RESULTS_FILE are (re)run and
# their results appended, leaving already-completed configs untouched.
declare -A COMPLETED_CONFIGS=()
if [ -f "$RESULTS_FILE" ]; then
    while read -r _done_key; do
        [ -n "$_done_key" ] && COMPLETED_CONFIGS["$_done_key"]=1
    done < <(python3 - "$RESULTS_FILE" <<'PY'
import json, sys
seen = set()
with open(sys.argv[1]) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        cn, bs = r.get("config_name"), r.get("batch_size")
        if cn is not None and bs is not None:
            seen.add(f"{cn}|{bs}")
for key in seen:
    print(key)
PY
)
fi

declare -a PENDING_CONFIGS=()
SKIPPED_CONFIGS=0
for cfg in "${BENCH_CONFIGS[@]}"; do
    _cfg_name=$(parse_bench_config "$cfg" 1)
    _cfg_bs=$(parse_bench_config "$cfg" 3)
    if [ -n "${COMPLETED_CONFIGS["${_cfg_name}|${_cfg_bs}"]:-}" ]; then
        echo "Skipping (already done): ${_cfg_name} (bs=${_cfg_bs}) -- use --force to re-run"
        SKIPPED_CONFIGS=$((SKIPPED_CONFIGS + 1))
    else
        PENDING_CONFIGS+=("$cfg")
    fi
done
BENCH_CONFIGS=("${PENDING_CONFIGS[@]}")
NUM_CONFIGS=${#BENCH_CONFIGS[@]}
if [ "$NUM_CONFIGS" -eq 0 ]; then
    echo "All configs already complete ($SKIPPED_CONFIGS skipped); regenerating plot only."
fi

# ---- Display ----
echo "=============================================="
echo "Batch Size Experiment"
echo "=============================================="
echo "GPU:              $GPU_SHORT_NAME"
echo "Attention Backend: $ATTENTION_BACKEND"
echo "Model:            $MODEL"
echo "TP Size:          $TP_SIZE"
echo "GPUs:             $NUM_GPUS ($NUM_SLOTS slots)"
echo "Input Length:     $INPUT_LEN"
echo "Output Length:    $OUTPUT_LEN"
echo "Batch Size:       $BATCH_SIZE"
echo "Configs:          $NUM_CONFIGS total"
echo "Extra Server Args: ${EXTRA_SERVER_ARGS:-<none>}"
echo "Output Dir:       $OUTPUT_DIR"
echo ""
echo "Configurations:"
for cfg in "${BENCH_CONFIGS[@]}"; do
    name=$(parse_bench_config "$cfg" 1)
    bs=$(parse_bench_config "$cfg" 3)
    dr=$(parse_bench_config "$cfg" 4)
    echo "  - $name (bs=$bs, det_ratio=$dr)"
done
echo "=============================================="
echo ""

# ---- Run Batches ----
NUM_BATCHES=$(( (NUM_CONFIGS + NUM_SLOTS - 1) / NUM_SLOTS ))
echo "Running $NUM_CONFIGS configs in $NUM_BATCHES batch(es) ($NUM_SLOTS slots per batch)"
echo ""

for ((batch=0; batch<NUM_BATCHES; batch++)); do
    START_IDX=$((batch * NUM_SLOTS))
    END_IDX=$((START_IDX + NUM_SLOTS))
    if [ $END_IDX -gt $NUM_CONFIGS ]; then END_IDX=$NUM_CONFIGS; fi
    BATCH_SIZE_CUR=$((END_IDX - START_IDX))

    echo "=============================================="
    echo "Batch $((batch+1))/$NUM_BATCHES (configs $((START_IDX+1))-$END_IDX)"
    echo "=============================================="

    # Launch servers for this batch
    SERVER_PIDS=()
    declare -a BATCH_URLS=()
    declare -a BATCH_CFG_NAMES=()
    declare -a BATCH_BS=()
    declare -a BATCH_DR=()

    echo "Launching $BATCH_SIZE_CUR servers..."
    for ((i=0; i<BATCH_SIZE_CUR; i++)); do
        CFG_IDX=$((START_IDX + i))
        CFG="${BENCH_CONFIGS[$CFG_IDX]}"
        CFG_NAME=$(parse_bench_config "$CFG" 1)
        CFG_ARGS=$(parse_bench_config "$CFG" 2)
        CFG_BS=$(parse_bench_config "$CFG" 3)
        CFG_DR=$(parse_bench_config "$CFG" 4)
        PORT=$((BASE_PORT + i))
        DEVICES=$(get_slot_devices $i)

        PID=$(launch_server "$PORT" "$CFG_NAME" "$CFG_ARGS" "$DEVICES")
        SERVER_PIDS+=("$PID")
        BATCH_URLS+=("http://127.0.0.1:${PORT}")
        BATCH_CFG_NAMES+=("$CFG_NAME")
        BATCH_BS+=("$CFG_BS")
        BATCH_DR+=("$CFG_DR")
        sleep 2
    done

    # Health check
    echo ""
    echo "Waiting for servers to be ready (timeout ${SERVER_STARTUP_TIMEOUT}s)..."
    for ((i=0; i<BATCH_SIZE_CUR; i++)); do
        URL="${BATCH_URLS[$i]}"
        echo -n "  Slot $i (${BATCH_CFG_NAMES[$i]}) $URL ... "
        if check_server_health "$URL" $((SERVER_STARTUP_TIMEOUT / 5)) 5; then
            echo "✓"
        else
            echo "✗ FAILED"
            echo "ERROR: Server on slot $i failed to start. Check logs in $OUTPUT_DIR"
            exit 1
        fi
    done
    echo "All servers ready!"
    echo ""

    # Run benchmarks in parallel
    declare -a BENCH_PIDS=()
    for ((i=0; i<BATCH_SIZE_CUR; i++)); do
        run_benchmark "${BATCH_URLS[$i]}" "${BATCH_CFG_NAMES[$i]}" "${BATCH_BS[$i]}" "${BATCH_DR[$i]}" &
        BENCH_PIDS+=($!)
    done
    for pid in "${BENCH_PIDS[@]}"; do
        wait "$pid"
    done

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

echo ""
echo "=============================================="
echo "Experiment Complete!"
echo "Results saved to: $RESULTS_FILE"
echo "=============================================="

# Generate plot
echo ""
echo "Generating plot..."
python "${ROOT}/plot.py" \
    --input "$OUTPUT_DIR" \
    --output "${OUTPUT_DIR}/throughput_vs_batchsize.pdf"
