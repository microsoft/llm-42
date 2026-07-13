#!/bin/bash
set -euo pipefail

# Raise open file limit for high-concurrency benchmarks
ulimit -n 65536 2>/dev/null || true

# GPU-balanced online (QPS-driven) benchmarks
#
# Generates all (config, det_ratio, qps, dataset) job pairs and distributes
# them across GPU slots via a shared job queue.  Each GPU worker claims a job,
# starts/reuses a server for the config, and runs the benchmark.
#
# Usage:
#   ./run_balanced_online_benchmark.sh                   # uses defaults
#   ./run_balanced_online_benchmark.sh --force           # force re-run completed experiments
#   NUM_GPUS=8 CONFIG_NAMES=... ./run_balanced_online_benchmark.sh
#   # Or via the orchestrator:
#   ./run.sh

# ---- Parse flags ----
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --force) FORCE=1 ;;
    esac
done

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ---- Configuration ----
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
if [ "$NUM_GPUS" -eq 0 ]; then
    echo "Error: No GPUs detected. Set NUM_GPUS manually."
    exit 1
fi
MODEL="${MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
HOST="${HOST:-0.0.0.0}"
BASE_PORT="${BASE_PORT:-30005}"
TP_SIZE="${TP_SIZE:-1}"

# Number of independent server slots (each uses TP_SIZE GPUs)
NUM_SLOTS=$((NUM_GPUS / TP_SIZE))
if [ "$NUM_SLOTS" -eq 0 ] || [ $((NUM_SLOTS * TP_SIZE)) -ne "$NUM_GPUS" ]; then
    echo "Error: NUM_GPUS ($NUM_GPUS) must be a positive multiple of TP_SIZE ($TP_SIZE)"
    exit 1
fi

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../gpu_utils.sh"
_GPU_TYPE="$GPU_SHORT_NAME"
_MODEL_TAG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]')

# Server configs: sglang_non_deterministic, sglang_deterministic, llm42
if [ "$_GPU_TYPE" = "b200" ]; then
    CONFIG_NAMES="${CONFIG_NAMES:-sglang_non_deterministic,sglang_deterministic,llm42_ws_32_bs_32}"
else
    CONFIG_NAMES="${CONFIG_NAMES:-sglang_non_deterministic,sglang_deterministic,llm42_ws_64_bs_8}"
fi

TOKENIZER=${TOKENIZER:-}
NUM_REQUESTS=${NUM_REQUESTS:-1024}
DATASET_NAME=${DATASET_NAME:-mixed_workload}
DATASET_PATH=${DATASET_PATH:-}
SHAREGPT_CONTEXT_LEN=${SHAREGPT_CONTEXT_LEN:-16384}
RANDOM_INPUT_LEN=${RANDOM_INPUT_LEN:-1024}
RANDOM_OUTPUT_LEN=${RANDOM_OUTPUT_LEN:-128}
DETERMINISTIC_SEED=${DETERMINISTIC_SEED:-42}
BACKEND=${BACKEND:-sglang}

BASELINE_RATIOS="1.0"
LLM42_RATIOS="${LLM42_RATIOS:-0.02,0.05,0.1,0.2,0.5,1.0}"

# QPS values to sweep
QPS_VALUES="${QPS_VALUES:-4,4.5,5,5.5}"

SERVER_STARTUP_TIMEOUT=${SERVER_STARTUP_TIMEOUT:-300}

# ---- Multi-dataset support ----
declare -a DATASET_TAGS
if [ -n "${DATASET_CONFIGS_LIST:-}" ]; then
    IFS=' ' read -ra DATASET_TAGS <<< "$DATASET_CONFIGS_LIST"
else
    if [ "$DATASET_NAME" = "random" ]; then
        DATASET_TAGS=("random_in${RANDOM_INPUT_LEN}_out${RANDOM_OUTPUT_LEN}")
    else
        DATASET_TAGS=("$DATASET_NAME")
    fi
fi

# Helper: run dir for a given dataset tag
# e.g. runs/h100_llama-3.3-70b-instruct-tp8_fa3_n1024_sharegpt_online
_RUN_DIR_BASE="${ROOT}/runs/${_GPU_TYPE}_${_MODEL_TAG}-tp${TP_SIZE}_${ATTENTION_BACKEND}_n${NUM_REQUESTS}"
get_dataset_run_dir() {
    echo "${_RUN_DIR_BASE}_${1}_online"
}
get_dataset_results_dir() {
    echo "${_RUN_DIR_BASE}_${1}_online/results"
}
get_dataset_server_log_dir() {
    echo "${_RUN_DIR_BASE}_${1}_online/server_logs"
}

# Python command
if command -v python &> /dev/null; then
    PYTHON_CMD="python"
elif command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    echo "Error: Python not found"
    exit 1
fi

IFS=',' read -ra CONFIG_ARRAY <<< "$CONFIG_NAMES"
IFS=',' read -ra QPS_ARRAY <<< "$QPS_VALUES"

# Create output dirs for all datasets
for _dtag in "${DATASET_TAGS[@]}"; do
    mkdir -p "$(get_dataset_results_dir "$_dtag")"
    mkdir -p "$(get_dataset_server_log_dir "$_dtag")"
done

# ---- Detect already-completed jobs from previous runs ----
# Key: config_name|det_ratio|qps|dataset_tag
declare -A COMPLETED_JOBS
if [ "$FORCE" -eq 1 ]; then
    echo "  --force: skipping completion detection, all jobs will run"
else
for _dtag in "${DATASET_TAGS[@]}"; do
    _ds_results_dir="$(get_dataset_results_dir "$_dtag")"
    _ds_results="${_ds_results_dir}/benchmark_results.jsonl"

    if [ -f "$_ds_results" ]; then
        while IFS= read -r _line; do
            _cname=$($PYTHON_CMD -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('config_name',''))" <<< "$_line" 2>/dev/null || true)
            _dratio=$($PYTHON_CMD -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('deterministic_ratio',''))" <<< "$_line" 2>/dev/null || true)
            _qps=$($PYTHON_CMD -c "import sys,json; d=json.loads(sys.stdin.read()); print(d.get('qps',''))" <<< "$_line" 2>/dev/null || true)
            if [ -n "$_cname" ] && [ -n "$_dratio" ] && [ -n "$_qps" ]; then
                COMPLETED_JOBS["${_cname}|${_dratio}|${_qps}|${_dtag}"]=1
            fi
        done < "$_ds_results"
    fi
done
fi

# ---- Job Queue Setup ----
QUEUE_DIR=$(mktemp -d "${TMPDIR:-/tmp}/bench_queue_XXXXXX")
mkdir -p "$QUEUE_DIR/pending" "$QUEUE_DIR/claimed" "$QUEUE_DIR/pids"

# Generate all (config, det_ratio, qps, dataset) jobs, skipping completed ones
declare -a ALL_JOBS=()
SKIPPED_JOBS=0
for dataset_tag in "${DATASET_TAGS[@]}"; do
    for qps in "${QPS_ARRAY[@]}"; do
        for config_name in "${CONFIG_ARRAY[@]}"; do
            if [[ "$config_name" == *"llm42"* ]]; then
                IFS=',' read -ra _RATIO_ARRAY <<< "$LLM42_RATIOS"
                for ratio in "${_RATIO_ARRAY[@]}"; do
                    if [ -n "${COMPLETED_JOBS["${config_name}|${ratio}|${qps}|${dataset_tag}"]:-}" ]; then
                        echo "  Skipping (already done): $config_name det_ratio=$ratio qps=$qps dataset=$dataset_tag"
                        SKIPPED_JOBS=$((SKIPPED_JOBS + 1))
                    else
                        ALL_JOBS+=("${config_name} ${ratio} ${qps} ${dataset_tag}")
                    fi
                done
            else
                if [ -n "${COMPLETED_JOBS["${config_name}|${BASELINE_RATIOS}|${qps}|${dataset_tag}"]:-}" ]; then
                    echo "  Skipping (already done): $config_name det_ratio=$BASELINE_RATIOS qps=$qps dataset=$dataset_tag"
                    SKIPPED_JOBS=$((SKIPPED_JOBS + 1))
                else
                    ALL_JOBS+=("${config_name} ${BASELINE_RATIOS} ${qps} ${dataset_tag}")
                fi
            fi
        done
    done
done

if [ ${#ALL_JOBS[@]} -eq 0 ]; then
    echo "All jobs already completed ($SKIPPED_JOBS skipped). Nothing to do."
    exit 0
fi

# Sort by config name so consecutive jobs share a server (avoids restarts).
IFS=$'\n' SORTED_JOBS=($(printf '%s\n' "${ALL_JOBS[@]}" | sort)); unset IFS

JOB_NUM=0
for job in "${SORTED_JOBS[@]}"; do
    printf '%s\n' "$job" > "$QUEUE_DIR/pending/$(printf 'job_%04d' $JOB_NUM)"
    ((++JOB_NUM))
done
TOTAL_JOBS=$JOB_NUM

# ---- Display ----
echo "=============================================="
echo "GPU-Balanced Online (QPS) Benchmark"
echo "=============================================="
echo "Model: $MODEL"
echo "GPUs: $NUM_GPUS (TP=$TP_SIZE, $NUM_SLOTS parallel slots)"
echo "Total Jobs: $TOTAL_JOBS (skipped $SKIPPED_JOBS already completed)"
echo "Num Requests: $NUM_REQUESTS"
echo "QPS Values: ${QPS_ARRAY[*]}"
echo "Datasets: ${DATASET_TAGS[*]}"
echo "Configs: $CONFIG_NAMES"
echo "Baseline Ratios: $BASELINE_RATIOS"
echo "LLM42 Ratios: $LLM42_RATIOS"
echo "Results Base: ${_RUN_DIR_BASE}_*_online"
echo ""
echo "Job Queue (sorted for server reuse):"
for f in "$QUEUE_DIR/pending/"*; do
    echo "  $(basename "$f"): $(cat "$f")"
done
echo "=============================================="
echo ""

# ---- Cleanup ----
declare -a WORKER_PIDS=()

cleanup() {
    echo ""
    echo "Cleaning up..."
    for pid in "${WORKER_PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid_file in "$QUEUE_DIR/pids/"server_*; do
        if [ -f "$pid_file" ]; then
            local spid
            spid=$(cat "$pid_file")
            kill_server "$spid"
        fi
    done
    wait 2>/dev/null || true
    rm -rf "$QUEUE_DIR"
    echo "Cleanup complete."
}
trap cleanup EXIT INT TERM

# ---- Helper Functions ----

get_config_args() {
    local config_name="$1"
    case "$config_name" in
        "sglang_non_deterministic")
            echo ""
            ;;
        "sglang_deterministic"|"sglang_global_deterministic")
            echo "--enable-deterministic-inference 2"
            ;;
        "sglang_global_deterministic_triton")
            echo "--enable-deterministic-inference 3"
            ;;
        llm42_ws_*_bs_*)
            local ws bs
            ws=$(echo "$config_name" | sed 's/.*ws_\([0-9]*\).*/\1/')
            bs=$(echo "$config_name" | sed 's/.*bs_\([0-9]*\).*/\1/')
            echo "--enable-llm42 3 --llm42-window-size $ws --llm42-verify-batch-size $bs"
            ;;
        *)
            echo "Error: Unknown config: $config_name" >&2
            return 1
            ;;
    esac
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

claim_next_job() {
    while true; do
        local next_job
        next_job=$(ls "$QUEUE_DIR/pending/" 2>/dev/null | sort | head -1)
        if [ -z "$next_job" ]; then
            return 1
        fi
        if mv "$QUEUE_DIR/pending/$next_job" "$QUEUE_DIR/claimed/$next_job" 2>/dev/null; then
            cat "$QUEUE_DIR/claimed/$next_job"
            return 0
        fi
    done
}

run_single_benchmark() {
    local url="$1"
    local config_name="$2"
    local det_ratio="$3"
    local slot_id="$4"
    local dataset_tag="$5"
    local qps="$6"

    # Parse dataset from tag
    local ds_name ds_input_len ds_output_len
    case "$dataset_tag" in
        random_in*_out*)
            ds_name="random"
            ds_input_len=$(echo "$dataset_tag" | sed 's/.*in\([0-9]*\).*/\1/')
            ds_output_len=$(echo "$dataset_tag" | sed 's/.*out\([0-9]*\).*/\1/')
            ;;
        *)
            ds_name="$dataset_tag"
            ds_input_len=0
            ds_output_len=0
            ;;
    esac

    local ds_results_dir
    ds_results_dir="$(get_dataset_results_dir "$dataset_tag")"

    local slot_results="${ds_results_dir}/results_slot${slot_id}.jsonl"
    local temp_result="${ds_results_dir}/temp_${config_name}_det${det_ratio}_qps${qps}.jsonl"

    echo "[Slot $slot_id] Running: $config_name det_ratio=$det_ratio qps=$qps dataset=$dataset_tag"

    # Build tokenizer arg
    TOKENIZER_ARG=""
    if [[ -n "${TOKENIZER}" ]]; then
        TOKENIZER_ARG="--tokenizer ${TOKENIZER}"
    fi

    # Build dataset args
    local _shared_cache="${HOME}/.cache/llm42_bench"
    local _bench_backend="$BACKEND"
    if [ "$ds_name" = "random" ]; then
        DATASET_ARGS="--dataset-name random --random-input-len $ds_input_len --random-output-len $ds_output_len --random-range-ratio 1.0"
        INPUT_LEN_FOR_RESULT=$ds_input_len
        OUTPUT_LEN_FOR_RESULT=$ds_output_len
    elif [[ -n "${DATASET_PATH}" ]]; then
        DATASET_ARGS="--dataset-name ${ds_name} --sharegpt-context-len $SHAREGPT_CONTEXT_LEN --dataset-path ${DATASET_PATH}"
        INPUT_LEN_FOR_RESULT=0
        OUTPUT_LEN_FOR_RESULT=0
    elif [ -f "${_shared_cache}/${ds_name}_sglang.jsonl" ]; then
        DATASET_ARGS="--dataset-name openai --dataset-path ${_shared_cache}/${ds_name}_sglang.jsonl"
        _bench_backend="sglang-oai-chat"
        INPUT_LEN_FOR_RESULT=0
        OUTPUT_LEN_FOR_RESULT=0
    else
        DATASET_ARGS="--dataset-name ${ds_name} --sharegpt-context-len $SHAREGPT_CONTEXT_LEN"
        INPUT_LEN_FOR_RESULT=0
        OUTPUT_LEN_FOR_RESULT=0
    fi
    EXTRA_BODY='{"ignore_eos": true, "temperature": 0}'

    local bench_rc=0
    curl -s "${url}/reset_llm42_stats" -X POST > /dev/null 2>&1 || true
    $PYTHON_CMD -m sglang.bench_serving \
        --backend "$_bench_backend" \
        --base-url "$url" \
        --model "$MODEL" \
        $TOKENIZER_ARG \
        $DATASET_ARGS \
        --num-prompts "$NUM_REQUESTS" \
        --request-rate "$qps" \
        --deterministic-ratio "$det_ratio" \
        --deterministic-seed "$DETERMINISTIC_SEED" \
        --extra-request-body "$EXTRA_BODY" \
        --output-file "$temp_result" \
        --output-details \
        2>&1 | tee "${ds_results_dir}/log_${config_name}_det${det_ratio}_qps${qps}.log" || bench_rc=$?

    if [ "$bench_rc" -ne 0 ]; then
        echo "[Slot $slot_id] WARNING: benchmark exited with code $bench_rc for $config_name det_ratio=$det_ratio qps=$qps"
    fi

    if [ -f "$temp_result" ]; then
        $PYTHON_CMD -c "
import json

with open('$temp_result', 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                result = json.loads(line)
                result['config_name'] = '$config_name'
                result['dataset_tag'] = '$dataset_tag'
                result['dataset_name'] = '$ds_name'
                result['input_len'] = $INPUT_LEN_FOR_RESULT
                result['output_len'] = $OUTPUT_LEN_FOR_RESULT
                result['deterministic_ratio'] = $det_ratio
                result['qps'] = $qps
                result['server_url'] = '$url'

                meta_info_list = result.get('meta_info', [])
                output_lens = result.get('output_lens', [])

                per_request_rollbacks = [m.get('llm42_num_rollbacks', 0) for m in meta_info_list if m] if meta_info_list else []
                per_request_tokens_rolled_back = [m.get('llm42_tokens_rolled_back', 0) for m in meta_info_list if m] if meta_info_list else []
                if per_request_rollbacks:
                    result['per_request_rollbacks'] = per_request_rollbacks
                    result['per_request_tokens_rolled_back'] = per_request_tokens_rolled_back

                if meta_info_list:
                    num_requests = len(per_request_rollbacks)
                    total_output_tokens = sum(output_lens) if output_lens else result.get('total_output_tokens', 0)
                    if num_requests > 0:
                        result['rollback_stats'] = {
                            'total_rollbacks': sum(per_request_rollbacks),
                            'total_tokens_rolled_back': sum(per_request_tokens_rolled_back),
                            'total_output_tokens': total_output_tokens,
                            'avg_rollbacks_per_request': sum(per_request_rollbacks) / num_requests,
                            'avg_tokens_rolled_back_per_request': sum(per_request_tokens_rolled_back) / num_requests,
                            'max_rollbacks_per_request': max(per_request_rollbacks) if per_request_rollbacks else 0,
                            'max_tokens_rolled_back_per_request': max(per_request_tokens_rolled_back) if per_request_tokens_rolled_back else 0,
                            'requests_with_rollbacks': sum(1 for x in per_request_rollbacks if x > 0),
                            'num_requests': num_requests,
                        }

                for key in ['meta_info', 'generated_texts', 'output_ids', 'itls', 'errors']:
                    result.pop(key, None)

                print(json.dumps(result))
            except json.JSONDecodeError:
                pass
" >> "$slot_results" || echo "[Slot $slot_id] WARNING: failed to process results for $config_name det_ratio=$det_ratio qps=$qps"
        rm -f "$temp_result"
    fi

    echo "[Slot $slot_id] Completed: $config_name det_ratio=$det_ratio qps=$qps dataset=$dataset_tag"
}

# ---- GPU Worker ----
get_slot_devices() {
    local slot_id=$1
    local first_gpu=$((slot_id * TP_SIZE))
    local devices=""
    for ((g=first_gpu; g<first_gpu+TP_SIZE; g++)); do
        devices="${devices:+${devices},}${g}"
    done
    echo "$devices"
}

gpu_worker() {
    local slot_id=$1
    local devices
    devices=$(get_slot_devices "$slot_id")
    local port=$((BASE_PORT + slot_id))
    local url="http://127.0.0.1:${port}"
    local current_config=""
    local server_pid=""
    local pid_file="$QUEUE_DIR/pids/server_slot${slot_id}"
    local label="Slot $slot_id [GPUs $devices]"

    trap 'if [ -n "$server_pid" ]; then kill_server "$server_pid"; fi' EXIT

    echo "[$label] Worker started (port $port)"

    while true; do
        local job_info
        if ! job_info=$(claim_next_job); then
            echo "[$label] No more jobs"
            break
        fi

        local config_name det_ratio qps dataset_tag
        config_name=$(echo "$job_info" | awk '{print $1}')
        det_ratio=$(echo "$job_info" | awk '{print $2}')
        qps=$(echo "$job_info" | awk '{print $3}')
        dataset_tag=$(echo "$job_info" | awk '{print $4}')

        echo "[$label] Claimed: $config_name det_ratio=$det_ratio qps=$qps dataset=$dataset_tag"

        # Ensure a healthy server is running for this config.
        local need_restart=0
        if [ "$config_name" != "$current_config" ]; then
            need_restart=1
        elif [ -z "$server_pid" ] || ! kill -0 "$server_pid" 2>/dev/null || \
             ! timeout 5 curl -s "${url}/v1/models" 2>/dev/null | grep -q '"object":"list"'; then
            echo "[$label] WARNING: server died unexpectedly, will restart $config_name"
            need_restart=1
        fi

        if [ "$need_restart" -eq 1 ]; then
            if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
                echo "[$label] Stopping server ($current_config -> $config_name)"
                kill_server "$server_pid"
                sleep 2
            fi

            if fuser "${port}/tcp" &>/dev/null; then
                echo "[$label] WARNING: port $port already in use, killing existing process..."
                fuser -k "${port}/tcp" &>/dev/null || true
                sleep 2
            fi

            local config_args
            config_args=$(get_config_args "$config_name")
            local log_file="$(get_dataset_server_log_dir "$dataset_tag")/server_slot${slot_id}_${config_name}.log"

            echo "[$label] Starting server: $config_name (port $port)..."

            CUDA_VISIBLE_DEVICES=$devices $PYTHON_CMD -m sglang.launch_server \
                --model-path "$MODEL" \
                --host "$HOST" \
                --port "$port" \
                --tp "$TP_SIZE" \
                --attention-backend "$ATTENTION_BACKEND" \
                --disable-radix-cache \
                --disable-chunked-prefix-cache \
                $([[ "$config_name" == llm42_* ]] && echo "--disable-overlap-schedule") \
                --enable-metrics \
                --random-seed 42 \
                --chunked-prefill-size -1 \
                $([ "${DISABLE_CUSTOM_ALL_REDUCE:-0}" = "1" ] && echo "--disable-custom-all-reduce" || true) \
                $([ "${ENABLE_TORCH_SYMM_MEM:-0}" = "1" ] && echo "--enable-torch-symm-mem" || true) \
                $config_args \
                > "$log_file" 2>&1 &

            server_pid=$!
            current_config="$config_name"
            echo "$server_pid" > "$pid_file"

            echo "[$label] Waiting for server (PID $server_pid)..."
            if ! wait_for_server "$url" "$SERVER_STARTUP_TIMEOUT"; then
                echo "[$label] ERROR: Server failed to start for $config_name (timeout ${SERVER_STARTUP_TIMEOUT}s)"
                echo "[$label] Skipping job: $config_name det_ratio=$det_ratio qps=$qps"
                kill_server "$server_pid"
                server_pid=""
                current_config=""
                continue
            fi
            echo "[$label] Server ready!"
        else
            echo "[$label] Reusing server for $config_name"
        fi

        run_single_benchmark "$url" "$config_name" "$det_ratio" "$slot_id" "$dataset_tag" "$qps" ||
            echo "[$label] WARNING: benchmark failed for $config_name det_ratio=$det_ratio qps=$qps dataset=$dataset_tag"
    done

    # Stop server
    if [ -n "$server_pid" ] && kill -0 "$server_pid" 2>/dev/null; then
        echo "[$label] Stopping server"
        kill_server "$server_pid"
    fi

    echo "[$label] Worker done"
}

# ---- Main Execution ----
NUM_WORKERS=$((NUM_SLOTS < TOTAL_JOBS ? NUM_SLOTS : TOTAL_JOBS))

# Pre-flight: kill any stale processes on our ports
echo "Checking for stale processes on ports ${BASE_PORT}..$(( BASE_PORT + NUM_WORKERS - 1 ))..."
for ((i=0; i<NUM_WORKERS; i++)); do
    port=$((BASE_PORT + i))
    if fuser "${port}/tcp" &>/dev/null; then
        echo "  WARNING: port $port already in use — killing leftover process"
        fuser -k "${port}/tcp" &>/dev/null || true
    fi
done
sleep 2

echo "Launching $NUM_WORKERS slot workers for $TOTAL_JOBS jobs..."
echo ""

for ((i=0; i<NUM_WORKERS; i++)); do
    gpu_worker $i &
    WORKER_PIDS+=($!)
done

# Wait for all workers
FAILED=0
for pid in "${WORKER_PIDS[@]}"; do
    wait "$pid" || FAILED=$((FAILED + 1))
done

# Merge per-GPU result files into the final results file per dataset
echo ""
echo "Merging results..."
for _dtag in "${DATASET_TAGS[@]}"; do
    _ds_results_dir="$(get_dataset_results_dir "$_dtag")"
    _ds_results_file="${_ds_results_dir}/benchmark_results.jsonl"
    for ((i=0; i<NUM_WORKERS; i++)); do
        slot_results="${_ds_results_dir}/results_slot${i}.jsonl"
        if [ -f "$slot_results" ]; then
            cat "$slot_results" >> "$_ds_results_file"
            rm -f "$slot_results"
        fi
    done
done

echo ""
echo "=============================================="
echo "Online Benchmarking Complete!"
if [ "$FAILED" -gt 0 ]; then
    echo "WARNING: $FAILED worker(s) had errors"
fi
echo "Results:"
for _dtag in "${DATASET_TAGS[@]}"; do
    echo "  $_dtag: $(get_dataset_results_dir "$_dtag")/benchmark_results.jsonl"
done
echo "=============================================="

# Export per-request data and summary CSV for each dataset
echo ""
echo "Exporting per-request data and summaries..."
for _dtag in "${DATASET_TAGS[@]}"; do
    _ds_results_dir="$(get_dataset_results_dir "$_dtag")"
    _ds_results_file="${_ds_results_dir}/benchmark_results.jsonl"

    if [ -f "$_ds_results_file" ]; then
        CSV_FILE="${_ds_results_dir}/per_request_data.csv"
        $PYTHON_CMD "${ROOT}/export_per_request_csv.py" \
            --input "$_ds_results_file" \
            --output "$CSV_FILE"

        SUMMARY_CSV="${_ds_results_dir}/summary.csv"
        $PYTHON_CMD "${ROOT}/summarize_online_csv.py" \
            --input "$_ds_results_file" \
            --output "$SUMMARY_CSV"

        echo "  $_dtag:"
        echo "    Per-request CSV: $CSV_FILE"
        echo "    Summary CSV:     $SUMMARY_CSV"
    fi
done
