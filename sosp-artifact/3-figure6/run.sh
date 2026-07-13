#!/usr/bin/env bash
set -euo pipefail

# Self-contained mismatch-indices benchmark (paper Figure 6).
# For each model it launches an SGLang server, runs a sequential vs Poisson
# ShareGPT comparison, plots the consistent-span CDF, and tears the server down.
#
# By default it runs BOTH models:
#   - Llama-3.1-8B-Instruct  (TP-1) -> llm42-plots/figure6a.pdf
#   - Llama-3.3-70B-Instruct (TP-8) -> llm42-plots/figure6b.pdf
# The 70B model is skipped entirely when fewer than 4 GPUs are visible; with
# >=4 GPUs its tensor-parallel size falls back to the largest power of two <=
# the visible GPU count (e.g. TP-4 on a 4-GPU node).
#
# By default a model whose mismatch data already exists is skipped (resume);
# pass --force to re-run and overwrite it.
#
# Environment overrides (all optional):
#   MODEL            run only this single model (default: both 8B and 70B)
#   TP_SIZE          tensor-parallel size for the MODEL override (default: 1)
#   PAPER_FIGURE     llm42-plots/ filename for the MODEL override (default: inferred)
#   NUM_GPUS         override detected GPU count (used for TP fallback)
#   ATTENTION_BACKEND, PORT, GPU_ID,
#   NUM_PROMPTS, QPS, SEED, SHAREGPT_OUTPUT_LEN,
#   SEQ_CONCURRENCY, POISSON_CONCURRENCY, EXTRA_REQUEST_BODY

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
# GPU detection
# ---------------------------------------------------------------------------
source "${ROOT}/../gpu_utils.sh"

# Number of visible GPUs (used for tensor-parallel fallback). 0 => unknown.
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"

# largest power of two <= n
largest_pow2_le() {
    local n=$1 p=1
    while (( p * 2 <= n )); do p=$((p * 2)); done
    echo "$p"
}

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PORT=${PORT:-30000}
GPU_ID=${GPU_ID:-0}
HOST="0.0.0.0"

EXTRA_SERVER_ARGS=""
if [ "${DISABLE_CUSTOM_ALL_REDUCE:-0}" = "1" ]; then
    EXTRA_SERVER_ARGS="$EXTRA_SERVER_ARGS --disable-custom-all-reduce"
fi
if [ "${ENABLE_TORCH_SYMM_MEM:-0}" = "1" ]; then
    EXTRA_SERVER_ARGS="$EXTRA_SERVER_ARGS --enable-torch-symm-mem"
fi

# Benchmark parameters
BACKEND=${BACKEND:-sglang}
NUM_PROMPTS=${NUM_PROMPTS:-128}
QPS=${QPS:-6}
SEED=${SEED:-42}
SHAREGPT_OUTPUT_LEN=${SHAREGPT_OUTPUT_LEN:-512}
SEQ_CONCURRENCY=${SEQ_CONCURRENCY:-1}
POISSON_CONCURRENCY=${POISSON_CONCURRENCY:-}
EXTRA_REQUEST_BODY=${EXTRA_REQUEST_BODY:-'{"temperature":0}'}
TOKENIZER=${TOKENIZER:-}
DATASET_PATH=${DATASET_PATH:-}

# Models to benchmark: "model_path requested_tp paper_figure"
# The 8B model exports figure6a.pdf; the 70B model exports figure6b.pdf.
if [ -n "${MODEL:-}" ]; then
    MODELS=("${MODEL} ${TP_SIZE:-1} ${PAPER_FIGURE:-}")
else
    MODELS=(
        "meta-llama/Llama-3.1-8B-Instruct 1 figure6a.pdf"
        "meta-llama/Llama-3.3-70B-Instruct 8 figure6b.pdf"
    )
fi

# ---------------------------------------------------------------------------
# Server lifecycle (SERVER_PID stays global for the cleanup trap)
# ---------------------------------------------------------------------------
SERVER_PID=""

cleanup() {
    echo ""
    echo "Cleaning up..."
    if [ -n "$SERVER_PID" ]; then
        echo "  Stopping server (PID $SERVER_PID)..."
        kill_server "$SERVER_PID"
    fi
    fuser -k "${PORT}/tcp" 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

check_server_health() {
    local base_url="$1"
    local max_retries="${2:-120}"
    local interval="${3:-5}"
    for ((i=1; i<=max_retries; i++)); do
        if curl -s "${base_url}/v1/models" 2>/dev/null | grep -q '"object":"list"'; then
            return 0
        fi
        sleep "$interval"
    done
    return 1
}

# Regenerate the consistent-span CDF and re-export the paper figure from an
# existing mismatch file (used both after a fresh run and when resuming).
replot_consistent_spans() {
    local output_dir="$1" mismatch_file="$2" paper_fig="$3"
    if [ ! -f "$mismatch_file" ]; then
        echo "Warning: $mismatch_file not found, skipping plot."
        return 0
    fi
    local -a plot_cmd=(
        python "${ROOT}/plot.py"
        --mismatch-file "$mismatch_file"
        --output "${output_dir}/mismatch_cdf.pdf"
    )
    [[ -n "$paper_fig" ]] && plot_cmd+=(--paper-figure "$paper_fig")
    "${plot_cmd[@]}"
}

run_model() {
    local model="$1" req_tp="$2" paper_fig="$3"
    local base_url="http://127.0.0.1:${PORT}"

    # Tensor-parallel fallback: cap to the largest power of two <= NUM_GPUS.
    local tp_size="$req_tp"
    if (( NUM_GPUS >= 1 )) && (( req_tp > NUM_GPUS )); then
        tp_size=$(largest_pow2_le "$NUM_GPUS")
        echo "NOTE: requested TP-${req_tp} for ${model} but only ${NUM_GPUS} GPU(s) visible; using TP-${tp_size}."
    fi

    local devices
    devices=$(seq -s, "$GPU_ID" $((GPU_ID + tp_size - 1)))

    local model_tag output_dir server_log mismatch_file
    model_tag=$(basename "$model" | tr '[:upper:]' '[:lower:]')
    output_dir="${ROOT}/runs/${GPU_SHORT_NAME}_${model_tag}_tp${tp_size}"
    mismatch_file="${output_dir}/mismatch_per_request.jsonl"

    # Resume: skip the (expensive) benchmark when the mismatch data already
    # exists, but still regenerate the plot/paper figure from it (unless --force).
    if [[ -f "$mismatch_file" && "$FORCE" -ne 1 ]]; then
        echo "Skipping benchmark (already done): ${model} -> ${output_dir}"
        echo "  Regenerating plot from existing data (use --force to re-run the benchmark)."
        replot_consistent_spans "$output_dir" "$mismatch_file" "$paper_fig"
        return 0
    fi

    # (Re)running: clear any partial output and recreate the directory.
    rm -rf "$output_dir"
    mkdir -p "$output_dir"
    server_log="${output_dir}/server.log"

    echo "=============================================="
    echo "Consistent Spans Benchmark"
    echo "=============================================="
    echo "GPU: ${GPU_SHORT_NAME}"
    echo "Model: ${model}"
    echo "TP Size: ${tp_size} (requested ${req_tp}, ${NUM_GPUS} GPU(s) visible)"
    echo "CUDA_VISIBLE_DEVICES: ${devices}"
    echo "Attention Backend: ${ATTENTION_BACKEND}"
    echo "Port: ${PORT}"
    echo "Num Prompts: ${NUM_PROMPTS}"
    echo "QPS: ${QPS}"
    echo "Extra Server Args: ${EXTRA_SERVER_ARGS:-<none>}"
    echo "Paper Figure: ${paper_fig:-<inferred>}"
    echo "Output Dir: ${output_dir}"
    echo "=============================================="
    echo ""

    # ---- Launch server ----
    echo "Launching SGLang server..."
    CUDA_VISIBLE_DEVICES="$devices" python -m sglang.launch_server \
        --model-path "$model" \
        --host "$HOST" \
        --port "$PORT" \
        --tp "$tp_size" \
        --attention-backend "$ATTENTION_BACKEND" \
        --disable-radix-cache \
        --disable-chunked-prefix-cache \
        --disable-overlap-schedule \
        --enable-metrics \
        --chunked-prefill-size -1 \
        --max-running-requests 64 \
        --random-seed "$SEED" \
        $EXTRA_SERVER_ARGS \
        > "$server_log" 2>&1 &
    SERVER_PID=$!
    echo "  Server PID: $SERVER_PID (log: $server_log)"

    echo -n "  Waiting for server to be ready... "
    if check_server_health "$base_url" 120 5; then
        echo "✓"
    else
        echo "✗ FAILED"
        echo "ERROR: Server failed to start. Check log: $server_log"
        if [ -n "$SERVER_PID" ]; then kill_server "$SERVER_PID"; fi
        fuser -k "${PORT}/tcp" 2>/dev/null || true
        SERVER_PID=""
        return 1
    fi

    # ---- Run comparison benchmark ----
    echo ""
    echo "Running sequential vs Poisson comparison..."

    local -a cmd=(
        python "${ROOT}/compare_sharegpt_runs.py"
        --backend "${BACKEND}"
        --base-url "${base_url}"
        --model "${model}"
        --num-prompts "${NUM_PROMPTS}"
        --qps "${QPS}"
        --seed "${SEED}"
        --deterministic-ratio 1.0
        --sequential-max-concurrency "${SEQ_CONCURRENCY}"
        --output-dir "${output_dir}"
        --extra-request-body "${EXTRA_REQUEST_BODY}"
        --ignore-eos
        --sharegpt-output-len "${SHAREGPT_OUTPUT_LEN}"
    )
    [[ -n "${TOKENIZER}" ]] && cmd+=(--tokenizer "${TOKENIZER}")
    [[ -n "${DATASET_PATH}" ]] && cmd+=(--dataset-path "${DATASET_PATH}")
    [[ -n "${POISSON_CONCURRENCY}" ]] && cmd+=(--max-concurrency "${POISSON_CONCURRENCY}")

    printf 'Running: %s\n' "${cmd[*]}"
    "${cmd[@]}" 2>&1 | tee "${output_dir}/benchmark.log"

    # ---- Generate the consistent-span CDF (Figure 6) ----
    echo ""
    echo "Generating consistent-span CDF plot..."
    replot_consistent_spans "$output_dir" "$mismatch_file" "$paper_fig"

    # ---- Tear down this model's server before the next one ----
    echo ""
    echo "Stopping server (PID $SERVER_PID)..."
    if [ -n "$SERVER_PID" ]; then kill_server "$SERVER_PID"; fi
    fuser -k "${PORT}/tcp" 2>/dev/null || true
    SERVER_PID=""

    echo ""
    echo "Results: ${output_dir}/"
}

# ---------------------------------------------------------------------------
# Run all models
# ---------------------------------------------------------------------------
for entry in "${MODELS[@]}"; do
    read -r m t f <<< "$entry"
    if [[ "${m,,}" == *70b* ]] && (( NUM_GPUS < 4 )); then
        echo "NOTE: skipping ${m} (70B) -- requires >=4 GPUs but only ${NUM_GPUS} visible."
        continue
    fi
    run_model "$m" "$t" "$f"
done

echo ""
echo "=============================================="
echo "Benchmark Complete!"
echo "=============================================="
