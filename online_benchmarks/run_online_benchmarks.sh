#!/bin/bash
# Online Serving Benchmarks - Measures TTFT, TPOT, E2E latency
# Datasets: random, sharegpt, arxiv (ccdv/arxiv-summarization from HuggingFace)

set -e

# Configuration
MODEL="${SGLANG_TEST_MODEL:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
BASE_URL="${SGLANG_BASE_URL:-http://localhost:30000}"
NUM_PROMPTS="${NUM_PROMPTS:-500}"
OUTPUT_DIR="$(dirname "$0")/results"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULTS_FILE="${OUTPUT_DIR}/results_${TIMESTAMP}.jsonl"

# Benchmark parameters
REQUEST_RATES=(1 4 8 16 32)
DET_RATIOS=(0.0 1.0)

mkdir -p "$OUTPUT_DIR"

echo "Online Benchmark Suite | Model: $MODEL | URL: $BASE_URL"
echo "Results: $RESULTS_FILE"

# Check server health
check_server() {
    for i in {1..5}; do
        curl -s "${BASE_URL}/health" > /dev/null 2>&1 && return 0
        echo "Waiting for server... ($i/5)"
        sleep 3
    done
    echo "ERROR: Server not responding at $BASE_URL"
    exit 1
}

check_server

run_bench() {
    local dataset=$1 rate=$2 det=$3 extra_args=$4
    echo ">>> $dataset | rate=$rate | det_ratio=$det"
    
    local tmp="${OUTPUT_DIR}/.tmp_${TIMESTAMP}.json"
    
    python -m sglang.bench_serving \
        --backend sglang --base-url "$BASE_URL" --model "$MODEL" \
        --dataset-name "$dataset" --num-prompts "$NUM_PROMPTS" \
        --request-rate "$rate" --deterministic-ratio "$det" \
        --output-file "$tmp" $extra_args 2>&1 | tail -5
    
    # Append with metadata
    [[ -f "$tmp" ]] && python -c "
import json
with open('$tmp') as f: r = json.load(f)
r.update({'dataset':'$dataset','rate':'$rate','det_ratio':$det})
print(json.dumps(r))
" >> "$RESULTS_FILE" && rm -f "$tmp"
}

# Random dataset benchmarks
for rate in "${REQUEST_RATES[@]}"; do
    for det in "${DET_RATIOS[@]}"; do
        run_bench random "$rate" "$det" "--random-input-len 1024 --random-output-len 256"
    done
done

# ShareGPT benchmarks
for rate in "${REQUEST_RATES[@]}"; do
    for det in "${DET_RATIOS[@]}"; do
        run_bench sharegpt "$rate" "$det" ""
    done
done

# Arxiv benchmarks (ccdv/arxiv-summarization from HuggingFace)
echo "Running arxiv benchmarks..."
for rate in "${REQUEST_RATES[@]}"; do
    for det in "${DET_RATIOS[@]}"; do
        echo ">>> arxiv | rate=$rate | det_ratio=$det"
        python "$(dirname "$0")/run_arxiv_benchmark.py" \
            --base-url "$BASE_URL" --model "$MODEL" --num-prompts "$NUM_PROMPTS" \
            --request-rate "$rate" --deterministic-ratio "$det" \
            --output-file "$RESULTS_FILE" 2>&1 | tail -8
    done
done

echo "Done! Results: $RESULTS_FILE"
python "$(dirname "$0")/plot_results.py" "$RESULTS_FILE" --output-dir "${OUTPUT_DIR}/plots_${TIMESTAMP}"
