#!/usr/bin/env bash
set -euo pipefail

# Simple launcher to run the sequential + Poisson ShareGPT comparison with deterministic sampling.
# Environment overrides:
#   BACKEND (default: sglang)
#   BASE_URL (default: http://127.0.0.1:30000)
#   MODEL (default: meta-llama/Llama-3.1-8B-Instruct)
#   TOKENIZER (default: empty = same as model)
#   DATASET_PATH (optional local ShareGPT JSON)
#   NUM_PROMPTS (default: 1000)
#   QPS (default: 6)
#   SEED (default: 42)
#   SEQ_CONCURRENCY (default: 1)
#   POISSON_CONCURRENCY (optional)
#   EXTRA_REQUEST_BODY (default: '{"temperature":0}')
#   PROMPT_SUFFIX (optional)
#   SHAREGPT_OUTPUT_LEN (optional int)
#   SHAREGPT_CONTEXT_LEN (optional int)
#   APPLY_CHAT_TEMPLATE (set to 1 to enable)
#   FLUSH_CACHE (set to 1 to enable)
#   OUTPUT_DIR (default: $ROOT/sharegpt_compare_out)

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${PYTHONPATH:-}:${ROOT}/python"

BACKEND=${BACKEND:-sglang}
BASE_URL=${BASE_URL:-http://127.0.0.1:30000}
MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
TOKENIZER=${TOKENIZER:-}
DATASET_PATH=${DATASET_PATH:-}
NUM_PROMPTS=${NUM_PROMPTS:-350}
QPS=${QPS:-6}
SEED=${SEED:-42}
SEQ_CONCURRENCY=${SEQ_CONCURRENCY:-1}
POISSON_CONCURRENCY=${POISSON_CONCURRENCY:-}
EXTRA_REQUEST_BODY=${EXTRA_REQUEST_BODY:-'{"temperature":0}'}
OUTPUT_DIR=${OUTPUT_DIR:-"${ROOT}/sharegpt_compare_out"}

cmd=(
  python "${ROOT}/compare_sharegpt_runs.py"
  --backend "${BACKEND}"
  --base-url "${BASE_URL}"
  --model "${MODEL}"
  --num-prompts "${NUM_PROMPTS}"
  --qps "${QPS}"
  --seed "${SEED}"
  --deterministic-ratio 1.0
  --sequential-max-concurrency "${SEQ_CONCURRENCY}"
  --output-dir "${OUTPUT_DIR}"
  --extra-request-body "${EXTRA_REQUEST_BODY}"
  --ignore-eos
  --sharegpt-output-len "${SHAREGPT_OUTPUT_LEN:-512}"
)

if [[ -n "${TOKENIZER}" ]]; then
  cmd+=(--tokenizer "${TOKENIZER}")
fi
if [[ -n "${DATASET_PATH}" ]]; then
  cmd+=(--dataset-path "${DATASET_PATH}")
fi
if [[ -n "${POISSON_CONCURRENCY}" ]]; then
  cmd+=(--max-concurrency "${POISSON_CONCURRENCY}")
fi

printf 'Running: %s\n' "${cmd[*]}"
"${cmd[@]}"
