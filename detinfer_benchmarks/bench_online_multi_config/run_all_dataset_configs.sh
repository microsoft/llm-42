#!/bin/bash
set -euo pipefail

# Run multi-config comparison for multiple dataset configurations
# This script runs 4 dataset configs and saves results for batch size analysis

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Common settings
export QPS=${QPS:-12}
export NUM_PROMPTS=${NUM_PROMPTS:-16384}

echo "=============================================="
echo "Running Multi-Config Comparison for Multiple Datasets"
echo "=============================================="
echo "QPS: $QPS"
echo "NUM_PROMPTS: $NUM_PROMPTS"
echo ""

# Dataset configurations
declare -a DATASET_CONFIGS=(
    "sharegpt"
    "random_in1024_out1"
    "random_in1024_out129"
    "random_in1024_out257"
)

# Initialize cumulative counts file
CUMULATIVE_COUNTS_FILE="${ROOT}/.cumulative_prefill_counts.txt"
> "$CUMULATIVE_COUNTS_FILE"  # Clear/create file

# Function to count prefill batch sizes from log and return as "1:count 2:count 3:count 4+:count"
count_prefill_batches() {
    local logfile="$1"
    if [ ! -f "$logfile" ]; then
        echo "0 0 0 0"
        return
    fi
    
    # Count occurrences of each batch size
    local count1=$(grep -oP '#new-seq: 1(?=,)' "$logfile" 2>/dev/null | wc -l)
    local count2=$(grep -oP '#new-seq: 2(?=,)' "$logfile" 2>/dev/null | wc -l)
    local count3=$(grep -oP '#new-seq: 3(?=,)' "$logfile" 2>/dev/null | wc -l)
    local count4plus=$(grep -oP '#new-seq: [4-9][0-9]*(?=,)|#new-seq: [1-9][0-9]+(?=,)' "$logfile" 2>/dev/null | wc -l)
    
    echo "$count1 $count2 $count3 $count4plus"
}

# Store previous cumulative counts (indexed by config name)
declare -A PREV_COUNTS

# Record INITIAL baseline counts BEFORE any experiments run
# This handles the case where servers already have logs from previous runs
echo "Recording initial baseline counts from server logs..."
if [ -d "${ROOT}/server_logs_multi_config" ]; then
    for logfile in "${ROOT}/server_logs_multi_config"/server_gpu*.log; do
        if [ -f "$logfile" ]; then
            basename_log=$(basename "$logfile")
            PREV_COUNTS["$basename_log"]="$(count_prefill_batches "$logfile")"
            echo "  $basename_log: ${PREV_COUNTS[$basename_log]}"
        fi
    done
fi
echo ""

# Run each dataset configuration
for config in "${DATASET_CONFIGS[@]}"; do
    echo ""
    echo "=============================================="
    echo "Running dataset config: $config"
    echo "=============================================="
    
    case "$config" in
        "sharegpt")
            export DATASET_NAME=sharegpt
            unset RANDOM_INPUT_LEN RANDOM_OUTPUT_LEN
            export OUTPUT_DIR="${ROOT}/results_${config}_qps${QPS}_n${NUM_PROMPTS}"
            ;;
        "random_in1024_out1")
            export DATASET_NAME=random
            export RANDOM_INPUT_LEN=1024
            export RANDOM_OUTPUT_LEN=1
            export OUTPUT_DIR="${ROOT}/results_${config}_qps${QPS}_n${NUM_PROMPTS}"
            ;;
        "random_in1024_out129")
            export DATASET_NAME=random
            export RANDOM_INPUT_LEN=1024
            export RANDOM_OUTPUT_LEN=129
            export OUTPUT_DIR="${ROOT}/results_${config}_qps${QPS}_n${NUM_PROMPTS}"
            ;;
        "random_in1024_out257")
            export DATASET_NAME=random
            export RANDOM_INPUT_LEN=1024
            export RANDOM_OUTPUT_LEN=257
            export OUTPUT_DIR="${ROOT}/results_${config}_qps${QPS}_n${NUM_PROMPTS}"
            ;;
    esac
    
    echo "Dataset: $DATASET_NAME"
    if [ "$DATASET_NAME" = "random" ]; then
        echo "Input Length: $RANDOM_INPUT_LEN"
        echo "Output Length: $RANDOM_OUTPUT_LEN"
    fi
    echo "Output Dir: $OUTPUT_DIR"
    echo ""
    
    # Run the comparison
    "${ROOT}/run_compare_mismatches_multi_config.sh"
    
    # Count prefill batches and compute delta from previous run
    mkdir -p "${OUTPUT_DIR}"
    COUNTS_FILE="${OUTPUT_DIR}/prefill_batch_counts.csv"
    echo "config_name,batch_1,batch_2,batch_3,batch_4plus" > "$COUNTS_FILE"
    
    if [ -d "${ROOT}/server_logs_multi_config" ]; then
        for logfile in "${ROOT}/server_logs_multi_config"/server_gpu*.log; do
            if [ -f "$logfile" ]; then
                basename_log=$(basename "$logfile")
                # Extract config name from filename (e.g., server_gpu0_port30005_sglang_non_deterministic.log)
                server_config=$(echo "$basename_log" | sed 's/server_gpu[0-9]*_port[0-9]*_//' | sed 's/\.log$//')
                
                # Get current cumulative counts
                read -r curr1 curr2 curr3 curr4plus <<< "$(count_prefill_batches "$logfile")"
                
                # Get previous counts (default to 0)
                prev_key="${basename_log}"
                read -r prev1 prev2 prev3 prev4plus <<< "${PREV_COUNTS[$prev_key]:-0 0 0 0}"
                
                # Compute delta (this run only)
                delta1=$((curr1 - prev1))
                delta2=$((curr2 - prev2))
                delta3=$((curr3 - prev3))
                delta4plus=$((curr4plus - prev4plus))
                
                # Store current as previous for next iteration
                PREV_COUNTS[$prev_key]="$curr1 $curr2 $curr3 $curr4plus"
                
                # Write to CSV
                echo "${server_config},${delta1},${delta2},${delta3},${delta4plus}" >> "$COUNTS_FILE"
                
                echo "  $server_config: bs=1:$delta1, bs=2:$delta2, bs=3:$delta3, bs>=4:$delta4plus"
            fi
        done
    fi
    
    echo ""
    echo "Saved prefill batch counts to: $COUNTS_FILE"
    echo "Completed: $config"
    echo ""
done

echo ""
echo "=============================================="
echo "All dataset configurations completed!"
echo "=============================================="
echo ""
echo "Results saved in:"
for config in "${DATASET_CONFIGS[@]}"; do
    echo "  - ${ROOT}/results_${config}_qps${QPS}_n${NUM_PROMPTS}"
done
echo ""
echo "To plot prefill batch size distribution:"
echo "  python plot_prefill_batch_sizes.py --results-dirs results_*_qps${QPS}_n${NUM_PROMPTS}"
