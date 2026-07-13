#!/usr/bin/env bash
# Shared GPU detection and B200-aware defaults.
# Source this from any benchmark script:
#   source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../gpu_utils.sh"
#
# After sourcing, the following variables are set:
#   GPU_SHORT_NAME      – short lowercase name (e.g. a100, h100, b200)
#   IS_B200             – "true" if Blackwell B200, "false" otherwise
#   ATTENTION_BACKEND          – "triton" on B200, "fa3" otherwise (respects env override)
#   DISABLE_CUSTOM_ALL_REDUCE  – "1" on B200 and SXM H100, "0" otherwise (respects env override)
#   ENABLE_TORCH_SYMM_MEM     – "1" on B200 and SXM H100, "0" otherwise (respects env override)

detect_gpu_short_name() {
    local full_name
    full_name=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || true)
    local lower
    lower=$(echo "$full_name" | tr '[:upper:]' '[:lower:]')

    # Match known GPU families (order matters — check specific names first)
    case "$lower" in
        *b200*)       echo "b200" ;;
        *b100*)       echo "b100" ;;
        *h200*)       echo "h200" ;;
        *h100*pcie*)  echo "h100_pcie" ;;
        *h100*)       echo "h100" ;;
        *h20*)        echo "h20"  ;;
        *a100*)       echo "a100" ;;
        *a10g*)       echo "a10g" ;;
        *a10*)        echo "a10"  ;;
        *a6000*)      echo "a6000" ;;
        *l40s*)       echo "l40s" ;;
        *l40*)        echo "l40"  ;;
        *l4*)         echo "l4"   ;;
        *rtx*6000*ada*) echo "rtx6000ada" ;;
        *rtx*4090*)   echo "rtx4090" ;;
        *rtx*3090*)   echo "rtx3090" ;;
        *v100*)       echo "v100" ;;
        *t4*)         echo "t4"   ;;
        # Fallback: sanitise to lowercase alnum + underscores
        *)  echo "$lower" | sed 's/[^a-z0-9]/_/g; s/__*/_/g; s/^_//; s/_$//' ;;
    esac
}

GPU_SHORT_NAME="${GPU_SHORT_NAME:-$(detect_gpu_short_name)}"
IS_B200=false
[[ "$GPU_SHORT_NAME" == b200* ]] && IS_B200=true

# GPU-aware defaults (scripts can still override via env before sourcing)
case "$GPU_SHORT_NAME" in
    b200*)
        ATTENTION_BACKEND="${ATTENTION_BACKEND:-flashinfer}"
        DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-1}"
        ENABLE_TORCH_SYMM_MEM="${ENABLE_TORCH_SYMM_MEM:-1}"
        ;;
    h100)
        # Regular (SXM) H100 supports symmetric memory
        ATTENTION_BACKEND="${ATTENTION_BACKEND:-fa3}"
        DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-1}"
        ENABLE_TORCH_SYMM_MEM="${ENABLE_TORCH_SYMM_MEM:-1}"
        ;;
    *)
        ATTENTION_BACKEND="${ATTENTION_BACKEND:-fa3}"
        DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-0}"
        ENABLE_TORCH_SYMM_MEM="${ENABLE_TORCH_SYMM_MEM:-0}"
        ;;
esac

# ---- Server process management ----
# Kill a server process and all its descendants (e.g. EngineCore → TP workers
# spawned via multiprocessing).  Sends SIGTERM first, waits up to KILL_GRACE
# seconds, then escalates to SIGKILL.
KILL_GRACE=${KILL_GRACE:-15}

# Recursively collect all descendant PIDs of a given process.
_collect_descendants() {
    local parent=$1
    local children
    children=$(ps -o pid= --ppid "$parent" 2>/dev/null || true)
    for child in $children; do
        echo "$child"
        _collect_descendants "$child"
    done
}

kill_server() {
    local pid=$1
    if ! kill -0 "$pid" 2>/dev/null; then return 0; fi

    # Collect entire descendant tree before killing (children may reparent)
    local descendants
    descendants=$(_collect_descendants "$pid")

    # Graceful shutdown
    kill "$pid" 2>/dev/null || true

    local elapsed=0
    while kill -0 "$pid" 2>/dev/null && [ "$elapsed" -lt "$KILL_GRACE" ]; do
        sleep 1
        elapsed=$((elapsed + 1))
    done

    # Force-kill parent + all descendants
    if kill -0 "$pid" 2>/dev/null; then
        kill -9 "$pid" 2>/dev/null || true
    fi
    for desc in $descendants; do
        if kill -0 "$desc" 2>/dev/null; then
            kill -9 "$desc" 2>/dev/null || true
        fi
    done

    wait "$pid" 2>/dev/null || true
}

# Kill any GPU processes on the specified devices (comma-separated GPU IDs).
# Useful for cleaning up orphaned workers from a previous crashed server.
kill_gpu_processes() {
    local devices="$1"
    local pids
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader \
               -i "$devices" 2>/dev/null | sort -u || true)
    for p in $pids; do
        [ -z "$p" ] && continue
        echo "  Killing orphaned GPU process PID $p on device(s) $devices"
        kill -9 "$p" 2>/dev/null || true
    done
}
