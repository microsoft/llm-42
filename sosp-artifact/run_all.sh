#!/usr/bin/env bash
# Run every paper benchmark in order (skips 0-basic).
#
# Usage:
#   ./run_all.sh [--duration quick|full] [--models 8b|70b|8b,70b] [--force]
#
# --duration (default quick) and --models (default 8b) are forwarded to the
# scripts that accept them (figure10, figure11-12, figure13). The other
# benchmarks have no such flags and run with their own defaults.
#
# By default every benchmark resumes -- already-completed work is skipped.
# --force is forwarded to all benchmarks to re-run and overwrite existing results.

DURATION="quick"
MODELS="8b"
FORCE=()   # (--force) when set; forwarded to every benchmark
while [ $# -gt 0 ]; do
    case "$1" in
        --duration)   [ $# -ge 2 ] || { echo "Error: --duration requires quick|full" >&2; exit 1; }; DURATION="$2"; shift ;;
        --duration=*) DURATION="${1#*=}" ;;
        --models)     [ $# -ge 2 ] || { echo "Error: --models requires 8b|70b|8b,70b" >&2; exit 1; }; MODELS="$2"; shift ;;
        --models=*)   MODELS="${1#*=}" ;;
        --force)      FORCE=(--force) ;;
        -h|--help)    echo "Usage: $0 [--duration quick|full] [--models 8b|70b|8b,70b] [--force]"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2
           echo "Usage: $0 [--duration quick|full] [--models 8b|70b|8b,70b] [--force]" >&2; exit 1 ;;
    esac
    shift
done

case "$DURATION" in quick|full) ;; *) echo "Error: --duration must be 'quick' or 'full' (got '$DURATION')" >&2; exit 1 ;; esac
IFS=',' read -ra _MODELS <<< "$MODELS"
for _m in "${_MODELS[@]}"; do
    case "$_m" in 8b|70b) ;; *) echo "Error: --models must be 8b|70b|8b,70b (got '$MODELS')" >&2; exit 1 ;; esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DM=(--duration "$DURATION" --models "$MODELS")   # forwarded to scripts that accept these flags

TIMING_LOG="${TIMING_LOG:-${ROOT}/run_all_timings.log}"
{
    echo "# run_all timings  (started $(date '+%F %T'))"
    echo "# duration=${DURATION}  models=${MODELS}  force=$([ ${#FORCE[@]} -gt 0 ] && echo yes || echo no)  gpus=$(nvidia-smi -L 2>/dev/null | wc -l)"
} | tee "$TIMING_LOG"
_ALL_START=$SECONDS

run() {
    local dir="$1"; shift
    echo
    echo "########## ${dir}/run.sh $* ##########"
    local _start=$SECONDS _status=0
    ( cd "${ROOT}/${dir}" && ./run.sh "$@" ) || _status=$?
    local _e=$(( SECONDS - _start ))
    printf '%-22s %02d:%02d:%02d  exit=%d\n' \
        "$dir" $((_e/3600)) $(((_e%3600)/60)) $((_e%60)) "$_status" | tee -a "$TIMING_LOG"
}
# Duration below is approximate assuming 8 GPUs are available. If fewer GPUs are available, it will be proportionally higher.

# Microbenchmarks and short-running so we don't customize them much
run 1-figure4     "${FORCE[@]}" # 1 minute
run 2-figure5     "${FORCE[@]}" # 2 minutes
run 3-figure6     "${FORCE[@]}" # 30 minutes
run 4-figure9a    "${FORCE[@]}" # 2 minutes
run 5-figure9b    "${FORCE[@]}" # 20 minutes

# Real longer-running jobs. Have customizable parameters (--duration quick|full, --models 8b|70b|8b,70b)
run 6-figure10    "${DM[@]}" "${FORCE[@]}" # quick mode: 15 minutes. full run: >24 hours
run 7-figure11-12 "${DM[@]}" "${FORCE[@]}" # quick mode: 15 minutes. full run: >12 hours
run 8-figure13    "${DM[@]}" "${FORCE[@]}" # quick mode: 10 minutes. full run: ~5 hours

echo
echo "All benchmarks done."
_e=$(( SECONDS - _ALL_START ))
printf '%-22s %02d:%02d:%02d\n' "TOTAL" $((_e/3600)) $(((_e%3600)/60)) $((_e%60)) | tee -a "$TIMING_LOG"
