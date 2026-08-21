#!/usr/bin/env bash
#
# wave12_lane.sh — run a QUEUE of wave-12 seeds with bounded concurrency
# (SOTA-944 WAVE 12, appendix AC + AD). Same evidence contract as
# wave11_lane.sh (per-cell log, .FAILED markers, lane.status heartbeat,
# lane.done sentinel). AD.7: on the post-resize 23 GiB box the registered
# default is max-concurrent 1 under `run-heavy --mem 18G` (the trainer's flat
# buffer is f64, ~9 GB data resident at wave-12's ~1.19 M rows); a second
# lane only if the first lane's measured peak RSS <= 8 GB.
#
#   scripts/wave12_lane.sh <log-dir> <max-concurrent> <seed> [<seed> ...]
set -uo pipefail

LOGDIR=${1:?usage: wave12_lane.sh <log-dir> <max-concurrent> <seed>...}
MAXC=${2:?usage: wave12_lane.sh <log-dir> <max-concurrent> <seed>...}
shift 2
[ $# -gt 0 ] || { echo "no seeds given" >&2; exit 2; }
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
RUN_HEAVY=${RUN_HEAVY:-$HOME/work/zen/scripts/run-heavy}
MEM=${WAVE12_MEM:-18G}
mkdir -p "$LOGDIR"

FAILED=0
finish() {
    rc=$?
    printf '%s lane exit rc=%s failed=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$rc" "$FAILED" \
        > "$LOGDIR/lane.done"
    exit $rc
}
trap finish EXIT

run_cell() {
    local seed=$1
    local log="$LOGDIR/W12_s${seed}.log"
    if "$RUN_HEAVY" --mem "$MEM" -- "$REPO_ROOT/scripts/wave12_seed.sh" "$seed" > "$log" 2>&1; then
        printf '%s OK   s%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$seed" >> "$LOGDIR/lane.log"
    else
        local rc=$?
        printf '%s FAIL s%s rc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$seed" "$rc" >> "$LOGDIR/lane.log"
        touch "$LOGDIR/W12_s${seed}.FAILED"
    fi
}

printf '%s lane start: %s seeds, max %s concurrent, mem %s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$#" "$MAXC" "$MEM" >> "$LOGDIR/lane.log"

for seed in "$@"; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAXC" ]; do
        printf '%s running=%s\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(jobs -rp | wc -l)" > "$LOGDIR/lane.status"
        sleep 20
    done
    printf '%s launch s%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$seed" >> "$LOGDIR/lane.log"
    run_cell "$seed" &
done
wait
FAILED=$(ls "$LOGDIR"/*.FAILED 2>/dev/null | wc -l)
printf '%s lane complete, failed=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$FAILED" >> "$LOGDIR/lane.log"
