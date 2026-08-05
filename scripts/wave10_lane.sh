#!/usr/bin/env bash
#
# wave10_lane.sh — run a QUEUE of wave-10 cells with bounded concurrency, and
# leave evidence for every one of them (SOTA-944 WAVE 10, appendix H).
#
#   scripts/wave10_lane.sh <log-dir> <max-concurrent> <arm:seed> [<arm:seed> ...]
#
# Contract (the same one wave-8/9 lanes established, kept because it works):
#   * every cell gets its own log file, named for the cell
#   * a cell that exits nonzero writes a `.FAILED` marker and is counted; the
#     lane's exit status is nonzero, so a failure cannot be silent
#   * a heartbeat line per poll into <log-dir>/lane.status, so "still working"
#     and "died 40 minutes ago" are distinguishable at a glance
#   * a `lane.done` sentinel on EXIT via trap — normal, error, or signal
set -uo pipefail

LOGDIR=${1:?usage: wave10_lane.sh <log-dir> <max-concurrent> <arm:seed>...}
MAXC=${2:?usage: wave10_lane.sh <log-dir> <max-concurrent> <arm:seed>...}
shift 2
[ $# -gt 0 ] || { echo "no cells given" >&2; exit 2; }
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
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
    local cell=$1 arm seed
    arm=${cell%%:*}; seed=${cell##*:}
    local log="$LOGDIR/${arm}_s${seed}.log"
    if "$REPO_ROOT/scripts/wave10_seed.sh" "$arm" "$seed" > "$log" 2>&1; then
        printf '%s OK   %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$cell" >> "$LOGDIR/lane.log"
    else
        local rc=$?
        printf '%s FAIL %s rc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$cell" "$rc" >> "$LOGDIR/lane.log"
        touch "$LOGDIR/${arm}_s${seed}.FAILED"
    fi
}

printf '%s lane start: %s cells, max %s concurrent\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$#" "$MAXC" >> "$LOGDIR/lane.log"

for cell in "$@"; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAXC" ]; do
        printf '%s running=%s queued-remaining\n' \
            "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$(jobs -rp | wc -l)" > "$LOGDIR/lane.status"
        sleep 20
    done
    printf '%s launch %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$cell" >> "$LOGDIR/lane.log"
    run_cell "$cell" &
done
wait
FAILED=$(ls "$LOGDIR"/*.FAILED 2>/dev/null | wc -l)
printf '%s lane complete, failed=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$FAILED" >> "$LOGDIR/lane.log"
[ "$FAILED" -eq 0 ]
