#!/usr/bin/env bash
#
# wave9_lane.sh <cell> [<cell> ...] — run N wave-9 cells CONCURRENTLY on this
# box and wait for all of them. A cell is `<arm><seed>` e.g. `A3301`.
#
# Each trainer is single-threaded (RAYON_NUM_THREADS=1) so N cells occupy N
# cores and the box stays responsive; the caller wraps this in run-heavy, which
# supplies the nice/ionice and the hard memory cap.
#
# --slot-limit N: do not launch a cell while the BOX already has N or more
#   zensim_mlp_train processes running, whoever started them. A full-mix 944
#   cell peaks near 12 GiB RSS, so concurrency is bounded by RAM, not cores:
#   on 2026-08-04 four concurrent cells hit the run-heavy 40G cgroup cap and
#   one was OOM-killed at epoch 0, and three on lianli's 29 GiB drove it to
#   0 available + 7 GiB of swap. Launching every cell at once was the bug;
#   this is the fix, in the driver rather than in an operator's head.
#   Liveness is counted with `pgrep -xc` on the 15-char comm, never `-f`
#   (which self-matches the invoking shell).
#
# Per-cell log: $WAVE9_LOG/<cell>.log. A failed cell leaves a .FAILED marker and
# makes the whole lane exit nonzero, so a silent partial lane is impossible.
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG=${WAVE9_LOG:-$HOME/tmp/wave9}
SLOT_LIMIT=0
SLOT_WAIT=${WAVE9_SLOT_WAIT:-30}
SLOT_TIMEOUT=${WAVE9_SLOT_TIMEOUT:-21600}
mkdir -p "$LOG"
while [ $# -gt 0 ] && [ "${1:0:2}" = "--" ]; do
    case "$1" in
        --slot-limit) SLOT_LIMIT=${2:?}; shift 2 ;;
        *) echo "unknown arg $1" >&2; exit 2 ;;
    esac
done
[ $# -gt 0 ] || { echo "usage: wave9_lane.sh [--slot-limit N] <arm><seed> ..." >&2; exit 2; }

wait_for_slot() {
    [ "$SLOT_LIMIT" -gt 0 ] || return 0
    local t0; t0=$(date +%s)
    while [ "$(pgrep -xc zensim_mlp_trai || echo 0)" -ge "$SLOT_LIMIT" ]; do
        if [ $(( $(date +%s) - t0 )) -ge "$SLOT_TIMEOUT" ]; then
            echo "$(date -u +%H:%M:%SZ) slot wait timed out (limit $SLOT_LIMIT)" >&2
            return 3
        fi
        sleep "$SLOT_WAIT"
    done
}

pids=(); cells=()
for cell in "$@"; do
    wait_for_slot || exit 3
    arm=${cell:0:1}; seed=${cell:1}
    case "$arm" in A|B|C) ;; *) echo "bad cell $cell" >&2; exit 2 ;; esac
    case "$seed" in ''|*[!0-9]*) echo "bad seed in $cell" >&2; exit 2 ;; esac
    rm -f "$LOG/$cell.FAILED"
    (
        RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
            "$REPO_ROOT/scripts/wave9_seed.sh" "$arm" "$seed" \
            > "$LOG/$cell.log" 2>&1
        rc=$?
        [ $rc -eq 0 ] || { echo "$cell rc=$rc" > "$LOG/$cell.FAILED"; }
        exit $rc
    ) &
    pids+=($!); cells+=("$cell")
    echo "$(date -u +%H:%M:%SZ) launched $cell pid ${pids[-1]}"
done

rc_all=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "$(date -u +%H:%M:%SZ) OK   ${cells[$i]}"
    else
        echo "$(date -u +%H:%M:%SZ) FAIL ${cells[$i]} (see $LOG/${cells[$i]}.log)" >&2
        rc_all=6
    fi
done
exit $rc_all
