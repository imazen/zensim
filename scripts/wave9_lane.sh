#!/usr/bin/env bash
#
# wave9_lane.sh <cell> [<cell> ...] — run N wave-9 cells CONCURRENTLY on this
# box and wait for all of them. A cell is `<arm><seed>` e.g. `A3301`.
#
# Each trainer is single-threaded (RAYON_NUM_THREADS=1) so N cells occupy N
# cores and the box stays responsive; the caller wraps this in run-heavy, which
# supplies the nice/ionice and the hard memory cap.
#
# Per-cell log: $WAVE9_LOG/<cell>.log. A failed cell leaves a .FAILED marker and
# makes the whole lane exit nonzero, so a silent partial lane is impossible.
set -uo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
LOG=${WAVE9_LOG:-$HOME/tmp/wave9}
mkdir -p "$LOG"
[ $# -gt 0 ] || { echo "usage: wave9_lane.sh <arm><seed> ..." >&2; exit 2; }

pids=(); cells=()
for cell in "$@"; do
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
