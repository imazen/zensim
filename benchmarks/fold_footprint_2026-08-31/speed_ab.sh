#!/usr/bin/env bash
# Paired before/after speed A/B for the fold-footprint lane.
#
# zenbench interleaves the arms WITHIN one process, so the before and after
# builds cannot share a group. Instead the two binaries are alternated
# (before, after, before, after) at each thread count so box drift cancels
# across the pair rather than inside it.
#
#   BEFORE_BIN=... AFTER_BIN=... ./speed_ab.sh <outdir> [threads...]
set -uo pipefail
: "${BEFORE_BIN:?}" "${AFTER_BIN:?}"
OUT="${1:?outdir}"; shift || true
THREADS="${*:-1 16}"
mkdir -p "$OUT"
export ZEN_FE_SIZES="${ZEN_FE_SIZES:-1152,2304}"
for round in 1 2; do
  for th in $THREADS; do
    for which in before after; do
      bin=$([ "$which" = before ] && echo "$BEFORE_BIN" || echo "$AFTER_BIN")
      echo "[$(date -u +%H:%M:%S)] round $round ${th}T $which  load=$(awk '{print $1}' /proc/loadavg)"
      RAYON_NUM_THREADS="$th" ~/work/zen/scripts/run-heavy --mem 16G --jobs 8 -- \
        "$bin" > "$OUT/${which}_${th}t_r${round}.txt" 2>&1
    done
  done
done
printf 'done %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > "$OUT/COMPLETE"
