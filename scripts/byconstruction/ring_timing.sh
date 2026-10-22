#!/bin/bash
# Paired timing of two byc_score_timing arms: alternating order per round, one pinned core, one thread.
# Arm A runs A_BIN on BAKE; arm B runs B_BIN on B_BAKE (default BAKE). Summarize with timing_summary.py.
# Usage: ring_timing.sh A_BIN B_BIN BAKE SIZE ITERS ROUNDS OUT_JSONL [B_BAKE]
set -euo pipefail
A=$1; B=$2; BAKE=$3; SIZE=$4; ITERS=$5; ROUNDS=$6; OUT=$7; B_BAKE=${8:-$3}
: > "$OUT"
for r in $(seq 1 "$ROUNDS"); do
  if (( r % 2 )); then order="A B"; else order="B A"; fi
  for which in $order; do
    bin=$A; bake=$BAKE; [ "$which" = B ] && { bin=$B; bake=$B_BAKE; }
    line=$(RAYON_NUM_THREADS=1 taskset -c 2 "$bin" --bake "$bake" --size "$SIZE" --iters "$ITERS" --warmup 3)
    echo "{\"round\":$r,\"arm\":\"$which\",\"result\":$line}" >> "$OUT"
  done
done
