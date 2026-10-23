#!/bin/bash
# Paired timing of two byc_score_timing arms: alternating order per round, pinned core(s), fixed thread count.
# Arm A runs A_BIN on BAKE; arm B runs B_BIN on B_BAKE (default BAKE). Summarize with timing_summary.py.
# Usage: ring_timing.sh A_BIN B_BIN BAKE SIZE ITERS ROUNDS OUT_JSONL [B_BAKE]
# Env: RING_A / RING_B set ZENSIM_RING_FORM per arm (raw|hp4, equal length; default raw; a build without the
#      form ignores it). THREADS (default 1) and CPUS (default 2) set RAYON_NUM_THREADS and the taskset list.
set -euo pipefail
A=$1; B=$2; BAKE=$3; SIZE=$4; ITERS=$5; ROUNDS=$6; OUT=$7; B_BAKE=${8:-$3}
RING_A=${RING_A:-raw}; RING_B=${RING_B:-raw}; THREADS=${THREADS:-1}; CPUS=${CPUS:-2}
: > "$OUT"
for r in $(seq 1 "$ROUNDS"); do
  if (( r % 2 )); then order="A B"; else order="B A"; fi
  for which in $order; do
    bin=$A; bake=$BAKE; ring=$RING_A
    [ "$which" = B ] && { bin=$B; bake=$B_BAKE; ring=$RING_B; }
    line=$(ZENSIM_RING_FORM=$ring RAYON_NUM_THREADS=$THREADS taskset -c "$CPUS" "$bin" --bake "$bake" --size "$SIZE" --iters "$ITERS" --warmup 3)
    echo "{\"round\":$r,\"arm\":\"$which\",\"ring\":\"$ring\",\"threads\":$THREADS,\"result\":$line}" >> "$OUT"
  done
done
