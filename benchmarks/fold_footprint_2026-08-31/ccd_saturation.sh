#!/usr/bin/env bash
# Per-CCD N-process saturation: does the fold's thread ceiling come from L3?
#
# This box (measured, NOT what the workspace docs say) is an AMD Ryzen 9
# 9950X3D with an ASYMMETRIC L3: CCD0 = cpus 0-7 (+SMT 16-23) with 96 MiB,
# CCD1 = cpus 8-15 (+SMT 24-31) with 32 MiB. If the fold's ceiling is an
# L3-per-thread effect, N independent single-threaded processes must saturate
# EARLIER on CCD1 than on CCD0; if it is DRAM bandwidth, the two CCDs behave
# the same.
#
#   FE_BIN=<bench binary> ./ccd_saturation.sh <out.tsv> [size] [iters]
#
# Columns: arm, ccd, cpus, n, iters, wall_s, throughput_cmp_per_s, speedup
set -uo pipefail
BIN="${FE_BIN:?set FE_BIN}"
OUT="${1:?out.tsv}"
SIZE="${2:-2304}"
ITERS="${3:-12}"

printf 'arm\tccd\tn\titers\twall_s\tthroughput\tspeedup\n' > "$OUT"
for arm in score_fold score_buffered; do
  for ccd in 0 1; do
    cpus=$([ "$ccd" = 0 ] && echo "0-7" || echo "8-15")
    base=""
    for n in 1 2 4 8; do
      # Warm the page cache / branch predictors once, untimed.
      taskset -c "$cpus" env ZEN_FE_RSS="$arm" ZEN_FE_SIZE="$SIZE" ZEN_FE_ITERS=2 \
        RAYON_NUM_THREADS=1 "$BIN" >/dev/null 2>&1
      t0=$(date +%s.%N)
      for i in $(seq 1 "$n"); do
        taskset -c "$cpus" env ZEN_FE_RSS="$arm" ZEN_FE_SIZE="$SIZE" ZEN_FE_ITERS="$ITERS" \
          RAYON_NUM_THREADS=1 "$BIN" >/dev/null 2>&1 &
      done
      wait
      t1=$(date +%s.%N)
      wall=$(echo "$t1 - $t0" | bc -l)
      thr=$(echo "$n * $ITERS / $wall" | bc -l)
      [ -z "$base" ] && base="$thr"
      sp=$(echo "$thr / $base" | bc -l)
      printf '%s\t%s\t%s\t%s\t%.3f\t%.3f\t%.3f\n' "$arm" "$ccd" "$n" "$ITERS" "$wall" "$thr" "$sp" >> "$OUT"
      printf '%-15s CCD%s n=%-2s wall=%7.3fs  thr=%7.3f cmp/s  speedup=%5.2fx\n' \
        "$arm" "$ccd" "$n" "$wall" "$thr" "$sp"
    done
  done
done
