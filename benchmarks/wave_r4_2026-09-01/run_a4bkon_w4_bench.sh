#!/bin/bash
# a4bkon lane §24.5: the direct W4 measurement for free156_peaks_raw.
# ASLR protocol: CCD0-pinned, nice -19, ASLR on, one arm-set per process,
# min-over-N starts. add/flag/free re-measured fresh in THIS process each
# start (never cited), per the "same interleaved process" requirement.
set -euo pipefail
REPO=/home/lilith/work/zen/zensim--a4bkon
BIN=/mnt/v/zen/cargo-targets/a4bkon/release/deps
BENCH=$(ls -t "$BIN"/ssim2_speed_bar-* 2>/dev/null | grep -v '\.d$' | head -1)
[ -x "$BENCH" ] || { echo "ABORT: no ssim2_speed_bar binary in $BIN"; exit 1; }
echo "bench binary: $BENCH"
OUT=/mnt/v/output/zensim/a4bkon-2026-09-01/speed
mkdir -p "$OUT"

export ZEN_HY_ADD=/mnt/v/output/zensim/wave-r4-2026-09-01/bakes/A2ctrl_r4_l0.3_packed.bin
export ZEN_HY_FREE=/mnt/v/output/zensim/wave-r4-2026-09-01/bakes/A4b_156_s4004_packed.bin
export ZEN_HY_MLP=/mnt/v/output/zensim/wave-r4-2026-09-01/bakes/A1v3_r4_s4004_packed.bin
export ZEN_S2_SIZES=576,1152,2304
export ZEN_S2_ROUNDS=40
export ZEN_S2_WALL_S=25

T="${1:?thread count required: 1, 8, or 16}"
N="${2:-3}"
case "$T" in
  1)  CORES="0" ;;
  8)  CORES="0-7" ;;
  16) CORES="0-15" ;;
  *) echo "ABORT: T must be 1, 8, or 16"; exit 1 ;;
esac
export RAYON_NUM_THREADS="$T"

for i in $(seq 1 "$N"); do
  LOG="$OUT/${T}t_start${i}.txt"
  echo "== ${T}T start $i/$N -> $LOG $(date -u +%H:%M:%SZ)"
  nice -19 taskset -c "$CORES" "$BENCH" --bench > "$LOG" 2>&1 || {
    echo "  (nonzero exit -- see $LOG)"; tail -5 "$LOG"; }
done
echo "${T}T DONE $(date -u +%H:%M:%SZ)"
