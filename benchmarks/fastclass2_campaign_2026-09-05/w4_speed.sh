#!/bin/bash
# fastclass2 — W4 (gate G5). The bar this campaign was briefed with is
# "<= 1.25x the 156 walk at 1T AND 8T"; the exam's own W4 clause is
# "speed >= fast_ssim2 at 1T". Both are read off ONE instrument run.
#
# PROTOCOL (benchmarks/ssim2_replacement_bar_2026-08-31.md + the two hazards
# benchmarks/profile_d_notax_2026-09-01.md section 4 documented):
#   * ONE binary, arms selected at RUNTIME by env -- never two builds.
#   * min over >= MIN_STARTS process starts, ASLR left ON.
#   * ZEN_S2_WALL_S scaled WITH image size. At 2304^2 a 6-arm group needs
#     ~60 s; at 8 s the group blows its budget and zenbench reports a
#     spuriously NEAR-ZERO mean for EVERY arm at once. min() will happily
#     select that corruption, so the reading is VALIDATED at collection: if
#     the stable fast_ssim2 arm reads below a plausible floor for its size,
#     the start is DISCARDED, not averaged in.
#   * NOTHING ELSE RUNS ON THE BOX. A single concurrent niced cargo build was
#     measured swinging fast_ssim2 128.9-633.6 ms inside one cell.
#   * ZEN_S2_EXTRACT_ONLY=1 adds walk-only siblings, so a W4 anomaly can be
#     attributed to the walk or to the forward pass without a second,
#     possibly-mismatched instrument.
#
# ARMS <- BAKES (each arm's toggles are fixed in the bench; the bake must match)
#   ZEN_HY_ADD   -> add156_156basic    v1_only, pools Off  (372-layout emit)
#   ZEN_HY_PEAKS -> peaks156_no_raw    pools Peaks, append+append2 (944 layout)
#   ZEN_HY_FREE  -> free156_peaks_raw  + V1FreeExtras::RawMoments (944 layout)
# A 944-declared-width bake CANNOT run in the ADD arm (the walk emits fewer
# columns and the forward fails loud). So ZEN_HY_ADD stays the shipped 372
# Profile D -- which is also exactly the denominator the 1.25x bar names.
# There is no class-C (289) arm in the instrument; see plan gate G5a.
#
# Usage: w4_speed.sh <peaks-bake.bin> <free-bake.bin> [out-dir]
set -euo pipefail
PEAKS_BAKE="${1:?peaks (228-slice) bake required}"
FREE_BAKE="${2:?free (261/265-slice) bake required}"
OUT="${3:-/mnt/v/output/zensim/fastclass2-2026-09-05/speed}"
REPO="${FC2_REPO:-/home/lilith/work/zen/zensim--fastclass2}"
BENCHDIR=/mnt/v/zen/cargo-targets/fastclass2-bench/release/deps
BENCH=$(ls -t "$BENCHDIR"/ssim2_speed_bar-* 2>/dev/null | grep -v '\.d$' | head -1)
[ -x "$BENCH" ] || { echo "ABORT: bench not built; cd $REPO/zensim-bench && CARGO_TARGET_DIR=/mnt/v/zen/cargo-targets/fastclass2-bench cargo build --release --bench ssim2_speed_bar"; exit 2; }
mkdir -p "$OUT"
export ZEN_HY_ADD="${ZEN_HY_ADD:-$REPO/zensim/weights/d_sdr_add156_id100_negrich_dial_2026-09-05.bin}"
export ZEN_HY_PEAKS="$PEAKS_BAKE"
export ZEN_HY_FREE="$FREE_BAKE"
STARTS="${FC2_STARTS:-10}"
for T in ${FC2_THREADS:-1 8}; do
  for S in ${FC2_SIZES:-576 1152 2304}; do
    case "$S" in 576) W=020 ;; 1152) W=030 ;; *) W=060 ;; esac
    for i in $(seq 1 "$STARTS"); do
      RAYON_NUM_THREADS="$T" ZEN_S2_SIZES="$S" ZEN_S2_WALL_S="$W" ZEN_S2_EXTRACT_ONLY=1 \
        "$BENCH" >> "$OUT/w4_t${T}_s${S}.log" 2>&1 || echo "start $i failed (t=$T s=$S)"
    done
    echo "collected $STARTS starts: t=$T size=$S -> $OUT/w4_t${T}_s${S}.log"
  done
done
python3 "$REPO/benchmarks/fastclass2_campaign_2026-09-05/w4_report.py" "$OUT"
