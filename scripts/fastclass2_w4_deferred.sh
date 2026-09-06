#!/bin/bash
# fastclass2 W4, DEFERRED — the measurement benchmarks/fastclass2_campaign_
# 2026-09-05.md §17 reports as NOT MEASURED because the box was at load 72-79
# when the campaign reached it. Self-contained: paths point at the MAIN
# checkout, so it survives the campaign workspace being cleaned up.
#
# Protocol (the coordinator's, and profile_d_notax_2026-09-01.md §4's):
#   ONE binary at HEAD, both candidate arms loaded into the SAME runs so they
#   interleave inside zenbench's round-robin; the two bit-identical controls
#   (fast_ssim2, zensim_B) present in every run; ZEN_S2_EXTRACT_ONLY=1 to split
#   walk from forward pass; min over N process starts with ASLR on; wall time
#   scaled with size; BOTH SIMD tiers. It REFUSES a busy box rather than
#   emitting a contaminated number — the same rule
#   scripts/kernel_fastclass_sweep.sh enforces at load 3.0.
set -uo pipefail
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
O=${W4_OUT:-/mnt/v/output/zensim/fastclass2-2026-09-05/speed}
MAXLOAD=${W4_MAXLOAD:-4.0}
STARTS=${W4_STARTS:-10}
CAND=${W4_CAND:-/mnt/v/output/zensim/fastclass2-2026-09-05/serv372/bakes/S372_S228_H128_p_s4004_packed.bin}
CAND156=${W4_CAND156:-/mnt/v/output/zensim/fastclass2-2026-09-05/serv372/bakes/S372_S156_H32_p_s4004_packed.bin}
load=$(awk '{print $1}' /proc/loadavg)
if awk -v l="$load" -v m="$MAXLOAD" 'BEGIN{exit !(l>m)}'; then
  echo "REFUSED: load $load > $MAXLOAD — a W4 number taken now is contaminated." >&2
  echo "profile_d_notax_2026-09-01.md §4 measured ONE concurrent niced build" >&2
  echo "swinging the stable fast_ssim2 arm 128.9-633.6 ms inside a single cell." >&2
  exit 3
fi
BENCH=$(ls -t /mnt/v/zen/cargo-targets/fastclass2-bench/release/deps/ssim2_speed_bar-* 2>/dev/null | grep -v '\.d$' | head -1)
if [ ! -x "${BENCH:-}" ]; then
  echo "building the bench at HEAD (the kernel lanes move the 156-walk baseline;" >&2
  echo "no W4 number may be compared across builds)" >&2
  ( cd "$REPO/zensim-bench" && CARGO_TARGET_DIR=/mnt/v/zen/cargo-targets/fastclass2-bench \
      nice -n19 ionice -c3 cargo build --release --bench ssim2_speed_bar ) || exit 2
  BENCH=$(ls -t /mnt/v/zen/cargo-targets/fastclass2-bench/release/deps/ssim2_speed_bar-* | grep -v '\.d$' | head -1)
fi
mkdir -p "$O"
export ZEN_HY_ADD="$REPO/zensim/weights/d_sdr_add156_id100_negrich_dial_2026-09-05.bin"
export ZEN_HY_PEAKS="$CAND"
export ZEN_HY_FREE="$CAND156"
for f in "$ZEN_HY_ADD" "$ZEN_HY_PEAKS" "$ZEN_HY_FREE"; do
  [ -f "$f" ] || { echo "missing arm bake: $f" >&2; exit 2; }
done
for TIER in native capv3; do
  for T in 1 8; do
    for S in 576 1152; do
      case $S in 576) W=015 ;; *) W=030 ;; esac
      LOG="$O/w4_${TIER}_t${T}_s${S}.log"; : > "$LOG"
      for i in $(seq 1 "$STARTS"); do
        if [ "$TIER" = capv3 ]; then
          RAYON_NUM_THREADS=$T ZEN_S2_SIZES=$S ZEN_S2_WALL_S=$W ZEN_S2_EXTRACT_ONLY=1 ZEN_S2_CAP_V3=1 "$BENCH" >> "$LOG" 2>&1
        else
          RAYON_NUM_THREADS=$T ZEN_S2_SIZES=$S ZEN_S2_WALL_S=$W ZEN_S2_EXTRACT_ONLY=1 "$BENCH" >> "$LOG" 2>&1
        fi
      done
      echo "$(date -u +%FT%TZ) $TIER t=$T s=$S" >> "$O/w4_progress.txt"
    done
  done
done
python3 "$REPO/benchmarks/fastclass2_campaign_2026-09-05/w4_report.py" "$O" | tee "$O/w4_table.txt"
