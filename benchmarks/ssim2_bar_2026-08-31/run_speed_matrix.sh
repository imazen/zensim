#!/usr/bin/env bash
# Speed row of the ssim2-replacement exam.
#   pass 1: zensim/extract_paths_bench   — 7 zensim walks + fast_ssim2, interleaved
#   pass 2: zensim-bench/ssim2_speed_bar — fast_ssim2 with and without its rayon
#           feature, anchored by an unchanged zensim_B arm in both builds
# One log per (bench, thread count). Record: benchmarks/ssim2_replacement_bar_2026-08-31.md
set -uo pipefail
ROOT="${ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
OUT="${OUT:-$ROOT/benchmarks/ssim2_bar_2026-08-31}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$ROOT/target-s2}"
WALL_S="${WALL_S:-150}"
cd "$ROOT"

xpbin() { ls -t "$CARGO_TARGET_DIR"/release/deps/extract_paths_bench-* 2>/dev/null | grep -v '\.d$' | head -1; }
s2bin() { ls -t "$CARGO_TARGET_DIR"/release/deps/ssim2_speed_bar-*    2>/dev/null | grep -v '\.d$' | head -1; }

echo "## pass 1 — extract_paths_bench (zensim walks + fast_ssim2 interleaved)"
nice -n19 ionice -c3 cargo build --release --bench extract_paths_bench -p zensim \
  --features custom-profiles,feature-regime-v2,threads,training >"$OUT/build_xp.log" 2>&1 || exit 2
for T in 1 8 16; do
  echo "=== extract_paths_bench T=$T"
  RAYON_NUM_THREADS=$T ZEN_XP_ROUNDS=40 ZEN_XP_WALL_S="$WALL_S" \
    nice -n19 ionice -c3 "$(xpbin)" > "$OUT/xp_${T}t.log" 2>&1
  echo "  rc=$? -> xp_${T}t.log"
done

echo "## pass 2 — ssim2_speed_bar (opponent threading)"
for FEAT in plain rayon; do
  if [ "$FEAT" = plain ]; then FL=(); else FL=(--features ssim2-rayon); fi
  # zensim-bench is NOT a workspace member — build from its own directory.
  ( cd "$ROOT/zensim-bench" && nice -n19 ionice -c3 cargo build --release \
      --bench ssim2_speed_bar "${FL[@]}" ) >"$OUT/build_s2_${FEAT}.log" 2>&1 \
    || { echo "build $FEAT FAILED"; continue; }
  for T in 1 8 16; do
    echo "=== ssim2_speed_bar[$FEAT] T=$T"
    RAYON_NUM_THREADS=$T ZEN_S2_ROUNDS=40 ZEN_S2_WALL_S="$WALL_S" \
      nice -n19 ionice -c3 "$(s2bin)" > "$OUT/s2bar_${FEAT}_${T}t.log" 2>&1
    echo "  rc=$? -> s2bar_${FEAT}_${T}t.log"
  done
done
echo DONE
