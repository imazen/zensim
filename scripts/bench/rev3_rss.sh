#!/usr/bin/env bash
# Peak-RSS half of the arithmetic-revision cost question (issue #61).
#
# The scorecard's memory clause is "peak incremental RSS, including
# caches/scratch/maps, <= 128 bytes/pixel + 64 MiB per worker". Revision 3 adds
# a per-strip f64 signal plane and the kernel's row-ring scratch, so the
# question is whether that moves the envelope.
#
# `/usr/bin/time -v` max RSS is the evidence (CLAUDE.md: RSS from `ps` or
# allocator-traced bytes are NOT substitutes). The bench's `ZEN_XP_RSS` mode
# runs ONE arm in a loop with no zenbench harness, which is what makes an
# external RSS reading attributable to that arm.
#
# Usage: scripts/bench/rev3_rss.sh <out-dir> [sizes] [arms] [cpu]
set -euo pipefail

OUT="${1:?usage: rev3_rss.sh <out-dir> [sizes] [arms] [cpu]}"
SIZES="${2:-1024 2048}"
ARMS="${3:-buf_v1_372 fold372_full fold944_full}"
CPU="${4:-8}"
ITERS="${ZEN_XP_ITERS:-8}"

mkdir -p "$OUT"
BIN=$(ls -t target/release/deps/extract_paths_bench-* 2>/dev/null | grep -v '\.d$' | head -1)
[ -n "${BIN:-}" ] || { echo "build the bench first (see rev3_cost_ab.sh)" >&2; exit 2; }
sha256sum "$BIN" | tee "$OUT/binary.sha256"

printf 'arm\tsize\trevision\tmax_rss_kb\tbytes_per_pixel\twall_s\n' > "$OUT/rss.tsv"
for size in $SIZES; do
  for arm in $ARMS; do
    for rev in 1 3; do
      log="$OUT/${arm}_${size}_rev${rev}.time"
      ZEN_XP_RSS="$arm" ZEN_XP_SIZE="$size" ZEN_XP_ITERS="$ITERS" \
      RAYON_NUM_THREADS=1 ZENSIM_FORMULA_REV="$rev" \
        taskset -c "$CPU" nice -n19 ionice -c3 \
        /usr/bin/time -v "$BIN" >"$log" 2>&1 || { cat "$log"; exit 3; }
      kb=$(awk -F': ' '/Maximum resident set size/{print $2}' "$log")
      wall=$(awk -F': ' '/Elapsed \(wall clock\)/{print $2}' "$log")
      bpp=$(python3 -c "print(f'{$kb*1024/($size*$size):.1f}')")
      printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$arm" "$size" "$rev" "$kb" "$bpp" "$wall" \
        | tee -a "$OUT/rss.tsv"
    done
  done
done
echo "wrote $OUT/rss.tsv"
