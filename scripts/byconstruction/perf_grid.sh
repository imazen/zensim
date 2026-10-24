#!/usr/bin/env bash
# Paired before/after timing grid for the speed pass: B (revision 5) and Tga (revision 6) at 512² and 2048², at the
# v4x and v3 tiers, one thread pinned to core 2. Each cell is ring_timing.sh (alternating arm order per round) and
# timing_summary.py (median per-round ratio with a bootstrap CI). Record: benchmarks/byconstruction_2026-10-22.md.
# Usage: perf_grid.sh A_BIN B_BIN OUT_DIR [ROUNDS]; RAW (default OUT_DIR) receives the per-round JSONL.
set -euo pipefail
A=$1
B=$2
OUT=$3
ROUNDS=${4:-10}
RAW=${RAW:-$OUT}
mkdir -p "$OUT" "$RAW"
cd ~/work/zen/zensim--byconstruction
S=scripts/byconstruction
BB=$PWD/zensim/weights/b_sdr_diff_frag_820_rev5_2026-10-19.bin
TGA=$HOME/tmp/byc/parity/tga_full_rev6.bin
unset ZENSIM_FORMULA_REV ZENSIM_RING_FORM
{
  echo "# A=$A sha256=$(sha256sum "$A" | cut -c1-16)"
  echo "# B=$B sha256=$(sha256sum "$B" | cut -c1-16)"
  echo "# host=$(hostname) date=$(date -u +%Y-%m-%dT%H:%M:%SZ) rounds=$ROUNDS load=$(cut -d' ' -f1-3 /proc/loadavg)"
} > "$OUT/grid.meta"
printf 'bake\ttier\tsize\tA_ms\tB_ms\tdelta\tci95_lo\tci95_hi\n' > "$OUT/grid.tsv"
for bake in "b:$BB" "tga:$TGA"; do
  for tier in v4x v3; do
    for size in 512 2048; do
      iters=40
      [ "$size" = 2048 ] && iters=8
      jsonl="$RAW/${bake%%:*}_${tier}_${size}.jsonl"
      TIER=$tier bash $S/ring_timing.sh "$A" "$B" "${bake#*:}" "$size" "$iters" "$ROUNDS" "$jsonl"
      python3 $S/timing_summary.py "$jsonl" | python3 -c "
import json, sys
d = json.load(sys.stdin)
lo, hi = d['ci95_pct']
print('\t'.join(['${bake%%:*}', '$tier', '$size', f\"{d['A_median_ms']:.4f}\", f\"{d['B_median_ms']:.4f}\",
                 f\"{d['paired_delta_pct']:+.2f}%\", f'{lo:+.2f}%', f'{hi:+.2f}%']))" >> "$OUT/grid.tsv"
      tail -1 "$OUT/grid.tsv"
    done
  done
done
