#!/usr/bin/env bash
# zgeom lane: Z2 stability + cost measurements (post-fit, under heavy lock).
set -uo pipefail
OUT=/mnt/v/output/zensim/zgeom-2026-09-21
BENCH=/home/lilith/tmp/devin/zgeom-target-bench/release/examples
fails=0

echo "=== stability: 3 kernels x 11 transforms x 192 pairs $(date +%H:%M:%S)"
RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-8} "$BENCH/zgeom_stability" \
    --pairs "$OUT/extract/stability192.tsv" --rows 192 \
    --kernels 'box2;bin121;bin1331' \
    --out "$OUT/stability/stability.csv" \
    || fails=$((fails+1))
python3 /home/lilith/work/zen/zensim--transplant/tools/zgeom/analyze_stability.py \
    "$OUT/stability/stability.csv" || fails=$((fails+1))

echo "=== cost: 3 kernels x {256,1024,2048}^2 x {1,8} threads $(date +%H:%M:%S)"
for t in 1 8; do
  RAYON_NUM_THREADS=$t "$BENCH/zgeom_cost" \
      --sizes 256,1024,2048 --kernels 'box2;bin121;bin1331' \
      --rounds 12 --warmup 3 --parallel 1 --threads-label $t \
      --out "$OUT/cost/cost_t$t.csv" || fails=$((fails+1))
done
python3 - "$OUT/cost/cost.csv" <<'EOF' || fails=$((fails+1))
import sys, csv
out = sys.argv[1]
rows = []
for t in (1, 8):
    rows += list(csv.DictReader(open(out.replace('cost.csv', f'cost_t{t}.csv'))))
w = csv.DictWriter(open(out, 'w'), fieldnames=rows[0].keys())
w.writeheader(); w.writerows(rows)
print('merged', len(rows), '->', out)
EOF
python3 /home/lilith/work/zen/zensim--transplant/tools/zgeom/analyze_cost.py \
    "$OUT/cost/cost.csv" || fails=$((fails+1))

echo "measurements done, fails=$fails"
