#!/bin/bash
# a4bkon lane: run K1 (kon-mass sweep, 4 runs) + K2 (mixed teacher w/ ttbig
# leg, 2 runs) sequentially. K3 is NOT here -- it depends on K1's results per
# the §24.3 mechanical selection rule and runs in a separate follow-up step.
set -euo pipefail
REPO=/home/lilith/work/zen/zensim--a4bkon
cd "$REPO"
export ZL_TRAIN=/mnt/v/zen/cargo-targets/waver4/release/zensim_mlp_train
export WR4_OUT=/mnt/v/output/zensim/a4bkon-2026-09-01/bakes
export WR4_KEEP="$REPO/scripts/sota944/slice_basic156_free.txt"
export WR4_DISTILL_SAFESYN=/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01/foldapp2_views/safesyn_distill_hya_r4.parquet
mkdir -p "$WR4_OUT"
SCRIPT="$REPO/benchmarks/wave_r4_2026-09-01/train_156_student.sh"

echo "=========================================================="
echo "K1 (kon-mass sweep): weights 1.8, 2.4 x seeds 4004, 4005"
echo "=========================================================="
for W in 1.8 2.4; do
  for SEED in 4004 4005; do
    OUT="$WR4_OUT/K1_w${W}_s${SEED}.bin"
    if [ -f "$OUT" ]; then
      echo "== SKIP (exists): $OUT"
      continue
    fi
    echo "---- K1 w=$W seed=$SEED $(date -u +%H:%M:%SZ) ----"
    WR4_KONJND_TRAIN_W="$W" "$SCRIPT" distill "$SEED" "$OUT"
  done
done

echo "=========================================================="
echo "K2 (mixed teacher, ttbig leg): seeds 4004, 4005"
echo "=========================================================="
export WR4_DISTILL_TBIG=/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01/recipe_views/ttbig_distill_hya_r4.parquet
for SEED in 4004 4005; do
  OUT="$WR4_OUT/K2_s${SEED}.bin"
  if [ -f "$OUT" ]; then
    echo "== SKIP (exists): $OUT"
    continue
  fi
  echo "---- K2 seed=$SEED $(date -u +%H:%M:%SZ) ----"
  "$SCRIPT" distill "$SEED" "$OUT"
done

echo "=========================================================="
echo "K1+K2 DONE $(date -u +%H:%M:%SZ)"
echo "=========================================================="
ls -la "$WR4_OUT"/K1_*.bin "$WR4_OUT"/K2_*.bin
