#!/bin/bash
# a4bkon lane: K3 = K2's recipe (ttbig leg present) + K1's WINNING kon weight,
# selected by the §24.3 mechanical rule (select_k3_weight.py). Run only after
# that selection exists.
set -euo pipefail
REPO=/home/lilith/work/zen/zensim--a4bkon
cd "$REPO"
SEL=/mnt/v/output/zensim/a4bkon-2026-09-01/k3_selection.json
[ -f "$SEL" ] || { echo "ABORT: run select_k3_weight.py first"; exit 1; }
W=$(python3 -c "import json; print(json.load(open('$SEL'))['selected'])")
echo "== K3 selected kon weight: $W (from $SEL)"

export ZL_TRAIN=/mnt/v/zen/cargo-targets/waver4/release/zensim_mlp_train
export WR4_OUT=/mnt/v/output/zensim/a4bkon-2026-09-01/bakes
export WR4_KEEP="$REPO/scripts/sota944/slice_basic156_free.txt"
export WR4_DISTILL_SAFESYN=/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01/foldapp2_views/safesyn_distill_hya_r4.parquet
export WR4_DISTILL_TBIG=/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01/recipe_views/ttbig_distill_hya_r4.parquet
export WR4_KONJND_TRAIN_W="$W"
SCRIPT="$REPO/benchmarks/wave_r4_2026-09-01/train_156_student.sh"

for SEED in 4004 4005; do
  OUT="$WR4_OUT/K3_s${SEED}.bin"
  if [ -f "$OUT" ]; then echo "== SKIP (exists): $OUT"; continue; fi
  echo "---- K3 (w=$W) seed=$SEED $(date -u +%H:%M:%SZ) ----"
  "$SCRIPT" distill "$SEED" "$OUT"
done
echo "K3 DONE $(date -u +%H:%M:%SZ)"
ls -la "$WR4_OUT"/K3_*.bin
