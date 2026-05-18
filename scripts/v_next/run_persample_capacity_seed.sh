#!/usr/bin/env bash
# EXP-PERSAMPLE-CAPACITY: V_24 per-sample-α recipe with --hidden swept.
# Identical to ex2-persample-alpha/run_per_sample_alpha_seed.sh EXCEPT for --hidden.
# Args: <seed> <hidden> <out_dir>
set -euo pipefail
SEED="${1:?seed}"
HIDDEN="${2:?hidden}"
OUT_DIR="${3:?out_dir}"
mkdir -p "$OUT_DIR"

BAKE="$OUT_DIR/h${HIDDEN}_s${SEED}_h${HIDDEN}.bin"
LOG="$OUT_DIR/h${HIDDEN}_s${SEED}_h${HIDDEN}.log"
TRAINER=/home/lilith/work/zen/zensim--exp-persample-capacity/target/release/zensim_mlp_train

"$TRAINER" \
  --group safesyn:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/safesyn_mix_300col.parquet:1.0:0.0 \
  --group kadid:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/kadid_mix_300col.parquet:0.3:1.0 \
  --group tid:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/tid_mix_300col.parquet:0.3:1.0 \
  --group konjnd:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/konjnd_mix_300col.parquet:0.02:1.0 \
  --group cvvdp_iwssim_large:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/cvvdp_iwssim_large_300col_v2.parquet:0.5:0.0 \
  --hidden "$HIDDEN" --epochs 300 --pairs-per-epoch 50000 --max-features 300 \
  --target-column mix_cv40_iw60 \
  --val-policy min --minibatch-size 256 --pwrc-pair-weight --pwrc-sensory-threshold 5.0 \
  --norm-in-norm-weight 0.1 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0 \
  --per-sample-alpha-head --seed "$SEED" --log-every 10 --early-stop-patience 0 \
  --out "$BAKE" 2>&1 | tee "$LOG"

echo "DONE seed=$SEED hidden=$HIDDEN bake=$BAKE log=$LOG"
