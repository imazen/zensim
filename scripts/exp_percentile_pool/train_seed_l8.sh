#!/usr/bin/env bash
# EX-PERCENTILE-POOL: L8 baseline — identical recipe to train_seed.sh
# but with the EXISTING L8 features from 2026-05-17-cvvdp-merged-trainer.
# This is the apples-to-apples comparison: same limited corpus (kadid+
# tid+konjnd, no safesyn, no cvvdp_iwssim_large), same recipe, only
# difference is L8 vs P² Block B features.
set -euo pipefail
SEED="${1:?seed}"
OUT_DIR="${2:?out_dir}"
mkdir -p "$OUT_DIR"

BAKE="$OUT_DIR/l8baseline_seed${SEED}.bin"
LOG="$OUT_DIR/l8baseline_seed${SEED}.log"
TRAINER=/home/lilith/work/zen/zensim--exp-percentile-pool/target/release/zensim_mlp_train

L8_DIR=/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer

"$TRAINER" \
  --group kadid:$L8_DIR/kadid_mix_300col.parquet:0.3:1.0 \
  --group tid:$L8_DIR/tid_mix_300col.parquet:0.3:1.0 \
  --group konjnd:$L8_DIR/konjnd_mix_300col.parquet:0.02:1.0 \
  --hidden 128 --epochs 300 --pairs-per-epoch 50000 --max-features 300 \
  --target-column mix_cv40_iw60 \
  --val-policy min --minibatch-size 256 --pwrc-pair-weight --pwrc-sensory-threshold 5.0 \
  --norm-in-norm-weight 0.1 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0 \
  --per-sample-alpha-head --seed "$SEED" --log-every 10 --early-stop-patience 0 \
  --out "$BAKE" 2>&1 | tee "$LOG"

echo "DONE seed=$SEED bake=$BAKE log=$LOG"
