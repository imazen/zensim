#!/usr/bin/env bash
# EX-1: V_24-thurstone+konjnd@0.02+LARGE+iwssim, 5-seed CI on V_22-mix-LARGE recipe.
# Matches the V_22 launch script (/tmp/run_iwssim_LARGE.sh) verbatim
# except for `--loss thurstone --thurstone-d 0.6745 --thurstone-eps
# 5.0` and `--norm-in-norm-weight 0` (Thurstone path forces NiN off).
set -e
SEED=$1
DATA_DIR=/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer
OUT_DIR=/mnt/v/output/zensim/ex1_thurstone_2026-05-18
LOG_DIR="$OUT_DIR/logs"
BAKE_DIR="$OUT_DIR/bakes"
OUT="$BAKE_DIR/v24_thurstone_konjnd_002_LARGE_iwssim_s${SEED}_h128.bin"

/home/lilith/work/zen/zensim/target/release/zensim_mlp_train \
  --group "safesyn:$DATA_DIR/safesyn_mix_300col.parquet:1.0:0.0" \
  --group "kadid:$DATA_DIR/kadid_mix_300col.parquet:0.3:1.0" \
  --group "tid:$DATA_DIR/tid_mix_300col.parquet:0.3:1.0" \
  --group "konjnd:$DATA_DIR/konjnd_mix_300col.parquet:0.02:1.0" \
  --group "cvvdp_iwssim_large:$DATA_DIR/cvvdp_large_300col.parquet:0.5:0.0" \
  --hidden 128 \
  --max-features 300 \
  --epochs 300 \
  --pairs-per-epoch 50000 \
  --lr 0.001 \
  --l2 0.00001 \
  --leaky-alpha 0.01 \
  --val-policy min \
  --seed "$SEED" \
  --log-every 30 \
  --early-stop-patience 120 \
  --minibatch-size 256 \
  --pwrc-pair-weight \
  --pwrc-sensory-threshold 5.0 \
  --loss thurstone \
  --thurstone-d 0.6745 \
  --thurstone-eps 5.0 \
  --norm-in-norm-weight 0.0 \
  --out "$OUT" \
  > "$LOG_DIR/seed${SEED}.log" 2>&1
echo "DONE seed=$SEED -> $OUT"
