#!/usr/bin/env bash
# Train V_24-α bakes for a single α value (seed=3 by default).
# Usage:
#   ./train_v24_alpha_sweep.sh <alpha_pct_int> [seed] [out_dir]
# Example:
#   ./train_v24_alpha_sweep.sh 5            -> α=0.05, seed=3, default out
#   ./train_v24_alpha_sweep.sh 10 1         -> α=0.10, seed=1
#   ./train_v24_alpha_sweep.sh 15 3 /tmp    -> custom out dir
#
# Recipe is V_22-mix-LARGE+iwssim verbatim, only training corpus changes.

set -euo pipefail

ALPHA_PCT="${1:?need alpha pct (e.g. 5 for α=0.05)}"
TRAIN_SEED="${2:-3}"
OUT_DIR_DEFAULT="/mnt/v/zen/zensim-eval/v24_alpha_2026-05-18"
OUT_DIR="${3:-$OUT_DIR_DEFAULT}"

ALPHA_SFX=$(printf "alpha%03d" "$ALPHA_PCT")
CORPUS_DIR="/mnt/v/zen/zensim-training/2026-05-18-v24-alpha"
TRAINER="/home/lilith/work/zen/zensim/target/release/zensim_mlp_train"
mkdir -p "$OUT_DIR"

BAKE_OUT="$OUT_DIR/v24_${ALPHA_SFX}_s${TRAIN_SEED}_h128.bin"
LOG_OUT="$OUT_DIR/v24_${ALPHA_SFX}_s${TRAIN_SEED}_h128.log"

echo "[$(date -Iseconds)] α=${ALPHA_PCT}% seed=${TRAIN_SEED}"
echo "  corpus: $CORPUS_DIR/{safesyn,kadid,tid,large}_${ALPHA_SFX}.parquet + konjnd.parquet"
echo "  out:    $BAKE_OUT"

"$TRAINER" \
  --group "safesyn:${CORPUS_DIR}/safesyn_${ALPHA_SFX}.parquet:1.0:1.0" \
  --group "kadid:${CORPUS_DIR}/kadid_${ALPHA_SFX}.parquet:0.3:1.0" \
  --group "tid:${CORPUS_DIR}/tid_${ALPHA_SFX}.parquet:0.3:1.0" \
  --group "konjnd:${CORPUS_DIR}/konjnd.parquet:0.02:0.0" \
  --group "large:${CORPUS_DIR}/large_${ALPHA_SFX}.parquet:0.5:0.0" \
  --target-column mix_target --target-scale 1.0 \
  --hidden 128 --epochs 300 --pairs-per-epoch 50000 \
  --lr 1e-3 --l2 1e-5 --leaky-alpha 0.01 --val-policy min --seed "$TRAIN_SEED" \
  --log-every 30 --max-features 300 \
  --minibatch-size 256 --pwrc-pair-weight --norm-in-norm-weight 0.1 \
  --early-stop-patience 0 \
  --out "$BAKE_OUT" \
  2>&1 | tee "$LOG_OUT"

echo "[$(date -Iseconds)] DONE α=${ALPHA_PCT}% seed=${TRAIN_SEED}"
echo "  bake md5: $(md5sum "$BAKE_OUT" | cut -d' ' -f1)"
ls -la "$BAKE_OUT"
