#!/bin/bash
# V_25-v2-perpair — same 5-group recipe as V_22-mix-LARGE+iwssim
# but with EX-4 24 per-ref XYB+LMS features (f300..f323) AND
# 19 per-pair CVVDP-shape features (f324..f342) appended.
# 343 inputs total, max_features=343.
# anchor groups (kadid/tid/konjnd/cid22-val/aic3) get REAL per-pair features.
# safesyn + cvvdp_iwssim_large get ZERO-FILLED per-pair (signal is constant
# within-pair → no ranking gradient contribution from those columns, but
# anchor groups carry the per-pair training signal).

set -u

TRAINER=/home/lilith/work/zen/zensim--ex4-extfeat/target/release/zensim_mlp_train
DATA=/mnt/v/zen/zensim-training/2026-05-18-extfeat
OUT_DIR=/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17
LOG_DIR=/tmp/v25_v2_extfeat_logs
mkdir -p "$LOG_DIR" "$OUT_DIR"

SEED=${1:-3}
TAG=${2:-h128_s${SEED}}

out="$OUT_DIR/v25_v2_extfeat_mix_cv40_konjnd_0_02_LARGE_iwssim_perpair_${TAG}.bin"
log="$LOG_DIR/${TAG}.log"

echo "[$(date -u +%H:%M:%S)] launch V_25-v2-perpair ${TAG} -> $out"
RAYON_NUM_THREADS=16 "$TRAINER" \
  --group safesyn:"$DATA/safesyn_extfeat_343.parquet":1.0:1.0 \
  --group kadid:"$DATA/kadid_extfeat_343.parquet":0.3:1.0 \
  --group tid:"$DATA/tid_extfeat_343.parquet":0.3:1.0 \
  --group konjnd:"$DATA/konjnd_extfeat_343.parquet":0.02:1.0 \
  --group cvvdp_large:"$DATA/cvvdp_iwssim_large_extfeat_343.parquet":0.5:0.0 \
  --target-column mix_cv40_iw60 --target-scale 1.0 --seed $SEED \
  --hidden 128 --epochs 300 --pairs-per-epoch 50000 \
  --lr 1e-3 --l2 1e-5 --leaky-alpha 0.01 --val-policy min \
  --log-every 30 --early-stop-patience 60 --max-features 343 \
  --minibatch-size 256 --pwrc-pair-weight --norm-in-norm-weight 0.1 \
  --out "$out" 2>&1 | tee "$log" | tail -10
echo "[$(date -u +%H:%M:%S)] done $TAG"
