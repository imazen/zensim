#!/bin/bash
# V_25-extfeat — same 5-group recipe as V_22-mix-LARGE+iwssim
# but with the EX-4 24 per-ref XYB+LMS features appended.
# 396 inputs (372 base + 24 EX-4 XYB+LMS), max_features=396.
# Per-pair CVVDP-shape features (19) are NOT included in this batch
# (dist image regeneration required, deferred).

set -u

TRAINER=/home/lilith/work/zen/zensim/target/release/zensim_mlp_train
DATA=/mnt/v/zen/zensim-training/2026-05-18-extfeat
OUT_DIR=/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17
LOG_DIR=/tmp/v25_extfeat_logs
mkdir -p "$LOG_DIR" "$OUT_DIR"

SEED=${1:-3}
TAG=${2:-h128_s${SEED}}

out="$OUT_DIR/v25_extfeat_mix_cv40_konjnd_0_02_LARGE_iwssim_${TAG}.bin"
log="$LOG_DIR/${TAG}.log"
if [[ -s "$out" ]]; then
    echo "SKIP $TAG (already exists at $out)"
    exit 0
fi

echo "[$(date -u +%H:%M:%S)] launch V_25 ${TAG} -> $out"
RAYON_NUM_THREADS=16 "$TRAINER" \
  --group safesyn:"$DATA/safesyn_extfeat_324.parquet":1.0:1.0 \
  --group kadid:"$DATA/kadid_extfeat_324.parquet":0.3:1.0 \
  --group tid:"$DATA/tid_extfeat_324.parquet":0.3:1.0 \
  --group konjnd:"$DATA/konjnd_extfeat_324.parquet":0.02:1.0 \
  --group cvvdp_large:"$DATA/cvvdp_iwssim_large_extfeat.parquet":0.5:0.0 \
  --target-column mix_cv40_iw60 --target-scale 1.0 --seed $SEED \
  --hidden 128 --epochs 300 --pairs-per-epoch 50000 \
  --lr 1e-3 --l2 1e-5 --leaky-alpha 0.01 --val-policy min \
  --log-every 30 --early-stop-patience 60 --max-features 324 \
  --minibatch-size 256 --pwrc-pair-weight --norm-in-norm-weight 0.1 \
  --out "$out" 2>&1 | tee "$log" | tail -5
echo "[$(date -u +%H:%M:%S)] done $TAG"
