#!/usr/bin/env bash
# joint-core-v1 leader reproduction: R915 basic228/h128 recipe, leader seeds,
# core legs substituted for the frozen train tables. Dev groups are the
# FROZEN development tables (read-only), identical to R915.
#
#   train:  safesyn->fresh_safesyn  cid22->cid22  human->human
#           codec->fresh_imazen26   (same weights/loss modes as R915)
#   val:    the frozen *_development.parquet groups, unchanged
#
# Args: $1 = seed, $2 = out prefix, $3 = optional table-suffix override
#       (perm30 variant tables), $4 = optional epochs override.
set -euo pipefail
SEED="$1"; OUT="$2"; SUFFIX="${3:-}"; EPOCHS="${4:-120}"
# R915 sample-seed = (leader-seed index + 1) * 1e9
case "$SEED" in
  17101) SSEED=1000000000 ;; 17103) SSEED=2000000000 ;;
  17107) SSEED=3000000000 ;; 17111) SSEED=4000000000 ;;
  17113) SSEED=5000000000 ;; *) SSEED=$((SEED + 10000)) ;;
esac
FEAT=${FEATDIR:-/mnt/v/output/zensim/joint-core-v1/features}
DEV=${DEVDIR:-/var/tmp/zensim-validation-2026-09-15/recovery/dedup-tables}
KEEP=$(seq -s, 0 227)
exec /home/lilith/work/zen/zensim/target/release/zensim_mlp_train \
  --group "safesyn:$FEAT/fresh_safesyn$SUFFIX.parquet:1.0168526508775275:0:withinref,both" \
  --group "safesyn_development:$DEV/safesyn_development.parquet:0:0.5:withinref,both" \
  --group "cid22:$FEAT/cid22$SUFFIX.parquet:1.0115735134169879:0:withinref,both" \
  --group "cid22_development:$DEV/cid22_development.parquet:0:2.0:withinref,both" \
  --group "human:$FEAT/human$SUFFIX.parquet:0.5041192364219411:0:withinref,rank" \
  --group "human_development:$DEV/human_development.parquet:0:1.0:withinref,rank" \
  --group "codec:$FEAT/fresh_imazen26$SUFFIX.parquet:0.6060902647942771:0:withinref,both" \
  --group "codec_development:$DEV/codec_development.parquet:0:1.0:withinref,both" \
  --target-column human_score --target-scale 1 \
  --hidden 128 --epochs "$EPOCHS" --pairs-per-epoch 50000 \
  --seed "$SEED" --init-seed "$SEED" --sample-seed "$SSEED" \
  --pair-sampling uniform --max-features 944 --keep-features "$KEEP" \
  --mse-weight 1 --early-stop-patience 0 \
  --val-policy mean --val-aggregate geomean3 \
  --out-dtype f32 --log-every 1 --no-auto-eval \
  --out "$OUT"
