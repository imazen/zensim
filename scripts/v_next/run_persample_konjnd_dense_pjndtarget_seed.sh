#!/usr/bin/env bash
# V_24-PS-α-dense-pjndtarget — densified KonJND-1k where the target column
# is the per-source PJND THRESHOLD (broadcast across all 20 distortion
# variants per source), NOT the per-pair ssim2 score. This directly
# matches the legacy KonJND-1k val parquet's encoding so the val SROCC
# can survive the densification.
#
# Args: <seed> <konjnd_w> <out_dir>
set -euo pipefail
SEED="${1:?seed}"
KONJND_W="${2:?konjnd_w}"
OUT_DIR="${3:?out_dir}"
mkdir -p "$OUT_DIR"

BAKE="$OUT_DIR/persample_dense_pjnd_konjnd${KONJND_W}_seed${SEED}.bin"
LOG="$OUT_DIR/persample_dense_pjnd_konjnd${KONJND_W}_seed${SEED}.log"
TRAINER=/home/lilith/work/zen/zensim/target/release/zensim_mlp_train

SAFESYN=/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/safesyn_mix_300col.parquet
KADID=/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/kadid_mix_300col.parquet
TID=/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/tid_mix_300col.parquet
CVVDP=/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/cvvdp_iwssim_large_300col_v2.parquet
KONJND_DENSE=/mnt/v/zen/zensim-training/2026-05-18-konjnd-dense/konjnd_dense_pjndtarget_300col.parquet

"$TRAINER" \
  --group "safesyn:${SAFESYN}:1.0:0.0" \
  --group "kadid:${KADID}:0.3:1.0" \
  --group "tid:${TID}:0.3:1.0" \
  --group "konjnd:${KONJND_DENSE}:${KONJND_W}:1.0" \
  --group "cvvdp_iwssim_large:${CVVDP}:0.5:0.0" \
  --hidden 128 --epochs 300 --pairs-per-epoch 50000 --max-features 300 \
  --target-column mix_cv40_iw60 --target-scale 100.0 \
  --val-policy min --minibatch-size 256 \
  --pwrc-pair-weight --pwrc-sensory-threshold 5.0 \
  --norm-in-norm-weight 0.1 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0 \
  --per-sample-alpha-head --seed "$SEED" --log-every 10 --early-stop-patience 0 \
  --out "$BAKE" 2>&1 | tee "$LOG"

echo "DONE seed=$SEED konjnd_w=$KONJND_W bake=$BAKE log=$LOG"
