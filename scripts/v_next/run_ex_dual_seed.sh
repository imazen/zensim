#!/usr/bin/env bash
# EX-DUAL: dual-target multi-task head training — single seed × single λ.
# Args: <seed> <pjnd_loss_weight> <out_dir>
#
# Recipe (minimal baseline, no NiN / no K-batch / no TV — this isolates
# the dual-target effect from V_22-LARGE's other knobs):
#   5-group mix (safesyn 1.0, kadid 0.3, tid 0.3, konjnd 0.02,
#                cvvdp_iwssim_LARGE 0.5)
#   + konjnd_pjnd (broadcast PJND targets, 20,160 rows) — train_w=0
#                                                        but referenced
#                                                        as pjnd-source
#   target = mix_cv40_iw60 (= 0.4·cvvdp + 0.6·iwssim) on all rank groups
#   100 epochs × 10,000 pairs/epoch (~5-6 min/run)
#   h=128, lr=1e-3, leaky=0.01, l2=1e-5
#   --dual-target-head --pjnd-loss-weight <λ> --pjnd-group-name konjnd_pjnd
set -euo pipefail
SEED="${1:?seed}"
LAMBDA="${2:?pjnd_loss_weight}"
OUT_DIR="${3:?out_dir}"
mkdir -p "$OUT_DIR"

BAKE="$OUT_DIR/exdual_l${LAMBDA}_seed${SEED}.bin"
LOG="$OUT_DIR/exdual_l${LAMBDA}_seed${SEED}.log"
TRAINER="/home/lilith/work/zen/zensim--dual-target/target/release/zensim_mlp_train"

"$TRAINER" \
  --group safesyn:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/safesyn_mix_300col.parquet:1.0:0.0 \
  --group kadid:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/kadid_mix_300col.parquet:0.3:1.0 \
  --group tid:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/tid_mix_300col.parquet:0.3:1.0 \
  --group konjnd:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/konjnd_mix_300col.parquet:0.02:1.0 \
  --group cvvdp_iwssim_large:/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer/cvvdp_iwssim_large_300col_v2.parquet:0.5:0.0 \
  --group konjnd_pjnd:/mnt/v/zen/zensim-training/2026-05-18-konjnd-dense/konjnd_dense_pjndtarget_300col.parquet:0.0:0.0 \
  --hidden 128 --epochs 100 --pairs-per-epoch 10000 --max-features 300 \
  --target-column mix_cv40_iw60 \
  --val-policy min --minibatch-size 1 \
  --dual-target-head --pjnd-loss-weight "$LAMBDA" --pjnd-group-name konjnd_pjnd \
  --seed "$SEED" --log-every 10 --early-stop-patience 0 \
  --out "$BAKE" 2>&1 | tee "$LOG"

echo "DONE seed=$SEED lambda=$LAMBDA bake=$BAKE"
