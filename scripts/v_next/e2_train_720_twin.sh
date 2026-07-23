#!/bin/bash
# E2 — ceiling model, TWIN ARM (the append-only feature-validation decision).
# Trains the SAME recipe at 720 (ext = v1-372 ++ v2-348) and at 372 (v1 only,
# --max-features cap); the ONLY diff between arms is --max-features + --out.
# Answers: does the appended v2-348 block improve an optimal global model?
#
# Feature-validation config (NOT the ship recipe): plain fp32 rank+mse, no
# monotone-mask / no QAT (those need a 720-wide sign mask that doesn't exist yet
# and are the E10 *ship* concern, not the rank question). Monotone dial shaping
# is deferred to the ship gate.
#
# CID22-SAFE: training groups are tbig/safesyn/cid201/kadid/tid only. The H
# holdouts (cid22val-49, aic3, aic4, ...) are NEVER training/val groups here —
# they are evaluated POST-HOC by bake_verdict. CID22-49 human MOS never trains
# and never drives early-stop.
#
# Usage: ARM={ext720|v1372} SEED=1 bash scripts/v_next/e2_train_720_twin.sh
set -u
ARM="${ARM:-ext720}"
SEED="${SEED:-1}"
ZM_BIN="${ZM_BIN:-target/release/zensim_mlp_train}"
D=/mnt/v/zen/zensim-training/ext720-canonical-2026-07-22
TBIG=/mnt/v/zen/tbig-720-2026-07-22/tbig_720_TRAIN_NN.parquet
OUTDIR=/mnt/v/output/zensim/bakes
mkdir -p "$OUTDIR"

case "$ARM" in
  ext720) MAXF=720 ;;
  v1372)  MAXF=372 ;;
  *) echo "ARM must be ext720|v1372"; exit 2 ;;
esac
OUT="$OUTDIR/e2_${ARM}_s${SEED}_2026-07-23.bin"

# Ladder groups (real-codec / synthetic per-ref ssim2 ladders) → withinref,both
# (rank within the ref + mse to the ssim2 target). kadid/tid are TRAIN-ONLY guards
# (val_w=0): they carry a different (DMOS) target distribution, so early-stop
# validates on the ssim2 groups (tbig/safesyn/cid201) with --val-policy mean — NOT
# the min-of-all-groups default, which pathologically peaked at epoch 0 as the model
# correctly specialized on the ssim2 target and "forgot" kadid's analytic ranking.
exec "$ZM_BIN" \
  --group "tbig:$TBIG:1.0:0.3:withinref,both" \
  --group "safesyn:$D/ext_safesyn_full.parquet:1.0:0.5:withinref,both" \
  --group "cid201:$D/ext_cid22_train201.parquet:1.5:2.0:withinref,both" \
  --group "kadid:$D/ext_kadid.parquet:0.5:0.0" \
  --group "tid:$D/ext_tid.parquet:0.5:0.0" \
  --hidden 128 --n-hidden-layers 2 --epochs 200 --seed "$SEED" --val-policy mean \
  --mse-weight 0.6 --ranknet-weight 0.6 \
  --target-column human_score --max-features "$MAXF" \
  --out "$OUT"
