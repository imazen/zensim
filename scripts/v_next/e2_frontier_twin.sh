#!/bin/bash
# E2 FRONTIER twin — 720 vs 372 with the best-MLP technique stack applied, so the
# v2 features are measured SHAPED (not raw). The floor twin (e2_train_720_twin.sh)
# fed raw features; the best models shape via Yeo-Johnson auto-transforms, and the
# v2 block is heavily shaping-dependent (338/348 features get a non-identity
# transform, lifting per-feature correlation up to +0.38). Without shaping,
# "frontier" = "floor" for v2.
#
# Technique stack (both arms, from the v47/Profile-B recipe): auto-transforms +
# per-sample-alpha head + 2-layer 128. The ONLY between-arm difference is
# --max-features {720|372} + which transform screen:
#   720 arm: merged screen (shipped v1 f0..f371 ++ new v2 f372..f719)
#   372 arm: shipped v1 screen (f0..f371) — IDENTICAL v1 shaping to the 720 arm
# so the 720 arm's edge = (sees v2) + (shapes v2), the honest "720 at the frontier".
#
# Monotone masking is DEFERRED (a dial property, E10 — needs a v2 sign mask); this
# twin isolates the feature-VALUE (rank) question at the shaping frontier.
#
# Usage: ARM={ext720|v1372} SEED=1 bash scripts/v_next/e2_frontier_twin.sh
set -u
ARM="${ARM:-ext720}"
SEED="${SEED:-1}"
ZM_BIN="${ZM_BIN:-target/release/zensim_mlp_train}"
D=/mnt/v/zen/zensim-training/ext720-canonical-2026-07-22
TBIG=/mnt/v/zen/tbig-720-2026-07-22/tbig_720_TRAIN_NN.parquet
SCREEN_720=benchmarks/v2_transform_screen_2026-07-23/screen_720_merged.tsv
SCREEN_372=benchmarks/yeo_johnson_screen_widest_2026-05-25/screen_results_cross_corpus_safe.tsv
OUTDIR=/mnt/v/output/zensim/bakes
mkdir -p "$OUTDIR"

case "$ARM" in
  ext720) MAXF=720; SCREEN="$SCREEN_720" ;;
  v1372)  MAXF=372; SCREEN="$SCREEN_372" ;;
  *) echo "ARM must be ext720|v1372"; exit 2 ;;
esac
OUT="$OUTDIR/e2f_${ARM}_s${SEED}_2026-07-23.bin"

exec "$ZM_BIN" \
  --group "tbig:$TBIG:1.0:0.3:withinref,both" \
  --group "safesyn:$D/ext_safesyn_full.parquet:1.0:0.5:withinref,both" \
  --group "cid201:$D/ext_cid22_train201.parquet:1.5:2.0:withinref,both" \
  --group "kadid:$D/ext_kadid.parquet:0.5:0.0" \
  --group "tid:$D/ext_tid.parquet:0.5:0.0" \
  --hidden 128 --n-hidden-layers 2 --epochs 200 --seed "$SEED" --val-policy mean \
  --mse-weight 0.6 --ranknet-weight 0.6 \
  --per-sample-alpha-head \
  --auto-transforms "$SCREEN" \
  --target-column human_score --max-features "$MAXF" \
  --out "$OUT"
