#!/usr/bin/env bash
# Train V0_6 + FiLM (Feature-wise Linear Modulation) on content class.
#
# Architecture: 228 zensim feats + 3 dct_hf feats → 64 hidden (FiLM-gated
# by 5-class one-hot) → 1. The cclass tail of the appended features
# only drives gamma/beta; the content path doesn't see it.
#
# Bake output: 5 ZNPR v3 models (one per class) with FiLM γ/β folded
# into the first layer. Eval-time dispatcher (dataset_metric_baseline
# with --film-manifest) routes each pair through the correct per-class
# bake based on its reference image's cclass.
#
# Usage: bash train_v06_film.sh

set -euo pipefail

ROOT=${ROOT:-/home/lilith/work/zen/zensim--v06-film}
EXTENDED_CSV=${EXTENDED_CSV:-/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic_extended.csv}
TS=$(date -u +%Y%m%dT%H%M%S)

VALIDATE=$ROOT/target/release/zensim-validate
TSV=/mnt/v/output/zensim/synthetic-v2/zenanalyze_union_v1_cclass.tsv
RUNS=/mnt/v/output/zensim/synthetic-v2/runs
KADID=/mnt/v/dataset/kadid10k
TID=/mnt/v/dataset/tid2013
KONJND=/mnt/v/datasets/KonJND-1k/KonJND-1k

# Same 3 dct_hf features as V0_6 baseline; cclass tail at the END.
DCT_HF_CCLASS_FEATS=dct_compressibility_y,dct_compressibility_uv,high_freq_energy_ratio,cclass_photo,cclass_screen,cclass_lineart,cclass_synthetic,cclass_document

[ -f "$EXTENDED_CSV" ] || { echo "missing $EXTENDED_CSV"; exit 1; }
[ -f "$TSV" ] || { echo "missing $TSV"; exit 1; }
[ -x "$VALIDATE" ] || { cd "$ROOT" && cargo build --release -p zensim-validate; }

n_pairs=$(($(wc -l < "$EXTENDED_CSV") - 1))
echo "[$(date +%H:%M:%S)] training V0_6+FiLM on $EXTENDED_CSV ($n_pairs pairs)"

NAME=v06_film_${TS}
BAKE=$RUNS/${NAME}.bin
LOG=/tmp/${NAME}.log

"$VALIDATE" \
  --dataset "$EXTENDED_CSV" \
  --format synthetic \
  --target-metric gpu-ssim2 \
  --feature-tier peaks \
  --train --algorithm mlp --mlp-hidden 64 --mlp-epochs 200 \
  --mlp-magnitude-match-lambda 0.001 --mlp-magnitude-match-alpha 30.0 \
  --mlp-zenanalyze-tsv "$TSV" \
  --mlp-zenanalyze-features "$DCT_HF_CCLASS_FEATS" \
  --mlp-film-onehot-content-class \
  --mlp-output "$BAKE" \
  --also kadid10k:$KADID,tid2013:$TID,konjnd1k:$KONJND \
  --mlp-validation-policy min \
  > "$LOG" 2>&1

echo "[$(date +%H:%M:%S)] done"
echo "  primary bake: $BAKE"
echo "  manifest:     ${BAKE%.bin}.film_manifest.tsv"
echo "  per-class bakes: ${BAKE%.bin}.c{0..4}_*.bin"
echo "  log:          $LOG"
v_best=$(grep -i "best validation" "$LOG" | tail -1)
echo "  $v_best"
