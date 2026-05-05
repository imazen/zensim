#!/usr/bin/env bash
# Train V0_6+content_class — same architecture as V0_6 dct_hf but with
# 5 additional content_class one-hot features per reference image.
#
# Usage: bash train_v06_cclass.sh

set -euo pipefail

ROOT=${ROOT:-/home/lilith/work/zen/zensim--v06-rebalance}
EXTENDED_CSV=${EXTENDED_CSV:-/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic_extended.csv}
TS=$(date -u +%Y%m%dT%H%M%S)

VALIDATE=$ROOT/target/release/zensim-validate
TSV=/mnt/v/output/zensim/synthetic-v2/zenanalyze_union_v1_cclass.tsv
RUNS=/mnt/v/output/zensim/synthetic-v2/runs
KADID=/mnt/v/dataset/kadid10k
TID=/mnt/v/dataset/tid2013
KONJND=/mnt/v/datasets/KonJND-1k/KonJND-1k

# V0_6 used 3 dct_hf features. V0_6+cclass adds 5 one-hot class columns.
DCT_HF_CCLASS_FEATS=dct_compressibility_y,dct_compressibility_uv,high_freq_energy_ratio,cclass_photo,cclass_screen,cclass_lineart,cclass_synthetic,cclass_document

[ -f "$EXTENDED_CSV" ] || { echo "missing $EXTENDED_CSV"; exit 1; }
[ -f "$TSV" ] || { echo "missing $TSV"; exit 1; }
[ -x "$VALIDATE" ] || { cd "$ROOT" && cargo build --release -p zensim-validate; }

n_pairs=$(($(wc -l < "$EXTENDED_CSV") - 1))
echo "[$(date +%H:%M:%S)] training V0_6+cclass on $EXTENDED_CSV ($n_pairs pairs)"

NAME=v06_cclass_${TS}
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
  --mlp-output "$BAKE" \
  --also kadid10k:$KADID,tid2013:$TID,konjnd1k:$KONJND \
  --mlp-validation-policy min \
  > "$LOG" 2>&1

echo "[$(date +%H:%M:%S)] done"
echo "  bake: $BAKE"
echo "  log:  $LOG"
v_best=$(grep "best validation mean" "$LOG" | awk '{print $NF}')
echo "  best validation min SROCC: $v_best"
