#!/usr/bin/env bash
# Retrain V0_6 baseline (dct_hf, no content_class) with the current
# zenpredict ZNPR v3 format so the eval can compare apples-to-apples
# against V0_6+cclass.

set -euo pipefail

ROOT=${ROOT:-/home/lilith/work/zen/zensim--v06-rebalance}
EXTENDED_CSV=${EXTENDED_CSV:-/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic_extended.csv}
TS=$(date -u +%Y%m%dT%H%M%S)

VALIDATE=$ROOT/target/release/zensim-validate
TSV=/mnt/v/output/zensim/synthetic-v2/zenanalyze_union_v1.tsv
RUNS=/mnt/v/output/zensim/synthetic-v2/runs
KADID=/mnt/v/dataset/kadid10k
TID=/mnt/v/dataset/tid2013
KONJND=/mnt/v/datasets/KonJND-1k/KonJND-1k

DCT_HF_FEATS=dct_compressibility_y,dct_compressibility_uv,high_freq_energy_ratio

NAME=v06_baseline_rebake_${TS}
BAKE=$RUNS/${NAME}.bin
LOG=/tmp/${NAME}.log

echo "[$(date +%H:%M:%S)] training V0_6 baseline (dct_hf, no cclass) on extended CSV"
"$VALIDATE" \
  --dataset "$EXTENDED_CSV" \
  --format synthetic \
  --target-metric gpu-ssim2 \
  --feature-tier peaks \
  --train --algorithm mlp --mlp-hidden 64 --mlp-epochs 200 \
  --mlp-magnitude-match-lambda 0.001 --mlp-magnitude-match-alpha 30.0 \
  --mlp-zenanalyze-tsv "$TSV" \
  --mlp-zenanalyze-features "$DCT_HF_FEATS" \
  --mlp-output "$BAKE" \
  --also kadid10k:$KADID,tid2013:$TID,konjnd1k:$KONJND \
  --mlp-validation-policy min \
  > "$LOG" 2>&1

echo "[$(date +%H:%M:%S)] done"
echo "  bake: $BAKE"
echo "  log:  $LOG"
v_best=$(grep "best validation mean" "$LOG" | awk '{print $NF}')
echo "  best validation min SROCC: $v_best"
