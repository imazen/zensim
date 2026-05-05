#!/usr/bin/env bash
# V0_7 e1-fill subsampling ablation: train + eval 6 variants.
#
# Each variant trains an MLP on (218k base + N% of e1 fill) and evaluates
# on KADID/TID/CID22/KonJND human-MOS holdouts.

set -euo pipefail

ROOT=/home/lilith/work/zen/zensim--v07-e1-ablation
VALIDATE=$ROOT/target/release/zensim-validate
EVAL=$ROOT/target/release/examples/dataset_metric_baseline
TSV=/mnt/v/output/zensim/synthetic-v2/zenanalyze_union_v1.tsv
KADID=/mnt/v/dataset/kadid10k
TID=/mnt/v/dataset/tid2013
CID22=/mnt/v/dataset/cid22/CID22_validation_set
KONJND=/mnt/v/datasets/KonJND-1k/KonJND-1k
DCT_HF_FEATS=dct_compressibility_y,dct_compressibility_uv,high_freq_energy_ratio
RUNS=$ROOT/runs
EVAL_DIR=$ROOT/eval_out
mkdir -p "$RUNS" "$EVAL_DIR"

PCTS=("$@")
if [ ${#PCTS[@]} -eq 0 ]; then
  PCTS=(0 5 10 20 50 100)
fi

for pct in "${PCTS[@]}"; do
  VARIANT_CSV=/mnt/v/output/zensim/synthetic-v2/v07_e1_${pct}pct.csv
  BAKE=$RUNS/v07_e1_${pct}pct.bin
  TRAIN_LOG=$RUNS/v07_e1_${pct}pct.train.log
  EVAL_CSV=$EVAL_DIR/v07_e1_${pct}pct_perpair.csv
  EVAL_LOG=$EVAL_DIR/v07_e1_${pct}pct_eval.log

  if [ -f "$BAKE" ] && [ -f "$EVAL_CSV" ]; then
    echo "[$(date +%H:%M:%S)] skip ${pct}pct (bake + eval exist)"
    continue
  fi

  if [ ! -f "$BAKE" ]; then
    echo "[$(date +%H:%M:%S)] TRAIN ${pct}pct -> $BAKE"
    "$VALIDATE" \
      --dataset "$VARIANT_CSV" \
      --format synthetic \
      --target-metric gpu-ssim2 \
      --feature-tier peaks \
      --train --algorithm mlp --mlp-hidden 64 --mlp-epochs 200 \
      --mlp-magnitude-match-lambda 0.001 --mlp-magnitude-match-alpha 30.0 \
      --mlp-zenanalyze-tsv "$TSV" \
      --mlp-zenanalyze-features "$DCT_HF_FEATS" \
      --mlp-output "$BAKE" \
      --also "kadid10k:$KADID,tid2013:$TID,konjnd1k:$KONJND" \
      --mlp-validation-policy min \
      > "$TRAIN_LOG" 2>&1 || { echo "TRAIN FAILED ${pct}pct"; tail -30 "$TRAIN_LOG"; exit 2; }
    best=$(grep "best validation mean\|best_val" "$TRAIN_LOG" | tail -1 || echo "")
    echo "  ${pct}pct done: ${best}"
  fi

  if [ ! -f "$EVAL_CSV" ]; then
    echo "[$(date +%H:%M:%S)] EVAL ${pct}pct -> $EVAL_CSV"
    "$EVAL" \
      --kadid "$KADID" --tid "$TID" --cid22 "$CID22" --konjnd "$KONJND" \
      --v04-bake "$BAKE" \
      --zenanalyze-tsv "$TSV" --zenanalyze-features "$DCT_HF_FEATS" \
      --max-pairs 1500 --per-pair-output "$EVAL_CSV" \
      > "$EVAL_LOG" 2>&1 || { echo "EVAL FAILED ${pct}pct"; tail -30 "$EVAL_LOG"; exit 3; }
    echo "  ${pct}pct eval done"
  fi
done

echo "[$(date +%H:%M:%S)] ALL VARIANTS DONE"
