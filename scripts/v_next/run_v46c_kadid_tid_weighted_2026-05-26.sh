#!/usr/bin/env bash
# V46c — V32-faithful 2-layer + --monotone-cbc + KADID/TID promoted to
# DOMINANT training signal for non-compression-distortion generalization.
#
# Hypothesis: V46b's KADID −0.120 and TID −0.139 SROCC drops vs V39
# come from under-weighting (train_w=0.5 each) the human-MOS non-
# compression corpora during monotone-cbc training. With the
# constraint forcing every feature ≥0 in distortion, the trainer
# needs explicit non-compression-distortion signal to learn the
# right feature semantics for those distortion types.
#
# Recipe vs V46b:
#   safesyn      1.0 → 1.0   (unchanged — compression-synthetic baseline)
#   cid22_train  1.5 → 1.0   (lowered — let KADID/TID dominate)
#   kadid        0.5 → 2.0   (key non-compression human MOS)
#   tid          0.5 → 2.0   (key non-compression human MOS)
#   konjnd_dense 1.2 → 1.2   (unchanged — HF anchor)
#
# CID22 stays validation-only (its train-fold is just for trainer
# anchoring; the 49-ref held-out validation is sacred per CLAUDE.md).
set -euo pipefail
TRAIN="/home/lilith/work/zen/zensim/target/release/zensim_mlp_train"
OUT="/mnt/v/output/zensim/bakes/v46c_kadid_tid_weighted_2026-05-26.bin"
LOG="/home/lilith/work/zen/zensim/benchmarks/v46c_kadid_tid_weighted_2026-05-26.log"
CANON="/mnt/v/zen/zensim-training/canonical-2026-05-21/train"
TXFORM="/home/lilith/work/zen/zensim/benchmarks/yeo_johnson_screen_widest_2026-05-25/screen_results_cross_corpus_safe.tsv"
mkdir -p "$(dirname "$OUT")"
"$TRAIN" \
  --group safesyn:$CANON/safesyn.parquet:1.0:0.5 \
  --group cid22_train:$CANON/cid22_train_norm.parquet:1.0:2.0 \
  --group kadid:$CANON/kadid.parquet:2.0:1.0 \
  --group tid:$CANON/tid.parquet:2.0:1.0 \
  --group konjnd_dense:$CANON/konjnd-dense-norm.parquet:1.2:1.5 \
  --hidden 128 --n-hidden-layers 2 --per-sample-alpha-head --epochs 200 \
  --lr 0.001 --l2 0.0001 --seed 17 --target-column human_score --max-features 372 \
  --auto-transforms "$TXFORM" \
  --val-aggregate geomean3 --out-dtype f32 \
  --mse-weight 0.6 --ranknet-weight 0.6 --monotonicity-reg 1.0 \
  --tanh-output-head-scale 30.0 \
  --anchor-parquet $CANON/multiband_anchor_dial100.parquet \
  --anchor-loss-weight 0.01 --anchor-step-p 0.05 \
  --minibatch-size 32 --log-every 25 \
  --monotone-cbc \
  --out "$OUT" --log-path "$LOG" 2>&1 | tee "${LOG}.stdout"
echo "DONE: $OUT"
