#!/usr/bin/env bash
# EXP-CROSS-CODEC-V9 conservative-recipe variant (2026-05-20).
#
# Reverts the V6→V9 hyperparameter widening: tanh-scale 20→15,
# σ-floor 25→15. The spline post-network can still extend range
# because it sees the network's actual distribution, not the
# theoretical tanh range. This tests whether mono is recovered
# at the original V6 hyperparameters with just the V9 anchors.
#
# Args: <seed>

set -euo pipefail

SEED="${1:?usage: $0 <seed>}"

OUT_DIR="/mnt/v/zen/zensim-eval/exp_cross_codec_v9_2026-05-20"
TRAINER="/home/lilith/work/zen/zensim/target/release/zensim_mlp_train"
PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-18/train"
ANCHOR="/mnt/v/zen/zensim-training/2026-05-20-v9-anchors/anchors_v9_372col.parquet"
EQUIV="/mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet"

BAKE="${OUT_DIR}/cc4v9_cons_s${SEED}.bin"
LOG="${OUT_DIR}/cc4v9_cons_s${SEED}.log"
STDOUT="${OUT_DIR}/cc4v9_cons_s${SEED}.stdout"

mkdir -p "${OUT_DIR}"

echo "EXP-CROSS-CODEC-V9 conservative-recipe: seed=${SEED}"

"${TRAINER}" \
    --group "safesyn:${PARQ_DIR}/safesyn.parquet:1.0:0.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr 5.66e-3 --l2 1e-5 \
    --leaky-alpha 0.01 --val-policy min --early-stop-patience 0 \
    --max-features 372 --minibatch-size 32 \
    --target-column mix_cv40_iw60 --target-scale 1.0 --out-dtype f32 \
    --per-sample-alpha-head \
    --tanh-output-head-scale 15.0 \
    --ranknet-weight 0.0 \
    --mse-weight 1.0 \
    --monotonicity-reg 1.0 \
    --monotonicity-margin 0.0 \
    --anchor-parquet "${ANCHOR}" \
    --anchor-loss-weight 1.0 \
    --anchor-target-score 60.0 \
    --anchor-step-p 0.30 \
    --cross-codec-eq-parquet "${EQUIV}" \
    --cross-codec-eq-weight 1.0 \
    --cross-codec-eq-step-p 0.10 \
    --cross-codec-rank-preserve-weight 0.2 \
    --dynamic-range-floor-weight 0.2 \
    --dynamic-range-sigma-threshold 15.0 \
    --dynamic-range-step-p 0.05 \
    --dynamic-range-probe-n 40 \
    --seed "${SEED}" --out "${BAKE}" --log-path "${LOG}" \
    2>&1 | tee "${STDOUT}"
