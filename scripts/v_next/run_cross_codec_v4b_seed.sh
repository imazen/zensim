#!/usr/bin/env bash
# EXP-CROSS-CODEC-V4b trainer driver (2026-05-19).
#
# V4 finding (n=6 bakes): multi-codec anchor at weight=1.0 + step_p=0.15
# pulls so hard toward score=63 that the q-sweep dynamic range collapses
# to 8-16 score units (gate ≥ 50). The cross-codec calibration is
# excellent (cc_std median 0.10-0.14) but mono/range fail.
#
# V4b: drop anchor_loss_weight and anchor_step_p to give the per-pair
# MSE objective room to spread the output range:
#   --anchor-loss-weight {0.05, 0.10}     (V4 used 1.0)
#   --anchor-step-p 0.05                  (V4 used 0.15)
#
# Hypothesis: the tanh pin + σ-floor structurally allow [0, 100] range
# expression; only the over-weighted anchor was compressing it. With
# a 10-20x weaker anchor, per-pair MSE drives the range to native
# [5, 95] while anchor still provides cross-codec calibration signal.
#
# Args: <seed> <anchor_w>
# Outputs: bake + log in OUT_DIR with name cc4v4b_s<SEED>_a<W>.bin
set -euo pipefail

SEED="${1:?usage: $0 <seed> <anchor_w>}"
ANCHOR_W="${2:?usage: $0 <seed> <anchor_w>}"

OUT_DIR="/mnt/v/zen/zensim-eval/exp_cross_codec_v4b_2026-05-19"
TRAINER="/home/lilith/work/zen/zensim--cross-codec-metric/target/release/zensim_mlp_train"
PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-18/train"
ANCHOR="/mnt/v/zen/zensim-training/2026-05-19-multi-codec-jnd-anchors/anchors_multi_codec_372col.parquet"
EQUIV="/mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet"

A_TAG="${ANCHOR_W/./_}"
BAKE="${OUT_DIR}/cc4v4b_s${SEED}_a${A_TAG}.bin"
LOG="${OUT_DIR}/cc4v4b_s${SEED}_a${A_TAG}.log"
STDOUT="${OUT_DIR}/cc4v4b_s${SEED}_a${A_TAG}.stdout"

mkdir -p "${OUT_DIR}"

echo "EXP-CROSS-CODEC-V4b: seed=${SEED} anchor_w=${ANCHOR_W}"
echo "  trainer:  ${TRAINER}"
echo "  out_bake: ${BAKE}"

"${TRAINER}" \
    --group "safesyn:${PARQ_DIR}/safesyn.parquet:1.0:0.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr 1e-3 --l2 1e-5 \
    --leaky-alpha 0.01 --val-policy min --early-stop-patience 0 \
    --max-features 372 --minibatch-size 1 \
    --target-column mix_cv40_iw60 --target-scale 1.0 --out-dtype f32 \
    --per-sample-alpha-head \
    --tanh-output-head-scale 10.0 \
    --ranknet-weight 0.0 \
    --mse-weight 1.0 \
    --monotonicity-reg 1.0 \
    --monotonicity-margin 0.0 \
    --anchor-parquet "${ANCHOR}" \
    --anchor-loss-weight "${ANCHOR_W}" \
    --anchor-target-score 63.0 \
    --anchor-step-p 0.05 \
    --cross-codec-eq-parquet "${EQUIV}" \
    --cross-codec-eq-weight 1.0 \
    --cross-codec-eq-step-p 0.10 \
    --cross-codec-rank-preserve-weight 0.2 \
    --dynamic-range-floor-weight 0.5 \
    --dynamic-range-sigma-threshold 20.0 \
    --dynamic-range-step-p 0.05 \
    --dynamic-range-probe-n 40 \
    --seed "${SEED}" --out "${BAKE}" --log-path "${LOG}" \
    2>&1 | tee "${STDOUT}"

echo "DONE seed=${SEED} anchor_w=${ANCHOR_W} bake=${BAKE} log=${LOG}"
