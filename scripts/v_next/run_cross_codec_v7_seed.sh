#!/usr/bin/env bash
# EXP-CROSS-CODEC-V7 trainer driver (2026-05-19).
#
# Replaces V6's rule-of-thumb anchor target_score column with the V7
# empirically-derived per-(codec, band) median ssim2 from the canonical
# ssim2 + cvvdp score parquets. Empirical medians live HIGHER than V6
# rules of thumb at every band except 0.3 — V6 was biasing the bake
# to predict scores BELOW where human-validated metrics land at the
# same butter level.
#
# Single change from V6: --anchor-parquet now points at
#   /mnt/v/zen/zensim-training/2026-05-19-empirical-band-anchors/anchors_empirical_372col.parquet
# Every other hyperparameter identical to V6 ship (cc4v6_w1p0_p0p30_s1)
# with anchor_w=1.0 step_p=0.30.
#
# Args: <seed>
# Outputs: cc4v7_s<SEED>.bin + .log + .stdout in OUT_DIR
# SPEED-B: KBATCH=1 default per the same constraint V6 ship has.
set -euo pipefail

SEED="${1:?usage: $0 <seed>}"
KBATCH="${KBATCH:-1}"

OUT_DIR="/mnt/v/zen/zensim-eval/exp_cross_codec_v7_2026-05-19"
TRAINER="/home/lilith/work/zen/zensim/target/release/zensim_mlp_train"
PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-18/train"
ANCHOR="/mnt/v/zen/zensim-training/2026-05-19-empirical-band-anchors/anchors_empirical_372col.parquet"
EQUIV="/mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet"

BAKE="${OUT_DIR}/cc4v7_s${SEED}.bin"
LOG="${OUT_DIR}/cc4v7_s${SEED}.log"
STDOUT="${OUT_DIR}/cc4v7_s${SEED}.stdout"

mkdir -p "${OUT_DIR}"

echo "EXP-CROSS-CODEC-V7: seed=${SEED} empirical-anchor=${ANCHOR}"
echo "  trainer:  ${TRAINER}"
echo "  out_bake: ${BAKE}"

"${TRAINER}" \
    --group "safesyn:${PARQ_DIR}/safesyn.parquet:1.0:0.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr 1e-3 --l2 1e-5 \
    --leaky-alpha 0.01 --val-policy min --early-stop-patience 0 \
    --max-features 372 --minibatch-size "${KBATCH}" \
    --target-column mix_cv40_iw60 --target-scale 1.0 --out-dtype f32 \
    --per-sample-alpha-head \
    --tanh-output-head-scale 15.0 \
    --ranknet-weight 0.0 \
    --mse-weight 1.0 \
    --monotonicity-reg 1.0 \
    --monotonicity-margin 0.0 \
    --anchor-parquet "${ANCHOR}" \
    --anchor-loss-weight 1.0 \
    --anchor-target-score 63.0 \
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

echo "DONE seed=${SEED} bake=${BAKE} log=${LOG}"
