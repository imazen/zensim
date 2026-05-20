#!/usr/bin/env bash
# EXP-CROSS-CODEC-V8 trainer driver (2026-05-19).
#
# V6 recipe with V8 anchor parquet (4 bands, ssim2=63 PJND anchored
# at butter=2.5 instead of V6/V7's butter=1.5). Heavy-distortion bands
# (4.0+ in V6, 6.0 in V7) dropped — only butter ∈ {0.5, 1.0, 2.5, 4.0}.
#
# All other hyperparameters identical to V6 ship recipe.
#
# Args: <seed>
# Outputs: cc4v8_s<SEED>.bin + .log + .stdout in OUT_DIR
set -euo pipefail

SEED="${1:?usage: $0 <seed>}"
KBATCH="${KBATCH:-1}"

OUT_DIR="/mnt/v/zen/zensim-eval/exp_cross_codec_v8_2026-05-19"
TRAINER="/home/lilith/work/zen/zensim--cross-codec-metric/target/release/zensim_mlp_train"
PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-18/train"
ANCHOR="/mnt/v/zen/zensim-training/2026-05-19-v8-anchors/anchors_v8_372col.parquet"
EQUIV="/mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet"

BAKE="${OUT_DIR}/cc4v8_s${SEED}.bin"
LOG="${OUT_DIR}/cc4v8_s${SEED}.log"
STDOUT="${OUT_DIR}/cc4v8_s${SEED}.stdout"

mkdir -p "${OUT_DIR}"

echo "EXP-CROSS-CODEC-V8: seed=${SEED}"
echo "  trainer:  ${TRAINER}"
echo "  anchor:   ${ANCHOR}"
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
