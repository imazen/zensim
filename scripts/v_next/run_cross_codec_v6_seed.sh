#!/usr/bin/env bash
# EXP-CROSS-CODEC-V6 trainer driver (2026-05-19).
#
# V5 (commit e0d869f) passes 5 of 6 Tuner gates spectacularly — only
# `median_range ≥ 50` fails (V5 outputs clustered in [37, 70]). Per
# V5 falsification doc V6 candidate #1: increase anchor pressure to
# force band targets to materialize as actual output values.
#
# Single architectural change from V5:
#   --anchor-loss-weight  →  ${ANCHOR_W}   (V5 used 0.05)
#   --anchor-step-p       →  ${ANCHOR_P}   (V5 used 0.15)
#
# Everything else identical to V5 (multi-band anchor parquet,
# cross-codec equiv pool, per-sample-α head, tanh scale 15.0,
# dyn-range floor, monotonicity reg).
#
# Args: <seed> <anchor_loss_weight> <anchor_step_p>
# Outputs: cc4v6_w<W>_p<P>_s<SEED>.bin + .log + .stdout in OUT_DIR
set -euo pipefail

SEED="${1:?usage: $0 <seed> <anchor_w> <anchor_p>}"
ANCHOR_W="${2:?usage: $0 <seed> <anchor_w> <anchor_p>}"
ANCHOR_P="${3:?usage: $0 <seed> <anchor_w> <anchor_p>}"

OUT_DIR="/mnt/v/zen/zensim-eval/exp_cross_codec_v6_2026-05-19"
TRAINER="/home/lilith/work/zen/zensim--cross-codec-metric/target/release/zensim_mlp_train"
PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-18/train"
ANCHOR="/mnt/v/zen/zensim-training/2026-05-19-multi-band-anchors/anchors_multi_band_372col.parquet"
EQUIV="/mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet"

# Encode anchor_w / anchor_p in the bake name: replace dots with 'p' for filesystem-friendliness
W_TAG=$(printf '%s' "${ANCHOR_W}" | tr '.' 'p')
P_TAG=$(printf '%s' "${ANCHOR_P}" | tr '.' 'p')
BAKE="${OUT_DIR}/cc4v6_w${W_TAG}_p${P_TAG}_s${SEED}.bin"
LOG="${OUT_DIR}/cc4v6_w${W_TAG}_p${P_TAG}_s${SEED}.log"
STDOUT="${OUT_DIR}/cc4v6_w${W_TAG}_p${P_TAG}_s${SEED}.stdout"

mkdir -p "${OUT_DIR}"

echo "EXP-CROSS-CODEC-V6: seed=${SEED} anchor_w=${ANCHOR_W} anchor_p=${ANCHOR_P}"
echo "  trainer:  ${TRAINER}"
echo "  out_bake: ${BAKE}"

"${TRAINER}" \
    --group "safesyn:${PARQ_DIR}/safesyn.parquet:1.0:0.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr 1e-3 --l2 1e-5 \
    --leaky-alpha 0.01 --val-policy min --early-stop-patience 0 \
    --max-features 372 --minibatch-size 1 \
    --target-column mix_cv40_iw60 --target-scale 1.0 --out-dtype f32 \
    --per-sample-alpha-head \
    --tanh-output-head-scale 15.0 \
    --ranknet-weight 0.0 \
    --mse-weight 1.0 \
    --monotonicity-reg 1.0 \
    --monotonicity-margin 0.0 \
    --anchor-parquet "${ANCHOR}" \
    --anchor-loss-weight "${ANCHOR_W}" \
    --anchor-target-score 63.0 \
    --anchor-step-p "${ANCHOR_P}" \
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

echo "DONE seed=${SEED} anchor_w=${ANCHOR_W} anchor_p=${ANCHOR_P} bake=${BAKE} log=${LOG}"
