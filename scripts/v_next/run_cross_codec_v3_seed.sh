#!/usr/bin/env bash
# EXP-CROSS-CODEC-V3 trainer driver (2026-05-19).
#
# Builds on V2 with three architectural additions to fix the
# "collapse to constant ~63" failure mode (V2 falsified for Tuner trail
# on dynamic range; see
# benchmarks/v_tuner_v2_cross_codec_2026-05-19_falsification.md):
#
#   1. --cross-codec-rank-preserve-weight 0.2: RankNet on equiv pairs
#      that have non-zero butter_diff, weighted by |butter_diff|.
#      Pushes back against (y_a − y_b)^2 collapse.
#
#   2. --dynamic-range-floor-weight 0.2: σ-floor probe (40 random
#      A-side equiv rows forwarded, penalty when σ < 15 score units).
#      Structurally requires output spread.
#
#   3. --monotonicity-reg 5.0: 10× stronger mono push (V2 used 0.5,
#      Tuner-v2 used 1.0 → range 89.68). Higher reg fights collapse.
#
# Args: <seed> <W>  (W = cross_codec_eq_weight)
# Outputs: bake + log in OUT_DIR with name cc4v3_s<SEED>_w<W>.bin
set -euo pipefail

SEED="${1:?usage: $0 <seed> <W>}"
WEIGHT="${2:?usage: $0 <seed> <W>}"

OUT_DIR="/mnt/v/zen/zensim-eval/exp_cross_codec_v3_2026-05-19"
TRAINER="/home/lilith/work/zen/zensim--cross-codec-metric/target/release/zensim_mlp_train"
PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-18/train"
ANCHOR="/mnt/v/zen/zensim-training/2026-05-19-jnd-anchors/anchors_372col.parquet"
EQUIV="/mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet"

W_TAG="${WEIGHT/./_}"
BAKE="${OUT_DIR}/cc4v3_s${SEED}_w${W_TAG}.bin"
LOG="${OUT_DIR}/cc4v3_s${SEED}_w${W_TAG}.log"
STDOUT="${OUT_DIR}/cc4v3_s${SEED}_w${W_TAG}.stdout"

mkdir -p "${OUT_DIR}"

echo "EXP-CROSS-CODEC-V3: seed=${SEED} W=${WEIGHT}"
echo "  trainer:  ${TRAINER}"
echo "  out_bake: ${BAKE}"

"${TRAINER}" \
    --group "safesyn:${PARQ_DIR}/safesyn.parquet:1.0:0.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr 1e-3 --l2 1e-5 \
    --leaky-alpha 0.01 --val-policy min --early-stop-patience 0 \
    --max-features 372 --minibatch-size 1 \
    --target-column mix_cv40_iw60 --target-scale 1.0 --out-dtype f32 \
    --per-sample-alpha-head \
    --ranknet-weight 0.0 \
    --mse-weight 1.0 \
    --monotonicity-reg 5.0 \
    --monotonicity-margin 0.0 \
    --anchor-parquet "${ANCHOR}" \
    --anchor-loss-weight 1.0 \
    --anchor-target-score 63.0 \
    --anchor-step-p 0.10 \
    --cross-codec-eq-parquet "${EQUIV}" \
    --cross-codec-eq-weight "${WEIGHT}" \
    --cross-codec-eq-step-p 0.10 \
    --cross-codec-rank-preserve-weight 0.2 \
    --dynamic-range-floor-weight 0.2 \
    --dynamic-range-sigma-threshold 15.0 \
    --dynamic-range-step-p 0.05 \
    --dynamic-range-probe-n 40 \
    --seed "${SEED}" --out "${BAKE}" --log-path "${LOG}" \
    2>&1 | tee "${STDOUT}"

echo "DONE seed=${SEED} W=${WEIGHT} bake=${BAKE} log=${LOG}"
