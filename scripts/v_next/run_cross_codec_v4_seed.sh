#!/usr/bin/env bash
# EXP-CROSS-CODEC-V4 trainer driver (2026-05-19).
#
# Builds on V3 with TWO architectural counterweights to close the V3
# mono-violation gap (best V3 strict mono 0.9100 vs gate 0.9378):
#
#   1. --tanh-output-head-scale 10.0 (V4-C from V3 falsification doc):
#      wraps the per-sample-α head output in `y_score = 100·σ(y_pre/scale)`,
#      eliminating the post-affine β-amplification path that V3 identified
#      as the dominant mono-violation cause.
#
#   2. --anchor-parquet anchors_multi_codec_372col.parquet (V4 user
#      directive): replaces V3's single-codec (zenjpeg) PJND anchor with a
#      multi-codec anchor — 1000 sources × 4 codecs at each codec's PJND-q,
#      all targeting score=63. Binds score=63 ↔ PJND across codecs.
#
# Plus V3 falsification finding: --monotonicity-reg 1.0 (NOT 5.0). The
# σ-floor already prevents collapse structurally, so strong mono-reg is
# counterproductive (V4-B from falsification doc).
#
# Args: <seed> <W>  (W = cross_codec_eq_weight)
# Outputs: bake + log in OUT_DIR with name cc4v4_s<SEED>_w<W>.bin
set -euo pipefail

SEED="${1:?usage: $0 <seed> <W>}"
WEIGHT="${2:?usage: $0 <seed> <W>}"

OUT_DIR="/mnt/v/zen/zensim-eval/exp_cross_codec_v4_2026-05-19"
TRAINER="/home/lilith/work/zen/zensim--cross-codec-metric/target/release/zensim_mlp_train"
PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-18/train"
# V4: multi-codec PJND anchor (4000 rows = 1000 sources × 4 codecs).
ANCHOR="/mnt/v/zen/zensim-training/2026-05-19-multi-codec-jnd-anchors/anchors_multi_codec_372col.parquet"
EQUIV="/mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet"

W_TAG="${WEIGHT/./_}"
BAKE="${OUT_DIR}/cc4v4_s${SEED}_w${W_TAG}.bin"
LOG="${OUT_DIR}/cc4v4_s${SEED}_w${W_TAG}.log"
STDOUT="${OUT_DIR}/cc4v4_s${SEED}_w${W_TAG}.stdout"

mkdir -p "${OUT_DIR}"

echo "EXP-CROSS-CODEC-V4: seed=${SEED} W=${WEIGHT}"
echo "  trainer:  ${TRAINER}"
echo "  out_bake: ${BAKE}"
echo "  anchor:   ${ANCHOR}"

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
    --anchor-loss-weight 1.0 \
    --anchor-target-score 63.0 \
    --anchor-step-p 0.15 \
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
