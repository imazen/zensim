#!/usr/bin/env bash
# EXP-CROSS-CODEC-V2 trainer driver (2026-05-19).
#
# Builds on EXP-CROSS-CODEC-METRIC (the W=1.0 seed=1 ship that landed at
# T=63 cross-codec butter 4.82/5.52 — close but not strict < 2.5). V2
# tightens the equivalence pool (max butter gap 0.3 vs 0.5), rebalances
# the avif↔X pool with a 2× row-weight oversample, and extends to 30
# butter levels giving a 68,788-pair pool (vs original 57,972).
#
# Args: <seed> <W>
# Outputs: bake + log in OUT_DIR with name cc4v2_s<SEED>_w<W>.bin
set -euo pipefail

SEED="${1:?usage: $0 <seed> <W>}"
WEIGHT="${2:?usage: $0 <seed> <W>}"

OUT_DIR="/mnt/v/zen/zensim-eval/exp_cross_codec_v2_2026-05-19"
TRAINER="/home/lilith/work/zen/zensim/target/release/zensim_mlp_train"
PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-18/train"
ANCHOR="/mnt/v/zen/zensim-training/2026-05-19-jnd-anchors/anchors_372col.parquet"
EQUIV="/mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet"

# Sanitize W for filename (replace . with _)
W_TAG="${WEIGHT/./_}"
BAKE="${OUT_DIR}/cc4v2_s${SEED}_w${W_TAG}.bin"
LOG="${OUT_DIR}/cc4v2_s${SEED}_w${W_TAG}.log"
STDOUT="${OUT_DIR}/cc4v2_s${SEED}_w${W_TAG}.stdout"

mkdir -p "${OUT_DIR}"

echo "EXP-CROSS-CODEC-V2: seed=${SEED} W=${WEIGHT}"
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
    --monotonicity-reg 0.5 \
    --monotonicity-margin 0.0 \
    --anchor-parquet "${ANCHOR}" \
    --anchor-loss-weight 1.0 \
    --anchor-target-score 63.0 \
    --anchor-step-p 0.10 \
    --cross-codec-eq-parquet "${EQUIV}" \
    --cross-codec-eq-weight "${WEIGHT}" \
    --cross-codec-eq-step-p 0.10 \
    --seed "${SEED}" --out "${BAKE}" --log-path "${LOG}" \
    2>&1 | tee "${STDOUT}"

echo "DONE seed=${SEED} W=${WEIGHT} bake=${BAKE} log=${LOG}"
