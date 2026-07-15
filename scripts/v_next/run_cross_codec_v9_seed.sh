#!/usr/bin/env bash
# EXP-CROSS-CODEC-V9 trainer driver (2026-05-20).
#
# V6/V8 recipe with V9 anchor parquet (8 bands, [0, 100] range, JND
# at 60, JOD at 30). The per-row target_score column in the V9 anchor
# parquet supplies the band-specific score targets; --anchor-target-score
# is the fallback for non-targeted rows.
#
# Hyperparameter deltas from V6 ship recipe:
#   --anchor-loss-weight 0.5 (down from 1.0): 8 bands vs 6 means more
#     anchor rows; reduce per-row pressure proportionally.
#   --tanh-output-head-scale 20.0 (up from 15.0): widen active linear
#     region to cover the full [0, 100] span without saturation.
#   --dynamic-range-sigma-threshold 25.0 (up from 15.0): encourage
#     wider score spread.
#
# Uses SPEED-B K=32 lr=5.66e-3 (verified clean by V6-RESHIP).
#
# Args: <seed>
# Env: KBATCH (default 32), LR_OVERRIDE (default 5.66e-3)

set -euo pipefail

SEED="${1:?usage: $0 <seed>}"
KBATCH="${KBATCH:-32}"
LR="${LR_OVERRIDE:-5.66e-3}"

OUT_DIR="/mnt/v/zen/zensim-eval/exp_cross_codec_v9_2026-05-20"
TRAINER="/home/lilith/work/zen/zensim/target/release/zensim_mlp_train"
PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-18/train"
ANCHOR="/mnt/v/zen/zensim-training/2026-05-20-v9-anchors/anchors_v9_372col.parquet"
EQUIV="/mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet"

BAKE="${OUT_DIR}/cc4v9_s${SEED}.bin"
LOG="${OUT_DIR}/cc4v9_s${SEED}.log"
STDOUT="${OUT_DIR}/cc4v9_s${SEED}.stdout"

mkdir -p "${OUT_DIR}"

echo "EXP-CROSS-CODEC-V9: seed=${SEED} KBATCH=${KBATCH} LR=${LR}"
echo "  trainer:  ${TRAINER}"
echo "  anchor:   ${ANCHOR}"
echo "  out_bake: ${BAKE}"

"${TRAINER}" \
    --group "safesyn:${PARQ_DIR}/safesyn.parquet:1.0:0.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr "${LR}" --l2 1e-5 \
    --leaky-alpha 0.01 --val-policy min --early-stop-patience 0 \
    --max-features 372 --minibatch-size "${KBATCH}" \
    --target-column mix_cv40_iw60 --target-scale 1.0 --out-dtype f32 \
    --per-sample-alpha-head \
    --tanh-output-head-scale 20.0 \
    --ranknet-weight 0.0 \
    --mse-weight 1.0 \
    --monotonicity-reg 1.0 \
    --monotonicity-margin 0.0 \
    --anchor-parquet "${ANCHOR}" \
    --anchor-loss-weight 0.5 \
    --anchor-target-score 60.0 \
    --anchor-step-p 0.30 \
    --cross-codec-eq-parquet "${EQUIV}" \
    --cross-codec-eq-weight 1.0 \
    --cross-codec-eq-step-p 0.10 \
    --cross-codec-rank-preserve-weight 0.2 \
    --dynamic-range-floor-weight 0.3 \
    --dynamic-range-sigma-threshold 25.0 \
    --dynamic-range-step-p 0.05 \
    --dynamic-range-probe-n 40 \
    --seed "${SEED}" --out "${BAKE}" --log-path "${LOG}" \
    2>&1 | tee "${STDOUT}"
