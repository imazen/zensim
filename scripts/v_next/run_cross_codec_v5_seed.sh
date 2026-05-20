#!/usr/bin/env bash
# EXP-CROSS-CODEC-V5 trainer driver (2026-05-19).
#
# V4b found: 4 of 5 Tuner gates pass with anchor_w=0.05 + step_p=0.05
# but range stalls at 35.25 vs gate 50. Root cause: single-target
# multi-codec anchor at score=63 pulls all outputs toward 63.
#
# V5: piecewise multi-band anchor. The anchor parquet has 6 anchor
# bands per (source, codec) at distinct butter levels × distinct
# target scores. The trainer reads target_score per-row from the
# parquet. Each band provides a calibration landmark at a different
# region of [0, 100], forcing the network to span the full range
# while preserving V4-style cross-codec parity at each band.
#
# Key V5 changes vs V4b:
#   --anchor-parquet      → multi-band anchors (6 bands × 4 codecs × ~1000 sources)
#   --tanh-output-head-scale 15.0   (V4b used 10.0 — wider linear region for [5, 95])
#   --anchor-loss-weight  → 0.05 (same as V4b; sum across 6 bands gives stronger signal)
#   --anchor-step-p 0.15  (V5 raise from V4b's 0.05 — match V4 design doc)
#
# The trainer auto-detects the `target_score` column on the parquet
# and switches to per-row target. No CLI change for per-row mode.
#
# SPEED-B (2026-05-19, task #165): default --minibatch-size 32.
# The K=1 asserts in mlp_train.rs aux loss steps were lifted; aux
# steps now fire on Adam-step boundaries and process K samples per
# fire, restoring the rayon parallel-batch speedup (T8.1-T8.11).
# Override KBATCH env var if a recipe needs different K.
#
# Args: <seed>
# Outputs: bake + log in OUT_DIR with name cc4v5_s<SEED>.bin
set -euo pipefail

SEED="${1:?usage: $0 <seed>}"
KBATCH="${KBATCH:-32}"

OUT_DIR="/mnt/v/zen/zensim-eval/exp_cross_codec_v5_2026-05-19"
TRAINER="/home/lilith/work/zen/zensim--cross-codec-metric/target/release/zensim_mlp_train"
PARQ_DIR="/mnt/v/zen/zensim-training/canonical-2026-05-18/train"
ANCHOR="/mnt/v/zen/zensim-training/2026-05-19-multi-band-anchors/anchors_multi_band_372col.parquet"
EQUIV="/mnt/v/zen/picker-training/2026-05-19-v2/cross_codec_equivalence_tight_v3.parquet"

BAKE="${OUT_DIR}/cc4v5_s${SEED}.bin"
LOG="${OUT_DIR}/cc4v5_s${SEED}.log"
STDOUT="${OUT_DIR}/cc4v5_s${SEED}.stdout"

mkdir -p "${OUT_DIR}"

echo "EXP-CROSS-CODEC-V5: seed=${SEED}"
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
    --anchor-loss-weight 0.05 \
    --anchor-target-score 63.0 \
    --anchor-step-p 0.15 \
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
