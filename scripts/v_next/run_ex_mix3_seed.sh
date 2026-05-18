#!/usr/bin/env bash
# EX-MIX3: 3-way cv+iw+sm target mix training
#
# Variant: $1 (one of: cv33_iw33_sm33, cv30_iw40_sm30, cv40_iw40_sm20)
# Seed:    $2
#
# Coverage gate (per task): konjnd-dense and cvvdp_iwssim_LARGE have 0% ssim2,
# DROP from EX-MIX3 training. Use safesyn + kadid + tid (3 groups).
# This deliberately differs from V_22-LARGE+iwssim baseline (which used 5 groups).
# All else matches the V_22 noLARGE recipe verbatim (h=128, epochs=300,
# mb=256, PWRC, NiN 0.1, target-scale 100.0).

set -euo pipefail

VARIANT="${1:?usage: $0 <cv33_iw33_sm33|cv30_iw40_sm30|cv40_iw40_sm20> <seed>}"
SEED="${2:?usage: $0 <variant> <seed>}"

case "${VARIANT}" in
    cv33_iw33_sm33) TARGET_COL="mix_cv33_iw33_sm33" ;;
    cv30_iw40_sm30) TARGET_COL="mix_cv30_iw40_sm30" ;;
    cv40_iw40_sm20) TARGET_COL="mix_cv40_iw40_sm20" ;;
    *) echo "ERROR: variant must be one of cv33_iw33_sm33, cv30_iw40_sm30, cv40_iw40_sm20" >&2; exit 2 ;;
esac

OUT_DIR="/mnt/v/zen/zensim-eval/ex_mix3_2026-05-18"
WORKSPACE="/home/lilith/work/zen/zensim--ex-mix3"
TRAINER="/home/lilith/work/zen/zensim/target/release/zensim_mlp_train"
PARQ_DIR="/mnt/v/zen/zensim-training/2026-05-18-mix3"

BAKE="${OUT_DIR}/exmix3_${VARIANT}_s${SEED}_h128.bin"
LOG="${OUT_DIR}/exmix3_${VARIANT}_s${SEED}_h128.log"
STDOUT="${OUT_DIR}/exmix3_${VARIANT}_s${SEED}.stdout"

mkdir -p "${OUT_DIR}"

"${TRAINER}" \
    --group "safesyn:${PARQ_DIR}/safesyn.parquet:1.0:0.0" \
    --group "kadid:${PARQ_DIR}/kadid.parquet:0.3:1.0" \
    --group "tid:${PARQ_DIR}/tid.parquet:0.3:1.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr 1e-3 --l2 1e-5 \
    --leaky-alpha 0.01 --val-policy min --early-stop-patience 0 \
    --max-features 372 --minibatch-size 256 \
    --pwrc-pair-weight --pwrc-sensory-threshold 5.0 \
    --norm-in-norm-weight 0.1 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0 \
    --target-column "${TARGET_COL}" --target-scale 100.0 --out-dtype f32 \
    --seed "${SEED}" --out "${BAKE}" --log-path "${LOG}" \
    2>&1 | tee "${STDOUT}"
