#!/usr/bin/env bash
# V_24-stdpool-nonin: pool-head ON, NiN OFF. Test EX-2 hypothesis that
# disabling NiN composition preserves CID22 lift while restoring KonJND.
# 5 seeds, identical recipe to V_24-stdpool-prod except --norm-in-norm-weight 0.0.

set -euo pipefail

SEED="${1:?usage: $0 <seed>}"
OUT_DIR="/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v24_stdpool_nonin"
WORKSPACE="/home/lilith/work/zen/zensim--ex2-stdpool-nonin"
TRAINER="${WORKSPACE}/target/release/zensim_mlp_train"
PARQ_DIR="/mnt/v/zen/zensim-training/2026-05-17-cvvdp"
PARQ2_DIR="/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer"

BAKE="${OUT_DIR}/v24_stdpool_nonin_s${SEED}_h128.bin"
LOG="${OUT_DIR}/v24_stdpool_nonin_s${SEED}_h128.log"
STDOUT="${OUT_DIR}/v24_stdpool_nonin_s${SEED}.stdout"

mkdir -p "${OUT_DIR}"

"${TRAINER}" \
    --pool-head \
    --group "safesyn:${PARQ_DIR}/safesyn_features_mix_targets_372col.parquet:1.0:1.0" \
    --group "kadid:${PARQ_DIR}/kadid_features_mix_targets_372col.parquet:0.3:1.0" \
    --group "tid:${PARQ_DIR}/tid_features_mix_targets_372col.parquet:0.3:1.0" \
    --group "konjnd:${PARQ_DIR}/konjnd_features_mix_targets_372col.parquet:0.02:0.0" \
    --group "cvvdp_iwssim_LARGE:${PARQ2_DIR}/cvvdp_iwssim_large_300col_v2.parquet:0.5:0.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr 1e-3 --l2 1e-5 \
    --leaky-alpha 0.01 --val-policy min --early-stop-patience 30 \
    --max-features 300 --minibatch-size 256 \
    --pwrc-pair-weight --pwrc-sensory-threshold 5.0 \
    --norm-in-norm-weight 0.0 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0 \
    --target-column mix_cv40_iw60 --target-scale 100.0 --out-dtype f32 \
    --seed "${SEED}" --out "${BAKE}" --log-path "${LOG}" \
    2>&1 | tee "${STDOUT}"
