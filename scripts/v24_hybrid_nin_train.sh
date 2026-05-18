#!/usr/bin/env bash
# V_24-hybrid-nin: hybrid pool+rank head + sigmoid α + NiN 0.1.
#
# Same recipe as V_24-hybrid (the prior agent's no-NiN run) BUT with
# --norm-in-norm-weight 0.1 to A/B fairly against V_22-mix-LARGE-iwssim
# (which uses NiN 0.1).
#
# 5 seeds. Per-seed wall ~12 min with parallel execution.

set -euo pipefail

SEED="${1:?usage: $0 <seed>}"
OUT_DIR="/mnt/v/zen/zensim-eval/v24_hybrid_nin_2026-05-18"
WORKSPACE="/home/lilith/work/zen/zensim--ex2-hybrid-head"
TRAINER="${WORKSPACE}/target/release/zensim_mlp_train"
PARQ_DIR="/mnt/v/zen/zensim-training/2026-05-17-cvvdp-merged-trainer"

BAKE="${OUT_DIR}/v24_hybrid_nin_konjnd002_LARGE_iwssim_s${SEED}_h128.bin"
LOG="${OUT_DIR}/v24_hybrid_nin_konjnd002_LARGE_iwssim_s${SEED}_h128.log"
STDOUT="${OUT_DIR}/seed${SEED}.log"

mkdir -p "${OUT_DIR}"

"${TRAINER}" \
    --hybrid-head \
    --group "safesyn:${PARQ_DIR}/safesyn_mix_300col.parquet:1.0:0.0" \
    --group "kadid:${PARQ_DIR}/kadid_mix_300col.parquet:0.3:1.0" \
    --group "tid:${PARQ_DIR}/tid_mix_300col.parquet:0.3:1.0" \
    --group "konjnd:${PARQ_DIR}/konjnd_mix_300col.parquet:0.02:1.0" \
    --group "cvvdp_iwssim_large:${PARQ_DIR}/cvvdp_iwssim_large_300col_v2.parquet:0.5:0.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr 1e-3 --l2 1e-5 \
    --leaky-alpha 0.01 --val-policy min --early-stop-patience 0 \
    --max-features 300 --minibatch-size 256 \
    --pwrc-pair-weight --pwrc-sensory-threshold 5.0 \
    --norm-in-norm-weight 0.1 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0 \
    --target-column mix_cv40_iw60 --target-scale 100.0 --out-dtype f32 \
    --seed "${SEED}" --out "${BAKE}" --log-path "${LOG}" \
    2>&1 | tee "${STDOUT}"
