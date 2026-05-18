#!/usr/bin/env bash
# V_22-mix-LARGE-372-noLARGE: same as V_22-mix-LARGE-372 but DROPS
# the LARGE group entirely. Ablation to isolate "does the IW signal
# from the 4 anchor groups beat V_22-mix-LARGE-300 on its own?"
#
# Output: ${OUT_DIR}/v22_372feat_noLARGE_s${SEED}_h128.bin

set -euo pipefail

SEED="${1:?usage: $0 <seed>}"
OUT_DIR="/mnt/v/zen/zensim-eval/v22_372feat_2026-05-18"
WORKSPACE="/home/lilith/work/zen/zensim--372feat"
TRAINER="${WORKSPACE}/target/release/zensim_mlp_train"
PARQ_DIR="/mnt/v/zen/zensim-training/2026-05-17-cvvdp"

BAKE="${OUT_DIR}/v22_372feat_noLARGE_s${SEED}_h128.bin"
LOG="${OUT_DIR}/v22_372feat_noLARGE_s${SEED}_h128.log"
STDOUT="${OUT_DIR}/v22_372feat_noLARGE_s${SEED}.stdout"

mkdir -p "${OUT_DIR}"

"${TRAINER}" \
    --group "safesyn:${PARQ_DIR}/safesyn_features_mix_targets_372col.parquet:1.0:0.0" \
    --group "kadid:${PARQ_DIR}/kadid_features_mix_targets_372col.parquet:0.3:1.0" \
    --group "tid:${PARQ_DIR}/tid_features_mix_targets_372col.parquet:0.3:1.0" \
    --group "konjnd:${PARQ_DIR}/konjnd_features_mix_targets_372col.parquet:0.02:1.0" \
    --hidden 128 --epochs 300 --pairs-per-epoch 50000 --lr 1e-3 --l2 1e-5 \
    --leaky-alpha 0.01 --val-policy min --early-stop-patience 0 \
    --max-features 372 --minibatch-size 256 \
    --pwrc-pair-weight --pwrc-sensory-threshold 5.0 \
    --norm-in-norm-weight 0.1 --norm-in-norm-p 1.0 --norm-in-norm-q 2.0 \
    --target-column mix_cv40_iw60 --target-scale 100.0 --out-dtype f32 \
    --seed "${SEED}" --out "${BAKE}" --log-path "${LOG}" \
    2>&1 | tee "${STDOUT}"
