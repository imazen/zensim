#!/usr/bin/env bash
# Eval all 5 V_24-stdpool-konjnd010 bakes via bake_verdict (full Mohammadi panel).

set -euo pipefail

OUT_DIR="/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v24_stdpool_konjnd010"
BIN="/home/lilith/work/zen/zensim--ex2-stdpool-nonin/target/release/bake_verdict"
FEATS="/mnt/v/zen/zensim-training/2026-05-15-full-features"

for SEED in 1 2 3 4 5; do
    BAKE="${OUT_DIR}/v24_stdpool_konjnd010_s${SEED}_h128.bin"
    OUT_MD="${OUT_DIR}/v24_stdpool_konjnd010_s${SEED}_verdict.md"
    if [[ -f "${BAKE}" ]]; then
        echo "=== seed=${SEED} ==="
        "${BIN}" --bake "${BAKE}" --features-root "${FEATS}" --output "${OUT_MD}" 2>&1 | tail -3
    else
        echo "=== seed=${SEED} MISSING ${BAKE} ==="
    fi
done
