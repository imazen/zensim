#!/usr/bin/env bash
# Evaluate all 5 V_24-stdpool-nonin bakes against full Mohammadi panel.
set -euo pipefail

OUT_DIR="/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v24_stdpool_nonin"
BAKE_VERDICT="/home/lilith/work/zen/zensim--ex2-stdpool-nonin/target/release/bake_verdict"

for SEED in 1 2 3 4 5; do
    BAKE="${OUT_DIR}/v24_stdpool_nonin_s${SEED}_h128.bin"
    VERDICT="${OUT_DIR}/v24_stdpool_nonin_s${SEED}_verdict.md"
    if [ -f "${BAKE}" ]; then
        echo "=== Evaluating seed ${SEED} ==="
        "${BAKE_VERDICT}" --bake "${BAKE}" --output "${VERDICT}" 2>&1 | tail -20
    else
        echo "MISSING: ${BAKE}"
    fi
done
