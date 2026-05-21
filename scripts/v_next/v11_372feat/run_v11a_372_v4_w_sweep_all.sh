#!/usr/bin/env bash
# V11-A-CC-EQ-WEIGHT-SWEEP (task #197, 2026-05-20).
# Loop driver: run all (weight, seed) combos sequentially.
# Sequential because of GPU memory (~6 GB / bake on RTX 5070 with 8 GB free).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_v11a_372_v4_w_sweep_seed.sh"

OUT_DIR="${OUT_DIR:-/mnt/v/zen/zensim-eval/exp_v11_cc_eq_sweep_2026-05-20}"
mkdir -p "${OUT_DIR}"

WEIGHTS="${WEIGHTS:-0.05 0.10 0.20 0.50}"
SEEDS="${SEEDS:-1 2 3 4 5}"

for w in ${WEIGHTS}; do
    for s in ${SEEDS}; do
        W_TAG="$(echo "$w" | sed 's/^0*\.//; s/\.//g')"
        BAKE="${OUT_DIR}/cc4v11_w${W_TAG}_s${s}.bin"
        if [ -f "${BAKE}" ]; then
            echo "SKIP w=$w s=$s (bake exists)"
            continue
        fi
        echo "=== START w=$w s=$s @ $(date -Iseconds) ==="
        CC_EQ_WEIGHT="$w" USE_GPU=cuda bash "${RUNNER}" "$s" \
            2>&1 | tail -5
        echo "=== END w=$w s=$s @ $(date -Iseconds) ==="
    done
done

echo "ALL DONE @ $(date -Iseconds)"
