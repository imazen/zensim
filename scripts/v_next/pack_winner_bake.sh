#!/usr/bin/env bash
# EX-MIX3: pack the winning seed-3 bake using zenpredict repack.
# (Seed 3 is typically the canonical "best seed" for packed bakes per
# V_22-LARGE+iwssim and other prior recipes.)
#
# Usage: pack_winner_bake.sh <variant>
# e.g., pack_winner_bake.sh cv33_iw33_sm33

set -euo pipefail

VARIANT="${1:?usage: $0 <variant>}"
SEED="${2:-3}"
OUT_DIR="/mnt/v/zen/zensim-eval/ex_mix3_2026-05-18"
ZENPREDICT="/home/lilith/work/zen/zenanalyze/target/release/zenpredict"

IN_BAKE="${OUT_DIR}/exmix3_${VARIANT}_s${SEED}_h128.bin"
PACKED="${OUT_DIR}/exmix3_${VARIANT}_s${SEED}_h128_packed.bin"

if [ ! -f "${IN_BAKE}" ]; then
    echo "ERROR: input bake not found: ${IN_BAKE}" >&2
    exit 1
fi

if [ ! -x "${ZENPREDICT}" ]; then
    echo "WARN: zenpredict binary not found at ${ZENPREDICT}"
    echo "  building... (cd /home/lilith/work/zen/zenanalyze && cargo build --release --bin zenpredict -p zenpredict-bake)"
    (cd /home/lilith/work/zen/zenanalyze && cargo build --release --bin zenpredict -p zenpredict-bake) 2>&1 | tail -5
fi

"${ZENPREDICT}" repack "${IN_BAKE}" "${PACKED}" \
    --dtype i8 --zerobias 0.005 --compress --optimize

echo "--- packed: ${PACKED}"
ls -la "${IN_BAKE}" "${PACKED}"

# Verify CID22 SROCC delta after pack
ORIG_VERDICT="${OUT_DIR}/verdicts/exmix3_${VARIANT}_s${SEED}.md"
PACKED_VERDICT="${OUT_DIR}/verdicts/exmix3_${VARIANT}_s${SEED}_packed.md"
/home/lilith/work/zen/zensim/target/release/bake_verdict \
    --bake "${PACKED}" \
    --corpora cid22,kadid,tid,konjnd,aic3 \
    --output "${PACKED_VERDICT}"

orig_cid=$(grep "^| CID22" "${ORIG_VERDICT}" | head -1 | awk -F'|' '{print $4}' | tr -d ' ')
pack_cid=$(grep "^| CID22" "${PACKED_VERDICT}" | head -1 | awk -F'|' '{print $4}' | tr -d ' ')
echo "  CID22 SROCC: orig=${orig_cid} packed=${pack_cid}"
