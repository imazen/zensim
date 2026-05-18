#!/usr/bin/env bash
# Eval V_24-hybrid-nin 5-seed CI vs V_22-mix-LARGE-iwssim baseline AND
# V_24-hybrid (no-NiN, the prior agent's run). Writes per-seed verdict
# .md files + aggregate summary.
#
# Usage: bash scripts/v24_hybrid_nin_eval.sh

set -euo pipefail

cd "$(dirname "$0")/.."

OUT_DIR="/mnt/v/zen/zensim-eval/v24_hybrid_nin_2026-05-18"
NONIN_DIR="/mnt/v/zen/zensim-eval/v24_hybrid_2026-05-18"
V22_BAKE="/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128_packed.bin"

VERDICT=./target/release/bake_verdict
COMPARE=./target/release/bake_compare

[ -x "$VERDICT" ] || { echo "bake_verdict not built"; exit 1; }
[ -x "$COMPARE" ] || { echo "bake_compare not built"; exit 1; }

# Per-seed bake_verdict
for s in 1 2 3 4 5; do
    BAKE="${OUT_DIR}/v24_hybrid_nin_konjnd002_LARGE_iwssim_s${s}_h128.bin"
    if [ ! -f "${BAKE}" ]; then
        echo "WARN: missing ${BAKE}"
        continue
    fi
    OUT="${OUT_DIR}/hybrid_nin_seed${s}.md"
    echo "[$(date -u +%H:%M:%S)] verdict seed ${s} ..."
    "${VERDICT}" --bake "${BAKE}" --output "${OUT}"
done

# Baseline V_22 verdict
echo "[$(date -u +%H:%M:%S)] verdict V_22-LARGE-iwssim baseline ..."
"${VERDICT}" --bake "${V22_BAKE}" --output "${OUT_DIR}/baseline_v22.md"

# Decisive bake_compare: pick seed=2 (the previous run's CID22 best seed) vs V_22 + vs no-NiN
SEED2_BAKE="${OUT_DIR}/v24_hybrid_nin_konjnd002_LARGE_iwssim_s2_h128.bin"
NONIN_S2_BAKE="${NONIN_DIR}/v24_hybrid_konjnd002_LARGE_iwssim_s2_h128.bin"

if [ -f "${SEED2_BAKE}" ] && [ -f "${V22_BAKE}" ]; then
    echo "[$(date -u +%H:%M:%S)] bake_compare seed=2 vs V_22-LARGE-iwssim ..."
    "${COMPARE}" --a "${SEED2_BAKE}" --b "${V22_BAKE}" \
        --output "${OUT_DIR}/bake_compare_s2_vs_v22.md" \
        --json "${OUT_DIR}/bake_compare_s2_vs_v22.json"
fi

if [ -f "${SEED2_BAKE}" ] && [ -f "${NONIN_S2_BAKE}" ]; then
    echo "[$(date -u +%H:%M:%S)] bake_compare seed=2 NiN vs no-NiN ..."
    "${COMPARE}" --a "${SEED2_BAKE}" --b "${NONIN_S2_BAKE}" \
        --output "${OUT_DIR}/bake_compare_s2_nin_vs_nonin.md" \
        --json "${OUT_DIR}/bake_compare_s2_nin_vs_nonin.json"
fi

# Aggregate summary
SUMMARY="${OUT_DIR}/aggregate_summary.md"
{
    echo "# V_24-hybrid-nin 5-seed CI summary ($(date -u +%Y-%m-%dT%H:%M:%SZ))"
    echo ""
    echo "Bakes:"
    for s in 1 2 3 4 5; do
        BAKE="${OUT_DIR}/v24_hybrid_nin_konjnd002_LARGE_iwssim_s${s}_h128.bin"
        if [ -f "${BAKE}" ]; then
            echo "- seed ${s}: $(basename "${BAKE}") ($(stat -c%s "${BAKE}") bytes)"
        fi
    done
    echo ""
    echo "## Aggregate SROCC per corpus"
    echo ""
    echo "| variant | CID22 | KADID | TID | KonJND | AIC-3 |"
    echo "|---|---:|---:|---:|---:|---:|"
    for s in 1 2 3 4 5; do
        OUT="${OUT_DIR}/hybrid_nin_seed${s}.md"
        if [ -f "${OUT}" ]; then
            SROCC=$(grep -E '^\| V_X bake \|' "${OUT}" | awk -F'|' '{print $3}' | tr -d ' ')
            echo -n "| hybrid_nin s${s} "
            for v in $SROCC; do echo -n "| $v "; done
            echo "|"
        fi
    done
    OUT="${OUT_DIR}/baseline_v22.md"
    if [ -f "${OUT}" ]; then
        SROCC=$(grep -E '^\| V_X bake \|' "${OUT}" | awk -F'|' '{print $3}' | tr -d ' ')
        echo -n "| V_22-LARGE-iwssim ship "
        for v in $SROCC; do echo -n "| $v "; done
        echo "|"
    fi
    echo ""
    echo "## Aggregate Z-RMSE per corpus"
    echo ""
    echo "| variant | CID22 | KADID | TID | KonJND | AIC-3 |"
    echo "|---|---:|---:|---:|---:|---:|"
    for s in 1 2 3 4 5; do
        OUT="${OUT_DIR}/hybrid_nin_seed${s}.md"
        if [ -f "${OUT}" ]; then
            ZRMSE=$(grep -E '^\| V_X bake \|' "${OUT}" | awk -F'|' '{print $8}' | tr -d ' ')
            echo -n "| hybrid_nin s${s} "
            for v in $ZRMSE; do echo -n "| $v "; done
            echo "|"
        fi
    done
    OUT="${OUT_DIR}/baseline_v22.md"
    if [ -f "${OUT}" ]; then
        ZRMSE=$(grep -E '^\| V_X bake \|' "${OUT}" | awk -F'|' '{print $8}' | tr -d ' ')
        echo -n "| V_22-LARGE-iwssim ship "
        for v in $ZRMSE; do echo -n "| $v "; done
        echo "|"
    fi
    echo ""
    echo "## α convergence per seed (from train log)"
    echo ""
    echo "| seed | final α | logit |"
    echo "|---:|---:|---:|"
    for s in 1 2 3 4 5; do
        LOG="${OUT_DIR}/v24_hybrid_nin_konjnd002_LARGE_iwssim_s${s}_h128.log"
        if [ -f "${LOG}" ]; then
            LINE=$(grep "final α=" "${LOG}" | tail -1 || true)
            ALPHA=$(echo "${LINE}" | sed -n 's/.*final α=\([0-9.]*\).*/\1/p')
            LOGIT=$(echo "${LINE}" | sed -n 's/.*logit=\([+\-0-9.]*\).*/\1/p')
            echo "| ${s} | ${ALPHA} | ${LOGIT} |"
        fi
    done
} > "${SUMMARY}"
cat "${SUMMARY}"
echo ""
echo "wrote ${SUMMARY}"
