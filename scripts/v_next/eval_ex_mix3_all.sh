#!/usr/bin/env bash
# EX-MIX3: evaluate all 15 bakes via bake_verdict + run bake_compare
# against baseline V_22 noLARGE s3 (best of 5 baselines).

set -euo pipefail

OUT_DIR="/mnt/v/zen/zensim-eval/ex_mix3_2026-05-18"
BAKE_VERDICT="/home/lilith/work/zen/zensim/target/release/bake_verdict"
BAKE_COMPARE="/home/lilith/work/zen/zensim/target/release/bake_compare"
BASELINE_NOLARGE_S3="/mnt/v/zen/zensim-eval/v22_372feat_2026-05-18/v22_372feat_noLARGE_s3_h128.bin"
BASELINE_LARGE_S3="/mnt/v/zen/zensim-eval/cvvdp_safesyn_2026-05-17/v22_mix_cv40_konjnd_0_02_LARGE_iwssim_s3_h128.bin"

VARIANTS=("cv33_iw33_sm33" "cv30_iw40_sm30" "cv40_iw40_sm20")
SEEDS=(1 2 3 4 5)

# Step 1: per-seed bake_verdict for each variant
mkdir -p "${OUT_DIR}/verdicts"
for v in "${VARIANTS[@]}"; do
    for s in "${SEEDS[@]}"; do
        bake="${OUT_DIR}/exmix3_${v}_s${s}_h128.bin"
        out="${OUT_DIR}/verdicts/exmix3_${v}_s${s}.md"
        if [ ! -f "${bake}" ]; then
            echo "  MISSING bake: ${bake} — skipping verdict"
            continue
        fi
        if [ -f "${out}" ]; then
            echo "  cached verdict: ${out}"
            continue
        fi
        echo "  verdict: ${bake}"
        "${BAKE_VERDICT}" --bake "${bake}" --corpora cid22,kadid,tid,konjnd,aic3 --output "${out}"
    done
done

# Step 2: bake_compare per-variant per-seed vs noLARGE baseline
mkdir -p "${OUT_DIR}/compares_vs_noLARGE_s3"
for v in "${VARIANTS[@]}"; do
    for s in "${SEEDS[@]}"; do
        bake="${OUT_DIR}/exmix3_${v}_s${s}_h128.bin"
        out="${OUT_DIR}/compares_vs_noLARGE_s3/exmix3_${v}_s${s}_vs_v22noLARGE_s3.md"
        if [ ! -f "${bake}" ]; then
            continue
        fi
        if [ -f "${out}" ]; then
            continue
        fi
        echo "  compare ${v} s${s} vs v22-noLARGE-s3"
        "${BAKE_COMPARE}" --a "${bake}" --b "${BASELINE_NOLARGE_S3}" --output "${out}" || echo "    FAIL"
    done
done

# Step 3: bake_compare per-variant per-seed vs LARGE+iwssim baseline (300-feat)
# NOTE: ex-mix3 is 372-feat; LARGE is 300-feat. bake_compare may handle mismatched n_features.
mkdir -p "${OUT_DIR}/compares_vs_LARGE_s3"
for v in "${VARIANTS[@]}"; do
    for s in "${SEEDS[@]}"; do
        bake="${OUT_DIR}/exmix3_${v}_s${s}_h128.bin"
        out="${OUT_DIR}/compares_vs_LARGE_s3/exmix3_${v}_s${s}_vs_v22LARGE_iwssim_s3.md"
        if [ ! -f "${bake}" ]; then
            continue
        fi
        if [ -f "${out}" ]; then
            continue
        fi
        echo "  compare ${v} s${s} vs v22-LARGE-iwssim-s3 (300feat)"
        "${BAKE_COMPARE}" --a "${bake}" --b "${BASELINE_LARGE_S3}" --output "${out}" 2>&1 || echo "    FAIL (likely 300vs372 mismatch)"
    done
done

echo "=== eval complete ==="
