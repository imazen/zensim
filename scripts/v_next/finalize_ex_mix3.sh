#!/usr/bin/env bash
# EX-MIX3 finalize: after all 15 verdicts done, run summarizer + Pareto verdict + (if winner) pack.

set -uo pipefail

OUT_DIR="/mnt/v/zen/zensim-eval/ex_mix3_2026-05-18"
WORKSPACE="/home/lilith/work/zen/zensim--ex-mix3"

cd "${WORKSPACE}"

# Ensure all verdicts exist (run any missing ones synchronously)
for v in cv33_iw33_sm33 cv30_iw40_sm30 cv40_iw40_sm20; do
    for s in 1 2 3 4 5; do
        bake="${OUT_DIR}/exmix3_${v}_s${s}_h128.bin"
        out="${OUT_DIR}/verdicts/exmix3_${v}_s${s}.md"
        if [ -f "${bake}" ] && [ ! -f "${out}" ]; then
            echo "  missing verdict, running: ${v} s${s}"
            /home/lilith/work/zen/zensim/target/release/bake_verdict \
                --bake "${bake}" \
                --corpora cid22,kadid,tid,konjnd,aic3 \
                --output "${out}"
        fi
    done
done

# 5-seed CI summary
python3 scripts/v_next/summarize_ex_mix3.py 2>&1 | tee "${OUT_DIR}/SUMMARY_5seed.log"

# Pareto verdict
python3 scripts/v_next/pareto_verdict.py 2>&1 | tee "${OUT_DIR}/PARETO_VERDICT.log"

# Per-seed bake_compare vs V_22 noLARGE s3
mkdir -p "${OUT_DIR}/compares_vs_noLARGE_s3"
for v in cv33_iw33_sm33 cv30_iw40_sm30 cv40_iw40_sm20; do
    for s in 1 2 3 4 5; do
        bake="${OUT_DIR}/exmix3_${v}_s${s}_h128.bin"
        out="${OUT_DIR}/compares_vs_noLARGE_s3/exmix3_${v}_s${s}_vs_v22noLARGE_s3.md"
        if [ -f "${bake}" ] && [ ! -f "${out}" ]; then
            /home/lilith/work/zen/zensim/target/release/bake_compare \
                --a "${bake}" \
                --b /mnt/v/zen/zensim-eval/v22_372feat_2026-05-18/v22_372feat_noLARGE_s3_h128.bin \
                --output "${out}" \
                --bootstrap-resamples 1000 2>&1 | tail -1
        fi
    done
done

echo "=== finalize complete ==="
echo "  - ${OUT_DIR}/SUMMARY_5seed.md (or .log)"
echo "  - ${OUT_DIR}/PARETO_VERDICT.md (or .log)"
echo "  - ${OUT_DIR}/compares_vs_noLARGE_s3/*.md"
