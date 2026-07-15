#!/usr/bin/env bash
# EXP-CROSS-CODEC-V4 evaluation harness (2026-05-19).
#
# V4 bakes carry the `zentrain.tanh_output_head` metadata — the runtime
# wraps the per-sample-α output natively in [0, 100]. NO affine
# calibration phase is needed (the V3 affine step was a workaround for
# the linear raw-output regime; V4 eliminates that step entirely).
#
# Phases:
#   1. qsweep_eval on the 50-image × 19-q JPEG sweep using mode `clamp`.
#      Tanh pin produces score directly; clamp just bounds extreme tails.
#   2. bake_verdict on KADID/TID/CID22/KonJND/AIC-3 for SROCC panel.
#   3. Cross-codec T=63 consistency (n=20 images, 4 codecs each) via
#      the run_cross_codec_v4b_consistency_inline.sh-style driver.
#   4. Multi-codec PJND score check using the anchor parquet built by
#      build_multi_codec_pjnd_anchors.py — predict score for every
#      (image, codec) anchor row, report cross-codec std at T=63.
#
# Output: /mnt/v/zen/zensim-eval/exp_cross_codec_v4b_2026-05-19/eval_summary.md
set -euo pipefail

V4B_DIR="/mnt/v/zen/zensim-eval/exp_cross_codec_v4b_2026-05-19"
QSWEEP_FEATURES="/mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep_features.csv"
QSWEEP_MANIFEST="/mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep/qsweep_manifest.tsv"
TUNER_BASELINE="/home/lilith/work/zen/zensim/zensim/weights/v_tuner_2026-05-18.bin"
QSWEEP_BIN="/home/lilith/work/zen/zensim/target/release/qsweep_eval"
VERDICT_BIN="/home/lilith/work/zen/zensim/target/release/bake_verdict"
MULTI_ANCHOR="/mnt/v/zen/zensim-training/2026-05-19-multi-codec-jnd-anchors/anchors_multi_codec_372col.parquet"

mkdir -p "${V4B_DIR}/verdicts"

echo "=== Phase 1: qsweep_eval (clamp mode — V4 native [0, 100]) ==="
RAW_OUT="${V4B_DIR}/qsweep_v4b.md"
BAKE_ARGS=("--features" "${QSWEEP_FEATURES}" "--manifest" "${QSWEEP_MANIFEST}")
BAKE_ARGS+=("--bake" "baseline_tuner=${TUNER_BASELINE}:clamp")
for bake in "${V4B_DIR}"/cc4v4b_*.bin; do
    [ -f "$bake" ] || continue
    name=$(basename "$bake" .bin)
    BAKE_ARGS+=("--bake" "${name}=${bake}:clamp")
done
BAKE_ARGS+=("--out" "${RAW_OUT}")
"${QSWEEP_BIN}" "${BAKE_ARGS[@]}"
echo "wrote ${RAW_OUT}"
echo

echo "=== Phase 2: bake_verdict (Mohammadi panel) on each V4 bake ==="
for bake in "${V4B_DIR}"/cc4v4b_*.bin; do
    [ -f "$bake" ] || continue
    name=$(basename "$bake" .bin)
    out_md="${V4B_DIR}/verdicts/${name}.md"
    "${VERDICT_BIN}" --bake "${bake}" --output "${out_md}" 2>&1 | tail -3 || echo "verdict failed for ${name}"
done

echo
echo "=== Phase 3: cross-codec T=63 consistency (n=20 images × 4 codecs) ==="
# Was: cp the `_inline` driver to /tmp, sed v3->v4b + cc4v3_->cc4v4b_ + strip
# /calibrated/, then run the result — the copy-and-sed pattern automated, at
# runtime. Two things were wrong with it. The `_inline` source has not existed
# for months, so the `if [ -x ]` guard meant Phase 3 SILENTLY SKIPPED on every
# run (a graceful skip, which CLAUDE.md forbids: a phase that quietly does
# nothing is worse than one that fails). And the file those seds were
# synthesizing already exists — run_cross_codec_v4b_consistency.sh is v4b-
# targeted with cc4v4b_ bakes and no /calibrated/ path. Call it directly; no
# guard, so a missing driver fails loudly instead of vanishing.
bash scripts/v_next/run_cross_codec_v4b_consistency.sh

echo
echo "=== Phase 4: multi-codec PJND score check ==="
python3 scripts/v_next/cross_codec_pjnd_check.py v4b "${V4B_DIR}" || echo "pjnd check failed"

echo
echo "All eval phases complete. See:"
echo "  ${RAW_OUT}                     (qsweep mono/tied/range/band-rmse)"
echo "  ${V4B_DIR}/verdicts/*.md        (Mohammadi panel per bake)"
echo "  ${V4B_DIR}/v4b_pjnd_check.md     (multi-codec PJND score std)"
