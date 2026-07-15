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
#      the run_cross_codec_v3_consistency.sh-style driver.
#   4. Multi-codec PJND score check using the anchor parquet built by
#      build_multi_codec_pjnd_anchors.py — predict score for every
#      (image, codec) anchor row, report cross-codec std at T=63.
#
# Output: /mnt/v/zen/zensim-eval/exp_cross_codec_v4_2026-05-19/eval_summary.md
set -euo pipefail

V4_DIR="/mnt/v/zen/zensim-eval/exp_cross_codec_v4_2026-05-19"
QSWEEP_FEATURES="/mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep_features.csv"
QSWEEP_MANIFEST="/mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep/qsweep_manifest.tsv"
TUNER_BASELINE="/home/lilith/work/zen/zensim/zensim/weights/v_tuner_2026-05-18.bin"
QSWEEP_BIN="/home/lilith/work/zen/zensim/target/release/qsweep_eval"
VERDICT_BIN="/home/lilith/work/zen/zensim/target/release/bake_verdict"
MULTI_ANCHOR="/mnt/v/zen/zensim-training/2026-05-19-multi-codec-jnd-anchors/anchors_multi_codec_372col.parquet"

mkdir -p "${V4_DIR}/verdicts"

echo "=== Phase 1: qsweep_eval (clamp mode — V4 native [0, 100]) ==="
RAW_OUT="${V4_DIR}/qsweep_v4.md"
BAKE_ARGS=("--features" "${QSWEEP_FEATURES}" "--manifest" "${QSWEEP_MANIFEST}")
BAKE_ARGS+=("--bake" "baseline_tuner=${TUNER_BASELINE}:clamp")
for bake in "${V4_DIR}"/cc4v4_*.bin; do
    [ -f "$bake" ] || continue
    name=$(basename "$bake" .bin)
    BAKE_ARGS+=("--bake" "${name}=${bake}:clamp")
done
BAKE_ARGS+=("--out" "${RAW_OUT}")
"${QSWEEP_BIN}" "${BAKE_ARGS[@]}"
echo "wrote ${RAW_OUT}"
echo

echo "=== Phase 2: bake_verdict (Mohammadi panel) on each V4 bake ==="
for bake in "${V4_DIR}"/cc4v4_*.bin; do
    [ -f "$bake" ] || continue
    name=$(basename "$bake" .bin)
    out_md="${V4_DIR}/verdicts/${name}.md"
    "${VERDICT_BIN}" --bake "${bake}" --output "${out_md}" 2>&1 | tail -3 || echo "verdict failed for ${name}"
done

echo
echo "=== Phase 3: cross-codec T=63 consistency (n=20 images × 4 codecs) ==="
# Was: cp the v3 driver to /tmp and sed v3->v4 + cc4v3_->cc4v4_ + strip
# /calibrated/ at runtime. That sed's exact output already exists on disk as
# run_cross_codec_v4_consistency.sh — v4-targeted, cc4v4_ bakes, no
# /calibrated/. Which is presumably where that file came from: someone ran the
# factory once and committed the result, and nobody removed the factory. So the
# generator and the generated both shipped, free to drift apart. Call the real
# one; drop the `if [ -x ]` so a missing driver fails loudly rather than
# skipping Phase 3 in silence.
bash scripts/v_next/run_cross_codec_v4_consistency.sh

echo
echo "=== Phase 4: multi-codec PJND score check ==="
python3 scripts/v_next/cross_codec_pjnd_check.py v4 "${V4_DIR}" || echo "pjnd check failed"

echo
echo "All eval phases complete. See:"
echo "  ${RAW_OUT}                     (qsweep mono/tied/range/band-rmse)"
echo "  ${V4_DIR}/verdicts/*.md        (Mohammadi panel per bake)"
echo "  ${V4_DIR}/v4_pjnd_check.md     (multi-codec PJND score std)"
