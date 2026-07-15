#!/usr/bin/env bash
# EXP-CROSS-CODEC-V5 evaluation harness (2026-05-19).
#
# V5 ships piecewise multi-band anchors. The eval has all V4b phases
# (qsweep, bake_verdict panels, T=63 cross-codec consistency, single-band
# PJND check) PLUS the new multi-band cross-codec check that walks every
# anchor band (butter ∈ {0.3, 0.8, 1.5, 2.5, 4.0, 6.0}) and reports
# cross-codec score std at each.
#
# V5 bakes use tanh_output_head_scale=15.0 (V4 used 10.0). The runtime
# dispatches on the `zentrain.tanh_output_head` metadata.
#
# Phases:
#   1. qsweep_eval on the 50-image × 19-q JPEG sweep using mode `clamp`.
#   2. bake_verdict on KADID/TID/CID22/KonJND/AIC-3 for SROCC panel.
#   3. Cross-codec T=63 consistency (n=20 images, 4 codecs each) — V4 gate.
#   4. Single-band multi-codec PJND score check (V4 eval; T=63 only).
#   5. NEW: multi-band cross-codec consistency check (V5 gate; all 6 bands).
#
# Output: /mnt/v/zen/zensim-eval/exp_cross_codec_v5_2026-05-19/
set -euo pipefail

V5_DIR="/mnt/v/zen/zensim-eval/exp_cross_codec_v5_2026-05-19"
QSWEEP_FEATURES="/mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep_features.csv"
QSWEEP_MANIFEST="/mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep/qsweep_manifest.tsv"
TUNER_BASELINE="/home/lilith/work/zen/zensim/zensim/weights/v_tuner_2026-05-18.bin"
QSWEEP_BIN="/home/lilith/work/zen/zensim/target/release/qsweep_eval"
VERDICT_BIN="/home/lilith/work/zen/zensim/target/release/bake_verdict"

mkdir -p "${V5_DIR}/verdicts"

echo "=== Phase 1: qsweep_eval (clamp mode — V5 native [0, 100]) ==="
RAW_OUT="${V5_DIR}/qsweep_v5.md"
BAKE_ARGS=("--features" "${QSWEEP_FEATURES}" "--manifest" "${QSWEEP_MANIFEST}")
BAKE_ARGS+=("--bake" "baseline_tuner=${TUNER_BASELINE}:clamp")
for bake in "${V5_DIR}"/cc4v5_*.bin; do
    [ -f "$bake" ] || continue
    name=$(basename "$bake" .bin)
    BAKE_ARGS+=("--bake" "${name}=${bake}:clamp")
done
BAKE_ARGS+=("--out" "${RAW_OUT}")
"${QSWEEP_BIN}" "${BAKE_ARGS[@]}"
echo "wrote ${RAW_OUT}"
echo

echo "=== Phase 2: bake_verdict (Mohammadi panel) on each V5 bake ==="
for bake in "${V5_DIR}"/cc4v5_*.bin; do
    [ -f "$bake" ] || continue
    name=$(basename "$bake" .bin)
    out_md="${V5_DIR}/verdicts/${name}.md"
    "${VERDICT_BIN}" --bake "${bake}" --output "${out_md}" 2>&1 | tail -3 || echo "verdict failed for ${name}"
done

echo
echo "=== Phase 3: cross-codec T=63 consistency (n=20 images × 4 codecs) ==="
TOOL=/home/lilith/work/zen/zensim/target/release/predict_features_with_bake
ZEN_METRICS=/home/lilith/work/zen/zenmetrics/target/release/zenmetrics
CONSISTENCY=/home/lilith/work/zen/zensim/scripts/v_next/cross_codec_consistency.py
T63_DIR="${V5_DIR}/cross_codec_t63"
mkdir -p "${T63_DIR}"
for bake in "${V5_DIR}"/cc4v5_*.bin; do
    [ -f "$bake" ] || continue
    name=$(basename "$bake" .bin)
    out_tsv="${T63_DIR}/${name}_t63_n20.tsv"
    if [ -f "$out_tsv" ]; then
        echo "=== ${name} (already exists, skip) ==="
        continue
    fi
    echo "=== ${name} ==="
    python3 "${CONSISTENCY}" \
        --target 63 \
        --bake "${bake}" \
        --bake-post clamp \
        --n-images 20 \
        --predict-tool "${TOOL}" \
        --zen-metrics "${ZEN_METRICS}" \
        --out "${out_tsv}" 2>&1 | tail -8 || echo "  driver failed for ${name}"
done

echo
echo "=== Phase 4: single-band multi-codec PJND score check (T=63) ==="
python3 scripts/v_next/cross_codec_pjnd_check.py v4b "${V5_DIR}" || echo "v4b pjnd check failed"

echo
echo "=== Phase 5: multi-band cross-codec consistency check (V5 gate, all 6 bands) ==="
python3 scripts/v_next/eval_v5_multi_band_check.py "${V5_DIR}" || echo "v5 multi-band check failed"

echo
echo "All eval phases complete. See:"
echo "  ${RAW_OUT}                                       (qsweep mono/tied/range/band-rmse)"
echo "  ${V5_DIR}/verdicts/*.md                            (Mohammadi panel per bake)"
echo "  ${V5_DIR}/cross_codec_t63/*.tsv                   (cross-codec T=63 raw)"
echo "  ${V5_DIR}/v4b_pjnd_check.md                       (V4-style PJND score std)"
echo "  ${V5_DIR}/v5_multi_band_check.md                  (V5-specific all-bands gate)"
