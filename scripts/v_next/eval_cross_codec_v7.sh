#!/usr/bin/env bash
# EXP-CROSS-CODEC-V7 evaluation harness (2026-05-19).
#
# Mirrors eval_cross_codec_v5.sh but globs cc4v7_*.bin in V7_DIR.
# Increased anchor pressure (anchor_loss_weight 0.5/1.0 vs V5's 0.05).
#
# Phases:
#   1. qsweep_eval on the 50-image × 19-q JPEG sweep using mode `clamp`.
#   2. bake_verdict on KADID/TID/CID22/KonJND/AIC-3 for SROCC panel.
#   3. Cross-codec T=63 consistency (n=20 images, 4 codecs each) — V4 gate.
#   4. Single-band multi-codec PJND score check (T=63 only).
#   5. Multi-band cross-codec consistency check (V5 gate, all 6 bands).
#
# Output: /mnt/v/zen/zensim-eval/exp_cross_codec_v7_2026-05-19/
set -euo pipefail

V7_DIR="/mnt/v/zen/zensim-eval/exp_cross_codec_v7_2026-05-19"
QSWEEP_FEATURES="/mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep_features.csv"
QSWEEP_MANIFEST="/mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep/qsweep_manifest.tsv"
TUNER_BASELINE="/home/lilith/work/zen/zensim/zensim/weights/v_tuner_2026-05-18.bin"
QSWEEP_BIN="/home/lilith/work/zen/zensim/target/release/qsweep_eval"
VERDICT_BIN="/home/lilith/work/zen/zensim/target/release/bake_verdict"

mkdir -p "${V7_DIR}/verdicts"

echo "=== Phase 1: qsweep_eval (clamp mode — V6 native [0, 100]) ==="
RAW_OUT="${V7_DIR}/qsweep_v7.md"
BAKE_ARGS=("--features" "${QSWEEP_FEATURES}" "--manifest" "${QSWEEP_MANIFEST}")
BAKE_ARGS+=("--bake" "baseline_tuner=${TUNER_BASELINE}:clamp")
for bake in "${V7_DIR}"/cc4v7_*.bin; do
    [ -f "$bake" ] || continue
    name=$(basename "$bake" .bin)
    BAKE_ARGS+=("--bake" "${name}=${bake}:clamp")
done
BAKE_ARGS+=("--out" "${RAW_OUT}")
"${QSWEEP_BIN}" "${BAKE_ARGS[@]}"
echo "wrote ${RAW_OUT}"
echo

echo "=== Phase 2: bake_verdict (Mohammadi panel) on each V6 bake ==="
for bake in "${V7_DIR}"/cc4v7_*.bin; do
    [ -f "$bake" ] || continue
    name=$(basename "$bake" .bin)
    out_md="${V7_DIR}/verdicts/${name}.md"
    "${VERDICT_BIN}" --bake "${bake}" --output "${out_md}" 2>&1 | tail -3 || echo "verdict failed for ${name}"
done

echo
echo "=== Phase 3: cross-codec T=63 consistency (n=20 images × 4 codecs) ==="
TOOL=/home/lilith/work/zen/zensim/target/release/predict_features_with_bake
ZEN_METRICS=/home/lilith/work/zen/zenmetrics/target/release/zenmetrics
CONSISTENCY=/home/lilith/work/zen/zensim/scripts/v_next/cross_codec_consistency.py
T63_DIR="${V7_DIR}/cross_codec_t63"
mkdir -p "${T63_DIR}"
for bake in "${V7_DIR}"/cc4v7_*.bin; do
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
python3 scripts/v_next/cross_codec_pjnd_check.py v7 "${V7_DIR}" || echo "v7 pjnd check failed"

echo
echo "=== Phase 5: multi-band cross-codec consistency check (V6 gate, all 6 bands) ==="
python3 scripts/v_next/cross_codec_multi_band_check.py v7 "${V7_DIR}" || echo "v7 multi-band check failed"

echo
echo "All eval phases complete. See:"
echo "  ${RAW_OUT}                                       (qsweep mono/tied/range/band-rmse)"
echo "  ${V7_DIR}/verdicts/*.md                          (Mohammadi panel per bake)"
echo "  ${V7_DIR}/cross_codec_t63/*.tsv                  (cross-codec T=63 raw)"
echo "  ${V7_DIR}/v7_pjnd_check.md                       (V4-style PJND score std)"
echo "  ${V7_DIR}/v7_multi_band_check.md                 (multi-band parity gate)"
