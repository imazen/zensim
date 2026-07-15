#!/usr/bin/env bash
# EXP-CROSS-CODEC-V8 evaluation harness (2026-05-19).
#
# Mirrors eval_cross_codec_v6.sh but globs cc4v8_*.bin in V8_DIR and
# uses the V8 multi-band check (4 bands, butter ∈ {0.5, 1.0, 2.5, 4.0}).
#
# Phases:
#   1. qsweep_eval on the 50-image × 19-q JPEG sweep using mode `clamp`.
#   2. bake_verdict on KADID/TID/CID22/KonJND/AIC-3 for Mohammadi panel.
#   3. Cross-codec T=63 consistency (n=20 images, 4 codecs each) — V4 gate.
#   4. Single-band multi-codec PJND score check (T=63 only).
#   5. Multi-band cross-codec consistency check (V8 gate, 4 bands).
set -euo pipefail

V8_DIR="/mnt/v/zen/zensim-eval/exp_cross_codec_v8_2026-05-19"
QSWEEP_FEATURES="/mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep_features.csv"
QSWEEP_MANIFEST="/mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep/qsweep_manifest.tsv"
TUNER_BASELINE="/home/lilith/work/zen/zensim/zensim/weights/v_tuner_2026-05-18.bin"
QSWEEP_BIN="/home/lilith/work/zen/zensim/target/release/qsweep_eval"
VERDICT_BIN="/home/lilith/work/zen/zensim/target/release/bake_verdict"

mkdir -p "${V8_DIR}/verdicts"

echo "=== Phase 1: qsweep_eval (clamp mode — V8 native [0, 100]) ==="
RAW_OUT="${V8_DIR}/qsweep_v8.md"
BAKE_ARGS=("--features" "${QSWEEP_FEATURES}" "--manifest" "${QSWEEP_MANIFEST}")
BAKE_ARGS+=("--bake" "baseline_tuner=${TUNER_BASELINE}:clamp")
for bake in "${V8_DIR}"/cc4v8_*.bin; do
    [ -f "$bake" ] || continue
    name=$(basename "$bake" .bin)
    BAKE_ARGS+=("--bake" "${name}=${bake}:clamp")
done
BAKE_ARGS+=("--out" "${RAW_OUT}")
"${QSWEEP_BIN}" "${BAKE_ARGS[@]}"
echo "wrote ${RAW_OUT}"
echo

echo "=== Phase 2: bake_verdict (Mohammadi panel) on each V8 bake ==="
for bake in "${V8_DIR}"/cc4v8_*.bin; do
    [ -f "$bake" ] || continue
    name=$(basename "$bake" .bin)
    out_md="${V8_DIR}/verdicts/${name}.md"
    "${VERDICT_BIN}" --bake "${bake}" --output "${out_md}" 2>&1 | tail -3 || echo "verdict failed for ${name}"
done

echo
echo "=== Phase 3: cross-codec T=63 consistency (n=20 images × 4 codecs) ==="
TOOL=/home/lilith/work/zen/zensim/target/release/predict_features_with_bake
ZEN_METRICS=/home/lilith/work/zen/zenmetrics/target/release/zenmetrics
CONSISTENCY=/home/lilith/work/zen/zensim/scripts/v_next/cross_codec_consistency.py
T63_DIR="${V8_DIR}/cross_codec_t63"
mkdir -p "${T63_DIR}"
for bake in "${V8_DIR}"/cc4v8_*.bin; do
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
python3 scripts/v_next/cross_codec_pjnd_check.py v8 "${V8_DIR}" || echo "v8 pjnd check failed"

echo
echo "=== Phase 5: multi-band cross-codec consistency check (V8 gate, 4 bands) ==="
python3 /home/lilith/work/zen/zensim/scripts/v_next/eval_v8_multi_band_check.py "${V8_DIR}" || echo "v8 multi-band check failed"

echo
echo "All eval phases complete. See:"
echo "  ${RAW_OUT}                                       (qsweep mono/tied/range/band-rmse)"
echo "  ${V8_DIR}/verdicts/*.md                          (Mohammadi panel per bake)"
echo "  ${V8_DIR}/cross_codec_t63/*.tsv                  (cross-codec T=63 raw)"
echo "  ${V8_DIR}/v8_pjnd_check.md                       (V4-style PJND score std)"
echo "  ${V8_DIR}/v8_multi_band_check.md                 (4-band parity gate)"
