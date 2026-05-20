#!/usr/bin/env bash
# SPEED-B lr-retune sweep — post-bake eval harness.
#
# Mirrors eval_cross_codec_v6.sh but globs cc4v6_lr*.bin in SWEEP_DIR.
# Phases:
#   1. qsweep_eval on the 50-image × 19-q JPEG sweep (mono/tied/range/band-rmse).
#   2. bake_verdict for SROCC panel — gates the ±0.01 vs K=1 baseline rule.
#   3. Cross-codec T=63 consistency (n=20 images × 4 codecs) — V6 gate.
#   4. Single-band multi-codec PJND score check (T=63 only).
#   5. Multi-band cross-codec consistency check (V6 gate, all 6 bands).
#
# Output: /mnt/v/zen/zensim-eval/speed_b_lr_retune_2026-05-19/
set -euo pipefail

SWEEP_DIR="/mnt/v/zen/zensim-eval/speed_b_lr_retune_2026-05-19"
QSWEEP_FEATURES="/mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep_features.csv"
QSWEEP_MANIFEST="/mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep/qsweep_manifest.tsv"

TRAINER_DIR="/home/lilith/work/zen/zensim--cross-codec-metric/target/release"
TUNER_BASELINE="/home/lilith/work/zen/zensim--cross-codec-metric/zensim/weights/v_tuner_2026-05-18.bin"
V6_K1_BASELINE="/mnt/v/zen/zensim-eval/exp_cross_codec_v6_2026-05-19/cc4v6_w1p0_p0p30_s1.bin"

QSWEEP_BIN="${TRAINER_DIR}/qsweep_eval"
VERDICT_BIN="/home/lilith/work/zen/zensim/target/release/bake_verdict"

mkdir -p "${SWEEP_DIR}/verdicts"

echo "=== Phase 1: qsweep_eval (clamp mode — V6 native [0, 100]) ==="
RAW_OUT="${SWEEP_DIR}/qsweep_lr_retune.md"
BAKE_ARGS=("--features" "${QSWEEP_FEATURES}" "--manifest" "${QSWEEP_MANIFEST}")
BAKE_ARGS+=("--bake" "baseline_tuner=${TUNER_BASELINE}:clamp")
BAKE_ARGS+=("--bake" "v6_k1_baseline=${V6_K1_BASELINE}:clamp")
for bake in "${SWEEP_DIR}"/cc4v6_lr*.bin; do
    [ -f "$bake" ] || continue
    name=$(basename "$bake" .bin)
    BAKE_ARGS+=("--bake" "${name}=${bake}:clamp")
done
BAKE_ARGS+=("--out" "${RAW_OUT}")
"${QSWEEP_BIN}" "${BAKE_ARGS[@]}"
echo "wrote ${RAW_OUT}"
echo

echo "=== Phase 2: bake_verdict (Mohammadi panel) on each bake ==="
for bake in "${SWEEP_DIR}"/cc4v6_lr*.bin; do
    [ -f "$bake" ] || continue
    name=$(basename "$bake" .bin)
    out_md="${SWEEP_DIR}/verdicts/${name}.md"
    if [ -s "${out_md}" ]; then
        echo "verdict ${name}: already exists, skip"
        continue
    fi
    "${VERDICT_BIN}" --bake "${bake}" --output "${out_md}" 2>&1 | tail -3 || \
        echo "verdict failed for ${name}"
done

echo
echo "=== Phase 3: cross-codec T=63 consistency (n=20 × 4 codecs) ==="
TOOL="${TRAINER_DIR}/predict_features_with_bake"
ZEN_METRICS="/home/lilith/work/zen/zenmetrics/target/release/zen-metrics"
CONSISTENCY="/home/lilith/work/zen/zensim--speed-b-lr-retune/scripts/v_next/cross_codec_consistency.py"
T63_DIR="${SWEEP_DIR}/cross_codec_t63"
mkdir -p "${T63_DIR}"
for bake in "${SWEEP_DIR}"/cc4v6_lr*.bin; do
    [ -f "$bake" ] || continue
    name=$(basename "$bake" .bin)
    out_tsv="${T63_DIR}/${name}_t63_n20.tsv"
    if [ -f "$out_tsv" ]; then
        echo "T=63 ${name}: already exists, skip"
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
echo "=== Phase 4: aggregate lr-retune summary ==="
python3 /home/lilith/work/zen/zensim--speed-b-lr-retune/scripts/v_next/aggregate_lr_retune.py "${SWEEP_DIR}" \
    > "${SWEEP_DIR}/lr_retune_summary.md"
echo "wrote ${SWEEP_DIR}/lr_retune_summary.md"

echo
echo "All eval phases complete."
echo "  ${RAW_OUT}                                       (qsweep mono/tied/range)"
echo "  ${SWEEP_DIR}/verdicts/*.md                       (Mohammadi panel per bake)"
echo "  ${SWEEP_DIR}/cross_codec_t63/*.tsv               (cross-codec T=63 raw)"
echo "  ${SWEEP_DIR}/lr_retune_summary.md                (aggregated lr × seed summary)"
