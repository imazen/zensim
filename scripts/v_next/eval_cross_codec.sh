#!/usr/bin/env bash
# EXP-CROSS-CODEC evaluation harness — any experiment generation.
#
# Replaces eval_cross_codec_v{4,4b,5,6,7,8}.sh (2026-07-15), 505 lines. They
# were one harness copied six times; the only real axis is which generation and
# whether the multi-band phase runs (v4/v4b predate it).
#
# What the copying cost, measured on this family:
#
#   * eval_cross_codec_v4b.sh titled itself "EXP-CROSS-CODEC-V4" and printed
#     "Phase 1: qsweep_eval (clamp mode — V4 native)" and "Phase 2 ... on each
#     V4 bake" while evaluating V4B.
#   * eval_cross_codec_v7.sh printed "V6" in THREE phase banners while
#     evaluating V7, and carried v6's "increased anchor pressure vs V5's 0.05"
#     rationale in its header, which is not what v7 changed.
#   Each sed updated the glob and the directory; none updated the sentences a
#   human reads while the thing runs.
#
#   * Phase 3 was factored out to run_cross_codec_v4_consistency.sh in v4/v4b,
#     then INLINED — identically — into v5, v6, v7 and v8. A duplication inside
#     the duplication.
#
#   * v4/v4b's Phase 3 was worse still: it copied a driver to /tmp and sed'd it
#     into the next generation AT RUNTIME, guarded by `if [ -x ]` over a file
#     that had not existed for months. So the phase silently skipped. Fixed
#     separately; see those scripts' history.
#
# Failure handling is deliberate. Every phase used to end in `|| echo "...
# failed"`, which swallows the error despite `set -e`, and then the script
# printed "All eval phases complete" unconditionally — a harness that reports
# success while phases fail is worse than one that crashes. Phases still
# continue on error (you want the rest of the panel), but failures are counted,
# named at the end, and exit nonzero.
#
# Usage:
#   scripts/v_next/eval_cross_codec.sh v6
#   scripts/v_next/eval_cross_codec.sh v4b --dir /some/other/exp/dir
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ZEN="$(dirname "${REPO}")"

EXP="${1:-}"
if [ -z "${EXP}" ]; then
    echo "usage: $(basename "$0") <exp: v4|v4b|v5|v6|v7|v8> [--dir DIR]" >&2
    exit 2
fi
shift

EXP_DIR="/mnt/v/zen/zensim-eval/exp_cross_codec_${EXP}_2026-05-19"
while [ $# -gt 0 ]; do
    case "$1" in
        --dir) EXP_DIR="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# The multi-band phase postdates v4/v4b — those generations have no multi-band
# anchor set, so running it would fail rather than tell you anything.
case "${EXP}" in
    v4|v4b) RUN_MULTI_BAND=0 ;;
    v5|v6|v7|v8) RUN_MULTI_BAND=1 ;;
    *) echo "unknown generation ${EXP}; add it to the case above" >&2; exit 2 ;;
esac

# v5 reuses v4b's PJND bake-glob (symlinked bakes) — see
# benchmarks/speed_a_hyperparam_test_2026-05-19.md.
PJND_EXP="${EXP}"
[ "${EXP}" = "v5" ] && PJND_EXP="v4b"

QSWEEP_FEATURES="/mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep_features.csv"
QSWEEP_MANIFEST="/mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep/qsweep_manifest.tsv"
TUNER_BASELINE="${REPO}/zensim/weights/v_tuner_2026-05-18.bin"
QSWEEP_BIN="${REPO}/target/release/qsweep_eval"
VERDICT_BIN="${REPO}/target/release/bake_verdict"
PREDICT_BIN="${REPO}/target/release/predict_features_with_bake"
ZEN_METRICS="${ZEN}/zenmetrics/target/release/zenmetrics"
CONSISTENCY="${REPO}/scripts/v_next/cross_codec_consistency.py"

UP="$(echo "${EXP}" | tr '[:lower:]' '[:upper:]')"
BAKE_GLOB="cc4${EXP}_*.bin"
RAW_OUT="${EXP_DIR}/qsweep_${EXP}.md"
T63_DIR="${EXP_DIR}/cross_codec_t63"

FAILED=()
note_fail() { FAILED+=("$1"); echo "  FAILED: $1" >&2; }

mkdir -p "${EXP_DIR}/verdicts" "${T63_DIR}"

shopt -s nullglob
BAKES=("${EXP_DIR}"/${BAKE_GLOB})
shopt -u nullglob
if [ ${#BAKES[@]} -eq 0 ]; then
    echo "no ${BAKE_GLOB} under ${EXP_DIR}" >&2
    exit 1
fi
echo "${UP}: ${#BAKES[@]} bakes under ${EXP_DIR}"
echo

echo "=== Phase 1: qsweep_eval (clamp mode — ${UP} native [0, 100]) ==="
BAKE_ARGS=("--features" "${QSWEEP_FEATURES}" "--manifest" "${QSWEEP_MANIFEST}")
BAKE_ARGS+=("--bake" "baseline_tuner=${TUNER_BASELINE}:clamp")
for bake in "${BAKES[@]}"; do
    BAKE_ARGS+=("--bake" "$(basename "$bake" .bin)=${bake}:clamp")
done
BAKE_ARGS+=("--out" "${RAW_OUT}")
"${QSWEEP_BIN}" "${BAKE_ARGS[@]}" || note_fail "phase1-qsweep"
echo "wrote ${RAW_OUT}"
echo

echo "=== Phase 2: bake_verdict (Mohammadi panel) on each ${UP} bake ==="
for bake in "${BAKES[@]}"; do
    name=$(basename "$bake" .bin)
    "${VERDICT_BIN}" --bake "${bake}" --output "${EXP_DIR}/verdicts/${name}.md" 2>&1 \
        | tail -3 || note_fail "phase2-verdict:${name}"
done

echo
echo "=== Phase 3: cross-codec T=63 consistency (n=20 images × 4 codecs) ==="
for bake in "${BAKES[@]}"; do
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
        --predict-tool "${PREDICT_BIN}" \
        --zen-metrics "${ZEN_METRICS}" \
        --out "${out_tsv}" 2>&1 | tail -8 || note_fail "phase3-consistency:${name}"
done

echo
echo "=== Phase 4: single-band multi-codec PJND score check (T=63) ==="
python3 "${REPO}/scripts/v_next/cross_codec_pjnd_check.py" "${PJND_EXP}" "${EXP_DIR}" \
    || note_fail "phase4-pjnd"

if [ "${RUN_MULTI_BAND}" -eq 1 ]; then
    echo
    echo "=== Phase 5: multi-band cross-codec consistency check (${UP} gate) ==="
    python3 "${REPO}/scripts/v_next/cross_codec_multi_band_check.py" "${EXP}" "${EXP_DIR}" \
        || note_fail "phase5-multi-band"
fi

echo
echo "Outputs:"
echo "  ${RAW_OUT}                          (qsweep mono/tied/range/band-rmse)"
echo "  ${EXP_DIR}/verdicts/*.md            (Mohammadi panel per bake)"
echo "  ${T63_DIR}/*.tsv                    (cross-codec T=63 raw)"
echo "  ${EXP_DIR}/${PJND_EXP}_pjnd_check.md  (PJND score std)"
[ "${RUN_MULTI_BAND}" -eq 1 ] && \
    echo "  ${EXP_DIR}/${EXP}_multi_band_check.md  (multi-band parity gate)"

echo
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All eval phases complete."
else
    echo "${#FAILED[@]} phase(s) FAILED — the outputs above are incomplete:" >&2
    printf '  %s\n' "${FAILED[@]}" >&2
    exit 1
fi
