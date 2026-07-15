#!/usr/bin/env bash
# EXP-CROSS-CODEC-V3 evaluation harness (2026-05-19).
#
# Runs the full Tuner-trail gate measurement on all 6 V3 bakes:
#   1. qsweep_eval on the 50-image × 19-q JPEG sweep → mono / tied / range.
#   2. Affine-calibrate each bake to span [5, 95] score units on the sweep.
#   3. Re-run qsweep_eval on the calibrated bakes to confirm range.
#   4. bake_verdict on KADID/TID/CID22/KonJND/AIC-3 for SROCC panel context.
#
# Output: /mnt/v/zen/zensim-eval/exp_cross_codec_v3_2026-05-19/eval_summary.md
set -euo pipefail

V3_DIR="/mnt/v/zen/zensim-eval/exp_cross_codec_v3_2026-05-19"
QSWEEP_FEATURES="/mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep_features.csv"
QSWEEP_MANIFEST="/mnt/v/output/zensim/exp_tuner_2026-05-18/qsweep/qsweep_manifest.tsv"
TUNER_BASELINE="/home/lilith/work/zen/zensim/zensim/weights/v_tuner_2026-05-18.bin"
QSWEEP_BIN="/home/lilith/work/zen/zensim/target/release/qsweep_eval"
AFFINE_PY="/home/lilith/work/zen/zensim/scripts/v_next/affine_per_sample_alpha.py"

mkdir -p "${V3_DIR}/calibrated"

echo "=== Phase 1: Raw qsweep_eval on all V3 bakes ==="
RAW_OUT="${V3_DIR}/qsweep_raw.md"
BAKE_ARGS=("--features" "${QSWEEP_FEATURES}" "--manifest" "${QSWEEP_MANIFEST}")
BAKE_ARGS+=("--bake" "baseline_tuner=${TUNER_BASELINE}:clamp")
for bake in "${V3_DIR}"/cc4v3_*.bin; do
    [ -f "$bake" ] || continue
    name=$(basename "$bake" .bin)
    BAKE_ARGS+=("--bake" "${name}=${bake}:raw")  # raw mode so we see real range
done
BAKE_ARGS+=("--out" "${RAW_OUT}")
"${QSWEEP_BIN}" "${BAKE_ARGS[@]}"
echo "wrote ${RAW_OUT}"
echo

echo "=== Phase 2: Extract per-bake q5/q95 medians + fit affine ==="
python3 <<'PYEOF'
import re, json, sys, subprocess, os
from pathlib import Path

raw_md = Path("/mnt/v/zen/zensim-eval/exp_cross_codec_v3_2026-05-19/qsweep_raw.md")
text = raw_md.read_text()

# qsweep_eval emits a per-bake per-q median table. Parse it.
# The format we expect is something like:
#   ## <bake_name>
#   ...
#   | q | p25 | p50 | p75 | n |
#   |---:|---:|---:|---:|---:|
#   | 5 | ... | M_q5 | ... | 50 |
#   ...
#   | 95 | ... | M_q95 | ... | 50 |

# Generic parser: find sections by `### <bake>` or `## <bake>` headers, then
# parse following tables to extract q -> p50.
sections = re.split(r'^#+\s+(\S+)', text, flags=re.MULTILINE)
# sections = [pre, name1, body1, name2, body2, ...]

if len(sections) < 3:
    print(f"[parse] only {len(sections)} sections in raw_md", file=sys.stderr)
    print(text[:500], file=sys.stderr)
    sys.exit(0)

results = {}
for i in range(1, len(sections), 2):
    name = sections[i].strip()
    body = sections[i+1] if i+1 < len(sections) else ""
    # Format from qsweep_eval:
    # | q | n | min | p25 | median | p75 | max |
    # | 5 | 50 | a | b | M | c | d |
    medians = {}
    for line in body.splitlines():
        # Need exactly 8 pipes (7 cells); first cell is q, fifth is median.
        if line.count('|') < 7:
            continue
        cells = [c.strip() for c in line.strip().strip('|').split('|')]
        if len(cells) < 7:
            continue
        try:
            q = int(cells[0])
        except ValueError:
            continue
        try:
            p50 = float(cells[4])
        except (ValueError, IndexError):
            continue
        medians[q] = p50
    if 5 in medians and 95 in medians:
        results[name] = medians

print(f"# Affine calibration plan", file=sys.stderr)
print(f"  bake | q=5 median | q=95 median | α | β", file=sys.stderr)
print(f"  ---  | --- | --- | --- | ---", file=sys.stderr)
out_plan = {}
for name, med in results.items():
    m5 = med[5]
    m95 = med[95]
    # y' = α + β · y, fit (m5 → 5), (m95 → 95)
    if abs(m95 - m5) < 1e-6:
        beta = 0.0
        alpha = 50.0
    else:
        beta = (95.0 - 5.0) / (m95 - m5)
        alpha = 5.0 - beta * m5
    out_plan[name] = {"q5": m5, "q95": m95, "alpha": alpha, "beta": beta}
    print(f"  {name} | {m5:.3f} | {m95:.3f} | {alpha:.3f} | {beta:.3f}", file=sys.stderr)

# Write plan as json for downstream calibration step
Path("/mnt/v/zen/zensim-eval/exp_cross_codec_v3_2026-05-19/affine_plan.json").write_text(json.dumps(out_plan, indent=2))
PYEOF

echo
echo "=== Phase 3: Apply affine to each V3 bake ==="
python3 <<'PYEOF'
import json, subprocess, sys
from pathlib import Path

plan = json.loads(Path("/mnt/v/zen/zensim-eval/exp_cross_codec_v3_2026-05-19/affine_plan.json").read_text())
in_dir = Path("/mnt/v/zen/zensim-eval/exp_cross_codec_v3_2026-05-19")
out_dir = in_dir / "calibrated"
out_dir.mkdir(exist_ok=True)
for name, cfg in plan.items():
    if name == "baseline_tuner":
        continue
    bake = in_dir / f"{name}.bin"
    out_bake = out_dir / f"{name}_calibrated.bin"
    if not bake.exists():
        print(f"  skip {name} (no bake)", file=sys.stderr)
        continue
    print(f"  calibrating {name}: α={cfg['alpha']:.3f} β={cfg['beta']:.3f}", file=sys.stderr)
    subprocess.run([
        "python3",
        "/home/lilith/work/zen/zensim/scripts/v_next/affine_per_sample_alpha.py",
        "--in-bake", str(bake),
        "--out-bake", str(out_bake),
        "--alpha", f"{cfg['alpha']:.6f}",
        "--beta", f"{cfg['beta']:.6f}",
        "--n-hidden", "128",
    ], check=True)
PYEOF

echo
echo "=== Phase 4: qsweep_eval on calibrated bakes ==="
CALIB_OUT="${V3_DIR}/qsweep_calibrated.md"
CALIB_ARGS=("--features" "${QSWEEP_FEATURES}" "--manifest" "${QSWEEP_MANIFEST}")
CALIB_ARGS+=("--bake" "baseline_tuner=${TUNER_BASELINE}:clamp")
for bake in "${V3_DIR}/calibrated"/*.bin; do
    [ -f "$bake" ] || continue
    name=$(basename "$bake" .bin)
    CALIB_ARGS+=("--bake" "${name}=${bake}:clamp")
done
CALIB_ARGS+=("--out" "${CALIB_OUT}")
"${QSWEEP_BIN}" "${CALIB_ARGS[@]}"
echo "wrote ${CALIB_OUT}"

echo
echo "=== Phase 5: bake_verdict (Mohammadi panel) ==="
mkdir -p "${V3_DIR}/verdicts"
for bake in "${V3_DIR}/calibrated"/*.bin; do
    [ -f "$bake" ] || continue
    name=$(basename "$bake" .bin)
    out_md="${V3_DIR}/verdicts/${name}.md"
    /home/lilith/work/zen/zensim/target/release/bake_verdict \
        --bake "${bake}" \
        --output "${out_md}" 2>&1 | tail -3 || echo "verdict failed for ${name}"
done

echo
echo "All eval phases complete. See:"
echo "  ${V3_DIR}/qsweep_raw.md       (raw, pre-affine)"
echo "  ${V3_DIR}/qsweep_calibrated.md (post-affine)"
echo "  ${V3_DIR}/verdicts/*.md       (Mohammadi panel per bake)"
echo "  ${V3_DIR}/affine_plan.json    (α/β per bake)"
