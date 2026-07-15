#!/usr/bin/env bash
# EXP-CROSS-CODEC-V9 master driver — runs the full pipeline:
# 1. Train 3 seeds (skipped if bakes exist).
# 2. Calibrate each via PCHIP spline.
# 3. Eval each calibrated bake (qsweep + Mohammadi + V9 gates).
# 4. Pick median bake by CID22 SROCC + render ship decision.

set -euo pipefail

OUT_DIR="/mnt/v/zen/zensim-eval/exp_cross_codec_v9_2026-05-20"
HERE="/home/lilith/work/zen/zensim"
mkdir -p "${OUT_DIR}"

echo "=== Step 1: train 3 seeds (parallel) ==="
for SEED in 1 2 3; do
    BAKE="${OUT_DIR}/cc4v9_s${SEED}.bin"
    if [ -f "${BAKE}" ]; then
        echo "  seed=${SEED} already trained: ${BAKE}"
    else
        bash "${HERE}/scripts/v_next/run_cross_codec_v9_seed.sh" "${SEED}" &
    fi
done
wait

echo "=== Step 2: calibrate (PCHIP spline) ==="
for SEED in 1 2 3; do
    BAKE="${OUT_DIR}/cc4v9_s${SEED}.bin"
    CAL="${OUT_DIR}/cc4v9_s${SEED}_calibrated.bin"
    SPLINE_CSV="${OUT_DIR}/cc4v9_s${SEED}.spline.csv"
    python3 "${HERE}/scripts/v_next/calibrate_v9_spline.py" \
        --bake "${BAKE}" --out "${CAL}" --spline-csv "${SPLINE_CSV}" \
        2>&1 | tee "${OUT_DIR}/cc4v9_s${SEED}_calibrate.log"
done

echo "=== Step 3: evaluate each calibrated bake ==="
for SEED in 1 2 3; do
    CAL="${OUT_DIR}/cc4v9_s${SEED}_calibrated.bin"
    VERDICT="${OUT_DIR}/cc4v9_s${SEED}_verdict.md"
    PANEL="${OUT_DIR}/cc4v9_s${SEED}_panel.md"
    python3 "${HERE}/scripts/v_next/eval_v9_bake.py" \
        --bake "${CAL}" --out-md "${VERDICT}" --verdict-md "${PANEL}" \
        2>&1 | tee "${OUT_DIR}/cc4v9_s${SEED}_eval.log"
done

echo "=== Step 4: aggregate + median selection ==="
python3 - <<'PYEOF'
import re
from pathlib import Path
OUT_DIR = Path("/mnt/v/zen/zensim-eval/exp_cross_codec_v9_2026-05-20")
results = []
for seed in [1, 2, 3]:
    verdict = OUT_DIR / f"cc4v9_s{seed}_verdict.md"
    if not verdict.exists():
        print(f"missing verdict for seed={seed}")
        continue
    text = verdict.read_text()
    # Extract gate verdicts + SROCC etc.
    cid22 = None
    konjnd = None
    n_pass_match = re.search(r"\*\*Total: (\d+)/(\d+) gates pass\*\*", text)
    n_pass = int(n_pass_match.group(1)) if n_pass_match else 0
    n_total = int(n_pass_match.group(2)) if n_pass_match else 0
    cid22_match = re.search(r"\| CID22 \| ([\d.]+) \|", text)
    if cid22_match: cid22 = float(cid22_match.group(1))
    konjnd_match = re.search(r"\| KonJND-1k \| ([\d.]+) \|", text)
    if konjnd_match: konjnd = float(konjnd_match.group(1))
    results.append({"seed": seed, "n_pass": n_pass, "n_total": n_total,
                    "cid22": cid22, "konjnd": konjnd})
# Sort by CID22 SROCC
results.sort(key=lambda r: r["cid22"] or 0.0)
print("\nPer-seed summary (sorted by CID22 SROCC):")
for r in results:
    print(f"  seed={r['seed']} gates={r['n_pass']}/{r['n_total']} "
          f"CID22={r['cid22']} KonJND={r['konjnd']}")
if len(results) == 3:
    median = results[1]  # middle of 3
    print(f"\nMedian by CID22: seed={median['seed']} "
          f"gates={median['n_pass']}/{median['n_total']} "
          f"CID22={median['cid22']}")
PYEOF

echo "=== DONE ==="
echo "Calibrated bakes: ${OUT_DIR}/cc4v9_s{1,2,3}_calibrated.bin"
echo "Verdicts: ${OUT_DIR}/cc4v9_s{1,2,3}_verdict.md"
