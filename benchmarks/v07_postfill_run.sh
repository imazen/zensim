#!/usr/bin/env bash
# Post-e1-fill retrain + 4-metric eval harness.
#
# Run this once the zenjpeg-420-e1 q-grid fill is complete and the
# extended synthetic CSV exists. Trains V0_7 (V0_6 dct_hf + sampler
# bias on the extended dataset), plus a no-bias control to isolate
# the bias impact, evaluates both on KADID/TID/CID22/KonJND, scores
# synthetic for smoothness comparison, and renders the updated
# 4-metric report.
#
# Usage:  bash v07_postfill_run.sh [<extended_csv>]
# Default CSV: /mnt/v/output/zensim/synthetic-v2/training_safe_synthetic_extended.csv
#
# Idempotent: feature cache is auto-detected on subsequent runs, and
# every artifact is timestamped so re-runs don't clobber.

set -euo pipefail

EXTENDED_CSV="${1:-/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic_extended.csv}"
TS=$(date -u +%Y%m%dT%H%M%S)

ROOT=${ROOT:-/home/lilith/work/zen/zensim}
VALIDATE=$ROOT/target/release/zensim-validate
EVAL=$ROOT/target/release/examples/dataset_metric_baseline
SCORE=$ROOT/target/release/examples/score_synthetic_with_mlp
RENDER=$ROOT/benchmarks/render_4metric_report.py
RANGE_PY=$ROOT/benchmarks/range_analysis.py
TSV=/mnt/v/output/zensim/synthetic-v2/zenanalyze_union_v1.tsv
RUNS=/mnt/v/output/zensim/synthetic-v2/runs
BENCH=$ROOT/benchmarks
KADID=/mnt/v/dataset/kadid10k
TID=/mnt/v/dataset/tid2013
CID22=/mnt/v/dataset/cid22/CID22_validation_set
KONJND=/mnt/v/datasets/KonJND-1k/KonJND-1k
DCT_HF_FEATS=dct_compressibility_y,dct_compressibility_uv,high_freq_energy_ratio

# Sanity checks ----------------------------------------------------------------
[ -f "$EXTENDED_CSV" ] || {
  echo "ERROR: extended CSV not found: $EXTENDED_CSV"
  echo "       Run the e1 fill generator first (see"
  echo "       benchmarks/zenjpeg_e1_fill_plan_2026-05-01.md)."
  exit 1
}
[ -x "$VALIDATE" ] || { cd $ROOT && cargo build --release -p zensim-validate; }
[ -x "$EVAL" ]     || { cd $ROOT && cargo build --release -p zensim-bench --example dataset_metric_baseline; }
[ -x "$SCORE" ]    || { cd $ROOT && cargo build --release -p zensim-bench --example score_synthetic_with_mlp; }
[ -f "$TSV" ]      || { echo "ERROR: zenanalyze TSV not found: $TSV"; exit 1; }

n_pairs=$(($(wc -l < "$EXTENDED_CSV") - 1))
echo "[$(date +%H:%M:%S)] Extended CSV: $EXTENDED_CSV ($n_pairs pairs)"

# 1. Train V0_7 dct_hf WITH sampler bias ---------------------------------------
V07_NAME=v07_dct_hf_lowband050_${TS}
V07_BAKE=$RUNS/${V07_NAME}.bin
V07_LOG=/tmp/${V07_NAME}.log
echo "[$(date +%H:%M:%S)] Training V0_7 (dct_hf + low-band oversample 0.5)"
echo "  bake: $V07_BAKE"
echo "  log:  $V07_LOG"
"$VALIDATE" \
  --dataset "$EXTENDED_CSV" \
  --format synthetic \
  --target-metric gpu-ssim2 \
  --feature-tier peaks \
  --train --algorithm mlp --mlp-hidden 64 --mlp-epochs 200 \
  --mlp-magnitude-match-lambda 0.001 --mlp-magnitude-match-alpha 30.0 \
  --mlp-zenanalyze-tsv "$TSV" \
  --mlp-zenanalyze-features "$DCT_HF_FEATS" \
  --mlp-low-band-oversample 0.5 \
  --mlp-output "$V07_BAKE" \
  --also kadid10k:$KADID,tid2013:$TID,konjnd1k:$KONJND \
  --mlp-validation-policy min \
  > "$V07_LOG" 2>&1 || { echo "FAILED: V0_7 train"; tail -20 "$V07_LOG"; exit 2; }
v07_best=$(grep "best validation mean" "$V07_LOG" | awk '{print $NF}')
echo "  V0_7 best_val_min_srocc = $v07_best"

# 2. Train V0_7-control (no sampler bias) --------------------------------------
V07C_NAME=v07_dct_hf_nobias_${TS}
V07C_BAKE=$RUNS/${V07C_NAME}.bin
V07C_LOG=/tmp/${V07C_NAME}.log
echo "[$(date +%H:%M:%S)] Training V0_7-control (dct_hf, no sampler bias)"
"$VALIDATE" \
  --dataset "$EXTENDED_CSV" \
  --format synthetic \
  --target-metric gpu-ssim2 \
  --feature-tier peaks \
  --train --algorithm mlp --mlp-hidden 64 --mlp-epochs 200 \
  --mlp-magnitude-match-lambda 0.001 --mlp-magnitude-match-alpha 30.0 \
  --mlp-zenanalyze-tsv "$TSV" \
  --mlp-zenanalyze-features "$DCT_HF_FEATS" \
  --mlp-output "$V07C_BAKE" \
  --also kadid10k:$KADID,tid2013:$TID,konjnd1k:$KONJND \
  --mlp-validation-policy min \
  > "$V07C_LOG" 2>&1 || { echo "FAILED: V0_7-control train"; tail -20 "$V07C_LOG"; exit 3; }
v07c_best=$(grep "best validation mean" "$V07C_LOG" | awk '{print $NF}')
echo "  V0_7-control best_val_min_srocc = $v07c_best"

# 3. Eval V0_7 + V0_7-control on holdouts (in parallel) ------------------------
EVAL_DIR=/tmp/eval_v07_${TS}
mkdir -p "$EVAL_DIR"
echo "[$(date +%H:%M:%S)] Eval V0_7 + V0_7-control on KADID/TID/CID22/KonJND (parallel)"
"$EVAL" \
  --kadid $KADID --tid $TID --cid22 $CID22 --konjnd $KONJND \
  --v04-bake "$V07_BAKE" \
  --zenanalyze-tsv "$TSV" --zenanalyze-features "$DCT_HF_FEATS" \
  --max-pairs 1500 --per-pair-output "$EVAL_DIR/v07_perpair.csv" \
  > "$EVAL_DIR/v07.log" 2>&1 &
EVAL_PID_A=$!
"$EVAL" \
  --kadid $KADID --tid $TID --cid22 $CID22 --konjnd $KONJND \
  --v04-bake "$V07C_BAKE" \
  --zenanalyze-tsv "$TSV" --zenanalyze-features "$DCT_HF_FEATS" \
  --max-pairs 1500 --per-pair-output "$EVAL_DIR/v07c_perpair.csv" \
  > "$EVAL_DIR/v07c.log" 2>&1 &
EVAL_PID_B=$!
wait $EVAL_PID_A $EVAL_PID_B
echo "  evals done → $EVAL_DIR"

# 4. Score synthetic for smoothness comparison ---------------------------------
SS_DIR=/tmp/synth_scored_v07_${TS}
mkdir -p "$SS_DIR"
# Auto-detect feature cache for the extended CSV.
EXT_CACHE=$(ls -1t ${EXTENDED_CSV}.features.*.bin 2>/dev/null | head -1 || true)
if [ -z "$EXT_CACHE" ]; then
  echo "  WARNING: no feature cache for $EXTENDED_CSV; skipping smoothness scoring."
else
  echo "[$(date +%H:%M:%S)] Scoring synthetic for smoothness comparison"
  echo "  cache: $EXT_CACHE"
  "$SCORE" --csv "$EXTENDED_CSV" --features-cache "$EXT_CACHE" \
    --bake "$V07_BAKE" \
    --zenanalyze-tsv "$TSV" --zenanalyze-features "$DCT_HF_FEATS" \
    --output "$SS_DIR/v07.csv" 2>&1 | tail -2
  "$SCORE" --csv "$EXTENDED_CSV" --features-cache "$EXT_CACHE" \
    --bake "$V07C_BAKE" \
    --zenanalyze-tsv "$TSV" --zenanalyze-features "$DCT_HF_FEATS" \
    --output "$SS_DIR/v07c.csv" 2>&1 | tail -2
fi

# 5. Render comparison report --------------------------------------------------
REPORT=$BENCH/v07_postfill_${TS}.md
echo "[$(date +%H:%M:%S)] Rendering comparison report → $REPORT"
python3 - "$EVAL_DIR/v07_perpair.csv" "$EVAL_DIR/v07c_perpair.csv" \
       "$BENCH/v04smooth_perpair_2026-05-01.csv" \
       "$BENCH/v05_perpair_2026-05-01.csv" \
       "$BENCH/v06_dct_hf_perpair_2026-05-01.csv" \
       "$REPORT" <<'PY'
import sys, csv
import numpy as np
from scipy import stats
from collections import defaultdict
from pathlib import Path

v07, v07c, v04s, v05, v06, out = sys.argv[1:7]
files = [
    ("V0_7 dct_hf + low-band-bias 0.5 (extended dataset)", v07),
    ("V0_7-control dct_hf no-bias (extended dataset)", v07c),
    ("V0_6 dct_hf (original 218k)", v06),
    ("V0_5 (original 218k)", v05),
    ("V0_4-smooth (original 218k)", v04s),
]

def load(p):
    by_ds = defaultdict(lambda: {"h":[], "v04":[], "ssim2":[], "butter":[]})
    with open(p) as f:
        for r in csv.DictReader(f):
            try:
                vs = [float(r[k]) for k in ("human_score","v04_distance","fast_ssim2_score","butter_3norm")]
            except Exception: continue
            if any(not np.isfinite(x) for x in vs): continue
            by_ds[r["dataset"]]["h"].append(vs[0])
            by_ds[r["dataset"]]["v04"].append(vs[1])
            by_ds[r["dataset"]]["ssim2"].append(vs[2])
            by_ds[r["dataset"]]["butter"].append(vs[3])
    return by_ds

def srocc(a,b):
    if len(a)<3: return float("nan")
    return abs(stats.spearmanr(a,b).correlation)

BANDS = [
    ("≤ 0",   lambda s: s < 0),
    ("0-25",  lambda s: 0 <= s < 25),
    ("25-40", lambda s: 25 <= s < 40),
    ("40-60", lambda s: 40 <= s < 60),
    ("60-75", lambda s: 60 <= s < 75),
    ("75-90", lambda s: 75 <= s < 90),
    ("≥ 90",  lambda s: s >= 90),
]

with open(out, "w") as f:
    f.write("# V0_7 post-fill 4-metric comparison\n\n")
    f.write("V0_7 = V0_6 dct_hf retrained on the extended dataset (218k base + ~140k\n")
    f.write("zenjpeg-420-e1 fills covering SSIM2 0-90), with `--mlp-low-band-oversample\n")
    f.write("0.5` so half of every batch comes from the SSIM2 [0, 60] band.\n\n")
    f.write("V0_7-control is the same training data and architecture without sampler\n")
    f.write("bias — isolates the data fill from the bias change.\n\n")

    f.write("## Per-(metric, dataset) |SROCC| vs human-MOS\n\n")
    all_ds = ["KADIK10k", "TID2013", "CID22"]
    f.write("| metric | " + " | ".join(all_ds) + " |\n")
    f.write("|---|" + "--:|" * len(all_ds) + "\n")
    for label, p in files:
        d = load(p)
        cells = []
        for ds in all_ds:
            if ds in d:
                v = srocc(d[ds]["v04"], d[ds]["h"])
                cells.append(f"{v:.4f}")
            else:
                cells.append("—")
        f.write(f"| {label} | " + " | ".join(cells) + " |\n")
    # SSIM2 reference
    d0 = load(files[0][1])
    f.write("| ref SSIMULACRA 2 | " + " | ".join(
        f"{srocc(d0[ds]['ssim2'], d0[ds]['h']):.4f}" if ds in d0 else "—"
        for ds in all_ds) + " |\n\n")

    f.write("## Per-band |SROCC| vs SSIM2 (synthetic ground truth)\n\n")
    f.write("Run `range_analysis.py` against the new bakes for the synthetic-side\n")
    f.write("breakdown; the human-MOS bands above are the user-facing test.\n\n")

    f.write("## V0_7 vs V0_6 dct_hf delta (human-MOS)\n\n")
    da = load(v07); db = load(v06)
    f.write("| dataset | V0_6 | V0_7 | Δ |\n")
    f.write("|---|--:|--:|--:|\n")
    for ds in all_ds:
        if ds in da and ds in db:
            a = srocc(da[ds]["v04"], da[ds]["h"])
            b = srocc(db[ds]["v04"], db[ds]["h"])
            sign = "+" if a-b > 0 else ""
            f.write(f"| {ds} | {b:.4f} | {a:.4f} | {sign}{a-b:.4f} |\n")
    f.write("\nPositive Δ = the e1 fill + sampler bias improved KADID/TID/CID22.\n")
    f.write("Negative Δ would indicate over-fitting to the new low-q pairs at the\n")
    f.write("expense of upper-band performance — re-evaluate sampler bias fraction.\n")
PY

echo ""
echo "[$(date +%H:%M:%S)] DONE"
echo "Report: $REPORT"
echo "V0_7 bake: $V07_BAKE  (val SROCC $v07_best)"
echo "V0_7-control bake: $V07C_BAKE  (val SROCC $v07c_best)"
echo "Eval CSVs: $EVAL_DIR"
echo "Smoothness CSVs: $SS_DIR"
echo ""
echo "Next:"
echo "  python3 $RANGE_PY  # update per-band breakdown"
echo "  jj describe + jj git push --change @"
