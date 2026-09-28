#!/usr/bin/env bash
# Phase 2d Part D — THE single frozen CID22-B read (prereg §5.4).
# Runs exactly once, after every fit artefact under fits/ is frozen.
# Unseals B labels (pairs/cid22b_unsealed.tsv via unseal_cid22b.py —
# refuses to run twice), then emits: scores/*__on__cid22b.csv for each
# fitted variant + K1, comparator per-row scores (fast-ssim2 audit, B, D,
# R915 ensembles), then metrics tables + paired bootstrap into tables/.
set -uo pipefail
OUT=/mnt/v/output/zensim/dvifm-screen2d-2026-09-19
FIT="python3 $OUT/tools/fit_standalone.py"
BIN=/home/lilith/work/zen/zensim/zensim-bench/target/release/examples
VALBIN=/home/lilith/work/zen/zensim/target/release
W=/home/lilith/work/zen/zensim/zensim/weights
REC=/var/tmp/zensim-validation-2026-09-15/recovery
export ZENSIM_FORMULA_REV=3 ZENSIM_SAMPLE_DIGEST=1
export RAYON_NUM_THREADS=8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$OUT/scores" "$OUT/tables" "$OUT/logs"
fails=0

# -- 0. THE unseal (refuses if pairs/cid22b_unsealed.tsv already exists) --
python3 "$OUT/tools/unseal_cid22b.py" \
  || { echo "UNSEAL-FAIL"; exit 1; }
tsv=$OUT/pairs/cid22b_unsealed.tsv

# -- 1. fitted variants + K1 on cid22b block caches ----------------------
dom=cid22b
binf() { echo "$OUT/cache/cid22b_${1}.bin"; }
for src in cid22a tidkadid imazen26 pooled; do
  for var in luma native3 xyb-y; do
    fit=$OUT/fits/${src}_${var}.json
    [ -s "$fit" ] || { echo "MISSING FIT $src $var"; fails=$((fails+1)); continue; }
    out=$OUT/scores/${src}_${var}__on__cid22b.csv
    [ -s "$out" ] && continue
    case $var in
      luma)    caches="--cache ycbcr_y=$(binf ycbcr_y)";;
      xyb-y)   caches="--cache xyb_y=$(binf xyb_y)";;
      native3) caches="--cache ycbcr_y=$(binf ycbcr_y) \
--cache ycbcr_cb=$(binf ycbcr_cb) --cache ycbcr_cr=$(binf ycbcr_cr)";;
    esac
    $FIT score --fit "$fit" $caches --pairs "$tsv" --out "$out" \
      || { echo "SCORE-FAIL $src $var cid22b"; fails=$((fails+1)); }
  done
done
out=$OUT/scores/k1__on__cid22b.csv
[ -s "$out" ] || $FIT score --fit "$OUT/fits/k1_control.json" \
  --cache xyb_y="$(binf xyb_y)" --pairs "$tsv" --out "$out" \
  || { echo "K1-FAIL cid22b"; fails=$((fails+1)); }

# -- 2. fast-ssim2 audit channel (pixel path, public surface) ------------
# Fed the UNSEALED tsv: emitted features CSV carries real labels, and the
# audit jsonl records peer_ssim2 per row.
aud=$OUT/scores/cid22b_audit_ssim2.jsonl
afeat=$OUT/cache/cid22b_audit.features.csv
if [ ! -s "$aud" ]; then
  "$BIN/extract_features_372col" --full-986 \
    --corpus pairs-tsv --path "$tsv" \
    --audit-ssim2 --audit-jsonl "$aud" \
    --out "$afeat" --allow-failures 0 \
    > "$OUT/logs/cid22b_audit.log" 2>&1 \
    || { echo "AUDIT-FAIL"; fails=$((fails+1)); }
fi
python3 "$OUT/tools/rows_to_scores.py" --pairs "$tsv" \
  --audit-jsonl "$aud" --out "$OUT/scores/fastssim2__on__cid22b.csv" \
  || { echo "SSIM2-JOIN-FAIL"; fails=$((fails+1)); }

# -- 3. bake comparators on the w944 parquet ------------------------------
pq=$OUT/pairs/cid22b_w944.parquet
if [ ! -s "$pq" ]; then
  python3 "$OUT/tools/csv_to_parquet.py" "$afeat" "$pq" \
    || { echo "PARQUET-FAIL"; fails=$((fails+1)); }
fi
score_bake() { # tag, bake...
  local tag=$1; shift
  local o=$OUT/scores/bake_${tag}__on__cid22b.tsv
  if [ ! -s "$o" ]; then
    "$VALBIN/ensemble_score_rows" $(printf -- "--bake %q " "$@") \
      --parquet "$pq" --output "$o" \
      || { echo "BAKE-FAIL $tag"; fails=$((fails+1)); return 1; }
  fi
  [ -s "$OUT/scores/bake_${tag}__on__cid22b.csv" ] && return 0
  python3 "$OUT/tools/rows_to_scores.py" --pairs "$tsv" \
    --bake-tsv "$o" --out "$OUT/scores/bake_${tag}__on__cid22b.csv" \
    || { echo "BAKE-JOIN-FAIL $tag"; fails=$((fails+1)); }
}
score_bake prof_b "$W/b_sdr_linear_cid80_inclwinsor_dense_dial_byid_2026-09-06.bin"
score_bake prof_d "$W/d_sdr_add156_id100_negrich_dial_byid_2026-09-06.bin"
score_bake r915_basic228 \
  $REC/calibrated/R915_basic228_h128_s17101.bin \
  $REC/calibrated/R915_basic228_h128_s17103.bin \
  $REC/calibrated/R915_basic228_h128_s17107.bin \
  $REC/calibrated/R915_basic228_h128_s17111.bin \
  $REC/calibrated/R915_basic228_h128_s17113.bin
score_bake r915_y60 \
  $REC/calibrated/R915_y60_h32_s17101.bin \
  $REC/calibrated/R915_y60_h32_s17103.bin \
  $REC/calibrated/R915_y60_h32_s17107.bin \
  $REC/calibrated/R915_y60_h32_s17111.bin \
  $REC/calibrated/R915_y60_h32_s17113.bin
echo "CID22B SCORE STAGE DONE fails=$fails"

# -- 4. metrics table + paired bootstrap vs fast-ssim2 -------------------
cd "$OUT/scores"
scores=(); tags=()
for f in cid22a_native3 cid22a_luma cid22a_xyb-y \
         tidkadid_native3 tidkadid_luma tidkadid_xyb-y \
         imazen26_native3 imazen26_luma imazen26_xyb-y \
         pooled_native3 pooled_luma pooled_xyb-y; do
  f2=${f}__on__cid22b.csv
  if [ -s "$f2" ]; then scores+=("$f2"); tags+=("$f"); fi
done
for f in k1__on__cid22b.csv fastssim2__on__cid22b.csv \
         bake_prof_b__on__cid22b.csv bake_prof_d__on__cid22b.csv \
         bake_r915_basic228__on__cid22b.csv bake_r915_y60__on__cid22b.csv; do
  [ -s "$f" ] && { scores+=("$f"); tags+=("${f%%__on__cid22b.csv}"); }
done
python3 - "$OUT" "${#scores[@]}" "${scores[@]}" "${tags[@]}" <<'PYEOF' > "$OUT/tables/cid22b_table.json" \
  || { echo "TABLE-FAIL"; fails=$((fails+1)); }
import json, subprocess, sys
out = sys.argv[1]; n = int(sys.argv[2])
paths = sys.argv[3:3+n]; tags = sys.argv[3+n:3+2*n]
args = ["python3", f"{out}/tools/metrics.py", "table"]
for t, p in zip(tags, paths):
    args += ["--tag", t, "--scores", p]
r = subprocess.run(args, capture_output=True, text=True)
sys.stderr.write(r.stderr); sys.stdout.write(r.stdout)
sys.exit(r.returncode)
PYEOF
python3 - "$OUT" <<'PYEOF' > "$OUT/tables/cid22b_delta_vs_fastssim2.json" \
  || { echo "DELTA-FAIL"; fails=$((fails+1)); }
import glob, subprocess, sys
out = sys.argv[1]
cands = sorted(glob.glob("*__on__cid22b.csv"))
cands = [c for c in cands if not c.startswith("fastssim2")]
args = ["python3", f"{out}/tools/metrics.py", "delta",
        "--base", "fastssim2__on__cid22b.csv",
        "--boots", "2000", "--seed", "20260919"]
for c in cands:
    args += ["--cand", c]
r = subprocess.run(args, capture_output=True, text=True)
sys.stderr.write(r.stderr); sys.stdout.write(r.stdout)
sys.exit(r.returncode)
PYEOF
echo "CID22B READ COMPLETE fails=$fails"
exit $fails
