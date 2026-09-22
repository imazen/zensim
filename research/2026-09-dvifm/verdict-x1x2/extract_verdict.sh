#!/usr/bin/env bash
# verdict-lane extraction driver.
# A) dev legs needing DVIFM caches: 3 ycbcr plane passes (f16, capped;
#    safesyn cap=512 for the disk cap, others 1024). Canonical features
#    csvs are NOT kept for these (their w944 dev parquets already exist);
#    csv + manifest go to ~/tmp/devin scratch and are removed per plane.
#    fast-ssim2 peer scores come from the --audit-ssim2 channel on the
#    ycbcr_y pass (audit jsonl kept).
# B) kadid_dev / konfig_val / cid22b: dvifm caches already exist in the
#    2d run; one --full-986 --audit-ssim2 pass each for the w944 parquet
#    + ssim2 peer channel.
set -uo pipefail
OUT=/mnt/v/output/zensim/dvifm-verdict-2026-09-20
SCR=/home/lilith/tmp/devin/verdict-scratch
SPEC=/mnt/v/output/zensim/dvifm-screen2d-2026-09-19/specs
BIN=/home/lilith/work/zen/zensim/zensim-bench/target/release/examples/extract_features_372col
export ZENSIM_FORMULA_REV=3 ZENSIM_SAMPLE_DIGEST=1
export RAYON_NUM_THREADS=8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /home/lilith/work/zen/zensim
mkdir -p "$SCR" "$OUT/cache" "$OUT/scores" "$OUT/features"
fails=0

extract_dom() { # $1 dom, $2 cap
  local dom=$1 cap=$2 tsv=$OUT/pairs/$1.tsv
  [ -s "$tsv" ] || { echo "MISSING $tsv"; return 1; }
  for pl in ycbcr_y ycbcr_cb ycbcr_cr; do
    local bin=$OUT/cache/${dom}_${pl}.bin
    if [ -s "$bin" ] && [ -s "${bin}.index.jsonl" ]; then
      echo "SKIP $dom $pl"; continue
    fi
    local extra=()
    if [ "$pl" = ycbcr_y ] && [ ! -s "$OUT/scores/${dom}_audit_ssim2.jsonl" ]; then
      extra=(--audit-ssim2 --audit-jsonl "$OUT/scores/${dom}_audit_ssim2.jsonl")
    fi
    echo "=== EXTRACT $dom $pl cap=$cap ==="
    "$BIN" --full-986 --dvifm-spec "$SPEC/dvifm-${pl}.json" \
      --dvifm-block-stats "$bin" --dvifm-cap "$cap" --dvifm-quant f16 \
      ${extra[@]+"${extra[@]}"} \
      --corpus pairs-tsv --path "$tsv" --out "$SCR/${dom}_${pl}.csv" \
      --allow-failures 0 || { echo "FAIL $dom $pl"; return 1; }
    rm -f "$SCR/${dom}_${pl}.csv" "$SCR/${dom}_${pl}.csv.manifest.json"
  done
  if [ ! -s "$OUT/scores/${dom}_audit_ssim2.jsonl" ]; then
    "$BIN" --full-986 --audit-ssim2 \
      --audit-jsonl "$OUT/scores/${dom}_audit_ssim2.jsonl" \
      --corpus pairs-tsv --path "$tsv" --out "$SCR/${dom}_audit.csv" \
      --allow-failures 0 || { echo "AUDIT-FAIL $dom"; return 1; }
    rm -f "$SCR/${dom}_audit.csv" "$SCR/${dom}_audit.csv.manifest.json"
  fi
  return 0
}

features_dom() { # $1 dom
  local dom=$1 tsv=$OUT/pairs/$1.tsv
  [ -s "$tsv" ] || { echo "MISSING $tsv"; return 1; }
  local csv=$OUT/features/${dom}.csv aud=$OUT/scores/${dom}_audit_ssim2.jsonl
  if [ -s "$csv" ] && [ -s "$aud" ]; then echo "SKIP $dom"; return 0; fi
  echo "=== FEATURES $dom ==="
  "$BIN" --full-986 --audit-ssim2 --audit-jsonl "$aud" \
    --corpus pairs-tsv --path "$tsv" --out "$csv" \
    --allow-failures 0 || { echo "FAIL $dom"; return 1; }
  return 0
}

extract_dom safesyn_dev 512  || fails=$((fails+1))
extract_dom cid22_dev  1024 || fails=$((fails+1))
extract_dom codec_dev  1024 || fails=$((fails+1))
extract_dom human_dev  1024 || fails=$((fails+1))
features_dom kadid_dev  || fails=$((fails+1))
features_dom konfig_val || fails=$((fails+1))
features_dom cid22b     || fails=$((fails+1))
# dvifm caches for the last three legs lived in the deleted screen2d
# cache dir — extract them into this lane's cache (kadid135 is the
# 30-pair {1,3,5} subset itself, so no rows-file needed downstream).
extract_dom kadid135   1024 || fails=$((fails+1))
extract_dom kadid_dev  1024 || fails=$((fails+1))
extract_dom konfig_val 1024 || fails=$((fails+1))
extract_dom cid22b     1024 || fails=$((fails+1))
echo "VERDICT EXTRACTION DONE fails=$fails"
