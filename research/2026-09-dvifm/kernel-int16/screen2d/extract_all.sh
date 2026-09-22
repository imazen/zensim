#!/usr/bin/env bash
# Phase 2d Part D block-cache extraction driver.
# One run-heavy job; sequential (domain x plane) extractions via the canonical
# extract_features_372col owner at --full-986 with per-plane DVIFM specs.
# Every run writes: cache/<dom>_<plane>.bin (+ .index.jsonl), features csv +
# manifest. RAYON threads capped at 8 (run-heavy --jobs 8).
set -uo pipefail
OUT=/mnt/v/output/zensim/dvifm-screen2d-2026-09-19
BIN=/home/lilith/work/zen/zensim/zensim-bench/target/release/examples/extract_features_372col
export ZENSIM_FORMULA_REV=3 ZENSIM_SAMPLE_DIGEST=1
export RAYON_NUM_THREADS=8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /home/lilith/work/zen/zensim

declare -A PAIRS=(
  [cid22a]=$OUT/pairs/cid22a.tsv
  [cid22b]=$OUT/pairs/cid22b.tsv
  [tid]=$OUT/pairs/tid_jp2kjpeg.tsv
  [kadid_train]=$OUT/pairs/kadid_train.tsv
  [kadid_dev]=$OUT/pairs/kadid_dev.tsv
  [konfig]=$OUT/pairs/konfig_val.tsv
  [imazen26]=$OUT/pairs/imazen26_pairs.tsv
)
PLANES=(xyb_y ycbcr_y ycbcr_cb ycbcr_cr)
DOMS=(cid22a tid kadid_train imazen26 cid22b kadid_dev konfig)

fails=0
for dom in "${DOMS[@]}"; do
  tsv="${PAIRS[$dom]}"
  if [ ! -s "$tsv" ]; then echo "MISSING $tsv"; fails=$((fails+1)); continue; fi
  for pl in "${PLANES[@]}"; do
    spec=$OUT/specs/dvifm-${pl}.json
    bin=$OUT/cache/${dom}_${pl}.bin
    csv=$OUT/cache/${dom}_${pl}.features.csv
    if [ -s "$bin" ] && [ -s "${bin}.index.jsonl" ]; then
      echo "SKIP $dom $pl (cache exists)"
      continue
    fi
    echo "=== EXTRACT $dom $pl <- $tsv ==="
    "$BIN" --full-986 --dvifm-spec "$spec" --dvifm-block-stats "$bin" \
      --corpus pairs-tsv --path "$tsv" --out "$csv" \
      --allow-failures 0 || { echo "FAIL $dom $pl"; fails=$((fails+1)); }
  done
done
echo "EXTRACTION DRIVER DONE fails=$fails"
