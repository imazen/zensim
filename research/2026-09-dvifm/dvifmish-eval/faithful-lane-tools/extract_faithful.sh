#!/usr/bin/env bash
# faithful-lane extraction driver — luma-only DVIFM block-stats caches.
# Two bands (laplacian=faithful, local=our drift) x the needed row sets.
# The --out features csv is a required byproduct -> scratch, removed.
set -uo pipefail
OUT=/mnt/v/output/zensim/dvifm-faithful-2026-09-22
SCR=/home/lilith/tmp/devin/faithful-scratch
BIN=/home/lilith/work/zen/zensim/zensim-bench/target/release/examples/extract_features_372col
export ZENSIM_FORMULA_REV=3 ZENSIM_SAMPLE_DIGEST=1
export RAYON_NUM_THREADS=8 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd /home/lilith/work/zen/zensim
mkdir -p "$SCR" "$OUT/cache"
fails=0

pass() { # $1 dom, $2 bandtag
  local dom=$1 band=$2 tsv=$OUT/pairs/$1.tsv
  local bin=$OUT/cache/${dom}_${band}_y.bin
  local spec=$OUT/specs/dvifm-ycbcr_y-${band}.json
  [ -s "$tsv" ] || { echo "MISSING $tsv"; return 1; }
  if [ -s "$bin" ] && [ -s "${bin}.index.jsonl" ]; then
    echo "SKIP $dom $band"; return 0
  fi
  echo "=== EXTRACT $dom $band ==="
  "$BIN" --full-986 --dvifm-spec "$spec" \
    --dvifm-block-stats "$bin" --dvifm-cap 1024 --dvifm-quant f16 \
    --corpus pairs-tsv --path "$tsv" --out "$SCR/${dom}_${band}.csv" \
    --allow-failures 0 || { echo "FAIL $dom $band"; return 1; }
  rm -f "$SCR/${dom}_${band}.csv" "$SCR/${dom}_${band}.csv.manifest.json" \
        "$SCR/${dom}_${band}.csv.producer.bin"
  return 0
}

for spec in "tid_full lap" "tid_full local" \
            "kadid_nt lap" "kadid_nt local" \
            "cid22a lap" "cid22a local" \
            "cid22_dev lap" "cid22_dev local"; do
  set -- $spec
  pass "$1" "$2" || fails=$((fails+1))
done
echo "DONE fails=$fails"
