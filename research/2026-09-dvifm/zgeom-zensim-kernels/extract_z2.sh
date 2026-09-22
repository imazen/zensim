#!/usr/bin/env bash
# zgeom lane Z2: three kernel arms over the same rows/row order.
#   extract_z2.sh [arm ...]   (default: box2 bin121 bin1331)
# Each arm writes extract/z2_<arm>_{train,<leg>_dev}.csv — 228-col basic+peaks
# surface, same recipe, row order = TSV order (pairs_core order for train).
set -uo pipefail
OUT=/mnt/v/output/zensim/zgeom-2026-09-21
EX=$OUT/extract
VDIR=/mnt/v/output/zensim/dvifm-verdict-2026-09-20/pairs
BIN=${BIN:-/home/lilith/work/zen/zensim--transplant/zensim-bench/target/release/examples/extract_features_372col}
export ZENSIM_FORMULA_REV=3
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-8} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
fails=0

ext() { # $1 out-name, $2 arm-token, $3 tsv
  local name=$1 arm=$2 tsv=$3
  if [ -s "$EX/$name.csv" ]; then echo "SKIP $name"; return 0; fi
  echo "=== EXT $name $(date +%H:%M:%S)"
  "$BIN" --path "$tsv" --corpus pairs-tsv --out "$EX/$name.csv" \
      --zgeom "$arm" --allow-failures 0 \
    || { echo "FAIL $name"; fails=$((fails+1)); }
}

for arm in "${@:-box2 bin121 bin1331}"; do
  ext "z2_${arm}_train" "$arm" "$EX/z2_train.tsv"
  for leg in safesyn cid22 human codec; do
    ext "z2_${arm}_${leg}_dev" "$arm" "$VDIR/${leg}_dev.tsv"
  done
done
echo "extract_z2 done, fails=$fails"
