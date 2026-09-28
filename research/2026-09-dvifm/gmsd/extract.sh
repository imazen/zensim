#!/usr/bin/env bash
# gmsd lane: one extraction pass over joint-core-v2 TRAIN + the four dev legs,
# emitting the production basic+peaks surface (box2:glob, 228) + 24 GMS
# std/mean columns + 96 map-deviation columns = 348 per row.
#   extract.sh            (all)
# Same row lists the zgeom/block5 lanes extracted, so base cols 0..227 must be
# bit-identical to zgeom's z2_box2 CSVs (checked by build_tables.py).
set -uo pipefail
L=/var/tmp/gmsd-lane
EX=$L/extract
ZG=/mnt/v/output/zensim/zgeom-2026-09-21
VDIR=/mnt/v/output/zensim/dvifm-verdict-2026-09-20/pairs
BIN=${BIN:-$L/zs-target/release/examples/extract_features_372col}
export ZENSIM_FORMULA_REV=3
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-8} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p "$EX"
fails=0
ext() { # $1 out-name, $2 pairs tsv
  local name=$1 tsv=$2
  if [ -s "$EX/$name.csv" ]; then echo "SKIP $name"; return 0; fi
  echo "=== EXT $name $(date +%H:%M:%S)"
  "$BIN" --corpus pairs-tsv --path "$tsv" --out "$EX/$name.csv" --allow-failures 0 \
      --zgeom box2:glob --zgeom-gmsdev --zgeom-mapdev \
    || { echo "FAIL $name"; fails=$((fails+1)); }
}
ext gm_train "$ZG/extract/z2_train.tsv"
for leg in safesyn cid22 human codec; do
  ext "gm_${leg}_dev" "$VDIR/${leg}_dev.tsv"
done
echo "extract done, fails=$fails"
