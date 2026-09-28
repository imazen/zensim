#!/usr/bin/env bash
# zgeom lane Z1: block-pooling arms over the same rows/row order.
#   extract_z1.sh <phase>
# phases:
#   hist   — b5max + --zgeom-hist over the full TRAIN corpus (the
#            c0-fit input); also emits the b5max arm table itself.
#   arms   — b5gate (--zgeom-c0 $SPEC/c0_z1.json) + b5max over train+dev.
#   dev    — dev legs only (for re-runs after the armed pass).
set -uo pipefail
OUT=/mnt/v/output/zensim/zgeom-2026-09-21
EX=$OUT/extract
SPEC=$OUT/specs
VDIR=/mnt/v/output/zensim/dvifm-verdict-2026-09-20/pairs
BIN=${BIN:-/home/lilith/work/zen/zensim--transplant/zensim-bench/target/release/examples/extract_features_372col}
export ZENSIM_FORMULA_REV=3
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-8} OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
fails=0

ext() { # $1 out-name, rest: extractor args
  local name=$1; shift
  if [ -s "$EX/$name.csv" ]; then echo "SKIP $name"; return 0; fi
  echo "=== EXT $name $(date +%H:%M:%S)"
  "$BIN" "$@" --corpus pairs-tsv --out "$EX/$name.csv" --allow-failures 0 \
    || { echo "FAIL $name"; fails=$((fails+1)); }
}

case "${1:-all}" in
hist)
  ext z1_b5max_train --path $EX/z2_train.tsv \
      --zgeom box2:b5max --zgeom-hist $EX/z1_hist.json
  ;;
arms)
  ext z1_b5gate_train --path $EX/z2_train.tsv \
      --zgeom box2:b5gate --zgeom-c0 $SPEC/c0_z1.json
  for leg in safesyn cid22 human codec; do
    ext "z1_b5gate_${leg}_dev" --path $VDIR/${leg}_dev.tsv \
        --zgeom box2:b5gate --zgeom-c0 $SPEC/c0_z1.json
    ext "z1_b5max_${leg}_dev" --path $VDIR/${leg}_dev.tsv \
        --zgeom box2:b5max
  done
  ;;
dev)
  for leg in safesyn cid22 human codec; do
    ext "z1_b5gate_${leg}_dev" --path $VDIR/${leg}_dev.tsv \
        --zgeom box2:b5gate --zgeom-c0 $SPEC/c0_z1.json
    ext "z1_b5max_${leg}_dev" --path $VDIR/${leg}_dev.tsv \
        --zgeom box2:b5max
  done
  ;;
all)
  $0 hist; $0 arms
  ;;
esac
echo "extract_z1 ${1:-all} done, fails=$fails"
