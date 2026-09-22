#!/usr/bin/env bash
# v2 block-record extraction for the loss lane.
# Each domain: 3 planes (ycbcr_{y,cb,cr}), cap 192 f16 records/row + hists.
# Features CSVs (a byproduct the lane does not need) go to /tmp scratch.
# Usage: extract_v2.sh <pairs.tsv> <tag> [max_pairs]
set -euo pipefail
PAIRS=$1; TAG=$2; MAXP=${3:-0}
LANE=/mnt/v/output/zensim/dvifm-loss-2026-09-20
SPECS=/mnt/v/output/zensim/dvifm-screen2d-2026-09-19/specs
CSVSCRATCH=/tmp/dvifm-loss-csv
BIN=/home/lilith/tmp/devin/loss-target/release/examples/extract_features_372col
export ZENSIM_FORMULA_REV=3
mkdir -p "$CSVSCRATCH" "$LANE/cache"
for pl in y cb cr; do
  spec=$SPECS/dvifm-ycbcr_$pl.json
  bin=$LANE/cache/${TAG}_ycbcr_$pl.bin
  hist=$LANE/cache/${TAG}_ycbcr_$pl.hist.bin
  args=(--corpus pairs-tsv --path "$PAIRS"
        --out "$CSVSCRATCH/${TAG}_ycbcr_$pl.features.csv"
        --full-986 --dvifm-spec "$spec"
        --dvifm-block-stats "$bin"
        --dvifm-cap 192 --dvifm-quant f16
        --dvifm-hist "$hist"
        --allow-failures 0)
  [ "$MAXP" != 0 ] && args+=(--max-pairs "$MAXP")
  echo "=== $TAG $pl -> $bin"
  ~/tmp/devin/heavy -- "$BIN" "${args[@]}"
  ls -la "$bin" "$hist"
done
