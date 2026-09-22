#!/usr/bin/env bash
# transplant lane: the extraction grid. Sequential inside the lock hold.
#   extract_all.sh <phase>
# phases: pilot | armed | chroma | dev | verify | all
set -uo pipefail
OUT=/mnt/v/output/zensim/dvifm-transplant-2026-09-20
EX=$OUT/extract
SPEC=$OUT/specs
VDIR=/mnt/v/output/zensim/dvifm-verdict-2026-09-20/pairs
BIN=${BIN:-$HOME/tmp/devin/transplant-target/release/examples/extract_features_372col}
export ZENSIM_FORMULA_REV=3 ZENSIM_SAMPLE_DIGEST=1
# thread count is an env override so the lane can run reduced (--jobs 4)
# when the supervisor authorizes an out-of-lock pass.
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
pilot)
  ext pilot --path $EX/pilot.tsv --full-986 \
      --transplant-spec $SPEC/transplant-pilot.json --transplant-only \
      --transplant-hist $EX/pilot_hist.json
  ;;
armed)
  ext v2fresh_986t --path $EX/v2fresh.tsv --full-986 \
      --transplant-spec $SPEC/transplant-armed.json
  ext rest_t --path $EX/rest.tsv --full-986 \
      --transplant-spec $SPEC/transplant-armed.json --transplant-only \
      --transplant-hist $EX/rest_hist.json
  ;;
chroma)
  ext all_cb --path $EX/all.tsv --full-986 \
      --dvifm-spec $SPEC/dvifm-ycbcr_cb.json --dvifm-only
  ext all_cr --path $EX/all.tsv --full-986 \
      --dvifm-spec $SPEC/dvifm-ycbcr_cr.json --dvifm-only
  ;;
dev)
  for leg in safesyn cid22 human codec; do
    ext dev_${leg}_t --path $VDIR/${leg}_dev.tsv --full-986 \
        --transplant-spec $SPEC/transplant-armed.json --transplant-only
    ext dev_${leg}_cb --path $VDIR/${leg}_dev.tsv --full-986 \
        --dvifm-spec $SPEC/dvifm-ycbcr_cb.json --dvifm-only
    ext dev_${leg}_cr --path $VDIR/${leg}_dev.tsv --full-986 \
        --dvifm-spec $SPEC/dvifm-ycbcr_cr.json --dvifm-only
  done
  ;;
verify)
  ext verify_986 --path $EX/verify.tsv --full-986
  ;;
all)
  $0 pilot; $0 armed; $0 chroma; $0 dev; $0 verify
  ;;
esac
echo "phase ${1:-all} done, fails=$fails"
