#!/usr/bin/env bash
# faithful lane — fit all arms, eval each on the TID2013-full / KADID10K-nt /
# CID22-A surfaces of its own band. Luma-only throughout.
set -uo pipefail
OUT=/mnt/v/output/zensim/dvifm-faithful-2026-09-22
PY=/mnt/v/output/zensim/dvifm-faithful-2026-09-22/tools/fit_faithful.py
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p "$OUT/fits" "$OUT/evals"
fails=0

run() { # $1 arm $2 fitsegs $3 evalsegs
  local arm=$1 fs=$2 es=$3
  local fit=$OUT/fits/$arm.json ev=$OUT/evals/$arm.json
  if [ ! -s "$fit" ]; then
    echo "=== FIT $arm <- $fs ==="
    python3 "$PY" fit --arm "$arm" --fitsegs "$OUT/segs/$fs.json" --out "$fit" \
      || { echo "FAIL fit $arm"; fails=$((fails+1)); return 1; }
  else echo "SKIP fit $arm"; fi
  if [ ! -s "$ev" ]; then
    echo "=== EVAL $arm -> $es ==="
    python3 "$PY" eval --arm "$arm" --fit-art "$fit" \
      --evalsegs "$OUT/segs/$es.json" --out "$ev" \
      || { echo "FAIL eval $arm"; fails=$((fails+1)); return 1; }
  else echo "SKIP eval $arm"; fi
}

run faithful   fitsegs_lap            evalsegs_lap
run edge_disc  fitsegs_lap            evalsegs_lap
run gate       fitsegs_lap            evalsegs_lap
run tied_pool  fitsegs_lap            evalsegs_lap
run band_local fitsegs_local          evalsegs_local
run fitmix     fitsegs_cid22dev       evalsegs_lap
run all_ours   fitsegs_local          evalsegs_local
run ours_full  fitsegs_cid22dev_local evalsegs_local
echo "ARMS DONE fails=$fails"
