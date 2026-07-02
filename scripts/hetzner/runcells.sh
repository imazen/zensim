#!/usr/bin/env bash
# Parallel trainer-cell runner for a Hetzner train box. 48-core CCX63 runs
# PAR cells concurrently (each cell ~4-8 threads); per-cell logs + status.tsv.
# Usage (on box):  runcells.sh cells.txt [PAR]
#   cells.txt = one manifest name per line (relative to zensim/weights/manifests/)
# Results land in /data/out/<cell>/ : bake .bin, .verdict.md, .kadissafety.md,
# hqpred tsv, train log. status.tsv is the ONLY file the workstation polls.
set -uo pipefail
CELLS_FILE="${1:?cells file}"; PAR="${2:-6}"
cd ~/work/zensim
OUT=/data/out; mkdir -p "$OUT"
STATUS="$OUT/status.tsv"
echo -e "cell\trc\tstart\tend" > "$STATUS"

run_cell() {
  local M="$1"; local D="$OUT/$M"; mkdir -p "$D"
  local t0=$(date -u +%FT%TZ)
  unset ZENSIM_DIAL_GRID
  RAYON_NUM_THREADS=6 nice -n10 target/release/zensim_mlp_train \
    --manifest "zensim/weights/manifests/${M}.toml" \
    > "$D/train.log" 2>&1
  local rc=$?
  B=$(grep -oE 'file *= *"[^"]+"' "zensim/weights/manifests/${M}.toml" | head -1 | grep -oE '/[^"]+')
  # manifests written for the box point [bake].file into /data/out/<cell>/
  if [ -f "$B" ]; then
    ZENSIM_DIAL_GRID=/data/grids/kadis_test_safetygrid.parquet nice -n10 target/release/bake_verdict \
      --bake "$B" --output "$D/kadissafety.md" >/dev/null 2>&1
    ZENSIM_DIAL_GRID=/data/grids/hq_codec_grid_2026-07-01.parquet ZENSIM_DIAL_PRED_OUT="$D/hqpred.tsv" \
      nice -n10 target/release/bake_verdict --bake "$B" --corpora aic3 --output "$D/hq.md" >/dev/null 2>&1
  fi
  echo -e "${M}\t${rc}\t${t0}\t$(date -u +%FT%TZ)" >> "$STATUS"
}

export -f run_cell; export OUT
xargs -a "$CELLS_FILE" -P "$PAR" -I{} bash -c 'run_cell "$@"' _ {}
echo -e "ALLDONE\t0\t-\t$(date -u +%FT%TZ)" >> "$STATUS"
