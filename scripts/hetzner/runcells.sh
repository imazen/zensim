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
  # preflight: validate every manifest input (sha/rows/schema/targets) BEFORE
  # burning a training run — fleet/versioning errors die here, loudly.
  if ! python3 scripts/v_next/validate_parquet.py --manifest "zensim/weights/manifests/${M}.toml" > "$D/preflight.log" 2>&1; then
    echo -e "${M}\tPREFLIGHT-FAIL\t${t0}\t$(date -u +%FT%TZ)" >> "$STATUS"
    return 1
  fi
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
    # COLLAPSE GATE (2026-07-03): the held-out-val selection is BLIND to
    # human-anchor collapse (w6 fan: collapsed seeds within 0.001 of healthy
    # on val geomean, while verdict CID22 fell 0.84->0.54 and KonJND
    # 0.35->0.12). Every cell self-reports: parse the verdict, flag below
    # floors so scoreboards/fleets surface it without a human.
    CID=$(grep -E '^\| CID22 ' "$D/kadissafety.md" 2>/dev/null | head -1 | awk -F'|' '{print $4}' | tr -d ' ')
    KJ=$(grep -E '^\| KonJND' "$D/kadissafety.md" 2>/dev/null | head -1 | awk -F'|' '{print $4}' | tr -d ' ')
    if [ -n "$CID" ] && awk -v c="$CID" -v k="${KJ:-1}" 'BEGIN{exit !(c<0.75 || k<0.20)}'; then
      echo -e "${M}-COLLAPSED\tcid22=$CID konjnd=$KJ\t-\t$(date -u +%FT%TZ)" >> "$STATUS"
      rc=9
    fi
    ZENSIM_DIAL_GRID=/data/grids/hq_codec_grid_2026-07-01.parquet ZENSIM_DIAL_PRED_OUT="$D/hqpred.tsv" \
      nice -n10 target/release/bake_verdict --bake "$B" --corpora aic3 --output "$D/hq.md" >/dev/null 2>&1
  fi
  echo -e "${M}\t${rc}\t${t0}\t$(date -u +%FT%TZ)" >> "$STATUS"
}

export -f run_cell; export OUT STATUS
xargs -a "$CELLS_FILE" -P "$PAR" -I{} bash -c 'run_cell "$@"' _ {}
NCELLS=$(grep -c . "$CELLS_FILE")
NROWS=$(( $(grep -c . "$STATUS") - 1 ))
if grep -qP "\t(PREFLIGHT-FAIL|[1-9][0-9]*)\t" "$STATUS" || [ "$NROWS" -ne "$NCELLS" ]; then
  echo -e "ALLDONE-WITH-FAILURES\t1\t-\t$(date -u +%FT%TZ)" >> "$STATUS"   # rows<cells = silent cell deaths
else
  echo -e "ALLDONE\t0\t-\t$(date -u +%FT%TZ)" >> "$STATUS"
fi
