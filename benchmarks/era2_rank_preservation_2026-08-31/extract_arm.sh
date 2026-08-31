#!/bin/bash
# era-2 rank-preservation lane: extract the 9 eval legs for ONE arm.
#   extract_arm.sh <arm-name> [<ZENSIM_H_TILE value, empty = era-1 default>]
set -u
ARM="$1"; TILE="${2:-}"
WT="${ZR_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BIN="$WT/target/release/examples/v2_ab_extract"
OUT=/mnt/v/output/zensim/era2-rank-2026-08-31/run-$ARM
MODE=foldapp2pools
mkdir -p "$OUT"
ts() { date -u +%H:%M:%SZ; }
echo "== arm=$ARM TILE='${TILE:-<unset>}' MODE=$MODE OUT=$OUT start $(ts)"
run_leg() {
  local name="$1" pairs="$2"
  local t0=$SECONDS
  if [ -n "$TILE" ]; then
    ZENSIM_H_TILE="$TILE" ZENSIM_AB_MODE="$MODE" "$BIN" "$pairs" "$OUT/$name.csv" >/dev/null 2>&1
  else
    ZENSIM_AB_MODE="$MODE" "$BIN" "$pairs" "$OUT/$name.csv" >/dev/null 2>&1
  fi
  local rc=$?
  local rows=-1 cols=-1
  [ -f "$OUT/$name.csv" ] && { rows=$(( $(wc -l < "$OUT/$name.csv") - 1 )); cols=$(head -1 "$OUT/$name.csv" | awk -F, '{print NF}'); }
  local want=$(( $(wc -l < "$pairs") - 1 ))
  echo "== $name rc=$rc rows=$rows/$want cols=$cols $((SECONDS-t0))s $(ts)"
  if [ "$rc" -ne 0 ] || [ "$rows" -ne "$want" ] || [ "$cols" -ne 946 ]; then
    echo "ABORT: $name failed"; exit 1
  fi
}
run_leg ext_sdr25            /mnt/v/output/zensim/v2-backfill-2026-07-20/sdr25_pairs.tsv
run_leg ext_aic4             /mnt/v/output/zensim/v2-backfill-2026-07-20/aic4_pairs.tsv
run_leg ext_konjnd_jpeg_val  /mnt/v/output/zensim/v2-backfill-2026-07-20/konjnd_jpeg_val_pairs.tsv
run_leg ext_aic3             /mnt/v/output/zensim/v2-ab-2026-07-19/aic3_pairs_ab.tsv
run_leg ext_live             /mnt/v/datasets/LIVE/live_r2_pairs.tsv
run_leg ext_csiq             /mnt/v/dataset/csiq/csiq_pairs.tsv
run_leg ext_tid              /mnt/v/dataset/tid2013/tid_pairs_ab.tsv
run_leg ext_cid22val         /mnt/v/dataset/cid22/CID22_validation_set/cid22val_pairs_ab.tsv
run_leg ext_kadid            /mnt/v/dataset/kadid10k/kadid_pairs_ab.tsv
echo "ARM-$ARM-DONE $(ts)"
