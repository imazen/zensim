#!/bin/bash
# wave-r4 A6: extract the 196,086-row big safesyn leg at era-2 x radius-4.
#
# Same extractor, same mode, same gates as extract_extra_legs.sh -- this is a
# thin driver, not a second extraction path. The pairs TSV is produced by
# stage_safesyn_big_r4.py (which gates the label join at 196086/196086) and its
# distorted sides are heterogeneous BY DESIGN:
#   111,068 JPEG rows  -> the original q<N>.jpg bitstreams (zen_io reads these)
#    85,018 other rows -> PNGs decoded by verify_bitstream_decode --decode-list
#                         (the decode owner), because zen_io cannot read
#                         avif/jxl/webp and the q<N>.png cache was deleted
#                         2026-06-22.
#
# Env: ZM944_BIN, ZM944_OUT, ZM944_MODE (default foldapp2pools), ZM944_PAIRS,
#      ZM944_NAME
set -u
BIN="${ZM944_BIN:?ZM944_BIN required}"
OUT="${ZM944_OUT:?ZM944_OUT required}"
MODE="${ZM944_MODE:-foldapp2pools}"
PAIRS="${ZM944_PAIRS:?ZM944_PAIRS required}"
NAME="${ZM944_NAME:-ext_safesyn_big}"
[ -x "$BIN" ] || { echo "ABORT: extractor missing: $BIN"; exit 1; }
[ -f "$PAIRS" ] || { echo "ABORT: pairs TSV missing: $PAIRS"; exit 1; }
mkdir -p "$OUT"
ts() { date -u +%H:%M:%SZ; }

want=$(( $(wc -l < "$PAIRS") - 1 ))
echo "== $NAME start $(ts) MODE=$MODE rows=$want"
echo "== pairs=$PAIRS"
t0=$SECONDS
ZENSIM_AB_MODE="$MODE" "$BIN" "$PAIRS" "$OUT/$NAME.csv"
rc=$?
rows=-1; cols=-1
if [ -f "$OUT/$NAME.csv" ]; then
  rows=$(( $(wc -l < "$OUT/$NAME.csv") - 1 ))
  cols=$(head -1 "$OUT/$NAME.csv" | awk -F, '{print NF}')
fi
echo "== $NAME done rc=$rc rows=$rows/$want cols=$cols $((SECONDS-t0))s $(ts)"
if [ "$rc" -ne 0 ] || [ "$rows" -ne "$want" ] || [ "$cols" -ne 946 ]; then
  echo "ABORT: $NAME failed (rc=$rc rows=$rows/$want cols=$cols want-cols=946)"; exit 1
fi
echo "BIGLEG-EXTRACT-DONE $NAME $(ts)"
