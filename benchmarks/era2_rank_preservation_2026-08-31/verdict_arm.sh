#!/bin/bash
# Score the era-2 rank-preservation roster on ONE feature root.
#   verdict_arm.sh <arm-label> <features-root>
set -u
ARM="$1"; ROOT="$2"
WT="${ZR_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BV="$WT/target/release/bake_verdict"
OUT=/mnt/v/output/zensim/era2-rank-2026-08-31/verdicts-$ARM
CORP=cid22,kadid,tid,konjnd,aic3,aic4,csiq,live,sdr25
mkdir -p "$OUT"
declare -A BAKES=(
  [B]="$WT/zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin"
  [C944]="$WT/zensim/weights/c_sdr_purity944_2026-08-29.bin"
  [WLIN7b_g020]="/mnt/v/output/zensim/wlin7b-2026-08-30/arms/Q7b_pools_g0.2_a0.2_b0.97.bin"
  [WLIN7b_g025]="/mnt/v/output/zensim/wlin7b-2026-08-30/arms/Q7b_pools_g0.25_a0.2_b0.97.bin"
  [ADD156]="/mnt/v/output/zensim/corr-lq/ADD156_safesyn_only_raw_lasso.bin"
  [BHdr]="$WT/zensim/weights/bhdr_linear_shaped_cvvdpmix_2026-07-12.bin"
)
echo "== verdicts arm=$ARM root=$ROOT out=$OUT"
for m in B C944 WLIN7b_g020 WLIN7b_g025 ADD156 BHdr; do
  b="${BAKES[$m]}"
  if [ ! -f "$b" ]; then echo "== $m SKIPPED (bake missing: $b)"; continue; fi
  "$BV" --bake "$b" --features-root "$ROOT" --regime 944 --corpora "$CORP" \
        --full-json "$OUT/$m.fulleval.json" --output "$OUT/$m.verdict.md" \
        > "$OUT/$m.stdout.txt" 2>&1
  rc=$?
  echo "== $m rc=$rc $( [ -f "$OUT/$m.fulleval.json" ] && echo json-ok || echo NO-JSON )"
  [ "$rc" -ne 0 ] && tail -3 "$OUT/$m.stdout.txt"
done
echo "VERDICTS-$ARM-DONE"
