#!/bin/bash
set -u
TAG="$1"; DGRID="$2"; CGRID="$3"
WT="${ZR_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BV="$WT/target/release/bake_verdict"
ROOT=/mnt/v/zen/zensim-training/era2rank-era1-2026-08-31
OUT=/mnt/v/output/zensim/era2-rank-2026-08-31/dialpanel-$TAG
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
for m in B C944 WLIN7b_g020 WLIN7b_g025 ADD156 BHdr; do
  "$BV" --bake "${BAKES[$m]}" --features-root "$ROOT" --regime 944 --corpora "$CORP" \
        --dial-grid "$DGRID" --corruption-grid "$CGRID" \
        --full-json "$OUT/$m.fulleval.json" --output "$OUT/$m.verdict.md" \
        > "$OUT/$m.stdout.txt" 2>&1 || echo "  $m rc=$?"
done
echo "DIALCORR-$TAG-DONE"
