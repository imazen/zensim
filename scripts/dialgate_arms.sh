#!/usr/bin/env bash
# Build one G-ADDR candidate arm (shared-anchor -> add-winsor -> extend-top) and
# score it through bake_verdict with the addressability gate.
#   usage: run_arms.sh <label> <anchor.parquet> [n_edges] [extend_top_anchor]
set -euo pipefail
REPO=/home/lilith/work/zen/zensim
D=/mnt/v/output/zensim/dialgate-2026-09-04
R=$REPO/target/release/bake_dial_refit
V=$REPO/target/release/bake_verdict
INC=/mnt/v/output/zensim-jxl-nearlossless/inclusive_winsor_corpus.parquet
GRID=/mnt/v/output/zensim/eval_panels_2026-05-29/dial_grid_372col_2026-05-29_quarantined_v2.parquet
BASE=$REPO/zensim/weights/b_sdr_linear_cid80_anchored_2026-07-04.bin
LABEL="$1"; ANCHOR="$2"; NEDGES="${3:-18}"; TOPANCHOR="${4:-$2}"
mkdir -p "$D/arms"
"$R" shared-anchor --in "$BASE" --out "$D/arms/${LABEL}_anch.bin" --anchor "$ANCHOR" --n-edges "$NEDGES"
"$R" add-winsor --in "$D/arms/${LABEL}_anch.bin" --out "$D/arms/${LABEL}_wins.bin" \
     --fit-corpus "$INC" --lo-pct 0.1 --hi-pct 99.9 >/dev/null
"$R" extend-top --in "$D/arms/${LABEL}_wins.bin" --out "$D/arms/${LABEL}.bin" \
     --anchor "$TOPANCHOR" --target-col target_score
"$V" --bake "$D/arms/${LABEL}.bin" --dial-grid "$GRID" \
     --negtail-probe "$D/negtail_probe_372_2026-09-04.parquet" \
     --identity-probe "$D/identity_probe_372_2026-09-04.parquet" \
     --corpora cid22,konjnd,kadid,tid,aic3 \
     --full-json "$D/arms/verdict_${LABEL}.json" > "$D/arms/verdict_${LABEL}.log" 2>&1
echo "== $LABEL =="
python3 - "$D/arms/verdict_${LABEL}.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); dl=d['dial']; a=dl['addressability']
print('  ', a['headline'])
print('   reach %.3f  min %.3f  max %.3f  p5 %.3f  p95 %.3f  DR %.3f  mono %.4f  tied %.4f'%(
    dl['reach'],dl['min'],dl['max'],dl['p5'],dl['p95'],dl['dynamic_range'],dl['mono_pct'],dl['tied_pct']))
m=a['measured']
print('   negtail min %.3f p1 %.3f frac<0 %.4f | identity %.4f | above-identity %d'%(
    m['negtail']['min'],m['negtail']['p1'],m['negtail']['frac_below_zero'],
    m['identity']['dial_max'],m['identity']['n_above_identity']))
print('   fails:', [c['id'] for c in a['checks'] if c['state']=='fail'])
print('   SROCC cid22 %.5f konjnd %.5f aic3 %.5f tid %.5f kadid %.5f'%tuple(
    d['rank'][k]['srocc_signed'] for k in ['cid22','konjnd','aic3','tid','kadid']))
PY
