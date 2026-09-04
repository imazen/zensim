#!/usr/bin/env bash
# G-ADDR candidate driver — TWO modes, one scoring path.
#
#   build:  dialgate_arms.sh <label> <anchor.parquet> [n_edges] [extend_top_anchor]
#             shared-anchor -> add-winsor -> extend-top, then grade.
#   score:  dialgate_arms.sh score <label> <bake.bin> [regime]
#             grade an EXISTING bake (a shipped profile, a prior arm) with no
#             rebuild. `regime` is 372 (default) / 720 / 944.
#
# Both modes end in the SAME `grade` function, so a re-grade after a bar change
# can never accidentally be a different measurement from the build-time one.
#
# Binaries come from the MAIN repo (never a sibling worktree — those get cleaned
# up and take committed scripts down with them); override with ZL_BV / ZL_BDR.
set -euo pipefail
REPO=/home/lilith/work/zen/zensim
D=/mnt/v/output/zensim/dialgate-2026-09-04
R="${ZL_BDR:-$REPO/target/release/bake_dial_refit}"
V="${ZL_BV:-$REPO/target/release/bake_verdict}"
INC=/mnt/v/output/zensim-jxl-nearlossless/inclusive_winsor_corpus.parquet
GRID=/mnt/v/output/zensim/eval_panels_2026-05-29/dial_grid_372col_2026-05-29_quarantined_v2.parquet
BASE=$REPO/zensim/weights/b_sdr_linear_cid80_anchored_2026-07-04.bin

grade() {  # <label> <bake.bin> [regime]
    local LABEL="$1" BAKE="$2" REGIME="${3:-372}"
    local EXTRA=() CORPORA=cid22,konjnd,kadid,tid,aic3
    [ "$REGIME" != "372" ] && EXTRA=(--regime "$REGIME")
    mkdir -p "$D/arms"
    # --gaddr-json carries the G-ADDR block at FULL f64 precision; the markdown
    # rounds to 4dp, which is not enough to append a registry pin from.
    "$V" --bake "$BAKE" "${EXTRA[@]}" \
         $([ "$REGIME" = "372" ] && echo --dial-grid "$GRID") \
         --negtail-probe "$D/negtail_probe_372_2026-09-04.parquet" \
         --identity-probe "$D/identity_probe_372_2026-09-04.parquet" \
         --corpora "$CORPORA" \
         --gaddr-json "$D/arms/gaddr_${LABEL}.json" \
         --full-json "$D/arms/verdict_${LABEL}.json" \
         > "$D/arms/verdict_${LABEL}.log" 2>&1
    echo "== $LABEL =="
    python3 - "$D/arms/verdict_${LABEL}.json" <<'PY'
import json,sys
d=json.load(open(sys.argv[1])); dl=d['dial']; a=dl['addressability']
print('  ', a['headline'])
print('   reach %.3f  min %.3f  max %.3f  p5 %.3f  p95 %.3f  DR %.3f  mono %.4f  tied %.4f'%(
    dl['reach'],dl['min'],dl['max'],dl['p5'],dl['p95'],dl['dynamic_range'],dl['mono_pct'],dl['tied_pct']))
m=a['measured']
if m.get('negtail') and m.get('identity'):
    print('   negtail min %.3f p1 %.3f frac<0 %.4f | identity %.4f | above-identity %d'%(
        m['negtail']['min'],m['negtail']['p1'],m['negtail']['frac_below_zero'],
        m['identity']['dial_max'],m['identity']['n_above_identity']))
print('   bar set:', a.get('reference'), '| incumbent:', a.get('incumbent_reference'))
print('   fails:', [c['id'] for c in a['checks'] if c['state']=='fail'],
      '| not measured:', [c['id'] for c in a['checks'] if c['state']=='not_measured'])
try:
    print('   SROCC cid22 %.5f konjnd %.5f aic3 %.5f tid %.5f kadid %.5f'%tuple(
        d['rank'][k]['srocc_signed'] for k in ['cid22','konjnd','aic3','tid','kadid']))
except KeyError:
    pass
PY
}

if [ "${1:-}" = "score" ]; then
    grade "$2" "$3" "${4:-372}"
    exit 0
fi

LABEL="$1"; ANCHOR="$2"; NEDGES="${3:-18}"; TOPANCHOR="${4:-$2}"
mkdir -p "$D/arms"
"$R" shared-anchor --in "$BASE" --out "$D/arms/${LABEL}_anch.bin" --anchor "$ANCHOR" --n-edges "$NEDGES"
"$R" add-winsor --in "$D/arms/${LABEL}_anch.bin" --out "$D/arms/${LABEL}_wins.bin" \
     --fit-corpus "$INC" --lo-pct 0.1 --hi-pct 99.9 >/dev/null
"$R" extend-top --in "$D/arms/${LABEL}_wins.bin" --out "$D/arms/${LABEL}.bin" \
     --anchor "$TOPANCHOR" --target-col target_score
grade "$LABEL" "$D/arms/${LABEL}.bin"
