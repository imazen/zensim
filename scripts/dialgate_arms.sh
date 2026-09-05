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
#
# ERA. `ZL_ERA` selects WHICH 372 instruments + features root the grade reads.
# The default `canonical` is the 2026-09-04 set and is byte-unchanged.
#   canonical  the 2026-05-29 grid + the 2026-09-04 probes + the DEFAULT root.
#              The grid's own pixels no longer exist (decode cache deleted
#              2026-06-22) and the root predates option C, so this reads the
#              published era, NOT the shipped runtime's.
#   postC      the 2026-09-05 re-extraction at HEAD -- THE RUNTIME ERA.
#   preC       the same pixels extracted at 27cfde15, the commit immediately
#              before option C. Exists so `postC - preC` is an era measurement
#              with the pixels held fixed.
# Individual overrides (ZL_GRID / ZL_NEGTAIL / ZL_IDENTITY / ZL_ROOT / ZL_OUT)
# still win, so a one-off instrument needs no new era name.
set -euo pipefail
REPO=/home/lilith/work/zen/zensim
D=/mnt/v/output/zensim/dialgate-2026-09-04
I=/mnt/v/output/zensim/dpeaks372-2026-09-05/instruments
R="${ZL_BDR:-$REPO/target/release/bake_dial_refit}"
V="${ZL_BV:-$REPO/target/release/bake_verdict}"
INC=/mnt/v/output/zensim-jxl-nearlossless/inclusive_winsor_corpus.parquet
BASE=$REPO/zensim/weights/b_sdr_linear_cid80_anchored_2026-07-04.bin

case "${ZL_ERA:-canonical}" in
  canonical)
    GRID_DEF=/mnt/v/output/zensim/eval_panels_2026-05-29/dial_grid_372col_2026-05-29_quarantined_v2.parquet
    NEGTAIL_DEF=$D/negtail_probe_372_2026-09-04.parquet
    IDENTITY_DEF=$D/identity_probe_372_2026-09-04.parquet
    ROOT_DEF=""; OUT_DEF=$D/arms ;;
  postC)
    GRID_DEF=$I/dial_grid_372col_postC_2026-09-05.parquet
    NEGTAIL_DEF=$I/negtail_probe_372_postC_2026-09-05.parquet
    IDENTITY_DEF=$I/identity_probe_372_postC_2026-09-05.parquet
    ROOT_DEF=/mnt/v/zen/zensim-training/2026-09-05-full-features-372-postC
    OUT_DEF=/mnt/v/output/zensim/dpeaks372-2026-09-05/arms/postC ;;
  preC)
    GRID_DEF=$I/dial_grid_372col_preC_2026-09-05.parquet
    NEGTAIL_DEF=$I/negtail_probe_372_preC_2026-09-05.parquet
    # byte-identical to the postC probe (the identity vector is the zero vector
    # at w372 in BOTH eras -- measured, sha256 e6f9096b…, 0 differing cells).
    IDENTITY_DEF=$I/identity_probe_372_postC_2026-09-05.parquet
    ROOT_DEF=/mnt/v/zen/zensim-training/2026-08-30-full-features-372
    OUT_DEF=/mnt/v/output/zensim/dpeaks372-2026-09-05/arms/preC ;;
  *) echo "unknown ZL_ERA=${ZL_ERA} (canonical|postC|preC)" >&2; exit 2 ;;
esac
GRID="${ZL_GRID:-$GRID_DEF}"
NEGTAIL="${ZL_NEGTAIL:-$NEGTAIL_DEF}"
IDENTITY="${ZL_IDENTITY:-$IDENTITY_DEF}"
ROOT="${ZL_ROOT:-$ROOT_DEF}"
ARMS="${ZL_OUT:-$OUT_DEF}"

grade() {  # <label> <bake.bin> [regime]
    local LABEL="$1" BAKE="$2" REGIME="${3:-372}"
    local EXTRA=() CORPORA=cid22,konjnd,kadid,tid,aic3
    [ "$REGIME" != "372" ] && EXTRA=(--regime "$REGIME")
    [ -n "$ROOT" ] && EXTRA+=(--features-root "$ROOT")
    mkdir -p "$ARMS"
    # --gaddr-json carries the G-ADDR block at FULL f64 precision; the markdown
    # rounds to 4dp, which is not enough to append a registry pin from.
    "$V" --bake "$BAKE" "${EXTRA[@]}" \
         $([ "$REGIME" = "372" ] && echo --dial-grid "$GRID") \
         --negtail-probe "$NEGTAIL" \
         --identity-probe "$IDENTITY" \
         --corpora "$CORPORA" \
         --gaddr-json "$ARMS/gaddr_${LABEL}.json" \
         --full-json "$ARMS/verdict_${LABEL}.json" \
         > "$ARMS/verdict_${LABEL}.log" 2>&1
    echo "== $LABEL (era ${ZL_ERA:-canonical}) =="
    python3 - "$ARMS/verdict_${LABEL}.json" <<'PY'
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
mkdir -p "$ARMS"
"$R" shared-anchor --in "$BASE" --out "$ARMS/${LABEL}_anch.bin" --anchor "$ANCHOR" --n-edges "$NEDGES"
"$R" add-winsor --in "$ARMS/${LABEL}_anch.bin" --out "$ARMS/${LABEL}_wins.bin" \
     --fit-corpus "$INC" --lo-pct 0.1 --hi-pct 99.9 >/dev/null
"$R" extend-top --in "$ARMS/${LABEL}_wins.bin" --out "$ARMS/${LABEL}.bin" \
     --anchor "$TOPANCHOR" --target-col target_score
grade "$LABEL" "$ARMS/${LABEL}.bin"
