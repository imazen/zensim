#!/bin/bash
# carrier_head_fit.sh — THE committed driver for the campaign's "carriers
# enable the 944-class linear kon head" result (ledger
# `benchmarks/balance_campaign_2026-08-28.md` §"LINEAR-QUESTION ANSWERED" /
# §"W-LIN 954 RESURRECTION"; recovery + verdict in
# `benchmarks/carrier_head_recipe_2026-08-30.md`).
#
# The original heads were fit ad hoc with no driver and no `--embed-repro`, so
# their argv was unrecoverable from the artifacts alone. It was recovered by
# fingerprinting: `bake_dial_refit fit-lasso --parity-fit` reproduces the stored
# `head954_kon.npz` w/bias/mu/sd BIT-EXACTLY under the argv below. Everything
# here goes through the owners (`bake_dial_refit gram` / `fit-lasso` /`gate`,
# `bake_verdict`); no fit or stat math lives in this file.
#
# RECOVERED RECIPE (see the doc's table for the source of every item):
#   legs      safesyn 1.0 + cid22t 1.5 + kadid 0.5 + tid 0.5   (mu/sd bit-exact)
#   space     shaped, screen = the 40-slot winsor_p99/signed_cbrt screen the
#             bakes carry in `zentrain.feature_transform*` (f0..f155 only) —
#             NOT scripts/sota944/screen944_monotone.tsv
#   target    human_score, target-scale 1.0, NO per-corpus min-max framing
#   solver    bvls + benchmarks/feature_sign_mask_2026-05-26.tsv (f372+ free)
#   lam 0 · n_sweeps 200 · tol 1e-10 (defaults; bit-exact confirms)
#   tau       0.005 — recovered from the packed weights (530 of 954 exactly zero,
#             |w| mean 0.0077320594 matched to 10 digits; tau=0 gives 340/0.0081902)
#
# Usage:
#   scripts/carrier_head_fit.sh <arm> <features-root> <n_feat> <screen.tsv> <out-dir>
# Env:
#   ZL_BDR / ZL_BV      binaries (default: $REPO/target/release/…)
#   CHF_LEG_<leg>       parquet basename override per leg (no .parquet)
#   CHF_GRAM_DIR        reuse frozen grams from this dir as <prefix><leg>.npz
#   CHF_GRAM_PREFIX     prefix for the above (e.g. l954_)
#   CHF_TAU             pre-pack zero threshold (default 0.005, the recovered value)
#   CHF_MM01=1          build/consume the PER-CORPUS MIN-MAX target frame instead of the
#                       raw one. The ledger's no-carrier arm used this and its carrier
#                       arms did not (see the doc's 2x2) — this knob exists so that
#                       asymmetry can be reproduced and priced, never repeated by
#                       accident.
#   CHF_SLICE           --slice-file: restrict the CD to these coordinates. Forcing
#                       w=0 on a coordinate is EXACTLY equivalent to zeroing that
#                       column in the table (its S[j,k]*w[k] terms vanish), so this
#                       gives a same-gram / same-rows / same-binary block ablation
#                       with no table surgery.
#   CHF_ANCHOR          anchor parquet basename (default: the safesyn leg)
#   CHF_EVAL            space-separated LABEL=<abs parquet> pairs -> `gate` |SROCC|
#   CHF_VERDICT_ROOT    features-root for `bake_verdict` (adds signed SROCC + bars)
#   CHF_CORPORA         bake_verdict --corpora (default the five-bar set)
set -euo pipefail
ARM="${1:?arm name}"; ROOT="${2:?features root}"; NFEAT="${3:?n_feat}"
SCREEN="${4:?screen tsv}"; OUT="${5:?out dir}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BDR="${ZL_BDR:-$REPO/target/release/bake_dial_refit}"
BV="${ZL_BV:-$REPO/target/release/bake_verdict}"
SIGNS="$REPO/benchmarks/feature_sign_mask_2026-05-26.tsv"
mkdir -p "$OUT/grams"
ts() { date -u +%H:%M:%SZ; }

ORDER="safesyn cid22t kadid tid"
declare -A W=( [safesyn]=1.0 [cid22t]=1.5 [kadid]=0.5 [tid]=0.5 )
declare -A DEF=( [safesyn]=ext_safesyn_full [cid22t]=ext_cid22_train201 \
                 [kadid]=ext_kadid [tid]=ext_tid )
leg_file() { local l="$1"; local v="CHF_LEG_${l}"; echo "${!v:-${DEF[$l]}}"; }

GARGS=()
for leg in $ORDER; do
  if [ -n "${CHF_GRAM_DIR:-}" ]; then
    g="$CHF_GRAM_DIR/${CHF_GRAM_PREFIX:-}${leg}.npz"
    [ -f "$g" ] || { echo "missing frozen gram $g" >&2; exit 2; }
  else
    g="$OUT/grams/${leg}_shaped.npz"
    if [ ! -f "$g" ]; then
      echo "== gram $ARM/$leg $(ts)"
      "$BDR" gram --parquet "$ROOT/$(leg_file "$leg").parquet" \
        --target human_score --space shaped --transforms-tsv "$SCREEN" \
        ${CHF_MM01:+--target-minmax01} \
        --expect-n-feat "$NFEAT" --out "$g"
    fi
  fi
  GARGS+=(--gram "$g" --weight "${W[$leg]}")
done

ANCHOR="$ROOT/${CHF_ANCHOR:-$(leg_file safesyn)}.parquet"
BAKE="$OUT/${ARM}.bin"
echo "== fit $ARM $(ts)"
"$BDR" fit-lasso "${GARGS[@]}" \
  --space shaped --target "human_score${CHF_MM01:+__mm01}" \
  --solver bvls --bounds-tsv "$SIGNS" --lam 0 --tau "${CHF_TAU:-0.005}" \
  ${CHF_SLICE:+--slice-file "$CHF_SLICE"} \
  --transforms-tsv "$SCREEN" \
  --anchor-parquet "$ANCHOR" --anchor-stride 37 \
  --anchor-target human_score --anchor-scale 100 \
  --emit-fit-npz "$OUT/${ARM}.npz" --embed-repro --out "$BAKE" \
  ${CHF_PARITY:+--parity-fit "$CHF_PARITY"}

for kv in ${CHF_EVAL:-}; do
  lbl="${kv%%=*}"; pq="${kv#*=}"
  echo "== gate $ARM/$lbl $(ts)"
  "$BDR" gate --bake "$BAKE" --corpus "$pq" 2>&1 | sed "s/^/  [$lbl] /" || true
done

if [ -n "${CHF_VERDICT_ROOT:-}" ]; then
  echo "== verdict $ARM $(ts)"
  "$BV" --bake "$BAKE" --regime 944 --features-root "$CHF_VERDICT_ROOT" \
    --corpora "${CHF_CORPORA:-cid22,konjnd,nonphoto,imazen26,hfnlproxy,kadid,tid}" \
    --full-json "$OUT/${ARM}.fulleval.json" --output "$OUT/${ARM}.verdict.md"
fi
echo "CARRIER-HEAD-DONE $ARM $(ts)"
