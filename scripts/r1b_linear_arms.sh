#!/bin/bash
# R1b — the matched-mix TRUE-linear arm, run identically on two (or three)
# feature roots that differ ONLY in the f156..f371 pool block.
#
# Registered in benchmarks/r1b_keyed_rebuild_2026-08-30.md §2b BEFORE any fit.
# Recipe frozen there: legs safesyn 1.0 + cid22t 1.5 + kadid 0.5 + tid 0.5,
# target human_score with per-corpus min-max frames, shaped space
# (scripts/sota944/screen944_monotone.tsv), solver BVLS + sign-mask
# (benchmarks/feature_sign_mask_2026-05-26.tsv; f372+ free). Owners only:
# `bake_dial_refit gram` and `bake_dial_refit fit-lasso` — no new fit code.
#
#   scripts/r1b_linear_arms.sh <arm-name> <features-root> <out-dir>
#
# Env: ZL_BDR (bake_dial_refit binary), ZL_BV (bake_verdict binary).
set -euo pipefail
ARM="${1:?arm name}"; ROOT="${2:?features root}"; OUT="${3:?out dir}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
BDR="${ZL_BDR:-$REPO/target/release/bake_dial_refit}"
BV="${ZL_BV:-$REPO/target/release/bake_verdict}"
SCREEN="$REPO/scripts/sota944/screen944_monotone.tsv"
SIGNS="$REPO/benchmarks/feature_sign_mask_2026-05-26.tsv"
mkdir -p "$OUT/grams"
ts() { date -u +%H:%M:%SZ; }

# leg -> parquet basename, in the frozen weight order
declare -A LEG=( [safesyn]=ext_safesyn_full [cid22t]=ext_cid22_train201 \
                 [kadid]=ext_kadid [tid]=ext_tid )
declare -A W=( [safesyn]=1.0 [cid22t]=1.5 [kadid]=0.5 [tid]=0.5 )
ORDER="safesyn cid22t kadid tid"

for leg in $ORDER; do
  g="$OUT/grams/${leg}_shaped.npz"
  [ -f "$g" ] && { echo "== gram $leg cached"; continue; }
  echo "== gram $ARM/$leg $(ts)"
  "$BDR" gram --parquet "$ROOT/${LEG[$leg]}.parquet" --target human_score \
    --target-scale 100 ${R1B_MM01:---target-minmax01} --space shaped \
    --transforms-tsv "$SCREEN" --expect-n-feat 944 --out "$g"
done

GARGS=()
for leg in $ORDER; do
  GARGS+=(--gram "$OUT/grams/${leg}_shaped.npz" --weight "${W[$leg]}")
done

BAKE="$OUT/${ARM}_head_kon.bin"
echo "== fit $ARM $(ts)"
"$BDR" fit-lasso "${GARGS[@]}" --space shaped --target "${R1B_FIT_TARGET:-human_score__mm01}" \
  --solver bvls --bounds-tsv "$SIGNS" --lam 0 \
  --transforms-tsv "$SCREEN" \
  --anchor-parquet "$ROOT/ext_safesyn_full.parquet" --anchor-stride 37 \
  --anchor-parquet "$ROOT/ext_kadid.parquet" --anchor-stride 5 \
  --anchor-target human_score --anchor-scale 100 \
  --emit-fit-npz "$OUT/${ARM}_head_kon.npz" --embed-repro --out "$BAKE"

echo "== verdict $ARM $(ts)"
"$BV" --bake "$BAKE" --regime 944 --features-root "$ROOT" \
  --corpora "${R1B_CORPORA:-cid22,konjnd,nonphoto,imazen26,hfnlproxy,kadid,tid}" \
  --full-json "$OUT/${ARM}_head_kon.fulleval.json" \
  --output "$OUT/${ARM}_head_kon.verdict.md"
echo "R1B-ARM-DONE $ARM $(ts)"
