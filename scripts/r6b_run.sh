#!/usr/bin/env bash
# R6b (F17): the lane driver — extraction, fits, evals, dial, decision.
#
# Owns NOTHING. Every step is one of the R6 owners, parameterised for F17's
# knob (`R6_ARM_ENV=ZENSIM_HF_GAIN`) and its pre-registered slice set
# (`R6_SLICES="156 228"`), plus the three pair-TSV legs the eval driver does not
# cover (anchor, identity, ladder). If a step here starts computing something,
# it belongs in the owner instead.
#
# Pre-registration: docs/PLAN_FEATURE_REV2_2026-09-05.md §11.
#
# Usage: r6b_run.sh <extras|arms|fit|eval|dial|delta|decide> [args]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="${R6B_ROOT:-/mnt/v/output/zensim/rev2-2026-09-05/r6b}"
R6="${R6_ROOT:-/mnt/v/output/zensim/rev2-2026-09-05/r6}"
EX="$REPO/zensim-bench/target/release/examples/extract_features_372col"
ARMS="${R6B_ARMS:-ratio bexcess log1p satexcess cap}"
NEW_ARMS="${R6B_NEW_ARMS:-bexcess log1p satexcess cap}"
VARIANTS="${R6B_VARIANTS:-s156_lasso s156_bvls s228_lasso s228_bvls}"
ANCHOR_TSV="${ANCHOR_TSV:-$HOME/tmp/r6/safesyn_r6.tsv.anchor.tsv}"
IDENTITY_TSV="${IDENTITY_TSV:-$HOME/tmp/r6/identity.tsv}"
LADDER_PAIRS="${LADDER_PAIRS:-$HOME/tmp/ladder_instr/ladder/ladder_pairs.tsv}"
export R6_ARM_ENV=ZENSIM_HF_GAIN
export R6_ARM_VALUES="$ARMS"
export R6_SLICES="156 228"
mkdir -p "$ROOT"/{tables,logs,grams,fits,bakes,slices,verdicts,perpair,dial,instruments,evalroot}
say() { printf '[%s] r6b %s\n' "$(date -u +%H:%M:%S)" "$*"; }

case "${1:?usage: r6b_run.sh <extras|arms|fit|eval|dial|delta|decide>}" in

# The three pair-TSV legs `r6_extract_arms.sh` does not cover. Same binary, same
# runtime arm switch; separate only because their corpora are ad-hoc TSVs.
extras)
  for a in ${2:-$ARMS}; do
    T="$ROOT/tables/$a"; mkdir -p "$T"
    for leg in anchor:"$ANCHOR_TSV" identity:"$IDENTITY_TSV" ladder:"$LADDER_PAIRS"; do
      name="${leg%%:*}"; path="${leg#*:}"
      [ -f "$path" ] || { echo "missing $name pairs: $path" >&2; exit 2; }
      say "$a $name"
      env ZENSIM_HF_GAIN="$a" nice -n19 ionice -c3 "$EX" --corpus pairs-tsv \
          --path "$path" --out "$T/$name.csv" >/dev/null
    done
  done ;;

# Full extraction of the arms that are NOT the revision-1 control.
arms)
  for a in ${2:-$NEW_ARMS}; do
    say "extract $a (eval + safesyn)"
    "$REPO/scripts/r6_extract_arms.sh" "$a" "$ROOT/tables" >"$ROOT/logs/extract_$a.log" 2>&1
    "$0" extras "$a" >>"$ROOT/logs/extract_$a.log" 2>&1
  done ;;

# Eval roots + fit-chain parquets, through the R6 packer with this lane's
# manifest keys. ZEN_BUILD_COMMIT names the commit the EXTRACTOR was built at.
pack)
  for a in ${2:-$ARMS}; do
    R6_ARM_KEY=r6b_f17_arm \
    R6_ARM_SELECTOR_ENV=ZENSIM_HF_GAIN \
    R6_ARM_SELECTOR_DESC="hf_gain_form::HfGainForm; ratio == the shipped revision-1 form" \
    R6_LANE="R6b F17-arm decision, docs/PLAN_FEATURE_REV2_2026-09-05.md section 11" \
    ZEN_BUILD_COMMIT="${R6B_BUILD_COMMIT:?set R6B_BUILD_COMMIT to the extractor commit}" \
    python3 "$REPO/scripts/r6_pack_arm.py" "$a" "$ROOT"
  done ;;

fit)   for a in ${2:-$ARMS}; do "$REPO/scripts/r6_fit_arms.sh" "$a" "$ROOT"; done ;;

eval)  R6_ARMS="${2:-$ARMS}" R6_VARIANTS="$VARIANTS" "$REPO/scripts/r6_eval_arms.sh" "$ROOT" ;;

dial)  R6_ARM_ENV=ZENSIM_HF_GAIN R6_ARMS="${3:-$ARMS}" R6_VARIANTS="$VARIANTS" \
         "$REPO/scripts/r6_dial_arms.sh" "${2:-grade}" "$ROOT" ;;

delta) python3 "$REPO/scripts/r6b_arm_delta.py" --tables "$ROOT/tables" \
         --arms "$(echo "$NEW_ARMS" | tr ' ' ,)" --out "$ROOT/arm_delta_all.json" ;;

decide) python3 "$REPO/scripts/r6b_decide.py" --root "$ROOT" \
         --arms "$(echo "$ARMS" | tr ' ' ,)" \
         --variants "$(echo "$VARIANTS" | tr ' ' ,)" \
         --out "$ROOT/decide.json" ;;

*) echo "unknown step: $1" >&2; exit 2 ;;
esac
say "step $1 done"
