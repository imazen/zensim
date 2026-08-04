#!/usr/bin/env bash
#
# wave9_seed.sh <arm> <seed> — ONE wave-9 training run (SOTA-944 WAVE 9,
# benchmarks/sota944_campaign_2026-08-03.md amendment 10, commit 0194aac3).
#
# WAVE 9 CHANGES EXACTLY ONE THING relative to wave-8 arm C: WHICH of the 54
# winsor windows the shared pooled fit is applied to. So this driver does not
# re-declare the recipe — it asks the committed `wave8_seed.sh C <seed>` for
# its argv (WAVE8_ECHO=1) with the screen substituted, and replaces only the
# `--out` token. Token-for-token identity with W8C is therefore STRUCTURAL,
# not a claim to be re-checked by eye.
#
#   arm A — the wave-8 refit screen, all 54 windows (replicates W8C at new seeds)
#   arm B — the 24 DEGENERATE (append-block) windows refit; the 30 fold ones
#           stay inherited          => "unstick the force-killed features"
#   arm C — the 30 NON-DEGENERATE (fold-block) windows refit; the 24 append
#           ones stay [0,0]         => "un-clip the fold block"
#
# B and C are exact set-complements over one shared fit, so their emitted
# screens satisfy  B (+) C = A  disjointly (§10.2, verified by file identity).
#
# Registered seeds: A/B {3301,3303,3307}; C {3301}. Tag: W9<arm>_s<seed>.
#
#   WAVE9_ECHO=1 scripts/wave9_seed.sh A 3301   # print the argv, one per line
set -euo pipefail

ARM=${1:?usage: wave9_seed.sh <A|B|C> <seed>}
SEED=${2:?usage: wave9_seed.sh <A|B|C> <seed>}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRAIN=${ZL_TRAIN:-${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/zensim_mlp_train}
OUT=${SOTA944_OUT:-/mnt/v/output/zensim/bakes/sota944/bakes}
mkdir -p "$OUT"
BAKE="$OUT/W9${ARM}_s${SEED}.bin"
[[ -f "$BAKE" ]] && { echo "exists: $BAKE"; exit 0; }

# The three frozen screens (§10.2). All three come from ONE pooled fit over the
# registered §9.1 corpus; they differ only in the applied subset.
case "$ARM" in
  A) SCREEN=${WAVE9_SCREEN_A:-$REPO_ROOT/benchmarks/wave8/refit_screen_tokens.txt} ;;
  B) SCREEN=${WAVE9_SCREEN_B:-$REPO_ROOT/benchmarks/wave9/w9_degen24_screen_tokens.txt} ;;
  C) SCREEN=${WAVE9_SCREEN_C:-$REPO_ROOT/benchmarks/wave9/w9_fold30_screen_tokens.txt} ;;
  *) echo "unknown arm $ARM (want A|B|C)" >&2; exit 2 ;;
esac
[[ -f "$SCREEN" ]] || { echo "missing screen $SCREEN" >&2; exit 2; }

# Ask the wave-8 owner for arm C's argv with our screen substituted. Its --out
# is pointed at a scratch dir so its own "already exists" short-circuit can
# never fire on a real wave-8 bake; we overwrite that token below anyway.
ECHO_OUT=${WAVE9_ECHO_OUT:-$HOME/tmp/wave9/echo-scratch}
mkdir -p "$ECHO_OUT"
mapfile -t ARGS < <(
  WAVE8_ECHO=1 WAVE8_SCREEN_REFIT="$SCREEN" SOTA944_OUT="$ECHO_OUT" \
    "$REPO_ROOT/scripts/wave8_seed.sh" C "$SEED"
)
[[ ${#ARGS[@]} -gt 4 ]] || { echo "wave8_seed.sh C echo produced no argv" >&2; exit 2; }
# The driver's last token is the --out value; assert that before replacing it.
LAST=$(( ${#ARGS[@]} - 1 ))
[[ "${ARGS[$((LAST-1))]}" == "--out" ]] || {
  echo "unexpected argv tail: ${ARGS[$((LAST-1))]} ${ARGS[$LAST]} (expected --out <path>)" >&2
  exit 2
}
ARGS[$LAST]="$BAKE"

if [[ "${WAVE9_ECHO:-0}" = "1" ]]; then
  printf '%s\n' "${ARGS[@]}"
  exit 0
fi
[[ -x "$TRAIN" ]] || { echo "missing $TRAIN" >&2; exit 2; }
exec "$TRAIN" "${ARGS[@]}"
