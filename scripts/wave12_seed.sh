#!/usr/bin/env bash
#
# wave12_seed.sh <seed> — ONE wave-12 training run (SOTA-944 WAVE 12,
# benchmarks/sota944_campaign_2026-08-03.md REGISTERED APPENDIX AC + AD).
#
# WAVE 12 = the wave-11 recipe (wave-10 arm L9, corrected mix) + exactly ONE
# new TRAIN-ONLY leg: avif944 (AC.2 + AC.R1 amendment 1). This driver
# re-declares NOTHING: it asks the committed wave11_seed.sh for its argv
# (WAVE11_ECHO=1) and changes exactly TWO things —
#   1. inserts ONE `--group avif944:<leg>:1.1043:0.0:both` pair immediately
#      after the last existing --group pair (weight = the bigcodec-leg
#      row-count convention, 0.5 * 459780/208169 = 1.1043; computed + recorded
#      by scripts/canonical_corpus/build_avif944_leg.py);
#   2. replaces the --out value (…/W11_s<seed>.bin → …/W12_s<seed>.bin).
# Token-for-token identity with the wave-11 driver is therefore STRUCTURAL;
# the recorded pre-flight is a token diff committed to
# benchmarks/wave12/echo_verify_2026-08-21.txt.
#
# Registered seeds (AD): 4201 4203 4205 4207 4209 4211 — disjoint from every
# prior campaign seed family (grep-verified; the doc's "4203" hits are
# coincidental 0.4203 KonJND values, same class as K.2's 0.4111 note).
#
#   WAVE12_ECHO=1 scripts/wave12_seed.sh 4201   # print the argv, one per line
set -euo pipefail

SEED=${1:?usage: wave12_seed.sh <seed>}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRAIN=${ZL_TRAIN:-${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/zensim_mlp_train}
OUT=${SOTA944_OUT:-/mnt/v/output/zensim/bakes/sota944/bakes}
LEG=${AVIF944_LEG:-/mnt/v/zen/zensim-training/avif944-2026-08-07/avif944_leg_944.parquet}
LEG_W=${AVIF944_W:-1.1043}          # AD.R outcome (c): half-weight follow-up = AVIF944_W=0.5522
mkdir -p "$OUT"
BAKE="$OUT/W12${WAVE12_TAG:-}_s${SEED}.bin"  # WAVE12_TAG=hw -> W12hw_s<seed>.bin (weight-variant, no collision)
[[ -f "$BAKE" ]] && { echo "exists: $BAKE"; exit 0; }

# Wave-11's own --out short-circuit must never fire on a real W11 bake; point
# its echo at scratch (we replace the token below anyway).
ECHO_OUT=${WAVE12_ECHO_OUT:-$HOME/tmp/wave12/echo-scratch}
mkdir -p "$ECHO_OUT"
mapfile -t BASE < <(
  WAVE11_ECHO=1 SOTA944_OUT="$ECHO_OUT" "$REPO_ROOT/scripts/wave11_seed.sh" "$SEED"
)
[[ ${#BASE[@]} -gt 4 ]] || { echo "wave11_seed.sh echo produced no argv" >&2; exit 2; }

# Insert the avif944 group pair immediately after the LAST existing --group
# pair, and assert exactly one insertion happened.
ARGS=(); INSERTED=0
last_group_val_idx=-1
for i in "${!BASE[@]}"; do
  [[ "${BASE[$i]}" == "--group" ]] && last_group_val_idx=$((i + 1))
done
[[ $last_group_val_idx -ge 1 ]] || { echo "no --group pairs found in wave-11 argv" >&2; exit 2; }
for i in "${!BASE[@]}"; do
  ARGS+=("${BASE[$i]}")
  if [[ $i -eq $last_group_val_idx ]]; then
    ARGS+=("--group" "avif944:${LEG}:${LEG_W}:0.0:both")
    INSERTED=$((INSERTED + 1))
  fi
done
[[ $INSERTED -eq 1 ]] || { echo "inserted $INSERTED avif944 groups (want exactly 1)" >&2; exit 2; }

# The argv's last token is the --out value; assert that before replacing it.
LAST=$(( ${#ARGS[@]} - 1 ))
[[ "${ARGS[$((LAST-1))]}" == "--out" ]] || {
  echo "unexpected argv tail: ${ARGS[$((LAST-1))]} ${ARGS[$LAST]} (expected --out <path>)" >&2
  exit 2
}
ARGS[$LAST]="$BAKE"

if [[ "${WAVE12_ECHO:-0}" = "1" ]]; then
  printf '%s\n' "${ARGS[@]}"
  exit 0
fi
[[ -f "$LEG" ]] || { echo "missing avif944 leg $LEG (run build_avif944_leg.py first)" >&2; exit 2; }
[[ -x "$TRAIN" ]] || { echo "missing $TRAIN" >&2; exit 2; }
exec "$TRAIN" "${ARGS[@]}"
