#!/usr/bin/env bash
#
# wave11_seed.sh <seed> — ONE wave-11 training run (SOTA-944 WAVE 11,
# benchmarks/sota944_campaign_2026-08-03.md REGISTERED APPENDIX K).
#
# WAVE 11 = wave-10 arm L9, the corrected-mix recipe (incumbent arm-H argv +
# the orientation-corrected KADID table + `tkadis` dropped), at seed depth.
# This driver re-declares NOTHING: it asks the committed wave10_seed.sh for
# L9's argv (WAVE10_ECHO=1) and replaces exactly ONE token — the --out value
# (…/W10L9_s<seed>.bin → …/W11_s<seed>.bin). Token-for-token identity with
# the L9 driver is therefore STRUCTURAL; the recorded pre-flight is a token
# diff of `WAVE11_ECHO=1 wave11_seed.sh <s>` against
# `WAVE10_ECHO=1 wave10_seed.sh L9 <s>` showing exactly that difference
# (benchmarks/wave11/echo_verify_2026-08-05.txt).
#
# Registered seeds (K.2): 4101 4103 4105 4107 4109 4111 — disjoint from every
# prior campaign family. Fleet nodes stage locally and override the data
# roots via the SOTA944_E/T/K/TEACHER env vars that wave7_armH_seed.sh
# already honors; SOTA944_OUT points at the node-local out dir.
#
#   WAVE11_ECHO=1 scripts/wave11_seed.sh 4101   # print the argv, one per line
set -euo pipefail

SEED=${1:?usage: wave11_seed.sh <seed>}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRAIN=${ZL_TRAIN:-${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/zensim_mlp_train}
OUT=${SOTA944_OUT:-/mnt/v/output/zensim/bakes/sota944/bakes}
mkdir -p "$OUT"
BAKE="$OUT/W11_s${SEED}.bin"
[[ -f "$BAKE" ]] && { echo "exists: $BAKE"; exit 0; }

mapfile -t BASE < <(WAVE10_ECHO=1 "$REPO_ROOT/scripts/wave10_seed.sh" L9 "$SEED")
[[ ${#BASE[@]} -gt 4 ]] || { echo "wave10_seed.sh L9 echo produced no argv" >&2; exit 2; }

# The L9 argv's last token is the --out value; assert that before replacing it.
LAST=$(( ${#BASE[@]} - 1 ))
[[ "${BASE[$((LAST-1))]}" == "--out" ]] || {
  echo "unexpected argv tail: ${BASE[$((LAST-1))]} ${BASE[$LAST]} (expected --out <path>)" >&2
  exit 2
}
BASE[$LAST]="$BAKE"

if [[ "${WAVE11_ECHO:-0}" = "1" ]]; then
  printf '%s\n' "${BASE[@]}"
  exit 0
fi
[[ -x "$TRAIN" ]] || { echo "missing $TRAIN" >&2; exit 2; }
exec "$TRAIN" "${BASE[@]}"
