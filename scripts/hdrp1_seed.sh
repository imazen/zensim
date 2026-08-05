#!/usr/bin/env bash
#
# hdrp1_seed.sh <seed> — ONE appendix-Q HDR-native training run (SOTA-944
# campaign, benchmarks/sota944_campaign_2026-08-03.md REGISTERED APPENDIX Q).
#
# THE RECIPE CHANGES EXACTLY TWO TOKEN PAIRS relative to the wave-11
# corrected-mix recipe (K.1 = wave-10 L9): it APPENDS the two Q.2 groups
#
#   --group hdr:<hdr944-leg traindigits>:1.2:0.0:both
#   --group hdr_val:<hdr944-leg valdigits>:0.0:2.0:both
#
# after the last existing --group pair, and replaces the --out value. It does
# not re-declare anything else: the base argv comes from the committed L9
# owner (`WAVE10_ECHO=1 scripts/wave10_seed.sh L9 <seed>`), so token-for-token
# identity with the corrected-mix recipe is STRUCTURAL (the K.1 pattern), and
# additionally echo-verified in benchmarks/hdrp1/echo_verify_2026-08-05.txt.
#
#   HDRP1_ECHO=1 scripts/hdrp1_seed.sh 6101   # print the argv, one per line
set -euo pipefail

SEED=${1:?usage: hdrp1_seed.sh <seed>}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRAIN=${ZL_TRAIN:-${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/zensim_mlp_train}
OUT=${HDRP1_OUT:-/mnt/v/output/zensim/bakes/hdrp1}
mkdir -p "$OUT"
BAKE="$OUT/Q_hdr944_s${SEED}.bin"
[[ -f "$BAKE" ]] && { echo "exists: $BAKE"; exit 0; }

HDR_LEG=/mnt/v/output/zensim/hdr944-leg
G1="hdr:$HDR_LEG/hdr_v3mix944_traindigits_2026-08-03.parquet:1.2:0.0:both"
G2="hdr_val:$HDR_LEG/hdr_v3mix944_valdigits_2026-08-03.parquet:0.0:2.0:both"

# Ask the committed L9 owner for its argv (scratch --out so its own
# "already exists" short-circuit can never fire on a real bake).
ECHO_OUT=${HDRP1_ECHO_OUT:-$HOME/tmp/hdrp1/echo-scratch}
mkdir -p "$ECHO_OUT"
mapfile -t BASE < <(
  WAVE10_ECHO=1 SOTA944_OUT="$ECHO_OUT" "$REPO_ROOT/scripts/wave10_seed.sh" L9 "$SEED"
)
[[ ${#BASE[@]} -gt 4 ]] || { echo "wave10_seed.sh echo produced no argv" >&2; exit 2; }

# Append the two Q.2 group pairs immediately after the LAST existing
# `--group <spec>` pair; assert exactly one insertion happened.
LASTG=-1
for i in "${!BASE[@]}"; do
  [[ "${BASE[$i]}" == "--group" ]] && LASTG=$i
done
[[ $LASTG -ge 0 ]] || { echo "no --group tokens in base argv" >&2; exit 2; }
ARGS=()
INSERTED=0
for i in "${!BASE[@]}"; do
  ARGS+=("${BASE[$i]}")
  if [[ $i -eq $((LASTG + 1)) ]]; then
    ARGS+=("--group" "$G1" "--group" "$G2")
    INSERTED=$((INSERTED + 1))
  fi
done
[[ $INSERTED -eq 1 ]] || { echo "inserted $INSERTED times (want 1)" >&2; exit 2; }

# The argv tail is `--out <path>`; assert, then replace.
LAST=$(( ${#ARGS[@]} - 1 ))
[[ "${ARGS[$((LAST-1))]}" == "--out" ]] || {
  echo "unexpected argv tail: ${ARGS[$((LAST-1))]} ${ARGS[$LAST]} (expected --out <path>)" >&2
  exit 2
}
ARGS[$LAST]="$BAKE"

if [[ "${HDRP1_ECHO:-0}" = "1" ]]; then
  printf '%s\n' "${ARGS[@]}"
  exit 0
fi
[[ -x "$TRAIN" ]] || { echo "missing $TRAIN" >&2; exit 2; }
exec "$TRAIN" "${ARGS[@]}"
