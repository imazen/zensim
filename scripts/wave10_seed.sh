#!/usr/bin/env bash
#
# wave10_seed.sh <arm> <seed> — ONE wave-10 training run (SOTA-944 WAVE 10,
# benchmarks/sota944_campaign_2026-08-03.md REGISTERED APPENDIX H).
#
# WAVE 10 CHANGES EXACTLY TWO THINGS relative to the incumbent arm-H recipe:
#   1. the `kadid` group points at the ORIENTATION-CORRECTED table (Appendix H
#      part 1 — the ext lineage stored (5-dmos)/4, the inverse of the correct
#      (dmos-1)/4). Since the corrected table took the canonical NAME, this is
#      in practice a no-op on the argv: the same path now holds fixed bytes.
#   2. for arms L1..L10, exactly ONE `--group` token is removed.
#
# So this driver does not re-declare the recipe. It asks the committed
# `wave7_armH_seed.sh` for arm H's argv (WAVE7_ECHO=1), drops at most one
# --group pair, and replaces the --out value. Token-for-token identity with
# arm H is therefore STRUCTURAL, not a claim to be re-checked by eye.
#
#   L0  — baseline, nothing dropped                  seeds 4001 4003 4007 (k=3)
#   L1  — drop safesyn          L6  — drop kadis     seeds 4001 4003      (k=2)
#   L2  — drop cid22_train      L7  — drop tsafesyn
#   L3  — drop kadid            L8  — drop ttbig
#   L4  — drop tid              L9  — drop tkadis
#   L5  — drop bigcodec         L10 — drop konjnd_bpg
#
# `konjnd_bpg_val` (train_w 0.0) is a validation-only leg, present in EVERY arm
# including L10. It is not one of the ten droppable legs.
#
#   WAVE10_ECHO=1 scripts/wave10_seed.sh L0 4001   # print the argv, one per line
set -euo pipefail

ARM=${1:?usage: wave10_seed.sh <L0|L1..L10> <seed>}
SEED=${2:?usage: wave10_seed.sh <L0|L1..L10> <seed>}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRAIN=${ZL_TRAIN:-${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/zensim_mlp_train}
OUT=${SOTA944_OUT:-/mnt/v/output/zensim/bakes/sota944/bakes}
mkdir -p "$OUT"
BAKE="$OUT/W10${ARM}_s${SEED}.bin"
[[ -f "$BAKE" ]] && { echo "exists: $BAKE"; exit 0; }

# arm -> the group NAME to drop ("" = drop nothing).
case "$ARM" in
  L0)  DROP="" ;;
  L1)  DROP="safesyn" ;;
  L2)  DROP="cid22_train" ;;
  L3)  DROP="kadid" ;;
  L4)  DROP="tid" ;;
  L5)  DROP="bigcodec" ;;
  L6)  DROP="kadis" ;;
  L7)  DROP="tsafesyn" ;;
  L8)  DROP="ttbig" ;;
  L9)  DROP="tkadis" ;;
  L10) DROP="konjnd_bpg" ;;
  *)   echo "unknown arm $ARM (want L0..L10)" >&2; exit 2 ;;
esac

# Ask the wave-7 owner for arm H's argv. Its --out is pointed at a scratch dir so
# its own "already exists" short-circuit can never fire on a real arm-H bake; we
# overwrite that token below anyway.
ECHO_OUT=${WAVE10_ECHO_OUT:-$HOME/tmp/wave10/echo-scratch}
mkdir -p "$ECHO_OUT"
mapfile -t BASE < <(
  WAVE7_ECHO=1 SOTA944_OUT="$ECHO_OUT" "$REPO_ROOT/scripts/wave7_armH_seed.sh" "$SEED"
)
[[ ${#BASE[@]} -gt 4 ]] || { echo "wave7_armH_seed.sh echo produced no argv" >&2; exit 2; }

# Drop exactly one `--group <name>:...` PAIR, and assert exactly one was dropped.
ARGS=(); DROPPED=0
i=0
while [ $i -lt ${#BASE[@]} ]; do
  tok=${BASE[$i]}
  if [[ -n "$DROP" && "$tok" == "--group" && $((i+1)) -lt ${#BASE[@]} \
        && "${BASE[$((i+1))]}" == "$DROP:"* ]]; then
    DROPPED=$((DROPPED+1)); i=$((i+2)); continue
  fi
  ARGS+=("$tok"); i=$((i+1))
done
if [[ -n "$DROP" ]]; then
  [[ $DROPPED -eq 1 ]] || { echo "arm $ARM: dropped $DROPPED groups named '$DROP' (want exactly 1)" >&2; exit 2; }
else
  [[ $DROPPED -eq 0 ]] || { echo "L0 must drop nothing, dropped $DROPPED" >&2; exit 2; }
fi

# The driver's last token is the --out value; assert that before replacing it.
LAST=$(( ${#ARGS[@]} - 1 ))
[[ "${ARGS[$((LAST-1))]}" == "--out" ]] || {
  echo "unexpected argv tail: ${ARGS[$((LAST-1))]} ${ARGS[$LAST]} (expected --out <path>)" >&2
  exit 2
}
ARGS[$LAST]="$BAKE"

if [[ "${WAVE10_ECHO:-0}" = "1" ]]; then
  printf '%s\n' "${ARGS[@]}"
  exit 0
fi
[[ -x "$TRAIN" ]] || { echo "missing $TRAIN" >&2; exit 2; }
exec "$TRAIN" "${ARGS[@]}"
