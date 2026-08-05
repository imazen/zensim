#!/usr/bin/env bash
#
# featsub_seed.sh <arm> <seed> — ONE feature-subset training run
# (SOTA-944 appendix J, benchmarks/sota944_campaign_2026-08-03.md).
#
# APPENDIX J CHANGES EXACTLY ONE THING relative to the incumbent arm-H recipe:
# it appends either `--keep-features <idx-file>` (Phase A) or `--group-l1
# <lambda>` (Phase B). Nothing else. So this driver does NOT re-declare the
# recipe — it asks the committed `wave7_armH_seed.sh` for arm H's argv
# (WAVE7_ECHO=1), swaps the --out value, and appends the one flag. Token-for-
# token identity with arm H is therefore STRUCTURAL, not a claim to re-check
# by eye. (Same construction as wave10_seed.sh.)
#
#   K944           — the full-width baseline, no flag       seeds 2501 2503 2507
#   K64 K128 K256 K512 K667 — --keep-features top<K>.idx    seeds 2501 2503
#   GL<lambda-tag> — --group-l1 <lambda>                    seeds 2501 2503
#                    (tag maps to a lambda below; e.g. GL0p3 -> 0.3)
#   PILOT<tag>     — GL<tag> with --epochs 12 (calibration only, NOT a result)
#
#   FEATSUB_ECHO=1 scripts/featsub/featsub_seed.sh K64 2501   # print the argv
set -euo pipefail

ARM=${1:?usage: featsub_seed.sh <K944|K64|K128|K256|K512|K667|GL<tag>|PILOT<tag>> <seed>}
SEED=${2:?usage: featsub_seed.sh <arm> <seed>}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TRAIN=${ZL_TRAIN:-${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/zensim_mlp_train}
IDX=${FEATSUB_IDX:-$REPO_ROOT/benchmarks/featsub/idx}
OUT=${FEATSUB_OUT:-/mnt/v/output/zensim/bakes/featsub}
mkdir -p "$OUT"
BAKE="$OUT/${ARM}_s${SEED}.bin"
[[ -f "$BAKE" ]] && { echo "exists: $BAKE"; exit 0; }

# arm -> the ONE appended flag pair ("" = none).
EXTRA=()
EPOCH_OVERRIDE=""
case "$ARM" in
  K944) ;;
  K64|K128|K256|K512|K667)
      F="$IDX/top${ARM#K}.idx"
      [[ -f "$F" ]] || { echo "missing index file $F (run scripts/featsub/topk_from_contrib.py)" >&2; exit 2; }
      EXTRA=(--keep-features "$F") ;;
  GL*|PILOT*)
      TAG=${ARM#PILOT}; TAG=${TAG#GL}
      # tag -> lambda: 'p' is the decimal point (0p3 -> 0.3), so the arm name
      # stays a legal filename while the lambda stays exact.
      LAM=${TAG//p/.}
      [[ "$LAM" =~ ^[0-9]+(\.[0-9]+)?$ ]] || { echo "bad lambda tag '$TAG' in arm $ARM" >&2; exit 2; }
      EXTRA=(--group-l1 "$LAM")
      [[ "$ARM" == PILOT* ]] && EPOCH_OVERRIDE=12 ;;
  *) echo "unknown arm $ARM" >&2; exit 2 ;;
esac

# Ask the wave-7 owner for arm H's argv. Its --out points at a scratch dir so
# its "already exists" short-circuit can never fire on a real arm-H bake; the
# token is overwritten below anyway.
ECHO_OUT=${FEATSUB_ECHO_OUT:-$HOME/tmp/featsub/echo-scratch}
mkdir -p "$ECHO_OUT"
mapfile -t BASE < <(
  WAVE7_ECHO=1 SOTA944_OUT="$ECHO_OUT" "$REPO_ROOT/scripts/wave7_armH_seed.sh" "$SEED"
)
[[ ${#BASE[@]} -gt 4 ]] || { echo "wave7_armH_seed.sh echo produced no argv" >&2; exit 2; }

# Replace the --out value; optionally replace --epochs (pilot only). Assert
# each replacement fired exactly once so a driver drift cannot go unnoticed.
ARGS=(); N_OUT=0; N_EP=0
i=0
while [ $i -lt ${#BASE[@]} ]; do
  tok=${BASE[$i]}
  if [[ "$tok" == "--out" && $((i+1)) -lt ${#BASE[@]} ]]; then
    ARGS+=(--out "$BAKE"); N_OUT=$((N_OUT+1)); i=$((i+2)); continue
  fi
  if [[ -n "$EPOCH_OVERRIDE" && "$tok" == "--epochs" && $((i+1)) -lt ${#BASE[@]} ]]; then
    ARGS+=(--epochs "$EPOCH_OVERRIDE"); N_EP=$((N_EP+1)); i=$((i+2)); continue
  fi
  ARGS+=("$tok"); i=$((i+1))
done
[[ $N_OUT -eq 1 ]] || { echo "expected exactly 1 --out in arm-H argv, saw $N_OUT" >&2; exit 2; }
if [[ -n "$EPOCH_OVERRIDE" ]]; then
  [[ $N_EP -eq 1 ]] || { echo "expected exactly 1 --epochs to override, saw $N_EP" >&2; exit 2; }
fi
ARGS+=("${EXTRA[@]}")

if [[ "${FEATSUB_ECHO:-0}" = "1" ]]; then
  printf '%s\n' "${ARGS[@]}"
  exit 0
fi
echo "[featsub] $ARM seed $SEED -> $BAKE"
exec "$TRAIN" "${ARGS[@]}"
