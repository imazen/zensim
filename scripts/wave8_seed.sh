#!/usr/bin/env bash
#
# wave8_seed.sh <arm> <seed> — ONE wave-8 training run (SOTA-944 WAVE 8,
# benchmarks/sota944_campaign_2026-08-03.md amendment 9, commit 4668f712).
#
# THE BASE RECIPE is arm H, verbatim: this driver is `wave7_armH_seed.sh` with
# the group set and the transform screen made selectable, and NOTHING else
# touched. `WAVE8_ECHO=1 wave8_seed.sh C <seed>` prints arm C's argv one token
# per line; diffed against `WAVE7_ECHO=1 wave7_armH_seed.sh <seed>` the only
# differences are the 54 refit winsor windows (arm C changes nothing else).
#
#   arm A — breadth-first: DROP bigcodec, kadis, tsafesyn, ttbig, tkadis;
#           kadid 0.5:1.0:rank -> 1.5:1.0:both; REFIT screen
#   arm B — arm A + the konjnd_bpg train/val legs (wave-7 weights); REFIT
#   arm C — base recipe UNCHANGED (all 11 groups, kadid untouched); REFIT
#   arm D — arm B's mix + kadid change; INHERITED screen (the 2x2's 4th corner)
#
# Registered seeds: A/B {3101,3103,3107}; C/D {3101}. Tag: W8<arm>_s<seed>.
set -euo pipefail

ARM=${1:?usage: wave8_seed.sh <A|B|C|D> <seed>}
SEED=${2:?usage: wave8_seed.sh <A|B|C|D> <seed>}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRAIN=${ZL_TRAIN:-${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/zensim_mlp_train}
E=${SOTA944_E:-/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01}
T=${SOTA944_T:-/mnt/v/zen/zensim-training}
K=${SOTA944_K:-/mnt/v/zen/zensim-training/kadis-944-2026-08-01}
TE=${SOTA944_TEACHER:-/mnt/v/output/zensim/bakes/sota944/teacher}
OUT=${SOTA944_OUT:-/mnt/v/output/zensim/bakes/sota944/bakes}
# The two frozen screens (§9.1). Both are committed under benchmarks/wave8/.
SCREEN_REFIT=${WAVE8_SCREEN_REFIT:-$REPO_ROOT/benchmarks/wave8/refit_screen_tokens.txt}
SCREEN_BASE=${WAVE8_SCREEN_BASE:-$REPO_ROOT/benchmarks/wave8/base_screen_tokens.txt}
mkdir -p "$OUT"
BAKE="$OUT/W8${ARM}_s${SEED}.bin"
[[ -f "$BAKE" ]] && { echo "exists: $BAKE"; exit 0; }

case "$ARM" in
  A|B) SCREEN=$SCREEN_REFIT ;;
  C)   SCREEN=$SCREEN_REFIT ;;
  D)   SCREEN=$SCREEN_BASE  ;;
  *)   echo "unknown arm $ARM (want A|B|C|D)" >&2; exit 2 ;;
esac
[[ -f "$SCREEN" ]] || { echo "missing screen $SCREEN" >&2; exit 2; }

FT_ARGS=()
while read -r t; do
  [[ -z "$t" || "$t" == \#* ]] && continue
  FT_ARGS+=(--feature-transform "$t")
done < "$SCREEN"

# kadid leg: arms A/B/D take the registered weight+loss lever, C does not.
case "$ARM" in
  A|B|D) KADID_LEG="kadid:$E/ext_kadid.parquet:1.5:1.0:both" ;;
  C)     KADID_LEG="kadid:$E/ext_kadid.parquet:0.5:1.0:rank" ;;
esac

GROUP_ARGS=(
  --group "safesyn:$E/ext_safesyn_full.parquet:1.0:0.5:both"
  --group "cid22_train:$E/ext_cid22_train201.parquet:1.0:2.0:both"
  --group "$KADID_LEG"
  --group "tid:$E/ext_tid.parquet:0.5:1.0:rank"
)
# The ssim2-mass block — arm C only (arms A/B/D drop it; that IS the wave).
if [[ "$ARM" == "C" ]]; then
  GROUP_ARGS+=(
    --group "bigcodec:$T/tbig_944_200k.parquet:0.5:1.0:both"
    --group "kadis:$K/kadis_944_ssim2_50k.parquet:0.15:1.0:both"
    --group "tsafesyn:$TE/safesyn_teacher944.parquet:0.5:1.0:both"
    --group "ttbig:$TE/tbig_teacher944.parquet:0.5:1.0:both"
    --group "tkadis:$TE/kadis_teacher944.parquet:0.5:1.0:both"
  )
fi
# The wave-7 KonJND legs — arms B, C, D (A is the no-konjnd breadth arm).
if [[ "$ARM" != "A" ]]; then
  GROUP_ARGS+=(
    --group "konjnd_bpg:$E/konjnd_bpg_train_944.parquet:1.2:0.0:both"
    --group "konjnd_bpg_val:$E/konjnd_bpg_val_944.parquet:0.0:1.5:both"
  )
fi

ARGS=(
  "${GROUP_ARGS[@]}"
  --n-hidden-layers 0 --target-column human_score --target-scale 100
  --epochs 120 --pairs-per-epoch 50000 --seed "$SEED"
  --max-features 944 --allow-narrow-features
  --coarse-decay 1e-5
  "${FT_ARGS[@]}"
  --out "$BAKE"
)

if [[ "${WAVE8_ECHO:-0}" = "1" ]]; then
  printf '%s\n' "${ARGS[@]}"
  exit 0
fi
[[ -x "$TRAIN" ]] || { echo "missing $TRAIN" >&2; exit 2; }
exec "$TRAIN" "${ARGS[@]}"
