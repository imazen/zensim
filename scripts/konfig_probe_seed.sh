#!/usr/bin/env bash
#
# konfig_probe_seed.sh <w25|w75> <seed> — ONE Appendix-L KonFiG weight-probe run
# (benchmarks/sota944_campaign_2026-08-03.md REGISTERED APPENDIX L §L.7,
# pre-reg e93eba04).
#
# The probe recipe = wave-10 arm L9 (the corrected mix wave-11 runs) with
# exactly ONE addition: the KonFiG training leg. This driver re-declares
# NOTHING: it asks the committed wave10_seed.sh for L9's argv (WAVE10_ECHO=1),
# APPENDS one `--group` token pair immediately before `--out`, and replaces
# the `--out` value. Token-for-token identity with L9 is therefore STRUCTURAL;
# the committed pre-flight echo diff shows exactly those two differences
# (benchmarks/konfig/echo_verify_2026-08-05.txt).
#
#   konfig group: konfig:<ext944-root>/konfig_944.parquet:<w>:0.0:rank
#     - train_w w ∈ {0.25, 0.75}  (registered doses, §L.7)
#     - val_w 0.0                 (validation objective IDENTICAL to W11's)
#     - loss_mode `rank`          (the kadid/tid human-label convention: the
#                                  JND-grid unit is not the score unit, so only
#                                  ordering is trainable; stated in L results)
#
#   Cells: KFG25_s4101 KFG25_s4103 KFG75_s4101 KFG75_s4103 — seeds are
#   wave-11's first two, so every comparison is paired by seed against
#   W11_s4101 / W11_s4103 (identical recipe, zero KonFiG).
#
#   KONFIG_ECHO=1 scripts/konfig_probe_seed.sh w25 4101   # print argv only
set -euo pipefail

DOSE=${1:?usage: konfig_probe_seed.sh <w25|w75> <seed>}
SEED=${2:?usage: konfig_probe_seed.sh <w25|w75> <seed>}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
TRAIN=${ZL_TRAIN:-${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/zensim_mlp_train}
OUT=${SOTA944_OUT:-/mnt/v/output/zensim/bakes/sota944/bakes}
KONFIG_TABLE=${KONFIG_TABLE:-/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/konfig_944.parquet}
mkdir -p "$OUT"

case "$DOSE" in
  w25) W=0.25; TAG=KFG25 ;;
  w75) W=0.75; TAG=KFG75 ;;
  *)   echo "unknown dose $DOSE (want w25|w75)" >&2; exit 2 ;;
esac
BAKE="$OUT/${TAG}_s${SEED}.bin"
[[ -f "$BAKE" ]] && { echo "exists: $BAKE"; exit 0; }
[[ -f "$KONFIG_TABLE" ]] || { echo "missing konfig table $KONFIG_TABLE" >&2; exit 2; }

mapfile -t BASE < <(WAVE10_ECHO=1 "$REPO_ROOT/scripts/wave10_seed.sh" L9 "$SEED")
[[ ${#BASE[@]} -gt 4 ]] || { echo "wave10_seed.sh L9 echo produced no argv" >&2; exit 2; }

# The L9 argv's last two tokens are `--out <path>`; assert, then splice the
# konfig group in front of them and replace the out value.
LAST=$(( ${#BASE[@]} - 1 ))
[[ "${BASE[$((LAST-1))]}" == "--out" ]] || {
  echo "unexpected argv tail: ${BASE[$((LAST-1))]} ${BASE[$LAST]} (expected --out <path>)" >&2
  exit 2
}
ARGS=("${BASE[@]:0:$((LAST-1))}")
ARGS+=("--group" "konfig:${KONFIG_TABLE}:${W}:0.0:rank")
ARGS+=("--out" "$BAKE")

# Assert exactly one konfig group and no other change in group count.
N_KONFIG=$(printf '%s\n' "${ARGS[@]}" | grep -c '^konfig:' || true)
[[ "$N_KONFIG" -eq 1 ]] || { echo "expected exactly 1 konfig group, got $N_KONFIG" >&2; exit 2; }

if [[ "${KONFIG_ECHO:-0}" = "1" ]]; then
  printf '%s\n' "${ARGS[@]}"
  exit 0
fi
[[ -x "$TRAIN" ]] || { echo "missing $TRAIN" >&2; exit 2; }
exec "$TRAIN" "${ARGS[@]}"
