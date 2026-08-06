#!/usr/bin/env bash
#
# sparsehf_seed.sh <CS<lambda-tag>> <seed> — ONE appendix-R arm-R2 training run
# (sparse distillation of C; benchmarks/sota944_campaign_2026-08-03.md APPENDIX R).
#
# R2 = the appendix-J group-lasso cell VERBATIM with ONLY the teacher swapped
# (EM4 -> C = W10L9_s4003 raw, twins built by build_teacher944.py --tag csparse).
# This driver does NOT re-declare the recipe: it asks the committed appendix-J
# owner (featsub_seed.sh, FEATSUB_ECHO=1) for the GL<tag> argv with
# SOTA944_TEACHER pointed at the csparse twins, swaps the --out token, and
# execs. Token-for-token identity with the matched GL cell is therefore
# STRUCTURAL: the only differing tokens are the 3 teacher parquet paths and
# --out. `SPARSEHF_CHECK=1` prints exactly that token diff instead of training.
#
#   CS0p3 -> --group-l1 0.3   CS1 -> 1   CS2 -> 2   CS4 -> 4
set -euo pipefail

ARM=${1:?usage: sparsehf_seed.sh <CS<tag>> <seed>}
SEED=${2:?usage: sparsehf_seed.sh <arm> <seed>}
[[ "$ARM" == CS* ]] || { echo "arm must be CS<lambda-tag>, got $ARM" >&2; exit 2; }
GLARM="GL${ARM#CS}"
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
TRAIN=${ZL_TRAIN:-${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/zensim_mlp_train}
TEACHER=${SPARSEHF_TEACHER:-/mnt/v/output/zensim/bakes/sota944/teacher_csparse}
OUT=${SPARSEHF_OUT:-/mnt/v/output/zensim/bakes/sparsehf}
mkdir -p "$OUT"
BAKE="$OUT/${ARM}_s${SEED}.bin"
[[ -f "$BAKE" ]] && { echo "exists: $BAKE"; exit 0; }
for t in safesyn tbig kadis; do
  [[ -s "$TEACHER/${t}_teacher944.parquet" ]] || { echo "missing teacher twin $TEACHER/${t}_teacher944.parquet (run build_teacher944.py --tag csparse)" >&2; exit 2; }
done

echo_argv() { # $1 = teacher dir
  SOTA944_TEACHER=$1 FEATSUB_ECHO=1 \
  FEATSUB_OUT=$HOME/tmp/sparsehf/echo-bakes FEATSUB_ECHO_OUT=$HOME/tmp/sparsehf/echo-scratch \
    "$REPO_ROOT/scripts/featsub/featsub_seed.sh" "$GLARM" "$SEED"
}
mkdir -p "$HOME/tmp/sparsehf/echo-bakes" "$HOME/tmp/sparsehf/echo-scratch"
mapfile -t BASE < <(echo_argv "$TEACHER")
[[ ${#BASE[@]} -gt 4 ]] || { echo "featsub_seed.sh echo produced no argv" >&2; exit 2; }

if [[ "${SPARSEHF_CHECK:-0}" = "1" ]]; then
  # Structural-identity audit: diff vs the matched appendix-J GL argv (EM4 teacher).
  mapfile -t GLBASE < <(echo_argv "/mnt/v/output/zensim/bakes/sota944/teacher")
  diff <(printf '%s\n' "${GLBASE[@]}") <(printf '%s\n' "${BASE[@]}") || true
  exit 0
fi

ARGS=(); N_OUT=0
i=0
while [ $i -lt ${#BASE[@]} ]; do
  tok=${BASE[$i]}
  if [[ "$tok" == "--out" && $((i+1)) -lt ${#BASE[@]} ]]; then
    ARGS+=(--out "$BAKE"); N_OUT=$((N_OUT+1)); i=$((i+2)); continue
  fi
  ARGS+=("$tok"); i=$((i+1))
done
[[ $N_OUT -eq 1 ]] || { echo "expected exactly 1 --out, saw $N_OUT" >&2; exit 2; }

if [[ "${SPARSEHF_ECHO:-0}" = "1" ]]; then
  printf '%s\n' "${ARGS[@]}"
  exit 0
fi
[[ -x "$TRAIN" ]] || { echo "missing $TRAIN" >&2; exit 2; }
echo "[sparsehf] $ARM seed $SEED (matched GL sibling: ${GLARM}_s${SEED}) -> $BAKE"
exec "$TRAIN" "${ARGS[@]}"
