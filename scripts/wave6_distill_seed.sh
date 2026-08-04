#!/usr/bin/env bash
# wave6_distill_seed.sh <teacher-tag> <seed> — train ONE arm-F student.
#
# The student is the co3a recipe VERBATIM: the argv is READ from the committed
# `C_co3a_s1301.bin.spec.json` embedded `zentrain.repro`, not re-typed, so the
# 40 WT transform flags + the 24 mask2 flags + every group weight are carried
# byte-for-byte. Exactly three things are substituted:
#
#   --seed <seed>
#   --out  <campaign bakes>/C_ens<tag>_s<seed>.bin
#   the three teacher-twin parquet paths  ->  teacher_<tag>/{safesyn,tbig,kadis}_teacher944.parquet
#
# Registered in SOTA-944 amendment 6 §6.1.
#
# Fleet-portable: the six REAL group paths in the spec argv are absolute
# /mnt/v paths, so a staged node remaps them by prefix through the same env
# vars the arm-C lane script uses (SOTA944_E / _T / _K / _OUT), plus
# SOTA944_TDIR for the teacher twins. Defaults = the local /mnt/v canonicals.
set -euo pipefail
TAG=${1:?teacher tag (ensk2|ensk5|ensG)}
SEED=${2:?seed}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
E=${SOTA944_E:-/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01}
T=${SOTA944_T:-/mnt/v/zen/zensim-training}
K=${SOTA944_K:-/mnt/v/zen/zensim-training/kadis-944-2026-08-01}
OUT_DIR=${SOTA944_OUT:-/mnt/v/output/zensim/bakes/sota944/bakes}
SPEC=${SOTA944_SPEC:-/mnt/v/output/zensim/bakes/sota944/bakes/C_co3a_s1301.bin.spec.json}
TDIR=${SOTA944_TDIR:-/mnt/v/output/zensim/bakes/sota944/teacher_$TAG}
LOGDIR=${SOTA944_LOGDIR:-$HOME/tmp/wave6/train}
TRAINER=${ZL_TRAIN:-${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/zensim_mlp_train}
mkdir -p "$OUT_DIR" "$LOGDIR"
OUT=$OUT_DIR/C_${TAG}_s${SEED}.bin
[[ -f $OUT ]] && { echo "exists: $OUT"; exit 0; }

for f in "$TDIR"/safesyn_teacher944.parquet "$TDIR"/tbig_teacher944.parquet \
         "$TDIR"/kadis_teacher944.parquet "$SPEC" "$TRAINER"; do
    [[ -e $f ]] || { echo "missing: $f" >&2; exit 2; }
done

mapfile -t ARGV < <(python3 - "$SPEC" "$TDIR" "$SEED" "$OUT" "$E" "$T" "$K" <<'PY'
import json, sys
spec, tdir, seed, out, E, T, K = sys.argv[1:8]
argv = json.load(open(spec))["argv"][1:]          # drop the trainer path
twin = {"tsafesyn": "safesyn", "ttbig": "tbig", "tkadis": "kadis"}
# Longest-prefix first so the ext944/kadis roots win over the bare T root.
REMAP = [("/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01", E),
         ("/mnt/v/zen/zensim-training/kadis-944-2026-08-01", K),
         ("/mnt/v/zen/zensim-training", T)]
res, i = [], 0
while i < len(argv):
    a = argv[i]
    if a == "--group":
        name, rest = argv[i + 1].split(":", 1)
        if name in twin:
            _, tail = rest.split(".parquet", 1)          # keep :w:vw:mode
            res += ["--group", f"{name}:{tdir}/{twin[name]}_teacher944.parquet{tail}"]
        else:
            for old, new in REMAP:
                if rest.startswith(old):
                    rest = new + rest[len(old):]
                    break
            res += ["--group", f"{name}:{rest}"]
        i += 2
    elif a == "--seed":
        res += ["--seed", seed]; i += 2
    elif a == "--out":
        res += ["--out", out]; i += 2
    else:
        res.append(a); i += 1
for a in res:
    print(a)
PY
)

echo "[wave6] $TAG seed $SEED -> $OUT  (${#ARGV[@]} args from $SPEC)"
printf '%s\n' "${ARGV[@]}" | grep -A1 -e '--group' || true
nice -n 19 ionice -c 3 "$TRAINER" "${ARGV[@]}" \
    > "$LOGDIR/C_${TAG}_s${SEED}.log" 2>&1
echo "[wave6] $TAG seed $SEED DONE rc=0"
