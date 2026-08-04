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
set -euo pipefail
TAG=${1:?teacher tag (ensk2|ensk5|ensG)}
SEED=${2:?seed}
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
B=/mnt/v/output/zensim/bakes/sota944/bakes
SPEC=$B/C_co3a_s1301.bin.spec.json
TDIR=/mnt/v/output/zensim/bakes/sota944/teacher_$TAG
TRAINER=${ZL_TRAIN:-${CARGO_TARGET_DIR:-$REPO_ROOT/target}/release/zensim_mlp_train}
OUT=$B/C_${TAG}_s${SEED}.bin

for f in "$TDIR"/safesyn_teacher944.parquet "$TDIR"/tbig_teacher944.parquet \
         "$TDIR"/kadis_teacher944.parquet "$SPEC" "$TRAINER"; do
    [[ -e $f ]] || { echo "missing: $f" >&2; exit 2; }
done

mapfile -t ARGV < <(python3 - "$SPEC" "$TDIR" "$SEED" "$OUT" <<'PY'
import json, sys
spec, tdir, seed, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
argv = json.load(open(spec))["argv"][1:]          # drop the trainer path
twin = {"tsafesyn": "safesyn", "ttbig": "tbig", "tkadis": "kadis"}
res, i = [], 0
while i < len(argv):
    a = argv[i]
    if a == "--group":
        name = argv[i + 1].split(":", 1)[0]
        if name in twin:
            _, rest = argv[i + 1].split(":", 1)
            _, tail = rest.split(".parquet", 1)          # keep :w:vw:mode
            res += ["--group", f"{name}:{tdir}/{twin[name]}_teacher944.parquet{tail}"]
        else:
            res += ["--group", argv[i + 1]]
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
mkdir -p ~/tmp/wave6/train
nice -n 19 ionice -c 3 "$TRAINER" "${ARGV[@]}" \
    > ~/tmp/wave6/train/C_${TAG}_s${SEED}.log 2>&1
echo "[wave6] $TAG seed $SEED DONE rc=0"
