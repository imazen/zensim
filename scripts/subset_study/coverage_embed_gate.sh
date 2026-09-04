#!/usr/bin/env bash
# Behavioural gate for `zentrain.sample_coverage` (2026-09-04 owner-fix lane).
#
# The block is computed by REPLAYING the pair sampler after training rather
# than by counting inside the training loop, so the claim that needs proving
# is that the replay describes the RUN — not something adjacent to it. These
# are the four checks that pin it:
#
#   1. the embedded digest EQUALS the real run's ZENSIM_SAMPLE_DIGEST=1 output
#      (the replay is exact, not an estimate);
#   2. same --sample-seed + different --init-seed -> byte-identical coverage
#      AND digest (coverage is a property of the SAMPLE stream alone);
#   3. different --sample-seed + same --init-seed -> different coverage
#      (it actually varies with the thing it claims to describe);
#   4. --no-sample-coverage embeds no block at all (absent, never a zero).
#
# Companion to split_seed_gate.sh, which pins the seed split itself.
#
#   TRAIN=target/release/zensim_mlp_train bash scripts/subset_study/coverage_embed_gate.sh
set -euo pipefail
V=${VIEWS:-/mnt/v/zen/zensim-training/ext944-era2r4-2026-09-01/foldapp2_views}
T=${TRAIN:?set TRAIN to a zensim_mlp_train}
O=${1:-$HOME/tmp/coverage_embed_gate}
mkdir -p "$O"

go () { # $1 = label, rest = seed flags
  local l=$1; shift
  ZENSIM_SAMPLE_DIGEST=1 "$T" \
    --group a:"$V/ext_aic4.parquet":1.0:1.0 \
    --group b:"$V/ext_konjnd_jpeg_val.parquet":0.5:1.0 \
    --group v:"$V/ext_sdr25.parquet":0.0:1.0 \
    --max-features 944 --hidden 16 --epochs 3 --pairs-per-epoch 2000 \
    --out "$O/$l.bin" "$@" > "$O/$l.log" 2>&1
}

go A --seed 100
go B --init-seed 999 --sample-seed 100
go C --init-seed 100 --sample-seed 999
go NOCOV --seed 100 --no-sample-coverage

python3 - "$O" <<'PYEOF'
import json, sys, os

O = sys.argv[1]

def block(path, key=b"zentrain.sample_coverage"):
    """Extract one embedded JSON metadata block by brace-matching from the
    first `{` after the key. The bake is a binary container; we deliberately
    do not link zenpredict here — the point is that an outside reader can
    find this block, which is what the board will do."""
    b = open(path, "rb").read()
    i = b.find(key)
    if i < 0:
        return None
    j = b.find(b"{", i)
    depth, e = 0, j
    while e < len(b):
        c = b[e:e + 1]
        if c == b"{":
            depth += 1
        elif c == b"}":
            depth -= 1
            if depth == 0:
                break
        e += 1
    return json.loads(b[j:e + 1].decode("utf-8"))

def run_digest(label):
    with open(os.path.join(O, f"{label}.log"), encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("ZENSIM_SAMPLE_DIGEST "):
                return line.split()[1]
    return None

A, B, C, N = (block(os.path.join(O, f"{x}.bin")) for x in ("A", "B", "C", "NOCOV"))
fail = 0

def chk(name, ok, detail=""):
    global fail
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  [{detail}]"))
    if not ok:
        fail = 1

chk("0 coverage block is embedded", A is not None)
if A is None:
    sys.exit(1)

chk("1 embedded digest == the real run's ZENSIM_SAMPLE_DIGEST",
    A["digest"] == run_digest("A"), f'{A["digest"]} vs {run_digest("A")}')
chk("1b every board-read field is present",
    all(k in A for k in ("schema", "init_seed", "sample_seed", "digest", "full", "early"))
    and all(k in A["full"] for k in ("pooled_row_coverage", "n_pairs", "per_group")))
same = json.dumps(A["full"], sort_keys=True) == json.dumps(B["full"], sort_keys=True)
chk("2 same sample-seed, different init-seed -> IDENTICAL coverage",
    same and A["digest"] == B["digest"], f'{A["digest"]} vs {B["digest"]}')
diff = json.dumps(A["full"], sort_keys=True) != json.dumps(C["full"], sort_keys=True)
chk("3 different sample-seed -> DIFFERENT coverage",
    diff and A["digest"] != C["digest"], f'{A["digest"]} vs {C["digest"]}')
chk("4 --no-sample-coverage embeds NO block", N is None)

print(f'\n# A pooled_row_coverage {A["full"]["pooled_row_coverage"]:.6f} '
      f'/ C {C["full"]["pooled_row_coverage"]:.6f} — two seeds, objectively '
      f'different subsets, which is the whole reason this block exists')
sys.exit(fail)
PYEOF
