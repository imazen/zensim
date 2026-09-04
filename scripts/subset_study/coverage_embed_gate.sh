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
# --pair-sampling stratified: coverage must stop depending on the sample seed.
go S1 --sample-seed 100 --pair-sampling stratified
go S2 --sample-seed 999 --pair-sampling stratified

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

A, B, C, N, S1, S2 = (block(os.path.join(O, f"{x}.bin"))
                      for x in ("A", "B", "C", "NOCOV", "S1", "S2"))
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

# --- 5/6: --pair-sampling stratified ----------------------------------------
# The mode exists to make coverage a property of the RECIPE rather than of the
# seed, so the pair of checks is the whole claim: the descriptors must stop
# moving with --sample-seed while the pair ORDER (the digest) keeps moving.
def cov_fields(b):
    """Coverage descriptors only — NOT the digest, which is order-sensitive and
    is SUPPOSED to differ (the seed permutes the within-stratum walk)."""
    return [{k: g[k] for k in ("name", "n_rows", "n_pairs", "rows_touched",
                               "row_coverage", "refs_touched", "cells_touched",
                               "band_pair_counts")}
            for g in b["full"]["per_group"]]

chk("5 stratified: coverage is seed-INVARIANT",
    cov_fields(S1) == cov_fields(S2),
    f'{S1["full"]["pooled_row_coverage"]} vs {S2["full"]["pooled_row_coverage"]}')
chk("5b stratified: the seed still permutes pair ORDER",
    S1["digest"] != S2["digest"], f'{S1["digest"]} == {S2["digest"]}')
chk("6 stratified is recorded in the block",
    S1.get("pair_sampling") == "stratified" and A.get("pair_sampling") == "uniform",
    f'{S1.get("pair_sampling")} / {A.get("pair_sampling")}')
# The contrast that makes check 5 evidence rather than a tautology: under
# UNIFORM the same two seeds DO move the descriptors.
chk("6b uniform (the control): coverage DOES move with the sample seed",
    cov_fields(A) != cov_fields(C))

print(f'\n# A pooled_row_coverage {A["full"]["pooled_row_coverage"]:.6f} '
      f'/ C {C["full"]["pooled_row_coverage"]:.6f} — two seeds, objectively '
      f'different subsets, which is the whole reason this block exists')
sys.exit(fail)
PYEOF
