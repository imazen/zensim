#!/usr/bin/env bash
# verify_verdict_identity.sh <a.full.json> <b.full.json> [rel_tol]
#
# THE numeric-identity gate for any change to bake_verdict / the eval path.
#
# Compares two `--full-json` verdicts field-by-field over the ENTIRE JSON
# tree (every scalar, every array element, at every depth) and fails loud on
# the first structural difference or any numeric mismatch above `rel_tol`
# (default 0.0 = bit-identical). Non-numeric fields that legitimately differ
# between runs — timestamps, the `name`, absolute output paths, wall time —
# are excluded by key name and listed in the output so the exclusion is
# never silent.
#
# Why it exists: an eval speedup that changes a number is a REGRESSION, not
# an optimization — every published SOTA-944 cell was produced by this
# binary, so "faster" is only admissible alongside "same". The 2026-08-04
# parallelization pass (rayon over corpora / rows / bands / bootstrap) is
# gated on this script reporting 0 mismatches across ~62k numeric fields.
#
#   scripts/verify_verdict_identity.sh before.json after.json
set -euo pipefail
A=${1:?usage: verify_verdict_identity.sh <a.full.json> <b.full.json> [rel_tol]}
B=${2:?usage: verify_verdict_identity.sh <a.full.json> <b.full.json> [rel_tol]}
TOL=${3:-0}
exec python3 - "$A" "$B" "$TOL" <<'PY'
import json, math, sys

a_path, b_path, tol = sys.argv[1], sys.argv[2], float(sys.argv[3])

# Keys whose values are run-scoped, not results. Anything NOT in here is
# compared; growing this list is a conscious act (and it prints below).
VOLATILE = {
    "generated",           # timestamp
    "wall_time_s",         # the thing we are changing
    "elapsed_s",
    "verdict_path", "json_path", "html_path", "output_path",
}
# Exact PATHS (not bare keys) that are run-scoped. `.name` is the `--name`
# label the caller chose; it must NOT be matched by bare key, or every
# `.repro.inputs[i].name` (the training group names — real content) would be
# skipped with it.
VOLATILE_PATHS = {".name"}

def walk(node, path, out):
    if isinstance(node, dict):
        for k, v in node.items():
            if k in VOLATILE or f"{path}.{k}" in VOLATILE_PATHS:
                out.setdefault("_skipped", set()).add(f"{path}.{k}")
                continue
            walk(v, f"{path}.{k}", out)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            walk(v, f"{path}[{i}]", out)
    else:
        out["flat"][path] = node

def flatten(doc):
    out = {"flat": {}}
    walk(doc, "", out)
    return out["flat"], out.get("_skipped", set())

fa, ska = flatten(json.load(open(a_path)))
fb, skb = flatten(json.load(open(b_path)))

ka, kb = set(fa), set(fb)
only_a, only_b = sorted(ka - kb), sorted(kb - ka)
mismatch, numeric, nonnum = [], 0, 0
for k in sorted(ka & kb):
    x, y = fa[k], fb[k]
    if isinstance(x, bool) or isinstance(y, bool) or not isinstance(x, (int, float)) \
       or not isinstance(y, (int, float)):
        nonnum += 1
        if x != y:
            mismatch.append((k, x, y, "non-numeric"))
        continue
    numeric += 1
    if x == y:
        continue
    if math.isnan(x) and math.isnan(y):
        continue
    if tol > 0:
        d = abs(x - y) / max(abs(x), abs(y), 1e-300)
        if d <= tol:
            continue
        mismatch.append((k, x, y, f"rel={d:.3e}"))
    else:
        mismatch.append((k, x, y, "not bit-identical"))

print(f"A = {a_path}")
print(f"B = {b_path}")
print(f"tolerance      : {'BIT-IDENTICAL' if tol == 0 else f'rel <= {tol:g}'}")
print(f"numeric fields : {numeric}")
print(f"other fields   : {nonnum}")
print(f"skipped (volatile): {len(ska | skb)} -> {sorted(ska | skb)}")
if only_a or only_b:
    print(f"STRUCTURE DIFFERS: {len(only_a)} only in A, {len(only_b)} only in B")
    for k in (only_a[:10] + only_b[:10]):
        print(f"  {k}")
if mismatch:
    print(f"MISMATCHES: {len(mismatch)}")
    for k, x, y, why in mismatch[:40]:
        print(f"  {k}: {x!r} != {y!r} ({why})")
    if len(mismatch) > 40:
        print(f"  ... and {len(mismatch) - 40} more")

ok = not mismatch and not only_a and not only_b
print("RESULT: " + ("PASS — identical" if ok else "FAIL"))
sys.exit(0 if ok else 1)
PY
