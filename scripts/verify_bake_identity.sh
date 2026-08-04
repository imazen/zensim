#!/usr/bin/env bash
# verify_bake_identity.sh <a.bin> <b.bin>
#
# THE numeric-identity gate for any change to the TRAINER.
#
# Two bakes produced by the same recipe + seed must be identical in every
# byte that describes the MODEL. They are legitimately allowed to differ in
# exactly two places inside the mandatory `zentrain.repro` metadata block:
#
#   * `timestamp_epoch` — when the run happened
#   * `argv` / `cwd` / `trainer_source_dir` — where the binary and outputs
#     lived (argv[0] and `--out` are recorded verbatim, by design)
#
# So this script compares (a) every byte OUTSIDE the repro JSON, and (b) the
# repro JSON itself with those provenance keys removed. Anything else — a
# single weight, `best_val`, a spline knot, a hyperparameter — fails.
#
# This exists because the naive `cmp a.bin b.bin` ALWAYS fails: the 2026-08-04
# trainer-perf pass burned a full measurement cycle discovering that two runs
# of the SAME binary differ by 3 bytes (the `--out` path in argv) and that a
# re-run differs by the timestamp. Without this script the real question —
# "did the weights change?" — is unanswerable from the bytes.
#
#   scripts/verify_bake_identity.sh before.bin after.bin
set -euo pipefail
A=${1:?usage: verify_bake_identity.sh <a.bin> <b.bin>}
B=${2:?usage: verify_bake_identity.sh <a.bin> <b.bin>}
exec python3 - "$A" "$B" <<'PY'
import json, sys

a_path, b_path = sys.argv[1], sys.argv[2]
KEY = b"zentrain.repro"
# Provenance keys allowed to differ between two runs of the same recipe.
VOLATILE = {"timestamp_epoch", "argv", "cwd", "trainer_source_dir"}

def split(path):
    """-> (bytes_outside_repro, repro_dict_or_None)"""
    d = open(path, "rb").read()
    k = d.find(KEY)
    if k < 0:
        return d, None
    # The value is the next '{'-rooted JSON object after the key. Scan
    # brace depth so an embedded '}' inside a string can't truncate it.
    start = d.find(b"{", k)
    if start < 0:
        return d, None
    depth, i, in_str, esc = 0, start, False, False
    while i < len(d):
        c = d[i]
        if in_str:
            if esc:
                esc = False
            elif c == 0x5C:
                esc = True
            elif c == 0x22:
                in_str = False
        elif c == 0x22:
            in_str = True
        elif c == 0x7B:
            depth += 1
        elif c == 0x7D:
            depth -= 1
            if depth == 0:
                i += 1
                break
        i += 1
    blob = d[start:i]
    try:
        obj = json.loads(blob.decode("utf-8"))
    except Exception as e:
        print(f"WARN: {path}: repro JSON did not parse ({e}); comparing raw")
        return d, None
    # Zero out the JSON span AND the 8 bytes before the key (the section
    # length header, which moves when argv lengths differ).
    return d[: max(0, k - 8)] + d[i:], obj

oa, ra = split(a_path)
ob, rb = split(b_path)

print(f"A = {a_path}")
print(f"B = {b_path}")

ok = True
if oa == ob:
    print(f"model bytes (outside zentrain.repro): IDENTICAL ({len(oa)} bytes)")
else:
    ok = False
    n = min(len(oa), len(ob))
    first = next((i for i in range(n) if oa[i] != ob[i]), n)
    ndiff = sum(1 for i in range(n) if oa[i] != ob[i]) + abs(len(oa) - len(ob))
    print(f"model bytes DIFFER: {ndiff} bytes, first at offset {first} "
          f"(sizes {len(oa)} vs {len(ob)})")
    if ndiff <= 8 and first < 128:
        print("  HINT: a handful of bytes this early is the ZNPR section-length "
              "table shifting because the embedded `argv` lengths differ. Re-run "
              "both trainings with the SAME binary path and the SAME --out path "
              "(move the output between runs) for a clean comparison.")

if ra is None or rb is None:
    print("zentrain.repro: absent or unparseable in at least one bake — "
          "compared raw above")
else:
    ka, kb = set(ra), set(rb)
    if ka != kb:
        ok = False
        print(f"repro KEYS differ: only-A={sorted(ka - kb)} only-B={sorted(kb - ka)}")
    bad = [k for k in sorted(ka & kb) if k not in VOLATILE and ra[k] != rb[k]]
    if bad:
        ok = False
        print(f"repro NON-VOLATILE fields differ: {bad}")
        for k in bad[:10]:
            print(f"  {k}: {ra[k]!r} != {rb[k]!r}")
    else:
        differing_volatile = sorted(k for k in (ka & kb) if k in VOLATILE and ra[k] != rb[k])
        print("repro non-volatile fields: IDENTICAL "
              f"(ignored provenance keys that differ: {differing_volatile})")
    # best_val is the selection signal — call it out explicitly.
    if "best_val" in ka & kb:
        same = ra["best_val"] == rb["best_val"]
        print(f"best_val: {ra['best_val']!r} vs {rb['best_val']!r} -> "
              f"{'SAME' if same else 'DIFFERENT'}")
        ok = ok and same

print("RESULT: " + ("PASS — same model" if ok else "FAIL"))
sys.exit(0 if ok else 1)
PY
