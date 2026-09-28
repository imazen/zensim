#!/usr/bin/env bash
# Replay the REGISTERED spatial coherence cells on the corrected extraction
# (issue #61's "re-run the registered failed/control spatial cells through the
# Rust surface and report the remaining failures honestly").
#
# WHAT THIS IS, AND IS NOT
# ------------------------
# The bakes in these cells were fit against REVISION 1 features. Revision 3
# changes what the extractor emits, so scoring them at revision 3 prices
# revision-1 coefficients against revision-3 features. `BakeScorer` refuses
# that by design, and this driver deliberately arms the diagnostic bypass
# (`cross-revision-diagnostic` feature + ZENSIM_CROSS_REVISION_DIAGNOSTIC=1) to
# get past it. Every process therefore prints a stderr line saying so, and it
# is captured beside the results.
#
# The resulting M2/M3 measure WHAT THE EXTRACTION CHANGE DOES TO A FIXED MODEL.
# They are NOT a qualification of revision 3 and NOT model quality. A pass here
# would not mean the spatial bars are met by a revision-3 candidate; only a
# refit and a full screen can say that.
#
# Usage: scripts/bench/rev3_spatial_replay.sh <out-dir> [commands.json]
set -euo pipefail

OUT="${1:?usage: rev3_spatial_replay.sh <out-dir> [commands.json]}"
CMDS="${2:-/mnt/v/output/zensim/nonmax-diagnosis-2026-09-08/COMMANDS.json}"
FEATURES="custom-profiles,feature-regime-v2,cross-revision-diagnostic"

mkdir -p "$OUT"
cargo build --release -p zensim --example diffmap_block_coherence --features "$FEATURES"
BIN="${CARGO_TARGET_DIR:-target}/release/examples/diffmap_block_coherence"
[ -x "$BIN" ] || { echo "missing $BIN" >&2; exit 2; }
sha256sum "$BIN" | tee "$OUT/binary.sha256"

python3 - "$CMDS" "$OUT" "$BIN" <<'PY'
import json, os, subprocess, sys
cmds_path, out, binary = sys.argv[1], sys.argv[2], sys.argv[3]
cells = json.load(open(cmds_path))
rows = []
for c in cells:
    argv = list(c["cmd"])
    cell = os.path.basename(argv[-1]).removesuffix(".json")
    argv[0] = binary
    for rev in ("1", "3"):
        dest = os.path.join(out, f"{cell}.rev{rev}.json")
        argv[-1] = dest
        env = dict(os.environ, ZENSIM_FORMULA_REV=rev,
                   ZENSIM_CROSS_REVISION_DIAGNOSTIC="1", RAYON_NUM_THREADS="8")
        log = os.path.join(out, f"{cell}.rev{rev}.log")
        with open(log, "w") as fh:
            rc = subprocess.call(argv, env=env, stdout=fh, stderr=subprocess.STDOUT)
        row = {"cell": cell, "revision": int(rev), "exit": rc}
        if rc == 0 and os.path.exists(dest):
            d = json.load(open(dest))
            row.update({k: d.get(k) for k in ("m2", "m3a", "m3f", "sse", "blocks")})
        rows.append(row)
        print(json.dumps(row), flush=True)
json.dump(rows, open(os.path.join(out, "SUMMARY.json"), "w"), indent=1)
print(f"wrote {out}/SUMMARY.json ({len(rows)} runs)")
PY
