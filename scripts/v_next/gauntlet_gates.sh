#!/bin/bash
# Gates for the gauntlet offline HTML — run on EVERY regen before shipping the file.
#   gate 1: `node --check` on EVERY extracted <script> block (the page carries two since
#           the ECharts migration: the vendored bundle + the app script — a bad escape in
#           the raw-Python-string app JS blanked the page once, 2026-07-29 e7f929ca, and a
#           corrupt vendor inline would blank it just as silently)
#   gate 2: the DOM-shim render harness (gauntlet_render_check.js) — executes the app
#           script and asserts a non-blank render: chips, scoreboard rows, sections,
#           sortability (real header clicks reorder the ATTACHED tables), ECharts mounts
#           with built option payloads, light+dark chart themes, and the JXL
#           loop-targeting panel when the payload carries it
#   gate 3: STRICT JSON validity of every `*.fulleval.json` the board is built from —
#           `node --check` for data. Python's `json` accepts a bare `NaN`/`Infinity` in
#           BOTH directions, so a producer that used the default `allow_nan=True` could
#           write a file this repo's Python read happily and every strict reader
#           rejected. MEASURED 2026-09-04: `peer_cvvdp.fulleval.json` carried 73 bare
#           `NaN`s, which meant `freeze_check --select` over the whole fulleval dir had
#           NEVER worked, silently, for as long as that row existed. Fixed at the
#           producer (`build_peer_fullevals.py` now dumps `allow_nan=False`); this gate
#           is what stops the next producer re-introducing it.
# Usage: scripts/v_next/gauntlet_gates.sh /path/to/summer_gauntlet.html [fulleval_dir]
#        (fulleval_dir defaults to the board's own; pass "skip" to run gates 1-2 only)
set -euo pipefail
HTML="${1:?usage: gauntlet_gates.sh <summer_gauntlet.html> [fulleval_dir|skip]}"
FULLEVAL_DIR="${2:-/mnt/v/output/zensim/reports/fulleval}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p ~/tmp
D=$(mktemp -d ~/tmp/gauntlet_gate.XXXXXX)
trap 'rm -rf "$D"' EXIT

N_BLOCKS=$(python3 - "$HTML" "$D" <<'PY'
import re, sys
html = open(sys.argv[1], encoding="utf-8").read()
blocks = re.findall(r"<script[^>]*>([\s\S]*?)</script>", html)
assert blocks, "no <script> blocks in the HTML"
for i, b in enumerate(blocks):
    open(f"{sys.argv[2]}/page_{i}.js", "w", encoding="utf-8").write(b)
print(len(blocks))
PY
)
for f in "$D"/page_*.js; do node --check "$f"; done
echo "GATE 1 PASS: node --check ($N_BLOCKS script blocks parse)"
node "$HERE/gauntlet_render_check.js" "$HTML"
echo "GATE 2 PASS: DOM-shim render harness"

# ── gate 3: strict JSON validity of the board's inputs ──────────────────────────────
if [ "$FULLEVAL_DIR" = "skip" ]; then
  echo "gate 3: SKIPPED (fulleval_dir=skip)"
elif [ ! -d "$FULLEVAL_DIR" ]; then
  # A missing dir is a SETUP failure, never a pass: a gate that silently
  # succeeds when it cannot see its inputs is worse than no gate.
  echo "gate 3 FAIL: no fulleval dir at $FULLEVAL_DIR (pass 'skip' to opt out deliberately)" >&2
  exit 2
else
  python3 - "$FULLEVAL_DIR" <<'PY'
import glob, json, os, re, sys

d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "*.fulleval.json")))
if not files:
    print(f"gate 3 FAIL: no *.fulleval.json under {d}", file=sys.stderr)
    sys.exit(2)

# `parse_constant` fires on exactly the three bare literals JSON forbids and
# CPython accepts, so this rejects what a strict reader rejects rather than
# guessing from a regex. The regex below is only for the ERROR MESSAGE.
def reject(tok):
    raise ValueError(f"bare `{tok}` — invalid JSON; serde_json (and every strict "
                     f"reader, `freeze_check --select` included) rejects it. Fix the "
                     f"PRODUCER to dump with allow_nan=False; never hand-edit the file.")

bad = []
for f in files:
    raw = open(f, encoding="utf-8").read()
    try:
        json.loads(raw, parse_constant=reject)
    except Exception as e:
        n = len(re.findall(r'(?<![\w."])(NaN|-?Infinity)(?![\w"])', raw))
        bad.append((os.path.basename(f), f"{e}" + (f" [{n} occurrence(s)]" if n else "")))

if bad:
    print(f"gate 3 FAIL: {len(bad)} of {len(files)} fulleval file(s) are not valid JSON",
          file=sys.stderr)
    for name, why in bad:
        print(f"  {name}: {why}", file=sys.stderr)
    sys.exit(1)
print(f"gate 3: {len(files)} fulleval file(s) are strict-valid JSON")
PY
fi
