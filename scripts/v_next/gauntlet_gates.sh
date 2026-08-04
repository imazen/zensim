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
# Usage: scripts/v_next/gauntlet_gates.sh /path/to/summer_gauntlet.html
set -euo pipefail
HTML="${1:?usage: gauntlet_gates.sh <summer_gauntlet.html>}"
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
