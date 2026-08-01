#!/bin/bash
# Gates for the gauntlet offline HTML — run on EVERY regen before shipping the file.
#   gate 1: `node --check` on the extracted inline <script> (JS parse — catches the
#           raw-Python-string escape class that blanked the page on 2026-07-29, e7f929ca)
#   gate 2: the DOM-shim render harness (gauntlet_render_check.js) — executes the page
#           script and asserts a non-blank render (chips, scoreboard rows, sections,
#           and the JXL loop-targeting panel when the payload carries it)
# Usage: scripts/v_next/gauntlet_gates.sh /path/to/summer_gauntlet.html
set -euo pipefail
HTML="${1:?usage: gauntlet_gates.sh <summer_gauntlet.html>}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p ~/tmp
D=$(mktemp -d ~/tmp/gauntlet_gate.XXXXXX)
trap 'rm -rf "$D"' EXIT

python3 - "$HTML" "$D/page.js" <<'PY'
import re, sys
html = open(sys.argv[1], encoding="utf-8").read()
m = re.search(r"<script>([\s\S]*)</script>", html)
assert m, "no <script> block in the HTML"
open(sys.argv[2], "w", encoding="utf-8").write(m.group(1))
PY

node --check "$D/page.js"
echo "GATE 1 PASS: node --check (inline JS parses)"
node "$HERE/gauntlet_render_check.js" "$HTML"
echo "GATE 2 PASS: DOM-shim render harness"
