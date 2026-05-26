#!/usr/bin/env python3
"""CI grep-gate enforcing that every pandas `pd.merge` / `df.merge` call in
scripts/ is either in join_safety.py / its tests, or carries a per-line
`# joinsafety-ok: <reason>` allow-list comment.

Used by .github/workflows/joinsafety.yml. Can also be run locally:

    python3 scripts/canonical_corpus/joinsafety_gate.py

Exits 0 on PASS, 1 on any unguarded call. Strips per-line comments BEFORE
pattern matching so narrative mentions of `.merge(` inside docstrings /
comments don't trip the gate (only real code does).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# scripts/ relative to the repo root (we're invoked from there in CI).
SCRIPTS_DIR = Path("scripts")
ALLOW_FILES = {"join_safety.py", "test_join_safety.py", "joinsafety_gate.py"}
PATTERN = re.compile(r"pd\.merge\(|\.merge\(")


def main() -> int:
    if not SCRIPTS_DIR.exists():
        print(
            f"joinsafety_gate: scripts/ not found at {SCRIPTS_DIR.absolute()} — "
            f"run from repo root.",
            file=sys.stderr,
        )
        return 2

    offenders: list[str] = []
    for path in SCRIPTS_DIR.rglob("*.py"):
        if path.name in ALLOW_FILES:
            continue
        try:
            text = path.read_text()
        except (OSError, UnicodeDecodeError) as e:
            print(f"WARN: could not read {path}: {e}", file=sys.stderr)
            continue
        for lineno, raw in enumerate(text.splitlines(), 1):
            # Strip line-end comments before the pattern test — comments
            # mentioning `.merge(` are common and harmless.
            code = raw.split("#", 1)[0]
            if PATTERN.search(code):
                # Per-line opt-out: '# joinsafety-ok: <reason>' anywhere
                # on the same line.
                if "joinsafety-ok" in raw:
                    continue
                offenders.append(f"{path}:{lineno}: {raw.strip()}")

    if offenders:
        print("::error::Unguarded pd.merge / .merge( calls in scripts/:")
        for o in offenders:
            print(o)
        print()
        print("::error::Route through scripts/canonical_corpus/join_safety.py:")
        print("  - safe_metric_join(...) — per-pair metric join with full key")
        print("  - attach_per_source_features(...) — per-source 1-to-many attach")
        print("  - attach_metric_positional(...) — positional alignment")
        print("  - guard_metric_table(...) — post-join Mode-A+B guard")
        print()
        print(
            "Or, if your call is genuinely safe (NOT a corpus/training data "
            "join), append on the same line: # joinsafety-ok: <reason>"
        )
        return 1

    print(f"PASS: scanned {SCRIPTS_DIR}/ — no unguarded pd.merge / .merge( found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
