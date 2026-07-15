#!/usr/bin/env python3
"""Fail the build on scripts that cannot possibly run.

WHY THIS EXISTS. On 2026-07-15 an audit of `scripts/v_next/` found 25 of 130
scripts pinned to binaries in sibling worktrees that had since been deleted
(`zensim--cross-codec-metric`, `--v10`, `--eval-accel`, `--picker-train`, ...).
They had been unrunnable for weeks. One more (`metric_compare_report.py`) had
not *parsed* since a bulk sed inserted an unescaped `<meta charset="utf-8">`
into a Python string literal — also unnoticed, because nobody ran it.

The mechanism is structural, not careless:

  1. an agent opens sibling worktree `zensim--foo`
  2. it writes a script hardcoding `/home/lilith/work/zen/zensim--foo/target/release/bar`
  3. the worktree is cleaned up (correctly! the cleanup rule is mandatory)
  4. the script stays, permanently dead, and nothing notices

CLAUDE.md's worktree-cleanup rule covers the worktree. It does not cover the
scripts that pointed INTO it. This linter closes that gap: a dead reference
fails here instead of rotting for weeks and inflating the directory into
something nobody can read.

    python3 scripts/lint_scripts.py            # report + exit 1 on failure
    python3 scripts/lint_scripts.py --list     # report only, always exit 0

Checks:
  PARSE     — the file is valid Python
  DEAD-WT   — references a `zensim--*` sibling worktree that no longer exists
  DEAD-BIN  — hardcodes a `target/{release,debug}/...` path that does not exist
              AND whose source cannot be found

A hardcoded build artifact under THIS repo's own `target/` is only a warning:
it may simply not be built yet. A path under a *deleted worktree* is fatal —
that one can never come back.
"""
from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
ZEN = ROOT.parent

PATH_RE = re.compile(r'["\'](/(?:home|mnt)/[^"\'\s]{6,})["\']')
WT_RE = re.compile(r"(zensim--[a-z0-9][a-z0-9-]*)")


def code_strings(tree: ast.AST) -> list[str]:
    """Every string literal that is CODE, not a docstring.

    Docstrings are excluded deliberately: prose legitimately names dead paths
    (this file's own docstring cites `zensim--cross-codec-metric` as the
    cautionary example). Linting prose would make documenting the failure mode
    impossible — the linter would flag the explanation of the linter.
    """
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = getattr(node, "body", None)
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))
    return [
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant)
        and isinstance(n.value, str)
        and id(n) not in docstrings
    ]


def check(p: Path) -> list[str]:
    src = p.read_text(errors="replace")
    fails: list[str] = []

    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return [f"PARSE line {e.lineno}: {e.msg}"]

    literals = code_strings(tree)
    seen: set[str] = set()
    for lit in literals:
        for wt in WT_RE.findall(lit):
            if wt in seen:
                continue
            seen.add(wt)
            if not (ZEN / wt).is_dir():
                fails.append(f"DEAD-WT references deleted sibling worktree {wt}/")

    for lit in literals:
        for raw in PATH_RE.findall(f'"{lit}"'):
            if any(c in raw for c in "{}*?"):
                continue
            if "/target/release/" not in raw and "/target/debug/" not in raw:
                continue
            if Path(raw).exists() or WT_RE.search(raw):
                continue  # worktree case already reported above
            # A missing artifact whose SOURCE still exists is just unbuilt —
            # `cargo build` fixes it. Only an artifact with no source anywhere
            # is a fossil. Failing the buildable case would train people to
            # ignore this linter, which is worse than not having it.
            if find_source(Path(raw).name):
                continue
            fails.append(f"DEAD-BIN missing binary with no source: {raw}")
    return sorted(set(fails))


def find_source(binary: str) -> Path | None:
    """Locate the .rs that would build `binary`, anywhere in the zen tree.

    Skips `target/` (build output) and `.claude/worktrees/` (ephemeral agent
    copies — finding a source only there means the source is NOT in the repo).
    """
    for repo in (ROOT, ZEN / "zenanalyze", ZEN / "zenmetrics"):
        if not repo.is_dir():
            continue
        for cand in repo.rglob(f"{binary}.rs"):
            parts = cand.parts
            if "target" in parts or ".claude" in parts or ".jj" in parts:
                continue
            return cand
    return None


def main() -> int:
    list_only = "--list" in sys.argv
    failures: dict[str, list[str]] = {}
    n = 0
    for p in sorted(SCRIPTS.rglob("*.py")):
        if "__pycache__" in p.parts or ".venv" in p.parts:
            continue
        n += 1
        f = check(p)
        if f:
            failures[str(p.relative_to(ROOT))] = f

    if not failures:
        print(f"lint_scripts: {n} scripts checked, all runnable")
        return 0

    for name, why in sorted(failures.items()):
        print(f"\n{name}")
        for w in why:
            print(f"    {w}")
    print(
        f"\nlint_scripts: {len(failures)} of {n} scripts cannot run.\n"
        "A script pinned to a deleted worktree is not a script, it is a fossil.\n"
        "Delete it (git history keeps it), or repoint it at a path that exists."
    )
    return 0 if list_only else 1


if __name__ == "__main__":
    sys.exit(main())
