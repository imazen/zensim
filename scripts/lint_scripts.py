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
  2. it writes a script hardcoding `<zen>/zensim--foo/target/release/bar`
  3. the worktree is cleaned up (correctly! the cleanup rule is mandatory)
  4. the script stays, permanently dead, and nothing notices

(That example is spelled `<zen>/` rather than the literal path because the
2026-07-15 mass-repoint sed — `zensim--*` -> `zensim` across 69 files — rewrote
this very docstring and silently deleted the illustration. Bulk seds do not
read context. That is the same mechanism that left `metric_compare_report.py`
unparseable for weeks.)

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
  DEAD-SCRIPT — shells to a `scripts/...` path that does not exist (this is the
              cross-language edge: a .sh calling a deleted .py)
  DEAD-LINK / DEAD-REF — a README under scripts/ indexes a file that is gone.
              The index is how people find tools; when it rots it sends the
              next session to rebuild something that already exists.

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
# A repo-relative script path, however it is reached: `python3 scripts/x.py`,
# a bare `scripts/x.py`, or one embedded in a longer command.
SCRIPT_RE = re.compile(r"(?<![\w/.-])(scripts/[\w/-]+\.(?:py|sh))")


def dead_script_refs(text: str) -> list[str]:
    """Repo-relative script paths that no longer exist.

    This check was missing at first, and the gap bit immediately: consolidating
    five `eval_v*_pjnd_check.py` into one broke six shell callers, and the
    linter still said "all runnable" because it only inspected
    `/target/release/` paths. It also surfaced a pre-existing break —
    `v0_20b/bake_v3.py` shelling to `affine_calibrate_znpr_v2.py`, deleted long
    ago. Cross-language edges are exactly where nothing else is looking.
    """
    return sorted(
        {
            f"DEAD-SCRIPT calls missing script: {ref}"
            for ref in SCRIPT_RE.findall(text)
            if not (ROOT / ref).exists()
        }
    )


MD_LINK_RE = re.compile(r"\[[^\]]*\]\(([^)#:]+?)\)")
MD_CODE_REF_RE = re.compile(r"`([\w./-]+\.(?:py|sh))`")


def check_markdown(p: Path) -> list[str]:
    """A README that indexes deleted scripts is worse than no README.

    scripts/v_next/README.md is how anyone finds these tools, and on 2026-07-15
    it still offered `affine_calibrate_znpr_v2.py` ("legacy v2... kept for
    reproducing pre-2026-05-13 bakes") and `verify_bake_srocc.py` as live
    options months after both were deleted. Nothing checks prose, so the index
    rots silently while the code it points at is long gone — and a stale index
    sends the next session to rebuild a tool that already exists, or to hunt for
    one that never will.

    Relative links are checked against the file's own directory; bare
    `code-quoted` script names are checked against that directory too, since
    that is how these tables cite tools.
    """
    fails: list[str] = []
    text = p.read_text(errors="replace")

    # A doc that DECLARES itself a record of the past legitimately names tools
    # that existed then; flagging those forever would make this linter noisy,
    # and a noisy linter gets muted. The exemption is deliberately opt-in and
    # visible: the doc must say so in its header. `HISTORY_v06_rebalance_
    # falsified.md` does ("**Status:** FALSIFIED 2026-05-18"). When this check
    # first ran, `CYCLE_7_DSSIM_COTRAIN_PLAN.md` did NOT — it still read as
    # "authorized" pending work for a hypothesis CLAUDE.md had recorded as
    # falsified two months earlier. That is exactly how a session retries a
    # dead idea, so it got a status header rather than an exemption.
    if re.search(r"\b(falsified|superseded|historical doc|concluded|abandoned)\b",
                 "\n".join(text.splitlines()[:12]), re.I):
        return []

    for target in MD_LINK_RE.findall(text):
        t = target.strip()
        if t.startswith(("http://", "https://", "mailto:", "/")):
            continue
        if not (p.parent / t).exists():
            fails.append(f"DEAD-LINK links to missing {t}")
    for name in MD_CODE_REF_RE.findall(text):
        # Only judge names that look like siblings in this directory; a path
        # into another repo or an illustrative name is not ours to resolve.
        if "/" in name:
            continue
        if (p.parent / name).exists():
            continue
        # Prose legitimately names a deleted file to say it IS deleted.
        for line in text.splitlines():
            if name in line and re.search(r"delet|remov|superseded|was\b|no longer|gone", line, re.I):
                break
        else:
            fails.append(f"DEAD-REF names missing script `{name}` without saying it is gone")
    return sorted(set(fails))


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
            if unverifiable(raw) or find_source(Path(raw).name):
                continue
            fails.append(f"DEAD-BIN missing binary with no source: {raw}")
    fails += dead_script_refs("\n".join(literals))
    return sorted(set(fails))


def unverifiable(raw: str) -> bool:
    """True when `raw` lives in a sibling repo that is not checked out here.

    A path into `../zenmetrics` or `../zenanalyze` can only be judged when that
    repo is actually present. CI checks out zensim ALONE, so on the runner every
    sibling path resolves to nothing and the DEAD-BIN check fired on 13
    perfectly good `zenmetrics` invocations — the linter passed locally (where
    the siblings exist) and failed in CI. A check whose verdict depends on the
    developer's directory layout is not a check.

    So: verify what is here, and say nothing about what is not. The local run
    still catches sibling-repo rot (that is how the `zen-metrics` -> `zenmetrics`
    rename was found); CI just declines to guess.
    """
    for sib in ("zenanalyze", "zenmetrics"):
        if f"/{sib}/" in raw and not (ZEN / sib).is_dir():
            return True
    return False


def find_source(binary: str) -> Path | None:
    """Locate the source that would build `binary`, anywhere in the zen tree.

    Two ways a binary gets its name, and both must be checked:
      - convention: `src/bin/<name>.rs`
      - declaration: `[[bin]] name = "<name>"` in a Cargo.toml, where the file
        is usually `main.rs` and shares no name with the binary at all.

    Missing the second gave a false DEAD-BIN on `zenmetrics`, which is declared
    that way. A linter that cries wolf on a shipping binary gets muted, and
    then it protects nothing.

    Skips `target/` (build output) and `.claude/` (ephemeral agent copies —
    finding a source only there means it is NOT in the repo).
    """
    def usable(p: Path) -> bool:
        parts = p.parts
        return not ("target" in parts or ".claude" in parts or ".jj" in parts)

    for repo in (ROOT, ZEN / "zenanalyze", ZEN / "zenmetrics"):
        if not repo.is_dir():
            continue
        for cand in repo.rglob(f"{binary}.rs"):
            if usable(cand):
                return cand
        for toml in repo.rglob("Cargo.toml"):
            if not usable(toml):
                continue
            if re.search(
                rf'\[\[bin\]\][^\[]*?name\s*=\s*"{re.escape(binary)}"',
                toml.read_text(errors="replace"),
                re.S,
            ):
                return toml
    return None


def check_shell(p: Path) -> list[str]:
    """Same DEAD-WT / DEAD-BIN checks for shell, minus the AST.

    Shell was NOT covered when this linter first landed, and it reported "all
    runnable" while three `eval_cross_codec_v{6,7,8}.sh` invoked a python
    checker through `zensim--cross-codec-*` worktrees that no longer exist. A
    linter that green-lights dead scripts is worse than no linter: it converts
    "nobody checked" into "something checked and it's fine."

    There is no docstring concept in shell, so `#` comments are stripped
    instead — prose in a comment may legitimately name a dead path.
    """
    fails: list[str] = []
    body = "\n".join(
        line.split("#", 1)[0] if not line.lstrip().startswith("#") else ""
        for line in p.read_text(errors="replace").splitlines()
    )
    seen: set[str] = set()
    for wt in WT_RE.findall(body):
        if wt in seen:
            continue
        seen.add(wt)
        if not (ZEN / wt).is_dir():
            fails.append(f"DEAD-WT references deleted sibling worktree {wt}/")
    for raw in re.findall(r"(/(?:home|mnt)/[^\s\"';|)]{6,})", body):
        if any(c in raw for c in "{}*?$"):
            continue
        if "/target/release/" not in raw and "/target/debug/" not in raw:
            continue
        if Path(raw).exists() or WT_RE.search(raw):
            continue
        if unverifiable(raw) or find_source(Path(raw).name):
            continue
        fails.append(f"DEAD-BIN missing binary with no source: {raw}")
    fails += dead_script_refs(body)
    return sorted(set(fails))


# Characters Windows refuses in a filename. A tracked file carrying one makes
# `git checkout` fail outright on a Windows runner -- before any compile, so
# the job's log shows only `error: invalid path` and looks like a toolchain
# problem rather than a repo one.
WINDOWS_FORBIDDEN = set('"*:<>?|')


def check_windows_paths() -> list[str]:
    """Tracked paths that Windows cannot check out.

    MEASURED 2026-07-15: a file literally named `"$LOG"` (a shell redirect to
    an unset `$LOG`, committed by accident in 9a4e2272) broke BOTH mandatory
    Windows runners at the checkout step for weeks. CLAUDE.md requires
    windows-11-arm CI, so this is not cosmetic -- it silently removed a
    required platform from coverage.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["git", "-c", "core.quotepath=off", "ls-files", "-z"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return []  # not a git checkout (or git absent) -- nothing to assert
    if out.returncode != 0:
        return []
    bad = []
    for path in out.stdout.split("\0"):
        if not path:
            continue
        hits = sorted(set(path) & WINDOWS_FORBIDDEN)
        if hits:
            bad.append(
                f"WIN-PATH: tracked file {path!r} contains {''.join(hits)!r}, "
                "which Windows forbids in a filename -- `git checkout` fails on "
                "the windows runners before anything compiles. Delete it (git "
                "history keeps it) or rename it."
            )
    return bad


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
    for p in sorted(SCRIPTS.rglob("*.sh")):
        n += 1
        f = check_shell(p)
        if f:
            failures[str(p.relative_to(ROOT))] = f
    for p in sorted(SCRIPTS.rglob("*.md")):
        n += 1
        f = check_markdown(p)
        if f:
            failures[str(p.relative_to(ROOT))] = f

    win = check_windows_paths()
    if win:
        failures["<tracked paths>"] = win

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
