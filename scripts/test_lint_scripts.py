#!/usr/bin/env python3
"""Self-tests for check_conflict_markers() in lint_scripts.py.

Run: python3 scripts/test_lint_scripts.py
Exits 0 if all pass, nonzero (with traceback) on the first failure.
No pytest dependency — plain asserts so it runs in minimal CI, matching
scripts/canonical_corpus/test_join_safety.py's style.

WHY THIS EXISTS: on 2026-09-05 two lanes each committed Markdown
(CHANGELOG.md, benchmarks/INDEX.md) carrying a stray leaked jj
conflict-marker line, found and fixed only by a human re-reading the file.
These tests pin the fix: the exact leaked shape must fail, and the
Markdown constructs that look similar (an H1 underline, a table rule, an
hr) must not.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lint_scripts import _scan_file, find_conflict_markers  # noqa: E402


def _write(tmp_dir: str, name: str, content: str) -> Path:
    fp = Path(tmp_dir) / name
    fp.write_text(content)
    return fp


def test_positive_leaked_trailing_marker_in_a_temp_file():
    """The exact 2026-09-05 incident shape: a lone trailing jj closing
    marker survives in committed Markdown, with no opening marker anywhere
    in the file (the rest of the conflict was resolved by hand; only the
    closing line was missed). Must fail, from a real file on disk.
    """
    content = (
        "# Changelog\n"
        "\n"
        "## [Unreleased]\n"
        "\n"
        "### Added\n"
        "- some entry\n"
        ">>>>>>> conflict 1 of 1 ends\n"
        "### Fixed\n"
        "- another entry\n"
    )
    with tempfile.TemporaryDirectory() as td:
        fp = _write(td, "CHANGELOG.md", content)
        hits = _scan_file(fp)
    assert hits == [(7, "conflict marker: '>>>>>>> conflict 1 of 1 ends'")], hits


def test_positive_full_jj_diff_style_block_in_a_temp_file():
    """A complete, still-unresolved jj conflict block (captured verbatim
    from a real `jj 0.44.0` 2-sided conflict) must fail on every marker
    line it contains, and only those lines.
    """
    content = (
        "line1\n"
        "<<<<<<< conflict 1 of 1\n"
        '%%%%%%% diff from: lnwxyltv 85a9ceff "base"\n'
        + ("\\" * 7)
        + '        to: rwyqvlux 8460a2c2 "sideA"\n'
        "-line2\n"
        "+SIDE_A\n"
        '+++++++ qssrszwy bf39d78d "sideB"\n'
        "SIDE_B\n"
        ">>>>>>> conflict 1 of 1 ends\n"
        "line3\n"
    )
    with tempfile.TemporaryDirectory() as td:
        fp = _write(td, "notes.md", content)
        hits = _scan_file(fp)
    lines_flagged = [ln for ln, _ in hits]
    # Every marker line (2,3,4,7,9) fires; ordinary diff content lines
    # (-line2, +SIDE_A at lines 5,6) and plain prose (1,8,10) do not.
    assert lines_flagged == [2, 3, 4, 7, 9], hits


def test_negative_markdown_headings_and_rules_in_a_temp_file():
    """A Markdown table separator and a `-------` horizontal rule must
    NOT fail: neither sits between a `<<<<<<<` and a `>>>>>>>`, and both
    are ordinary Markdown used throughout this repo's own docs.
    """
    content = (
        "Heading\n"
        "=======\n"
        "\n"
        "| a | b |\n"
        "|---|---|\n"
        "| 1 | 2 |\n"
        "\n"
        "-------\n"
        "\n"
        "Some prose that quotes a unified diff hunk for illustration:\n"
        "+++ b/file.txt\n"
        "--- a/file.txt\n"
    )
    with tempfile.TemporaryDirectory() as td:
        fp = _write(td, "README.md", content)
        hits = _scan_file(fp)
    assert hits == [], hits


def test_inner_markers_are_ignored_outside_a_block():
    """%%%%%%%/+++++++/-------/=======/||||||| only mean something between
    an open and a close marker. Bare, they must never fire.
    """
    text = (
        "%%%%%%% not a conflict, just seven percent signs\n"
        "+++++++ also not one\n"
        "------- an hr, not a conflict\n"
        "======= an H1 underline, not a conflict\n"
        "||||||| not a diff3 marker either without a block around it\n"
    )
    assert find_conflict_markers(text) == []


def test_bare_close_marker_with_no_opener_anywhere_still_fires():
    """<<<<<<< and >>>>>>> are unconditional -- this is the shape that
    actually leaked: the opener and the diff body were already resolved
    away by hand, leaving only the closer.
    """
    text = "normal line\n>>>>>>> conflict 1 of 1 ends\nmore normal lines\n"
    hits = find_conflict_markers(text)
    assert hits == [(2, "conflict marker: '>>>>>>> conflict 1 of 1 ends'")], hits


def test_bare_open_marker_with_no_closer_also_fires():
    """An unterminated block (someone deleted the closing line instead of
    the opening one) is just as much a leak."""
    text = "before\n<<<<<<< conflict 1 of 1\nstill in the block\n"
    hits = find_conflict_markers(text)
    assert hits == [(2, "conflict marker: '<<<<<<< conflict 1 of 1'")], hits


def test_git_style_conflict_without_jj_extras():
    """A plain git merge conflict (no jj involved at all) must fail on
    its own three markers."""
    text = (
        "before\n"
        "<<<<<<< HEAD\n"
        "ours\n"
        "=======\n"
        "theirs\n"
        ">>>>>>> feature-branch\n"
        "after\n"
    )
    hits = find_conflict_markers(text)
    lines_flagged = [ln for ln, _ in hits]
    assert lines_flagged == [2, 4, 6], hits


def test_diff3_common_ancestor_marker_inside_a_block():
    """git's diff3/zdiff3 style adds a ||||||| common-ancestor section
    between the open marker and =======; it must fire only there."""
    text = (
        "<<<<<<< HEAD\n"
        "ours\n"
        "||||||| merged common ancestors\n"
        "base\n"
        "=======\n"
        "theirs\n"
        ">>>>>>> feature-branch\n"
    )
    hits = find_conflict_markers(text)
    lines_flagged = [ln for ln, _ in hits]
    assert lines_flagged == [1, 3, 5, 7], hits


def test_binary_file_is_skipped_not_scanned():
    """A NUL in the first bytes marks a file binary; _scan_file must
    return [] even if it happens to contain marker-shaped bytes later."""
    with tempfile.TemporaryDirectory() as td:
        fp = Path(td) / "weights.bin"
        fp.write_bytes(b"\x00\x01\x02" + b">>>>>>> conflict 1 of 1 ends\n")
        hits = _scan_file(fp)
    assert hits == [], hits


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nALL {len(tests)} lint_scripts conflict-marker tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
