#!/usr/bin/env python3
"""Self-tests for check_conflict_markers() and the hygiene patterns in lint_scripts.py.

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

The hygiene: address/identifier tests below pin the OTHER shared gate --
scripts/lib/hygiene_patterns.txt, read by both this linter and
scripts/safe_push.sh. Every flagged literal in them is ASSEMBLED at runtime
from parts, never written out, so this file cannot itself carry the shapes
it exists to ban (and so the linter does not flag its own test suite).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lint_scripts import _hygiene_patterns, _scan_file, find_conflict_markers  # noqa: E402


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


# ---------------------------------------------------------------- hygiene


def _hygiene_hits(line: str) -> list[str]:
    """Names of every hygiene class matching one line. Uses the real loader, so
    a pattern-file edit that breaks a class fails here rather than in a push."""
    return [name for name, rx in _hygiene_patterns() if rx.search(line)]


def test_hygiene_patterns_file_actually_loads():
    """A missing or malformed pattern file must not silently disarm the gate:
    lint_scripts reports it as a failure, and that path starts here."""
    pats = _hygiene_patterns()
    names = {n for n, _ in pats}
    assert "private-network-address" in names, names
    assert "hardware-address" in names, names
    assert "household-framing" in names, names


def test_hygiene_positive_private_network_addresses():
    """All three RFC1918 ranges, each assembled from parts."""
    for octets in (("192", "168", "50", "44"), ("10", "1", "2", "3"), ("172", "20", "0", "5")):
        line = "see http://%s.%s.%s.%s:3300/report" % octets
        assert "private-network-address" in _hygiene_hits(line), line


def test_hygiene_positive_hardware_address():
    mac = ":".join(["04", "7c", "16", "b3", "18", "51"])
    line = f'  "hw": "{mac}",'
    assert "hardware-address" in _hygiene_hits(line), line


def test_hygiene_positive_household_framing():
    for line in ("the %s PCs stay on Windows" % ("kid" + "s'"),
                 "borrowed a %s box overnight" % ("child" + "'s")):
        assert "household-framing" in _hygiene_hits(line), line


def test_hygiene_negative_canonical_and_loopback_forms():
    """The documented canonical form, loopback, and bind-all must all pass --
    otherwise the gate refuses the very thing it tells you to use instead."""
    for line in ("browse http://localhost:3300/zensim/reports/",
                 "curl http://127.0.0.1:3300/health",
                 "python3 -m http.server 3400 --bind 0.0.0.0",
                 "worker on r7900x, secondary on tower"):
        assert _hygiene_hits(line) == [], (line, _hygiene_hits(line))


def test_hygiene_negative_version_and_identifier_lookalikes():
    """The classes that made a naive keyword grep useless: a dotted version, a
    timestamp, and `kids` as a plain code identifier (a knob-id set, which a
    real script in a sibling repo genuinely has). None may fire."""
    for line in ("zenjxl 0.4.0, build 10.0.19045 on the runner",
                 "elapsed 01:02:03:04:05 in the log",
                 "per_img_front_kids = {}   # img -> set(kid) on front",
                 "topk_kids = sorted(keep)"):
        assert _hygiene_hits(line) == [], (line, _hygiene_hits(line))


def test_hygiene_public_patterns_carry_no_specific_values():
    """The public pattern file must describe CLASSES, never the values it
    protects -- a pattern list that spelled them out would leak exactly what it
    exists to keep out. Checked by feeding the file to its own patterns."""
    from lint_scripts import HYGIENE_PATTERNS_FILE

    text = HYGIENE_PATTERNS_FILE.read_text(encoding="utf-8")
    for lineno, line in enumerate(text.splitlines(), 1):
        hits = _hygiene_hits(line)
        assert hits == [], (lineno, line, hits)


def main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\nALL {len(tests)} lint_scripts tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
