"""Small, strict readers for the markdown structures dialviz consumes.

They raise `SourceShapeError` when a source file no longer has the structure
the generator was written against, so a changed source fails the build and
the tests instead of silently rendering a partial page.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


class SourceShapeError(RuntimeError):
    pass


@dataclass
class MdTable:
    path: str
    line: int  # 1-based line of the header row
    headers: list[str]
    rows: list[list[str]]
    row_lines: list[int] = field(default_factory=list)

    def col(self, name: str) -> int:
        for i, h in enumerate(self.headers):
            if h.strip().lower() == name.lower():
                return i
        raise SourceShapeError(f"{self.path}:{self.line}: no column {name!r} in {self.headers}")


def split_row(line: str) -> list[str]:
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|"):
        s = s[:-1]
    # GFM splits table cells on every unescaped pipe, including inside code spans.
    cells, cur = [], []
    i = 0
    while i < len(s):
        ch = s[i]
        if ch == "\\" and i + 1 < len(s) and s[i + 1] == "|":
            cur.append("|")
            i += 2
            continue
        if ch == "|":
            cells.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
        i += 1
    cells.append("".join(cur).strip())
    return cells


def tables(text: str, path: str) -> list[MdTable]:
    lines = text.splitlines()
    out = []
    i = 0
    while i < len(lines) - 1:
        if lines[i].lstrip().startswith("|") and re.match(r"^\s*\|?\s*:?-{2,}", lines[i + 1]):
            hdr = split_row(lines[i])
            t = MdTable(path, i + 1, hdr, [], [])
            j = i + 2
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                t.rows.append(split_row(lines[j]))
                t.row_lines.append(j + 1)
                j += 1
            out.append(t)
            i = j
        else:
            i += 1
    return out


def table_with_headers(text: str, path: str, required: list[str]) -> MdTable:
    want = [r.lower() for r in required]
    for t in tables(text, path):
        have = [h.strip().lower() for h in t.headers]
        if all(w in have for w in want):
            return t
    raise SourceShapeError(f"{path}: no markdown table with columns {required}")


def headings(text: str) -> list[tuple[int, int, str]]:
    """(line, level, title) for every ATX heading outside fenced code."""
    out, fence = [], False
    for n, line in enumerate(text.splitlines(), 1):
        if line.startswith("```"):
            fence = not fence
            continue
        if fence:
            continue
        m = re.match(r"^(#{1,6})\s+(.*?)\s*#*\s*$", line)
        if m:
            out.append((n, len(m.group(1)), m.group(2)))
    return out


def section(text: str, title_re: str, path: str) -> tuple[int, str]:
    """Body of the first heading matching `title_re` up to the next heading of equal or higher level."""
    lines = text.splitlines()
    hs = headings(text)
    for k, (n, lvl, title) in enumerate(hs):
        if re.search(title_re, title):
            end = len(lines)
            for n2, lvl2, _ in hs[k + 1:]:
                if lvl2 <= lvl:
                    end = n2 - 1
                    break
            return n, "\n".join(lines[n:end])
    raise SourceShapeError(f"{path}: no heading matching {title_re!r}")


def first_line_matching(text: str, pattern: str, path: str) -> tuple[int, str]:
    rx = re.compile(pattern)
    for n, line in enumerate(text.splitlines(), 1):
        if rx.search(line):
            return n, line
    raise SourceShapeError(f"{path}: no line matching {pattern!r}")


def strip_md(s: str) -> str:
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)
    return s.replace("**", "").replace("`", "").strip()
