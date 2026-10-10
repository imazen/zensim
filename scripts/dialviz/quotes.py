"""Resolve catalogue locators to verbatim quotes, table cells, code excerpts and code references."""
from __future__ import annotations

import re

from .mdparse import SourceShapeError, tables


def _paragraph(lines: list[str], i: int) -> tuple[str, int]:
    """Text of the paragraph, bullet or table row starting at line index i."""
    first = lines[i]
    if first.lstrip().startswith("|"):
        return first.strip(), i + 1
    out = [first.strip()]
    bullet = re.match(r"^\s*([-*]|\d+\.)\s", first)
    heading = first.startswith("#")
    if heading:
        # a heading quote takes the heading plus its first paragraph
        j = i + 1
        while j < len(lines) and not lines[j].strip():
            j += 1
        body, end = _paragraph(lines, j) if j < len(lines) else ("", j)
        return (first.strip() + "\n" + body).strip(), end
    j = i + 1
    while j < len(lines):
        l = lines[j]
        if not l.strip() or l.startswith("#") or l.lstrip().startswith("|"):
            break
        if bullet and re.match(r"^\s*([-*]|\d+\.)\s", l) and not l.startswith("  "):
            break
        if not bullet and re.match(r"^[-*]\s", l):
            break
        out.append(l.strip())
        j += 1
    return " ".join(out), j


def resolve(ctx, loc: dict) -> dict:
    if "quote" in loc:
        path = loc["quote"]
        lines = ctx.text(path, "quote").splitlines()
        rx = re.compile(loc["match"])
        for i, l in enumerate(lines):
            if rx.search(l):
                text, _ = _paragraph(lines, i)
                m = rx.search(text)
                if m and m.start() > 0 and not text.startswith(("#", "*", "-", "|")):
                    # the match starts mid-paragraph: quote from the start of its sentence
                    cut = max(text.rfind(". ", 0, m.start()), text.rfind(": ", 0, m.start()))
                    text = text[cut + 2:] if cut >= 0 else text
                return {"kind": "quote", "path": path, "line": i + 1, "text": text, **_chip(loc, text)}
        raise SourceShapeError(f"{path}: no line matching {loc['match']!r}")
    if "row" in loc:
        path = loc["row"]
        want = [h.lower() for h in loc["headers"]]
        rx = re.compile(loc["key"])
        candidates = [t for t in tables(ctx.text(path, "quote"), path)
                      if all(w in [h.strip().lower() for h in t.headers] for w in want)]
        if not candidates:
            raise SourceShapeError(f"{path}: no markdown table with columns {loc['headers']}")
        for t in candidates:
            for row, ln in zip(t.rows, t.row_lines):
                if row and rx.search(row[0]):
                    col = t.col(loc["col"]) if loc.get("col") else None
                    text = row[col] if col is not None else " | ".join(row)
                    return {"kind": "cell", "path": path, "line": ln, "text": text, "row_key": row[0],
                            "column": loc.get("col"), "note": loc.get("note"), **_chip(loc, text)}
        raise SourceShapeError(f"{path}: no {loc['headers']} table row whose first cell matches {loc['key']!r}")
    if "code" in loc:
        path = loc["code"]
        lines = ctx.text(path, "code").splitlines()
        srx, erx = re.compile(loc["start"]), re.compile(loc["end"])
        for i, l in enumerate(lines):
            if srx.search(l):
                for j in range(i, min(len(lines), i + 120)):
                    if erx.search(lines[j]):
                        return {"kind": "code", "path": path, "line": i + 1, "end": j + 1,
                                "text": "\n".join(lines[i:j + 1])}
                raise SourceShapeError(f"{path}:{i + 1}: excerpt end {loc['end']!r} not found within 120 lines")
        raise SourceShapeError(f"{path}: excerpt start {loc['start']!r} not found")
    if "symbol" in loc:
        path = loc["symbol"]
        lines = ctx.text(path, "code-ref").splitlines()
        rx = re.compile(loc["match"])
        for i, l in enumerate(lines):
            if rx.search(l):
                return {"kind": "symbol", "path": path, "line": i + 1, "label": loc.get("label") or path,
                        "text": l.strip()}
        raise SourceShapeError(f"{path}: symbol {loc['match']!r} not found")
    raise SourceShapeError(f"unknown locator {loc}")


def _chip(loc: dict, text: str) -> dict:
    if "fail" in loc and re.search(loc["fail"], text):
        return {"status": "fail"}
    if "pass" in loc and re.search(loc["pass"], text):
        return {"status": "pass"}
    return {"status": "info"}


def threshold(ctx, spec: dict) -> dict:
    """A numeric rule value read from the one source line that states it."""
    lines = ctx.text(spec["path"], "threshold").splitlines()
    hits = [i for i, l in enumerate(lines) if re.search(spec["line"], l)]
    if len(hits) != 1:
        raise SourceShapeError(f"{spec['path']}: threshold line /{spec['line']}/ matched {len(hits)} lines, expected 1")
    m = re.search(spec["value"], lines[hits[0]])
    if not m:
        raise SourceShapeError(f"{spec['path']}:{hits[0] + 1}: no value matching /{spec['value']}/")
    v = float(m.group(1)) * spec.get("scale", 1)
    return {"value": round(v, 10), "text": m.group(0), "path": spec["path"], "line": hits[0] + 1}
