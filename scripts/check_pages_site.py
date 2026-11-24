#!/usr/bin/env python3
"""Whole-site privacy, size, and integrity gate for the zensim GitHub Pages artifact."""
from __future__ import annotations

import base64
import json
import re
import sys
from pathlib import Path

MAX_EVAL_DATA_BYTES = 85_000
TEXT_EXT = {".html", ".js", ".mjs", ".css", ".json", ".svg", ".txt", ".md"}

FORBIDDEN = re.compile(
    r"/home/|/mnt/|/var/|/srv/|/tmp/|source_paths|zenpapers|imazen-26|"
    r"\b(?:tower|r7900x|r5900xt|i265|i270|i134|r3500|r3800x|lilith-wsl)\b|"
    r"\.lan\b|aic[-_ ]?2026|public_test|normalized_pred",
    re.I,
)


def decode_dot_count_sum(b64_str: str, dot_grid: int) -> int:
    buf = base64.b64decode(b64_str, validate=True)
    pos = 0
    total = 0
    prev_ix = -1
    while pos < len(buf):
        ix = buf[pos]
        if ix <= prev_ix or ix >= dot_grid:
            raise SystemExit(f"invalid column ix={ix} after {prev_ix}")
        prev_ix = ix
        n_single = buf[pos + 1]
        pos += 2
        for _ in range(n_single):
            iy = buf[pos]
            if iy >= dot_grid:
                raise SystemExit(f"invalid iy={iy}")
            pos += 1
            total += 1
        n_multi = buf[pos]
        pos += 1
        for _ in range(n_multi):
            iy = buf[pos]
            c0 = buf[pos + 1]
            if c0 < 255:
                cnt = c0
                pos += 2
            else:
                cnt = int.from_bytes(buf[pos + 2 : pos + 4], "little")
                pos += 4
            if iy >= dot_grid or cnt < 2:
                raise SystemExit(f"invalid multi cell iy={iy} cnt={cnt}")
            total += cnt
    if pos != len(buf):
        raise SystemExit("trailing bytes in dot stream")
    return total


def main() -> None:
    site = Path(sys.argv[1])
    if not site.is_dir():
        raise SystemExit(f"missing site dir: {site}")

    required = [
        "index.html",
        "evaluations.html",
        "evaluations.js",
        "evaluations.css",
        "evaluations-data.json",
        "design/tokens.css",
    ]
    for rel in required:
        if not (site / rel).is_file():
            raise SystemExit(f"missing required site file: {rel}")

    data_path = site / "evaluations-data.json"
    data_bytes = data_path.stat().st_size
    if data_bytes >= MAX_EVAL_DATA_BYTES:
        raise SystemExit(f"evaluations-data.json is {data_bytes} bytes (>= {MAX_EVAL_DATA_BYTES})")

    data = json.loads(data_path.read_text(encoding="utf-8"))
    dot_grid = int(data.get("dot_grid", 256))
    for v in data["views"]:
        n = v["rows"]
        if set(v["sc"].keys()) != {"A", "S2", "BA"}:
            raise SystemExit(f"scatter keys in {v['id']} must be A,S2,BA; got {list(v['sc'].keys())}")
        for m, sc in v["sc"].items():
            cells = sc["c"]
            cell_sum = sum(cells[i + 1] for i in range(0, len(cells), 2))
            q_sum = sum(q[4] for q in sc["q"])
            d_sum = decode_dot_count_sum(sc["d"], dot_grid)
            if cell_sum != n or q_sum != n or d_sum != n:
                raise SystemExit(
                    f"count mismatch in {v['id']}/{m}: cells={cell_sum} q={q_sum} dots={d_sum} expected={n}"
                )

    clic_n = data["clic22"]["rows"]
    for m, md in data["clic22"]["models"].items():
        m_sum = sum(b[2] for b in md["bins"])
        if m_sum != clic_n:
            raise SystemExit(f"CLIC margin bin sum mismatch in {m}: {m_sum} != {clic_n}")

    index_html = (site / "index.html").read_text(encoding="utf-8")
    eval_html = (site / "evaluations.html").read_text(encoding="utf-8")
    if index_html.count('href="evaluations.html"') != 1:
        raise SystemExit("index.html must contain exactly one link to evaluations.html")
    if 'href="index.html"' not in eval_html:
        raise SystemExit("evaluations.html must link back to index.html")

    scanned = 0
    for p in sorted(site.rglob("*")):
        if p.is_file() and p.suffix.lower() in TEXT_EXT:
            txt = p.read_text(encoding="utf-8")
            m = FORBIDDEN.search(txt)
            if m:
                raise SystemExit(f"forbidden token {m.group(0)!r} in {p.relative_to(site)}")
            scanned += 1

    print(json.dumps({"site_files_scanned": scanned, "evaluations_data_bytes": data_bytes, "passed": True}))


if __name__ == "__main__":
    main()
