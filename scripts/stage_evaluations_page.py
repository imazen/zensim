#!/usr/bin/env python3
"""Copy verified evaluations assets into site/ and link from index.html."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

NAV_ANCHOR = '<button data-view="weights">Weight audit</button>'
NAV_LINK = (
    '<a href="evaluations.html" '
    'style="padding:6px 8px;border:1px solid var(--border);border-radius:4px;'
    'text-decoration:none;color:var(--ink);background:var(--surface-1);'
    'display:inline-flex;align-items:center">Evaluations &amp; benchmarks</a>'
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--self", dest="self_dir", type=Path, required=True)
    ap.add_argument("--site", type=Path, required=True)
    args = ap.parse_args()

    src_pages = args.self_dir / "pages"
    for name in ("evaluations.html", "evaluations.css", "evaluations.js", "evaluations-data.json"):
        shutil.copyfile(src_pages / name, args.site / name)

    index_path = args.site / "index.html"
    html = index_path.read_text(encoding="utf-8")
    if 'href="evaluations.html"' in html:
        raise SystemExit("index.html already contains evaluations.html link")
    if html.count(NAV_ANCHOR) != 1:
        raise SystemExit(f"expected exactly 1 {NAV_ANCHOR!r} in index.html")
    patched = html.replace(NAV_ANCHOR, NAV_ANCHOR + NAV_LINK, 1)
    assert patched.count('href="evaluations.html"') == 1
    index_path.write_text(patched, encoding="utf-8")


if __name__ == "__main__":
    main()
