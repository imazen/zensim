#!/usr/bin/env python3
"""Build the scoring pairs TSV from E5A per-source manifests.

Reads `/var/tmp/e5a-render/fixtures/<origin>/_MANIFEST.json` for each origin
dir under FIXDIR and writes a tab-separated file:

    key  origin  kind  family  variant  severity  index  width  height  ref_path  dist_path

- kind = corruption | benign
- ref_path = correct output (a_file); dist_path = broken/alternative (b_file)
- key = <origin>_<index:04d> — stable join key and map filename stem.
- width/height come from the manifest (a and b dims are verified equal).
"""
import json
import sys
from pathlib import Path

FIXDIR = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/var/tmp/e5a-render/fixtures")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("/var/tmp/e5a-render/pairs.tsv")

rows = []
for d in sorted(FIXDIR.iterdir()):
    mfile = d / "_MANIFEST.json"
    if not mfile.exists():
        continue
    origin = d.name
    for it in json.loads(mfile.read_text()):
        key = f"{origin}_{it['index']:04d}"
        rows.append(
            [
                key,
                origin,
                it["kind"],
                it["family"],
                it["variant"],
                str(it["severity"]),
                str(it["index"]),
                str(it["width"]),
                str(it["height"]),
                str(d / it["a_file"]),
                str(d / it["b_file"]),
            ]
        )

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w") as f:
    f.write(
        "key\torigin\tkind\tfamily\tvariant\tseverity\tindex\twidth\theight\tref_path\tdist_path\n"
    )
    for r in rows:
        f.write("\t".join(r) + "\n")
print(f"wrote {OUT}: {len(rows)} pairs", file=sys.stderr)
