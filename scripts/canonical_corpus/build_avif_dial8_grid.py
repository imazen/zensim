#!/usr/bin/env python3
"""Build the WAVE-12 AVIF dial-ladder eval grid from eval8_944 (appendix AD).

THE G-AC2 AVIF-facing dial instrument (campaign appendix AC.2 amendment 2 +
APPENDIX AD): the leg-side eval holdout `eval8_944.parquet` (origins ending 8,
never trained on), restricted to the corpus's DEFAULT STRATUM — knob cell
`s4` (plan `rd_core`; speed-4 default knobs, the same stratum the G-Z5
orientation bar is defined on) — reshaped into the canonical dial-grid schema
that `bake_verdict --dial-grid` already consumes (`parquet_loader::
load_dial_grid`: image_id / codec / q / codec_param / param_kind / f0..fN).
This EXTENDS the existing dial-grid tooling (the dial mono/tied statistics are
computed by bake_verdict's own dial_panel — no stat code here, per the
no-duplication rule); it only derives the ladder VIEW.

Derivation (registered):
  - rows: knob_tuple_json.cell == "s4" exactly (fp varies per rendition —
    zenavif sizes some knobs to the input; the ladder key is (rendition,
    cell), matching the corpus's 19,146-(rendition x stratum)-ladder gate)
  - expected shape: 269 renditions x 30 q (q = 1,5..70 step5 + 72..100 step2)
    = 8,070 rows, asserted exactly, per-rendition ladder completeness asserted
  - image_id = image_path verbatim; codec = "zenavif_s4"; codec_param = q;
    param_kind = "q"; f<i> = feat_<i> (f64)

Usage:  python3 scripts/canonical_corpus/build_avif_dial8_grid.py
Exit 0 = built + gated; nonzero = gate failed (no partial output kept).
"""

import hashlib
import json
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

IN_PATH = Path("/mnt/v/zen/zensim-training/avif944-2026-08-07/eval8_944.parquet")
IN_SHA = "e47091fa24d953d87a0b16c70ea6bb1235ef59d7d4070172d3e8b02493d97724"
OUT_DIR = Path("/mnt/v/output/zensim/v2-eval-944-2026-08-01")
OUT_PATH = OUT_DIR / "avif_dial8_944col_2026-08-21.parquet"
MANIFEST_FRAG = OUT_DIR / "_MANIFEST_avif_dial8.json"
EXPECT_ROWS = 8_070
EXPECT_RENDITIONS = 269
EXPECT_Q = sorted(list(range(5, 71, 5)) + [1] + list(range(72, 101, 2)))
CELL = "s4"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    assert len(EXPECT_Q) == 30, "q ladder definition drifted"
    print(f"sha256-gating {IN_PATH} ...", flush=True)
    got = sha256_file(IN_PATH)
    if got != IN_SHA:
        print(f"ABORT: input sha {got} != registered {IN_SHA}", file=sys.stderr)
        return 1

    t = pq.read_table(IN_PATH)
    knobs = t.column("knob_tuple_json").to_pylist()
    mask = []
    plans = set()
    for k in knobs:
        d = json.loads(k)
        hit = d.get("cell") == CELL
        mask.append(hit)
        if hit:
            plans.add(d.get("plan"))
    sel = t.filter(pa.array(mask))
    if plans != {"rd_core"}:
        print(f"ABORT: unexpected plans in {CELL}: {plans}", file=sys.stderr)
        return 1
    if sel.num_rows != EXPECT_ROWS:
        print(f"ABORT: rows {sel.num_rows} != {EXPECT_ROWS}", file=sys.stderr)
        return 1

    imgs = sel.column("image_path").to_pylist()
    qs = [float(q) for q in sel.column("q").to_pylist()]
    ladders = defaultdict(list)
    for i, q in zip(imgs, qs):
        ladders[i].append(q)
    if len(ladders) != EXPECT_RENDITIONS:
        print(f"ABORT: renditions {len(ladders)} != {EXPECT_RENDITIONS}",
              file=sys.stderr)
        return 1
    for img, lq in ladders.items():
        if sorted(lq) != [float(q) for q in EXPECT_Q]:
            print(f"ABORT: incomplete ladder for {img}: {sorted(lq)}",
                  file=sys.stderr)
            return 1

    cols = {
        "image_id": pa.array(imgs, pa.string()),
        "codec": pa.array(["zenavif_s4"] * sel.num_rows, pa.string()),
        "q": pa.array(qs, pa.float64()),
        "codec_param": pa.array(qs, pa.float64()),
        "param_kind": pa.array(["q"] * sel.num_rows, pa.string()),
    }
    for i in range(944):
        cols[f"f{i}"] = sel.column(f"feat_{i}").cast(pa.float64())
    out = pa.table(cols)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_PATH.with_suffix(".parquet.tmp")
    pq.write_table(out, tmp, compression="zstd")
    os.replace(tmp, OUT_PATH)
    out_sha = sha256_file(OUT_PATH)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    ).stdout.strip()
    frag = {
        "_": "wave-12 AVIF dial-ladder eval grid (G-AC2 instrument; appendix AD)",
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "build_commit": commit,
        "input": {"path": str(IN_PATH), "sha256": IN_SHA},
        "output": {"path": str(OUT_PATH), "sha256": out_sha,
                   "rows": out.num_rows},
        "derivation": f"knob cell == '{CELL}' (default stratum, plan rd_core); "
        f"{EXPECT_RENDITIONS} renditions x {len(EXPECT_Q)} q; ladder key = "
        "(image_id, codec); schema = load_dial_grid contract",
        "population": "eval8 holdout (origins ending 8; never trained on)",
        "consumer": "bake_verdict --dial-grid <this file> (dial_panel owner)",
    }
    MANIFEST_FRAG.write_text(json.dumps(frag, indent=1) + "\n")
    print(f"OK: {OUT_PATH} rows={out.num_rows} sha={out_sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
