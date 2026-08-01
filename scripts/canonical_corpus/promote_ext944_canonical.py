#!/usr/bin/env python3
"""Promote the 944-feature (folded+append+append2 STREAMING regime) local-leg
extraction into the CANONICAL dataset directory + manifest — the ext944
counterpart of promote_ext924_canonical.py (P1 of PLAN_SOTA944, task #9).

Source (all 11 legs, one wave, one extractor rev):
  $EXT944_RUN/<corpus>.csv   (driver: zensim/examples/v2_ab_extract,
  ZENSIM_AB_MODE=foldapp2, profile codec_target, default toggles + append2,
  raw unpadded slices; extractor commit = $ZENSIM_COMMIT — REQUIRED env,
  full sha, because the append2 constants are PROVISIONAL per
  benchmarks/append2_bandvis_gates_2026-07-27.md and a later constants
  change must be detectable from the manifest)

Destination: /mnt/v/zen/zensim-training/ext944-canonical-<date>/

Per-corpus roles / target semantics / bans INHERITED verbatim from the ext924
canonical manifest (corpus facts, not regime facts). Schema: ref_basename
(utf8), human_score (f64), f0..f943 (f64), ZSTD-7, NaN/null hard abort, row
counts MUST equal the ext924 counts exactly.

Run gate_backfill944.py per leg (f0..f923 bitwise vs ext924) AFTER this
promote; the promote is not a substitute for the gate.
"""

import csv
import hashlib
import json
import os
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SRC = Path(os.environ.get("EXT944_RUN", "/mnt/v/output/zensim/ext944-run-2026-08-01"))
DEST = Path(
    os.environ.get(
        "EXT944_DEST",
        f"/mnt/v/zen/zensim-training/ext944-canonical-{date.today()}",
    )
)
OLD_MANIFEST = Path(
    "/mnt/v/zen/zensim-training/ext924-canonical-2026-07-27/_MANIFEST.json"
)
ZENSIM_COMMIT = os.environ.get("ZENSIM_COMMIT") or sys.exit(
    "ABORT: ZENSIM_COMMIT env required (full sha of the extractor build)"
)
N_FEATURES = 944
LAYOUT = (
    "f0..f943 folded+append+append2 STREAMING regime: f0..f155 folded v1-basic "
    "bands; f156..f371 structurally zero (fold slots); f372..f719 v2-348 "
    "bounded; f720..f923 append-204; f924..f943 append2-20 (BANDVIS_GAIN/LOSS, "
    "LUMA_MEAN_REF, HL_BIN1/2 x 4 scales, Y-only; f = 924 + scale*5 + local; "
    "constants PROVISIONAL per benchmarks/append2_bandvis_gates_2026-07-27.md)"
)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_to_parquet(src: Path, dst: Path, want_rows: int) -> int:
    with open(src, newline="") as f:
        r = csv.reader(f)
        names = next(r)
        if len(names) != 2 + N_FEATURES:
            sys.exit(f"ABORT {src.name}: {len(names)} cols, want {2 + N_FEATURES}")
        cols = {n: [] for n in names}
        for row in r:
            cols[names[0]].append(row[0])
            for n, v in zip(names[1:], row[1:]):
                x = float(v)
                if x != x or x in (float("inf"), float("-inf")):
                    sys.exit(f"ABORT {src.name}: non-finite in {n}")
                cols[n].append(x)
    n = len(cols[names[0]])
    if n != want_rows:
        sys.exit(f"ABORT {src.name}: {n} rows, want {want_rows}")
    arrays = [pa.array(cols[names[0]], type=pa.utf8())] + [
        pa.array(cols[n2], type=pa.float64()) for n2 in names[1:]
    ]
    pq.write_table(
        pa.table(arrays, names=names), dst, compression="zstd", compression_level=7
    )
    return n


def main() -> None:
    old = json.load(open(OLD_MANIFEST))
    DEST.mkdir(parents=True, exist_ok=True)
    entries = []
    total = 0
    for oe in old["entries"]:
        corpus = oe["corpus"]
        src = SRC / f"{corpus}.csv"
        if not src.is_file():
            sys.exit(f"ABORT: missing extraction {src}")
        dst = DEST / f"{corpus}.parquet"
        rows = csv_to_parquet(src, dst, oe["rows"])
        total += rows
        entries.append(
            {
                "corpus": corpus,
                "parquet": str(dst),
                "sha256": sha256_file(dst),
                "rows": rows,
                "n_features": N_FEATURES,
                "layout": LAYOUT,
                "role": oe["role"],
                "target_semantics": oe["target_semantics"],
                "notes": oe.get("notes", ""),
                "source_csv": str(src),
                "extractor": "zensim/examples/v2_ab_extract ZENSIM_AB_MODE=foldapp2",
            }
        )
        print(f"ok {corpus}: {rows} rows -> {dst.name}", flush=True)
    manifest = {
        "description": (
            "CANONICAL 944-feature (folded+append+append2 STREAMING regime) "
            "extraction datasets — the 11 local legs re-extracted at the "
            "PLAN_SOTA944 P1 rev with append2 ON (task #9 backfill). Row "
            "counts match ext924-canonical-2026-07-27 exactly; f0..f923 "
            "bitwise-identical to it (gate_backfill944.py per leg). append2 "
            "constants are PROVISIONAL — see build_commit. REGIME PURITY: "
            "never column-mix with 924/720/v1 parquets."
        ),
        "date": str(date.today()),
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "build_commit": ZENSIM_COMMIT,
        "driver": "v2_ab_extract (foldapp2), profile codec_target, default toggles + append2",
        "n_corpora": len(entries),
        "total_rows": total,
        "bans": old["bans"],
        "inherited_from": str(OLD_MANIFEST),
        "entries": entries,
    }
    (DEST / "_MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(f"MANIFEST: {len(entries)} corpora, {total} rows -> {DEST}/_MANIFEST.json")
    if total != old["total_rows"]:
        sys.exit(f"ABORT: total {total} != 924-era {old['total_rows']}")
    print("G-row-count parity vs ext924: PASS")


if __name__ == "__main__":
    main()
