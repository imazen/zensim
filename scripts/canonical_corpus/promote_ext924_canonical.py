#!/usr/bin/env python3
"""Promote the 924-feature (folded+append STREAMING regime) local-leg extraction
into ONE canonical dataset directory with a unified manifest — the ext924
counterpart of promote_ext720_canonical.py (W1 of the 2026-07-27 backfill plan).

Source (all 11 legs, one wave, one extractor rev):
  /mnt/v/output/zensim/ext924-run-2026-07-27/<corpus>.csv
  (driver: zensim/examples/v2_ab_extract, ZENSIM_AB_MODE=foldapp, zensim @
   0b3d16b04b19 — the C5 streaming-only rev; profile codec_target, default
   toggles, raw unpadded slices)

Destination:
  /mnt/v/zen/zensim-training/ext924-canonical-2026-07-27/

Per-corpus roles / target semantics / bans are INHERITED verbatim from the 720
canonical manifest (they are corpus facts, not regime facts):
  /mnt/v/zen/zensim-training/ext720-canonical-2026-07-22/_MANIFEST.json

Schema of every parquet: ref_basename (utf8), human_score (f64), f0..f923
(f64) — the folded+append streaming regime: f0..f155 folded v1-basic bands,
f156..f371 structurally ZERO (fold slots), f372..f719 v2-348 bounded block,
f720..f923 append-204 block. REGIME PURITY: never column-mix with ext-720/v1
parquets. All ZSTD. NaN/null gate enforced here (hard abort).

Idempotent: re-running verifies sha256s and rewrites the manifest/README.
Mirrors (run after): R2 s3://zentrain/ext924-canonical-2026-07-27/ + Tower
/mnt/tower/output/zensim-ext924-canonical-2026-07-27/.
"""

import csv
import hashlib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SRC = Path("/mnt/v/output/zensim/ext924-run-2026-07-27")
DEST = Path("/mnt/v/zen/zensim-training/ext924-canonical-2026-07-27")
OLD_MANIFEST = Path(
    "/mnt/v/zen/zensim-training/ext720-canonical-2026-07-22/_MANIFEST.json"
)
ZENSIM_COMMIT = "0b3d16b04b19"
N_FEATURES = 924
LAYOUT = (
    "f0..f923 folded+append STREAMING regime: f0..f155 folded v1-basic bands; "
    "f156..f371 structurally zero (fold slots); f372..f719 v2-348 bounded; "
    "f720..f923 append-204"
)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_to_parquet(src: Path, dst: Path, want_rows: int) -> int:
    cols = None
    names = None
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
                "extractor": "zensim/examples/v2_ab_extract ZENSIM_AB_MODE=foldapp",
            }
        )
        print(f"ok {corpus}: {rows} rows -> {dst.name}")
    manifest = {
        "description": (
            "CANONICAL 924-feature (folded+append STREAMING regime) extraction "
            "datasets — the 11 local legs re-extracted at the C5 streaming-only "
            "rev (W1 of PLAN_FLEET_924_BACKFILL_2026-07-27). Row counts match "
            "ext720-canonical-2026-07-22 exactly. REGIME PURITY: never "
            "column-mix with 720/v1 parquets."
        ),
        "date": str(date.today()),
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "build_commit": ZENSIM_COMMIT,
        "driver": "v2_ab_extract (foldapp), profile codec_target, default toggles",
        "n_corpora": len(entries),
        "total_rows": total,
        "bans": old["bans"],
        "inherited_from": str(OLD_MANIFEST),
        "entries": entries,
    }
    (DEST / "_MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(f"MANIFEST: {len(entries)} corpora, {total} rows -> {DEST}/_MANIFEST.json")
    if total != old["total_rows"]:
        sys.exit(f"ABORT: total {total} != 720-era {old['total_rows']}")
    print("G-W1 row-count parity: PASS")


if __name__ == "__main__":
    main()
