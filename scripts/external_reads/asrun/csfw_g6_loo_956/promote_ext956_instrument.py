#!/usr/bin/env python3
# ===========================================================================
# AS-RUN PROVENANCE COPY — frozen 2026-07-31 (decision-surface audit gap 3:
# the seven-domain external-read runners were previously uncommitted).
# Source:        /mnt/v/output/zensim/csfw-g6-loo-2026-07-29/harness/promote_ext956_instrument.py
# sha256(source): 1a3257d206e7bd4d2da2865d329e59a2476541b4275a16c2c9b220a43d838f23
# build_commit:  7bfd511de78f85e8fcd618df15716ca56575bb60
# Protocol doc:  benchmarks/csfw_g6_loo_2026-07-29.md
# Everything below the marker line is BYTE-IDENTICAL to the source file
# (verify: strip through the marker, sha256 the rest). Do NOT extend this
# file — it is an archival record of the exact as-run analysis (it may call
# scipy directly; it predates the stats-rule batch migration and is kept
# verbatim). New analyses go through scripts/external_reads/run_external_reads.py.
# Data paths inside are as-run: /mnt/v artifact paths persist; ~/tmp
# FEATS_DIR-style inputs were session scratch — the pooled/persisted tables
# in the artifact dir are the stored equivalents (see ../README.md).
# ===== byte-identical source below this line ================================
"""Promote the 956-feature (folded+append+append2+csfw STREAMING regime)
local-leg extraction into the instrument dataset dir + manifest — adapted from
the bandvis-loo harness promote_ext944_instrument.py (N_FEATURES 944 -> 956,
mode foldapp2 -> foldcsfw), roles inherited from the ext944-instrument
manifest (which inherited them from ext924-canonical).

Role: instrument-grade (G6 LOO at 956); regenerable ~30 min; NOT canonical
— no R2/Tower mirroring by design.

Schema: ref_basename (utf8), human_score (f64), f0..f955 (f64), ZSTD-7,
NaN/null hard abort, row counts MUST equal the ext944-instrument counts
exactly (which equal ext924-canonical).
"""

import csv
import hashlib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SRC = Path("/mnt/v/output/zensim/ext956-run-2026-07-29")
DEST = Path("/mnt/v/zen/zensim-training/ext956-instrument-2026-07-29")
EXT944_MANIFEST = Path(
    "/mnt/v/zen/zensim-training/ext944-instrument-2026-07-28/_MANIFEST.json"
)
ZENSIM_COMMIT = "7bfd511de78f85e8fcd618df15716ca56575bb60"
N_FEATURES = 956
LAYOUT = (
    "f0..f955 folded+append+append2+csfw STREAMING regime: f0..f155 folded "
    "v1-basic bands; f156..f371 structurally zero (fold slots); f372..f719 "
    "v2-348 bounded; f720..f923 append-204; f924..f943 append2-20 "
    "(BANDVIS_GAIN/LOSS, LUMA_MEAN_REF, HL_BIN1/2 x 4 scales, Y-only, "
    "f = 924 + scale*5 + local); f944..f955 csfw tier-1 12 "
    "(W_GLOBAL_DMEAN/W_GLOBAL_CGAIN/W_GLOBAL_CLOSS x 4 scales, Y-only, "
    "f = 944 + scale*3 + local)"
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
    old = json.load(open(EXT944_MANIFEST))
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
                "extractor": "zensim/examples/v2_ab_extract ZENSIM_AB_MODE=foldcsfw",
            }
        )
        print(f"ok {corpus}: {rows} rows -> {dst.name}", flush=True)
    manifest = {
        "description": (
            "INSTRUMENT-GRADE 956-feature (folded+append+append2+csfw STREAMING "
            "regime) extraction — the 11 local legs re-extracted with "
            "ZENSIM_AB_MODE=foldcsfw for the CSFW tier-1 pre-registered G6 LOO "
            "gate + P12 residual-boost. Row counts match "
            "ext944-instrument-2026-07-28 (= ext924-canonical-2026-07-27) "
            "exactly; first 944 columns verified byte-equal to the "
            "ext944-instrument parquets (byte_gate_956.py). REGIME PURITY: "
            "never column-mix with 944/924/720/v1 parquets."
        ),
        "role": (
            "instrument-grade (G6 LOO at 956); regenerable ~30 min; NOT "
            "canonical — no R2/Tower mirroring by design"
        ),
        "date": str(date.today()),
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "build_commit": ZENSIM_COMMIT,
        "driver": "v2_ab_extract (foldcsfw), profile codec_target, default toggles",
        "n_corpora": len(entries),
        "total_rows": total,
        "bans": old["bans"],
        "pairs_provenance": (
            "same per-corpus pairs TSVs as the ext944-instrument run (itself "
            "the W1 ext924 pairs) — see "
            + str(EXT944_MANIFEST)
            + " and /mnt/v/output/zensim/ext924-run-2026-07-27/run.log"
        ),
        "inherited_from": str(EXT944_MANIFEST),
        "entries": entries,
    }
    (DEST / "_MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(f"MANIFEST: {len(entries)} corpora, {total} rows -> {DEST}/_MANIFEST.json")
    if total != old["total_rows"]:
        sys.exit(f"ABORT: total {total} != ext944 {old['total_rows']}")
    print("row-count parity vs ext944-instrument: PASS")


if __name__ == "__main__":
    main()
