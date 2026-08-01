#!/usr/bin/env python3
# ===========================================================================
# AS-RUN PROVENANCE COPY — frozen 2026-07-31 (decision-surface audit gap 3:
# the seven-domain external-read runners were previously uncommitted).
# Source:        /mnt/v/output/zensim/bandvis-loo-2026-07-28/harness/promote_ext944_instrument.py
# sha256(source): 0f7c6990feb7807e248f93ab9617a6930b4d02dd178a1530e8fa2fe5239cd67d
# build_commit:  b1d4bc257e57f7c3215ec8a237e9f87cdad8e35f
# Protocol doc:  benchmarks/bandvis_loo_944_2026-07-28.md
# Everything below the marker line is BYTE-IDENTICAL to the source file
# (verify: strip through the marker, sha256 the rest). Do NOT extend this
# file — it is an archival record of the exact as-run analysis (it may call
# scipy directly; it predates the stats-rule batch migration and is kept
# verbatim). New analyses go through scripts/external_reads/run_external_reads.py.
# Data paths inside are as-run: /mnt/v artifact paths persist; ~/tmp
# FEATS_DIR-style inputs were session scratch — the pooled/persisted tables
# in the artifact dir are the stored equivalents (see ../README.md).
# ===== byte-identical source below this line ================================
"""Promote the 944-feature (folded+append+append2 STREAMING regime) local-leg
extraction into the instrument dataset dir + manifest — adapted from
zensim scripts/canonical_corpus/promote_ext924_canonical.py (the W1 promote),
N_FEATURES 924 -> 944, roles inherited from the ext924 canonical manifest.

Role: instrument-grade (LOO/P12 at 944); regenerable in ~30 min; NOT canonical
— no R2/Tower mirroring by design.

Schema: ref_basename (utf8), human_score (f64), f0..f943 (f64), ZSTD-7,
NaN/null hard abort, row counts MUST equal the ext924 counts exactly.
"""

import csv
import hashlib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

SRC = Path("/mnt/v/output/zensim/ext944-run-2026-07-28")
DEST = Path("/mnt/v/zen/zensim-training/ext944-instrument-2026-07-28")
EXT924_MANIFEST = Path(
    "/mnt/v/zen/zensim-training/ext924-canonical-2026-07-27/_MANIFEST.json"
)
ZENSIM_COMMIT = "b1d4bc257e57f7c3215ec8a237e9f87cdad8e35f"
N_FEATURES = 944
LAYOUT = (
    "f0..f943 folded+append+append2 STREAMING regime: f0..f155 folded v1-basic "
    "bands; f156..f371 structurally zero (fold slots); f372..f719 v2-348 "
    "bounded; f720..f923 append-204; f924..f943 append2-20 (BANDVIS_GAIN/LOSS, "
    "LUMA_MEAN_REF, HL_BIN1/2 x 4 scales, Y-only; f = 924 + scale*5 + local)"
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
    old = json.load(open(EXT924_MANIFEST))
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
            "INSTRUMENT-GRADE 944-feature (folded+append+append2 STREAMING "
            "regime) extraction — the 11 local legs re-extracted with "
            "ZENSIM_AB_MODE=foldapp2 for the append2/BANDVIS pre-registered "
            "LOO gate + P12 residual-boost. Row counts match "
            "ext924-canonical-2026-07-27 exactly; first 924 columns verified "
            "byte-equal to the canonical 924 parquets (byte_gate_944.py). "
            "REGIME PURITY: never column-mix with 720/v1 parquets."
        ),
        "role": (
            "instrument-grade (LOO/P12 at 944); regenerable in ~30 min; NOT "
            "canonical — no R2/Tower mirroring by design"
        ),
        "date": str(date.today()),
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "build_commit": ZENSIM_COMMIT,
        "driver": "v2_ab_extract (foldapp2), profile codec_target, default toggles",
        "n_corpora": len(entries),
        "total_rows": total,
        "bans": old["bans"],
        "pairs_provenance": (
            "same per-corpus pairs TSVs as the W1 ext924 run — see "
            "/mnt/v/output/zensim/ext924-run-2026-07-27/run.log and "
            + str(EXT924_MANIFEST)
        ),
        "inherited_from": str(EXT924_MANIFEST),
        "entries": entries,
    }
    (DEST / "_MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(f"MANIFEST: {len(entries)} corpora, {total} rows -> {DEST}/_MANIFEST.json")
    if total != old["total_rows"]:
        sys.exit(f"ABORT: total {total} != ext924 {old['total_rows']}")
    print("row-count parity vs ext924: PASS")


if __name__ == "__main__":
    main()
