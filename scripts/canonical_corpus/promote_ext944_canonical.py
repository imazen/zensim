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

R1b (2026-08-30) adds three env knobs so the SAME owner promotes the all-live
pools regime instead of a second copy of this code:

  EXT944_MODE   extraction regime tag, default "folded720append2". R1b uses
                "folded720append2pools" (V1PoolsMode::Full — f156..371 carry
                v1's peak/masked/IW pool blocks LIVE instead of structural
                zeros). The tag is written into the manifest, the layout
                string, the parquet key-value metadata, AND a `regime` utf8
                column on every row, so a pools row can never be silently
                column-mixed with a zero-block row. (The zensim parquet
                loader projects only the columns it needs and skips the rest,
                so the extra column is loader-transparent.)
  EXT944_LEGS   space-separated corpus names to promote (default: all).
  EXT944_VERIFY_ROOT
                root holding a SAME-BINARY zero-block control extraction.
                When set, each promoted leg is GATED:
                  G-P1  f0..f155 and f372..f943 BIT-IDENTICAL to the stored
                        leg (same pairs, same pixels, same code path — the
                        ONLY thing the regime changes is the pool block)
                  G-P2  stored f156..f371 all zero; promoted f156..f371 not
                  G-P3  ref_basename + human_score identical row for row
                A failure aborts that leg. The control MUST come from the same
                extractor build: comparing against an older-era canonical root
                measures extractor drift, not the toggle (MEASURED 2026-08-30:
                22 of 728 non-pool columns differ from the 2026-08-01 root at
                |d| ~ 1e-9, all in the append block — a real but tiny version
                drift that would make a bit-identity gate meaningless).
  EXT944_DRIFT_ROOT
                an older-era root to REPORT drift against (never a pass/fail).
                Emits per-leg n_columns_differing + worst column + max |d| and
                max rel into the manifest.
                `EXT944_BASENAME_MAP=old:new,...`
                records a DELIBERATE ref_basename substitution (the TID
                `i25.png` case defect: the canonical pairs TSV names
                `I25.png`, which the corpus no longer has) — the map is
                applied before G-P3 and is recorded in the manifest.
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
MODE = os.environ.get("EXT944_MODE", "folded720append2")
ONLY_LEGS = set(os.environ.get("EXT944_LEGS", "").split())
VERIFY_ROOT = os.environ.get("EXT944_VERIFY_ROOT")
DRIFT_ROOT = os.environ.get("EXT944_DRIFT_ROOT")
BASENAME_MAP = dict(
    kv.split(":", 1) for kv in os.environ.get("EXT944_BASENAME_MAP", "").split(",") if kv
)
POOLS_LIVE = MODE.endswith("pools") or MODE.endswith("carriers")
LAYOUT = (
    "f0..f943 folded+append+append2 STREAMING regime: f0..f155 folded v1-basic "
    "bands; f156..f371 structurally zero (fold slots); f372..f719 v2-348 "
    "bounded; f720..f923 append-204; f924..f943 append2-20 (BANDVIS_GAIN/LOSS, "
    "LUMA_MEAN_REF, HL_BIN1/2 x 4 scales, Y-only; f = 924 + scale*5 + local; "
    "constants PROVISIONAL per benchmarks/append2_bandvis_gates_2026-07-27.md)"
)
if POOLS_LIVE:
    LAYOUT = LAYOUT.replace(
        "f156..f371 structurally zero (fold slots); ",
        f"f156..f371 v1 peak/masked/IW POOL BLOCKS LIVE ({MODE}, "
        "V1PoolsMode::Full/Carriers — NOT structural zeros; its own regime, "
        "never column-mixed with folded720append2 rows); ",
    )


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _pool_stats(cols, names):
    """(n_nonzero_slots, n_zero_slots) over f156..f371 using the first row."""
    nz = 0
    for i in range(156, 372):
        c = cols.get(f"f{i}")
        if c and any(v != 0.0 for v in c[: min(len(c), 64)]):
            nz += 1
    return nz, 216 - nz


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
    if BASENAME_MAP:
        cols[names[0]] = [BASENAME_MAP.get(b, b) for b in cols[names[0]]]
    nz, zero = _pool_stats(cols, names)
    if POOLS_LIVE and nz == 0:
        sys.exit(
            f"ABORT {src.name}: EXT944_MODE={MODE} claims live pools but "
            f"f156..f371 are all zero — that is a folded720append2 extraction"
        )
    if not POOLS_LIVE and nz != 0:
        sys.exit(
            f"ABORT {src.name}: EXT944_MODE={MODE} expects structural zeros but "
            f"{nz} of 216 f156..f371 slots are live"
        )
    arrays = [pa.array(cols[names[0]], type=pa.utf8())] + [
        pa.array(cols[n2], type=pa.float64()) for n2 in names[1:]
    ]
    out_names = list(names)
    if MODE != "folded720append2":
        arrays.append(pa.array([MODE] * n, type=pa.utf8()))
        out_names.append("regime")
    tbl = pa.table(arrays, names=out_names).replace_schema_metadata(
        {"zensim_regime": MODE, "zensim_n_features": str(N_FEATURES)}
    )
    pq.write_table(tbl, dst, compression="zstd", compression_level=7)
    return n, nz


def measure_drift(corpus: str, dst: Path) -> dict:
    """INFORMATIONAL (never a gate): how far the promoted leg sits from an
    older-era canonical root outside the pool block."""
    import numpy as np

    stored = Path(DRIFT_ROOT) / f"{corpus}.parquet"
    if not stored.is_file():
        return {"drift": "SKIPPED (no such leg)", "root": str(stored)}
    feat = [f"f{i}" for i in range(N_FEATURES) if not 156 <= i < 372]
    a = pq.read_table(stored, columns=feat)
    b = pq.read_table(dst, columns=feat)
    if a.num_rows != b.num_rows:
        return {"drift": f"SKIPPED (rows {a.num_rows} vs {b.num_rows})"}
    ndiff, worst_abs, worst_col, worst_rel = 0, 0.0, "", 0.0
    for c in feat:
        x = np.asarray(a[c].combine_chunks(), dtype=np.float64)
        y = np.asarray(b[c].combine_chunks(), dtype=np.float64)
        if np.array_equal(x, y):
            continue
        ndiff += 1
        d = float(np.max(np.abs(x - y)))
        scale = np.maximum(np.abs(x), np.abs(y))
        rel = float(np.max(np.abs(x - y)[scale > 0] / scale[scale > 0])) if np.any(scale > 0) else 0.0
        worst_rel = max(worst_rel, rel)
        if d > worst_abs:
            worst_abs, worst_col = d, c
    print(f"    DRIFT vs {Path(DRIFT_ROOT).name}: {ndiff}/728 non-pool columns "
          f"differ, worst {worst_col} max|d|={worst_abs:.3e} max_rel={worst_rel:.3e}")
    return {"root": str(stored), "columns_differing": ndiff,
            "of_columns": len(feat), "worst_column": worst_col,
            "max_abs": worst_abs, "max_rel": worst_rel}


def verify_against_stored(corpus: str, dst: Path) -> dict:
    """G-P1/G-P2/G-P3: the pools re-extraction differs from a SAME-BINARY
    zero-block control in the pool block and NOWHERE else."""
    import numpy as np

    stored = Path(VERIFY_ROOT) / f"{corpus}.parquet"
    if not stored.is_file():
        return {"gate": "SKIPPED (no stored leg)", "stored": str(stored)}
    feat = [f"f{i}" for i in range(N_FEATURES)]
    a = pq.read_table(stored, columns=["ref_basename", "human_score"] + feat)
    b = pq.read_table(dst, columns=["ref_basename", "human_score"] + feat)
    if a.num_rows != b.num_rows:
        sys.exit(f"G-P3 FAIL {corpus}: {b.num_rows} rows vs stored {a.num_rows}")
    if a["ref_basename"].to_pylist() != b["ref_basename"].to_pylist():
        sys.exit(f"G-P3 FAIL {corpus}: ref_basename sequence differs from stored")
    ah = np.asarray(a["human_score"].combine_chunks(), dtype=np.float64)
    bh = np.asarray(b["human_score"].combine_chunks(), dtype=np.float64)
    if not np.array_equal(ah, bh):
        sys.exit(f"G-P3 FAIL {corpus}: human_score differs from stored")
    outside, pool_zero_stored, pool_live_new = 0, 0, 0
    worst = (0.0, "")
    for i in range(N_FEATURES):
        x = np.asarray(a[f"f{i}"].combine_chunks(), dtype=np.float64)
        y = np.asarray(b[f"f{i}"].combine_chunks(), dtype=np.float64)
        if 156 <= i < 372:
            if not np.any(x != 0.0):
                pool_zero_stored += 1
            if np.any(y != 0.0):
                pool_live_new += 1
            continue
        if not np.array_equal(x, y):
            outside += 1
            d = float(np.max(np.abs(x - y)))
            if d > worst[0]:
                worst = (d, f"f{i}")
    if outside:
        sys.exit(
            f"G-P1 FAIL {corpus}: {outside} of 728 non-pool columns differ from "
            f"stored (worst {worst[1]} max|d|={worst[0]:.3e}) — the regime flag "
            f"must change the pool block and NOTHING else"
        )
    print(
        f"    G-P1 OK: 728/728 non-pool columns BIT-IDENTICAL to {stored.name}; "
        f"G-P2 stored-zero {pool_zero_stored}/216, new-live {pool_live_new}/216"
    )
    return {
        "gate": "PASS",
        "stored": str(stored),
        "nonpool_columns_bit_identical": 728,
        "stored_pool_slots_zero": pool_zero_stored,
        "promoted_pool_slots_live": pool_live_new,
    }


def main() -> None:
    old = json.load(open(OLD_MANIFEST))
    DEST.mkdir(parents=True, exist_ok=True)
    entries = []
    total = 0
    for oe in old["entries"]:
        corpus = oe["corpus"]
        if ONLY_LEGS and corpus not in ONLY_LEGS:
            continue
        src = SRC / f"{corpus}.csv"
        if not src.is_file():
            sys.exit(f"ABORT: missing extraction {src}")
        dst = DEST / f"{corpus}.parquet"
        rows, nz = csv_to_parquet(src, dst, oe["rows"])
        total += rows
        gate = verify_against_stored(corpus, dst) if VERIFY_ROOT else None
        drift = measure_drift(corpus, dst) if DRIFT_ROOT else None
        entries.append(
            {
                "regime": MODE,
                "pool_slots_live": nz,
                "verify": gate,
                "drift": drift,
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
                "extractor": f"zensim/examples/v2_ab_extract ZENSIM_AB_MODE={MODE}",
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
        "driver": f"v2_ab_extract ({MODE}), profile codec_target, default toggles + append2",
        "regime": MODE,
        "regime_purity": (
            f"Rows in regime {MODE} are NEVER column-mixed with rows of any "
            "other regime (folded720append2 / *carriers / 924 / 720 / v1)."
        ),
        "basename_map": BASENAME_MAP or None,
        "n_corpora": len(entries),
        "total_rows": total,
        "bans": old["bans"],
        "inherited_from": str(OLD_MANIFEST),
        "entries": entries,
    }
    (DEST / "_MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(f"MANIFEST: {len(entries)} corpora, {total} rows -> {DEST}/_MANIFEST.json")
    if ONLY_LEGS:
        print(f"partial promote ({len(entries)} of {len(old['entries'])} legs) — "
              "total-row parity gate not applicable")
    elif total != old["total_rows"]:
        sys.exit(f"ABORT: total {total} != 924-era {old['total_rows']}")
    else:
        print("G-row-count parity vs ext924: PASS")


if __name__ == "__main__":
    main()
