#!/usr/bin/env python3
"""Build the WAVE-12 avif944 TRAIN-ONLY leg parquet (campaign appendix AD).

Registered rule (benchmarks/sota944_campaign_2026-08-03.md APPENDIX AC.2 +
AC.R1 amendment 1 + APPENDIX AD): the leg = the registered rescue-rebuilt
training view `train_944.parquet` (sha-gated below) with the codec-leg-
convention target column appended:

    human_score = ssim2_gpu / 100        (f64, quality-oriented, ~[0,1];
                                          negative tail allowed by design)

matching the cid22_train / konjnd_bpg / kadis codec-leg precedents (target
stored as `human_score` = ssim2/100 so the shared `--target-column
human_score --target-scale 100` recipe applies unchanged). NOTHING else is
altered: every original column rides along; feat_0..feat_943 stay contiguous
from feat_0 (the parquet_loader contract); `human_score` is appended last.

Orientation provenance: ssim2_gpu's within-ladder orientation is owned by the
corpus-level G-Z5 gate (PASSED 0.999313 on the rescue-rebuilt scores; recorded
in the corpus _MANIFEST.json, sha-pinned in the view manifest this build
verifies). This builder therefore gates by CONSTRUCTION IDENTITY
(human_score*100 == ssim2_gpu bit-for-bit as f64 ops allow) rather than
re-implementing the ladder gate.

Registered leg weight (computed here, recorded in the manifest fragment and in
every run's embedded zentrain.repro): the bigcodec-leg row-count convention =
equal per-row train mass to the bigcodec leg,
    w = 0.5 * rows_avif / rows_bigcodec = 0.5 * 459780 / 208169 = 1.1043 (4dp)

Usage:  python3 scripts/canonical_corpus/build_avif944_leg.py
Exit 0 = built + gated; nonzero = any gate failed (no partial output kept).
"""

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

VIEW_DIR = Path("/mnt/v/zen/zensim-training/avif944-2026-08-07")
IN_PATH = VIEW_DIR / "train_944.parquet"
IN_SHA = "fa35d5cbb4c84e35d16c9341f6fce87a43b38eeea1055c8bcf59762fe15c77c8"
OUT_PATH = VIEW_DIR / "avif944_leg_944.parquet"
MANIFEST_FRAG = VIEW_DIR / "_MANIFEST_avif944_leg.json"
EXPECT_ROWS = 459_780
EXPECT_ORIGINS = 873
TRAIN_DIGITS = {"0", "2", "4", "6"}
ROWS_BIGCODEC = 208_169  # tbig_944_200k.parquet, the convention's yardstick
WEIGHT_RAW = 0.5 * EXPECT_ROWS / ROWS_BIGCODEC
WEIGHT_4DP = round(WEIGHT_RAW, 4)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    print(f"sha256-gating {IN_PATH} ...", flush=True)
    got = sha256_file(IN_PATH)
    if got != IN_SHA:
        print(f"ABORT: input sha {got} != registered {IN_SHA}", file=sys.stderr)
        return 1

    pf = pq.ParquetFile(IN_PATH)
    names = pf.schema_arrow.names
    if "human_score" in names:
        print("ABORT: input already has human_score", file=sys.stderr)
        return 1
    if "ssim2_gpu" not in names or "origin" not in names:
        print("ABORT: expected columns missing", file=sys.stderr)
        return 1
    # feat contiguity from feat_0 (parquet_loader contract)
    f0 = names.index("feat_0")
    for i in range(944):
        if names[f0 + i] != f"feat_{i}":
            print(f"ABORT: feat_{i} not contiguous at {f0 + i}", file=sys.stderr)
            return 1

    out_schema = pa.schema(
        list(pf.schema_arrow) + [pa.field("human_score", pa.float64())]
    )
    tmp = OUT_PATH.with_suffix(".parquet.tmp")
    n_rows = 0
    origins = set()
    writer = pq.ParquetWriter(tmp, out_schema, compression="zstd")
    try:
        for rg in range(pf.num_row_groups):
            t = pf.read_row_group(rg)
            ss = pc.cast(t.column("ssim2_gpu"), pa.float64())
            hs = pc.divide(ss, pa.scalar(100.0, pa.float64()))
            # construction-identity gate: hs*100 must round-trip to ssim2 exactly
            back = pc.multiply(hs, pa.scalar(100.0, pa.float64()))
            neq = pc.sum(
                pc.cast(pc.not_equal(back, ss), pa.int64())
            ).as_py() or 0
            # x/100*100 is not always bit-exact in f64; allow <=1e-9 rel drift
            if neq:
                diff = pc.max(pc.abs(pc.subtract(back, ss))).as_py()
                if diff > 1e-9:
                    print(f"ABORT: identity gate max|Δ|={diff}", file=sys.stderr)
                    return 1
            t = t.append_column("human_score", hs)
            writer.write_table(t)
            n_rows += t.num_rows
            origins.update(t.column("origin").to_pylist())
            print(f"  rg {rg + 1}/{pf.num_row_groups} rows={n_rows}", flush=True)
    finally:
        writer.close()

    if n_rows != EXPECT_ROWS:
        print(f"ABORT: rows {n_rows} != {EXPECT_ROWS}", file=sys.stderr)
        tmp.unlink()
        return 1
    if len(origins) != EXPECT_ORIGINS:
        print(f"ABORT: origins {len(origins)} != {EXPECT_ORIGINS}", file=sys.stderr)
        tmp.unlink()
        return 1
    bad = {o for o in origins if str(o)[-1] not in TRAIN_DIGITS}
    if bad:
        print(f"ABORT: non-train-digit origins: {sorted(bad)[:5]}", file=sys.stderr)
        tmp.unlink()
        return 1

    os.replace(tmp, OUT_PATH)
    out_sha = sha256_file(OUT_PATH)
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
    ).stdout.strip()
    frag = {
        "_": "wave-12 avif944 TRAIN-ONLY leg (campaign appendix AD)",
        "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "build_commit": commit,
        "input": {"path": str(IN_PATH), "sha256": IN_SHA},
        "output": {"path": str(OUT_PATH), "sha256": out_sha, "rows": n_rows},
        "target_rule": "human_score = ssim2_gpu/100 (codec-leg convention; "
        "quality-oriented; negative tail allowed)",
        "orientation_provenance": "corpus G-Z5 gate PASSED 0.999313 "
        "(rescue-rebuilt scores; see corpus _MANIFEST.json)",
        "registered_weight": {
            "convention": "equal per-row train mass to the bigcodec leg",
            "formula": "0.5 * rows_avif / rows_bigcodec",
            "rows_avif": EXPECT_ROWS,
            "rows_bigcodec": ROWS_BIGCODEC,
            "value_raw": WEIGHT_RAW,
            "value_argv": WEIGHT_4DP,
        },
        "loss_mode": "both",
        "val_weight": 0.0,
        "origins": len(origins),
    }
    MANIFEST_FRAG.write_text(json.dumps(frag, indent=1) + "\n")
    print(f"OK: {OUT_PATH} rows={n_rows} sha={out_sha}")
    print(f"registered weight argv value: {WEIGHT_4DP}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
