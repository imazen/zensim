#!/usr/bin/env python3
"""Promote the KADIS negative-rich sample into the canonical set.

WHY. The blend champion's recipe (`_HON = {safesyn: 1, bigcodec: 1.5,
kadis: 0.3}`) trains on `/mnt/v/output/zensim/reports/b_negatives/
kadis_sample_negrich.parquet`. That file cannot be used by the Rust trainer
alongside the other groups, because `--target-column` is GLOBAL and this is the
only corpus in the mix whose target is named `score_ssim2_gpu` rather than
`human_score`. Normalizing the target here is what lets one trainer invocation
express the whole recipe.

PROVENANCE GAP — RECORDED, NOT FIXED (2026-07-15). The source parquet has no
builder anywhere in the repo: four scripts READ it (`blend_lib.py`,
`train_mlp_negatives.py`, `train_mlp_diverse.py`,
`mlp_piecewise_negatives_probe.py`), none WRITES it. It carries no
`_MANIFEST.json`, no `build_commit`, and — the sharp one — **no `source_id`**.

That last omission matters beyond bookkeeping. KADIS-700k ships `source_id`
precisely because DATA_SPLITS.md requires splitting on it ("split on this, never
on row, for leak-free train/val/test"): one reference image contributes 5
severity levels, so a row-wise split puts the same source on both sides. This
sample dropped the column, so **its split cannot be verified to be leak-free.**

It is used train-only (`val_w = 0`, role `neg`), so no leak can reach a
validation number through this corpus today — which is why this file promotes it
rather than blocking. But any future recipe that gives kadis a val weight is
unverifiable until the sample is rebuilt from
`/mnt/v/datasets/kadis700k/canonical/kadis700k_canonical_2026-06-30.parquet`
(which has `source_id`) with the selection recorded. Doing that rebuild now
would change the corpus and break reproduction of the §8.39 / round-7 result,
which is the thing being reproduced this session — so the rebuild is deliberately
NOT bundled here. The manifest carries `provenance_gap` so this is greppable
rather than folklore.

Single owner: the canonical-corpus builder family.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

SRC = Path("/mnt/v/output/zensim/reports/b_negatives/kadis_sample_negrich.parquet")
OUT_DIR = Path("/mnt/v/zen/zensim-training/canonical-2026-07-15/train")
OUT = OUT_DIR / "kadis_negrich.parquet"
N_FEAT = 372
SRC_TARGET = "score_ssim2_gpu"


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True, text=True, check=True,
    ).stdout.strip()


def main() -> int:
    if not SRC.exists():
        print(f"FATAL: missing {SRC}")
        return 1

    t = pq.read_table(SRC)
    names = set(t.schema.names)
    if SRC_TARGET not in names:
        print(f"FATAL: {SRC} lacks {SRC_TARGET!r}")
        return 1
    feat_in = [f"feat_{i}" for i in range(N_FEAT)]
    missing = [c for c in feat_in if c not in names]
    if missing:
        print(f"FATAL: lacks {len(missing)} feature columns (first {missing[0]})")
        return 1

    tgt = np.asarray(t[SRC_TARGET], dtype=np.float64)
    finite = np.isfinite(tgt)
    if not finite.all():
        print(f"FATAL: {(~finite).sum()} non-finite targets — refusing to guess")
        return 1

    # The canonical convention: `human_score` is the [0,1]-scaled anchor and the
    # trainer multiplies by --target-scale (default 100). safesyn holds exactly
    # human_score == ssim2_gpu / 100. Keep the raw value under its canonical
    # name too, so a reader never has to know which scale they are holding.
    cols = {
        "human_score": pa.array(tgt / 100.0, type=pa.float64()),
        "ssim2_gpu": pa.array(tgt, type=pa.float64()),
    }
    # Canonical schema is f<i>; this sidecar emits feat_<i>.
    for i in range(N_FEAT):
        cols[f"f{i}"] = t[f"feat_{i}"].cast(pa.float64())
    out = pa.table(cols)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pq.write_table(out, OUT, compression="zstd", compression_level=15)

    lo, hi = float(tgt.min()), float(tgt.max())
    manifest = {
        "corpus": "kadis_negrich",
        "canonical_role": "train-only negatives (role=neg, val_w=0)",
        "built_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "build_commit": git_commit(),
        "builder": "scripts/canonical_corpus/build_kadis_negrich_2026-07-15.py",
        "rows": out.num_rows,
        "features": N_FEAT,
        "bytes": OUT.stat().st_size,
        "sha256": sha256_file(OUT),
        "source_path": str(SRC),
        "source_sha256": sha256_file(SRC),
        "target_column": "human_score (= score_ssim2_gpu/100, canonical [0,1] scale)",
        "provenance_gap": {
            "no_builder": "No script in the repo writes the source parquet; 4 read it. "
                          "Its selection rule (266,111 rows out of KADIS-700k) is unrecorded.",
            "no_source_id": "The source dropped KADIS's `source_id`. DATA_SPLITS.md requires "
                            "splitting on source_id (one ref = 5 severities), so this corpus's "
                            "split CANNOT be verified leak-free.",
            "why_promoted_anyway": "Used train-only (val_w=0), so no leak reaches a validation "
                                   "number today. Rebuilding from kadis700k_canonical_2026-06-30 "
                                   "(which has source_id) would change the corpus and break "
                                   "reproduction of the round-7/§8.39 result.",
            "fix": "Rebuild from /mnt/v/datasets/kadis700k/canonical/"
                   "kadis700k_canonical_2026-06-30.parquet carrying source_id, record the "
                   "selection rule, and re-measure before giving kadis any val weight.",
        },
        "diagnostics": {
            "target_raw_range": [lo, hi],
            "target_scaled_range": [lo / 100.0, hi / 100.0],
            "pct_negative_ssim2": float((tgt < 0).mean() * 100),
        },
    }
    mpath = OUT_DIR / "_MANIFEST_kadis_negrich.json"
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"wrote {OUT}")
    print(f"  rows={out.num_rows:,} feats={N_FEAT} bytes={OUT.stat().st_size:,}")
    print(f"  sha256={manifest['sha256']}")
    print(f"  human_score [{lo/100:.4f}, {hi/100:.4f}]  (raw ssim2 [{lo:.2f}, {hi:.2f}])")
    print(f"  negative-ssim2 rows: {manifest['diagnostics']['pct_negative_ssim2']:.1f}%")
    print(f"  PROVENANCE GAP recorded in {mpath.name}: no builder, no source_id")
    return 0


if __name__ == "__main__":
    sys.exit(main())
