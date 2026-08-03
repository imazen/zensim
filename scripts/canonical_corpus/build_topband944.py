#!/usr/bin/env python3
"""build_topband944.py — the near-top anchor TRAINING group (SOTA-944
near-top arm; benchmarks/sota944_campaign_2026-08-03.md amendment 2).

From the 4 lossy bigcodec-944 TRAIN views (train origins ONLY — asserted):
cells with score_ssim2 >= 91 (the near-lossless band the raw output
saturates over, issue #50), global-stride subsampled to ~TARGET rows,
human_score = clip(score_ssim2/100, 0, 1). This is the INSTRUMENT-CLEAN
top-band mass: sdr25 (the selection oracle) and ext_hfnlproxy (the bar
row, TEST origins) are deliberately NOT training sources.
"""

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

VIEWS = ["zenjpeg_lossy", "zenwebp_lossy", "zenjxl_lossy", "zenavif_lossy"]
TARGET_ROWS = 30000


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views-root", required=True)
    ap.add_argument("--n-feat", type=int, default=944)
    ap.add_argument("--band-min", type=float, default=91.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    feat_cols = [f"f{i}" for i in range(a.n_feat)]
    cols = ["ref_filename", "score_ssim2"] + feat_cols
    parts = []
    prov = []
    for ds in VIEWS:
        p = Path(a.views_root) / ds / "train_944.parquet"
        t = pq.read_table(p, columns=cols)
        s2 = np.asarray(t["score_ssim2"], dtype=np.float64)
        idx = np.nonzero(s2 >= a.band_min)[0]
        parts.append(t.take(pa.array(idx)))
        prov.append({"view": str(p), "rows": t.num_rows, "band_rows": int(len(idx))})
        print(f"  {ds}: {len(idx):,} of {t.num_rows:,} rows >= {a.band_min}", flush=True)
    full = pa.concat_tables(parts)
    # train-origin assertion (picker split: even last digits train)
    refs = full["ref_filename"].to_pylist()
    digits = {int(re.match(r"o_(\d+)\.png", r).group(1)) % 10 for r in refs}
    assert digits <= {0, 2, 4, 6, 8}, f"non-train origins in TRAIN views: {digits}"
    step = max(1, full.num_rows // TARGET_ROWS)
    take = np.arange(0, full.num_rows, step, dtype=np.int64)
    sub = full.take(pa.array(take))
    hs = pc.min_element_wise(pc.max_element_wise(pc.divide(sub["score_ssim2"], 100.0), 0.0), 1.0)
    out_cols = {"ref_basename": sub["ref_filename"],
                "human_score": pc.cast(hs, pa.float64())}
    for c in feat_cols:
        out_cols[c] = sub[c]
    out = Path(a.out)
    pq.write_table(pa.table(out_cols), out, compression="zstd", compression_level=7)
    commit = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parent.parent.parent),
         "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True).stdout.strip()
    (out.parent / (out.name + "._MANIFEST.json")).write_text(json.dumps({
        "group": "topband944 near-top anchor training group",
        "sha256": sha256_file(out), "rows": sub.num_rows, "stride": step,
        "band": f"score_ssim2 >= {a.band_min}", "build_commit": commit,
        "origins": "TRAIN views only (even digits asserted); sdr25/hfnlproxy NOT sources (oracle/bar integrity)",
        "views": prov}, indent=1))
    print(f"wrote {out}: {sub.num_rows:,} rows (pool {full.num_rows:,}, stride {step})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
