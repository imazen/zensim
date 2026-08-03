#!/usr/bin/env python3
"""build_tbig_200k.py — the committed rebuild of the E-M campaign's bigcodec
training slice (`tbig_924_200k`) at any regime width.

The 924-era builder was uncommitted scratch (`~/tmp`-class, coh924 session); this
is its durable owner (SOTA-944 pre-reg §5/§8, benchmarks/sota944_campaign_2026-08-03.md).
Mechanism (recovered + verified against the surviving
/mnt/v/zen/zensim-training/tbig_924_200k.parquet, 208,169 rows; the view ORDER
and key form were pinned empirically from that file's segment boundaries):

  for each of the 4 lossy TRAIN views in the order
  (zenjpeg_lossy, zenwebp_lossy, zenjxl_lossy, zenavif_lossy):
      step = n_rows // 50000
      keep rows 0, step, 2*step, ...          (deterministic global stride)
      human_score = clip(score_ssim2 / 100, 0, 1)
      ref_basename = ref_filename             (already carries ".png")
  concat -> one parquet: ref_basename, human_score, encoded_filename, f0..f<N-1>

`encoded_filename` is NEW vs the 924 slice (the free join-key fix the 924 build
forgot); feature columns come from the view verbatim.

Gates (all hard, run in-process):
  G-T1: (ref_basename, human_score) sequence EXACTLY equals the reference
        slice's (row-for-row) when --verify-against is given.
  G-T2: f0..f923 bitwise-identical to the reference slice on every row
        (columnar equality; f64 bit pattern via cast to int64 view).

Usage:
  python3 scripts/canonical_corpus/build_tbig_200k.py \
      --views-root /mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/bigcodec \
      --suffix _944 --n-feat 944 \
      --verify-against /mnt/v/zen/zensim-training/tbig_924_200k.parquet \
      --out /mnt/v/zen/zensim-training/tbig_944_200k.parquet
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

VIEWS = ["zenjpeg_lossy", "zenwebp_lossy", "zenjxl_lossy", "zenavif_lossy"]
TARGET_PER_VIEW = 50000


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views-root", required=True)
    ap.add_argument("--suffix", default="_944", help="view filename suffix (train<suffix>.parquet)")
    ap.add_argument("--n-feat", type=int, default=944)
    ap.add_argument("--verify-against", help="reference slice for G-T1/G-T2 (the 924 file)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    feat_cols = [f"f{i}" for i in range(a.n_feat)]
    parts = []
    prov = []
    for ds in VIEWS:
        p = Path(a.views_root) / ds / f"train{a.suffix}.parquet"
        n = pq.read_metadata(p).num_rows
        step = n // TARGET_PER_VIEW
        idx = np.arange(0, n, step, dtype=np.int64)
        cols = ["ref_filename", "encoded_filename", "score_ssim2"] + feat_cols
        t = pq.read_table(p, columns=cols).take(pa.array(idx))
        hs = pc.min_element_wise(
            pc.max_element_wise(pc.divide(t["score_ssim2"], 100.0), 0.0), 1.0
        )
        out_cols = {
            "ref_basename": t["ref_filename"],
            "human_score": pc.cast(hs, pa.float64()),
            "encoded_filename": t["encoded_filename"],
        }
        for c in feat_cols:
            out_cols[c] = t[c]
        parts.append(pa.table(out_cols))
        prov.append({"view": str(p), "sha256": sha256_file(p), "n_rows": n, "step": step,
                     "picked": len(idx)})
        print(f"  {ds}: n={n} step={step} picked={len(idx)}", flush=True)

    full = pa.concat_tables(parts)
    print(f"total rows: {full.num_rows}")

    if a.verify_against:
        ref = pq.read_table(a.verify_against)
        assert full.num_rows == ref.num_rows, (
            f"G-T1 FAIL: rows {full.num_rows} != ref {ref.num_rows}")
        rb_new = full["ref_basename"].combine_chunks()
        rb_old = ref["ref_basename"].combine_chunks()
        assert rb_new.equals(rb_old), "G-T1 FAIL: ref_basename sequence differs"
        hs_new = np.asarray(full["human_score"].combine_chunks(), dtype=np.float64)
        hs_old = np.asarray(ref["human_score"].combine_chunks(), dtype=np.float64)
        assert (hs_new.view(np.int64) == hs_old.view(np.int64)).all(), (
            "G-T1 FAIL: human_score bit patterns differ")
        n_ref_feat = sum(1 for c in ref.schema.names if c.startswith("f") and c[1:].isdigit())
        mism = []
        for i in range(n_ref_feat):
            c = f"f{i}"
            x = np.asarray(full[c].combine_chunks(), dtype=np.float64).view(np.int64)
            y = np.asarray(ref[c].combine_chunks(), dtype=np.float64).view(np.int64)
            if not (x == y).all():
                mism.append(c)
        assert not mism, f"G-T2 FAIL: {len(mism)} feature cols differ vs ref: {mism[:8]}"
        print(f"G-T1 + G-T2 PASS: keys + f0..f{n_ref_feat - 1} bitwise-identical "
              f"to {a.verify_against}")

    out = Path(a.out)
    pq.write_table(full, out, compression="zstd", compression_level=7)
    commit = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parent.parent.parent),
         "rev-parse", "--short=12", "HEAD"],
        capture_output=True, text=True).stdout.strip() or "unknown"
    manifest = {
        "file": str(out), "sha256": sha256_file(out), "rows": full.num_rows,
        "n_feat": a.n_feat, "build_commit": commit,
        "mechanism": "global-row-stride step=n//50000 per lossy TRAIN view; "
                     "human_score=clip(score_ssim2/100,0,1); row-identical to "
                     "tbig_924_200k (G-T1/G-T2 gated when --verify-against)",
        "views": prov,
        "verified_against": a.verify_against,
    }
    mp = out.with_suffix(".parquet._MANIFEST.json")
    mp.write_text(json.dumps(manifest, indent=1))
    print(f"wrote {out} ({out.stat().st_size} B) + {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
