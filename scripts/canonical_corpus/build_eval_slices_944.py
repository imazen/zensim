#!/usr/bin/env python3
"""build_eval_slices_944.py — the 944-era imazen26 / nonphoto bake_verdict eval
slices, from the canonical bigcodec-944 TEST views (SOTA-944 pre-reg §8.3;
docs/FULL_EVAL.md "924-era eval slices" rule carried to 944: fingerprint/NN
tables cannot cross regimes, so the slices are exact-key subsets of the frozen
TEST views, held-out origins {7,9}).

  imazen26  = all-content stride subsample of the 4 lossy TEST views;
              human_score = score_ssim2 (as-is, 372-era convention).
  nonphoto  = the same views filtered to NON-PHOTO content classes (below),
              then stride-subsampled; human_score = score_ssim2 / 100
              (372-era convention).

NON-PHOTO class set (defined HERE, by class semantics, recorded in the
_MANIFEST — the 372-era slice used a per-image tag over VAL origins and cannot
be carried across the origin-split change; the FULL_EVAL rule says class
filter via imazen26_manifest.tsv):
  2200-unsplash-renders, 5000-national-park-service-brochures,
  5200-epa-climate-impact-2021-report, 5300-noaa-hurricane-documents,
  6000-lilith-scans-public-patents, 6600-ia-scans-manuscript-illustrations,
  6800-ia-scans-manuscript-text, 7000-lilith-plots,
  8000-lilith-mobile-screenshots, 8100-lilith-web-screenshots,
  9000-lilith-ai-clipart, 9094-lilith-ai-illustrations, 9226-lilith-ai-products

Deterministic (global stride per slice to ~TARGET rows). Output columns:
ref_basename, human_score, f0..f943 (f64, zstd) — the ext-leg convention.
"""

import argparse
import csv
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

VIEWS = ["zenavif_lossy", "zenjpeg_lossy", "zenjxl_lossy", "zenwebp_lossy"]
NONPHOTO_CLASSES = {
    "2200-unsplash-renders",
    "5000-national-park-service-brochures",
    "5200-epa-climate-impact-2021-report",
    "5300-noaa-hurricane-documents",
    "6000-lilith-scans-public-patents",
    "6600-ia-scans-manuscript-illustrations",
    "6800-ia-scans-manuscript-text",
    "7000-lilith-plots",
    "8000-lilith-mobile-screenshots",
    "8100-lilith-web-screenshots",
    "9000-lilith-ai-clipart",
    "9094-lilith-ai-illustrations",
    "9226-lilith-ai-products",
}
TARGET_ROWS = 10000


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def origin_of(ref: str) -> int:
    m = re.match(r"o_(\d+)\.png", ref)
    return int(m.group(1)) if m else -1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--views-root", required=True)
    ap.add_argument("--manifest", default="/mnt/v/output/imazen-26-features/imazen26_manifest.tsv")
    ap.add_argument("--suffix", default="_944")
    ap.add_argument("--n-feat", type=int, default=944)
    ap.add_argument("--split", default="test", choices=["test", "validate"],
                    help="picker-view split to slice (2026-08-28: 'validate' added so "
                         "SELECTION runs on validate-family slices and the test-family "
                         "slices retire to touch-once terminal reads)")
    ap.add_argument("--out-root", required=True)
    a = ap.parse_args()

    cls_of = {}
    with open(a.manifest) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            cls_of[int(row["stem"])] = row["content_class"]  # col1 = the 4-digit corpus id (header fixed 2026-08-27; was mislabeled "sha256")

    feat_cols = [f"f{i}" for i in range(a.n_feat)]
    cols = ["ref_filename", "score_ssim2"] + feat_cols
    tabs = []
    prov = []
    for ds in VIEWS:
        p = Path(a.views_root) / ds / f"{a.split}{a.suffix}.parquet"
        t = pq.read_table(p, columns=cols)
        tabs.append(t)
        prov.append({"view": str(p), "sha256": sha256_file(p), "rows": t.num_rows})
        print(f"  {ds}: test rows {t.num_rows}", flush=True)
    full = pa.concat_tables(tabs)
    refs = full["ref_filename"].to_pylist()
    origins = np.array([origin_of(r) for r in refs])
    classes = np.array([cls_of.get(o, "?") for o in origins])
    assert (np.char.not_equal(classes, "?")).all(), "unmapped origins in TEST views"
    odd = np.unique(origins % 10)
    assert set(odd.tolist()) <= {7, 9}, f"TEST views must be origins {{7,9}}, saw digits {odd}"

    def emit(name: str, mask: np.ndarray, scale: float):
        idx_all = np.nonzero(mask)[0]
        step = max(1, len(idx_all) // TARGET_ROWS)
        idx = idx_all[::step]
        t = full.take(pa.array(idx))
        hs = pc.cast(pc.multiply(t["score_ssim2"], scale), pa.float64())
        out_cols = {"ref_basename": t["ref_filename"], "human_score": hs}
        for c in feat_cols:
            out_cols[c] = t[c]
        out = Path(a.out_root) / f"ext_{name}.parquet"
        pq.write_table(pa.table(out_cols), out, compression="zstd", compression_level=7)
        n_refs = len(set(np.array(refs)[idx].tolist()))
        print(f"  ext_{name}: {len(idx)} rows / {n_refs} refs (pool {len(idx_all)}, step {step})")
        return {"file": str(out), "sha256": sha256_file(out), "rows": len(idx),
                "distinct_refs": n_refs, "pool_rows": int(len(idx_all)), "stride": step,
                "human_score": f"score_ssim2 x {scale}"}

    m_all = np.ones(len(refs), dtype=bool)
    m_np = np.isin(classes, sorted(NONPHOTO_CLASSES))
    print(f"nonphoto pool: {int(m_np.sum())} of {len(refs)} rows "
          f"({len(set(origins[m_np].tolist()))} origins)")
    e1 = emit("imazen26", m_all, 1.0)
    e2 = emit("nonphoto", m_np, 0.01)

    # HF-NL-proxy (SOTA-944 pre-reg §1b): the near-lossless band of the SAME
    # held-out TEST views — cells with score_ssim2 >= 91 (the 372-era hf
    # corpus's ssim2 91..100 band), restricted to refs with >= 6 such cells
    # so the per-ref SROCC (the registered headline, `per_ref_mean`) is
    # well-posed. human_score = score_ssim2/100 (band convention).
    ssim2 = np.asarray(full["score_ssim2"].combine_chunks(), dtype=np.float64)
    m_hf = ssim2 >= 91.0
    ref_arr = np.array(refs)
    import collections
    cnt = collections.Counter(ref_arr[m_hf].tolist())
    good_refs = {r for r, c in cnt.items() if c >= 6}
    m_hf &= np.isin(ref_arr, sorted(good_refs))
    print(f"hfnlproxy pool: {int(m_hf.sum())} rows / {len(good_refs)} refs (>=6 cells each)")
    e3 = emit("hfnlproxy", m_hf, 0.01)

    commit = subprocess.run(
        ["git", "-C", str(Path(__file__).resolve().parent.parent.parent),
         "rev-parse", "--short=12", "HEAD"],
        capture_output=True, text=True).stdout.strip() or "unknown"
    manifest = {
        "description": "944-era imazen26/nonphoto bake_verdict eval slices from the "
                       "canonical bigcodec-944 TEST views (origins {7,9}); FULL_EVAL "
                       "'924-era eval slices' rule at 944. SOTA-944 campaign "
                       "(benchmarks/sota944_campaign_2026-08-03.md).",
        "build_commit": commit,
        "nonphoto_classes": sorted(NONPHOTO_CLASSES),
        "views": prov,
        "slices": [e1, e2, e3],
    }
    mp = Path(a.out_root) / "_MANIFEST_eval_slices.json"
    mp.write_text(json.dumps(manifest, indent=1))
    print(f"wrote {mp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
