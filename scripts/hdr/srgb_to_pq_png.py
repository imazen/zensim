#!/usr/bin/env python3
"""Convert sRGB u8 PNGs to 203-nit PQ-PNGs (16-bit, cICP-spliced) — the
G-A / R1 sub-domain-identity instrument's input converter.

The nits mapping MUST match zensim's own SDR→PU-linear convention
(`zensim/src/metric.rs` feature-layout parity test): sRGB piecewise EOTF,
linear × 203 cd/m² (BT.2408 graphics white), then PQ-encode (SMPTE 2084,
peak 10,000). Writing via kadis_distort.io.save_dist_pq (chunked drvfs-safe,
cICP spliced) so the resulting files flow through the standard
`score-pairs --hdr` ingest.

  usage: srgb_to_pq_png.py <list.tsv> <out_dir> [--jobs 8]
         list.tsv: one absolute sRGB PNG path per line; output keeps the
         basename with .pq.png suffix; existing outputs are skipped.
"""
import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, os.path.expanduser("~/work/kadis-distort"))

ap = argparse.ArgumentParser()
ap.add_argument("list_tsv")
ap.add_argument("out_dir")
ap.add_argument("--jobs", type=int, default=8)
a = ap.parse_args()

# cICP payload: PQ (16) transfer, BT.709 primaries (1), full range —
# matches what hdr.rs's PNG ingest expects for PQ-PNG (transfer==16 gate).
CICP = bytes([1, 16, 0, 1])


def convert(src: str) -> str:
    import numpy as np
    import cv2
    from kadis_distort.io import save_dist_pq

    # Unique output name from the last 3 path components (dataset__sub__name)
    # — LIVE and TID2013 share basenames (i01.png), a flat mapping collides.
    parts = src.rstrip("/").split("/")
    stem = "__".join(parts[-3:]).rsplit(".png", 1)[0]
    dst = os.path.join(a.out_dir, stem + ".pq.png")
    if os.path.exists(dst):
        return "skip"
    img = cv2.imread(src, cv2.IMREAD_COLOR)  # BGR u8
    assert img is not None, src
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
    lin = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    nits = lin * 203.0
    # SMPTE 2084 PQ encode, peak 10,000 cd/m²
    y = np.clip(nits / 10000.0, 0.0, 1.0)
    m1, m2 = 2610 / 16384, 2523 / 4096 * 128
    c1, c2, c3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
    ym1 = y ** m1
    pq = ((c1 + c2 * ym1) / (1 + c3 * ym1)) ** m2
    save_dist_pq(pq.astype(np.float32), CICP, dst)
    return "ok"


paths = [l.strip() for l in open(a.list_tsv) if l.strip()]
os.makedirs(a.out_dir, exist_ok=True)
if a.jobs <= 1:
    # SERIAL path (2026-08-28): --jobs 1 avoids multiprocessing entirely —
    # forkserver children can resolve a mixed interpreter in venv setups and
    # die at import (measured: PU21 experiment, all 849 conversions lost).
    res = [convert(p) for p in paths]
else:
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        res = list(ex.map(convert, paths, chunksize=8))
print(f"converted {res.count('ok')}, skipped {res.count('skip')}, total {len(paths)}")
