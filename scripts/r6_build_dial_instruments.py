#!/usr/bin/env python3
"""Cut each arm's dial instruments from that arm's own extracted tables.

* `<arm>_ladder.parquet`   — the floor-dense ladder grid, as extracted.
* `<arm>_negtail.parquet`  — the registered negative-tail SELECTION RULE
  (`ssim2 < 0`) applied to the arm's own safesyn table, capped at the registered
  probe size (2,000) by a fixed-seed draw so every arm gets the SAME rows.
  `ssim2_gpu` is written because A8r's reachability guard reads the
  INSTRUMENT's truth, not the scorer's.
* `<arm>_identity.parquet` — the 400 self-pairs, with the `entry` label column
  the probe loader expects.

The row SET is chosen once, on the base arm, and reused verbatim: an arm probe
whose membership depended on the arm would confound "which rows" with "what the
form does to them".
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "canonical_corpus"))
from pack_eval372_root import read_fresh_csv, sha256  # noqa: E402

BASE_ARM = "ssim2"
NEGTAIL_N = 2000
SEED = 20260905


def main() -> int:
    root = sys.argv[1] if len(sys.argv) > 1 else "/mnt/v/output/zensim/rev2-2026-09-05/r6"
    arms = (os.environ.get("R6_ARMS") or "ssim2 c1 lorentz clamp").split()
    out = os.path.join(root, "instruments")
    os.makedirs(out, exist_ok=True)

    base = read_fresh_csv(os.path.join(root, "tables", BASE_ARM, "safesyn.csv"))
    y = np.asarray(base.column("human_score").to_pylist(), dtype=np.float64)
    neg = np.nonzero(y < 0.0)[0]
    rng = np.random.default_rng(SEED)
    if len(neg) > NEGTAIL_N:
        neg = np.sort(rng.choice(neg, NEGTAIL_N, replace=False))
    print(f"negtail rows: {len(neg)} of {len(y)} safesyn rows have ssim2 < 0 "
          f"(min {y.min():.4f})")

    for arm in arms:
        t = os.path.join(root, "tables", arm)
        lad = read_fresh_csv(os.path.join(t, "ladder.csv"))
        p = os.path.join(out, f"{arm}_ladder.parquet")
        pq.write_table(lad, p, compression="zstd")
        print(f"{arm}/ladder    rows={lad.num_rows:6d} sha256 {sha256(p)[:16]}…")

        ss = read_fresh_csv(os.path.join(t, "safesyn.csv")).take(pa.array(neg))
        cols = {n: ss.column(n) for n in ss.schema.names}
        cols["ssim2_gpu"] = ss.column("human_score")
        cols["entry"] = pa.array([f"neg{i:05d}" for i in range(ss.num_rows)])
        p = os.path.join(out, f"{arm}_negtail.parquet")
        pq.write_table(pa.table(cols), p, compression="zstd")
        print(f"{arm}/negtail   rows={ss.num_rows:6d} sha256 {sha256(p)[:16]}…")

        idt = read_fresh_csv(os.path.join(t, "identity.csv"))
        cols = {n: idt.column(n) for n in idt.schema.names}
        cols["entry"] = pa.array([f"id{i:04d}" for i in range(idt.num_rows)])
        p = os.path.join(out, f"{arm}_identity.parquet")
        pq.write_table(pa.table(cols), p, compression="zstd")
        print(f"{arm}/identity  rows={idt.num_rows:6d} sha256 {sha256(p)[:16]}…")
    return 0


if __name__ == "__main__":
    sys.exit(main())
