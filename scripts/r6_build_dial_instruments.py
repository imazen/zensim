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
LADDER_GRID = ("/mnt/v/output/zensim/ladder-2026-09-05/instruments/"
               "dial_grid_372col_ladder.parquet")
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
        # The dial grid's CELL IDENTITY (image_id / codec / q / codec_param /
        # param_kind) is a property of the instrument, not of the arm, so it is
        # carried over from the registered ladder grid verbatim and only the
        # features are replaced.
        #
        # The join is BY `row_id`, not positional. The ladder pairs TSV carries
        # `row_id` = the grid's row index, and it is NOT in ascending order:
        # the pairs list is grouped by IMAGE while the grid is grouped by CODEC,
        # so they agree for the first 51 rows and then diverge. A positional
        # join was tried first and its gate caught this — 9,287 of 9,593 rows
        # would have been labelled with the wrong cell. Keeping a real gate on
        # a join you believe is trivially correct is what made that a five-minute
        # fix instead of a silently wrong dial panel.
        #
        # The gate is now STRONGER than the positional one it replaces: every
        # row must agree with `grid[row_id]` on BOTH `image_id` AND
        # `codec_param`, so a permuted, truncated or mis-keyed pairs list all
        # fail loudly.
        lad = read_fresh_csv(os.path.join(t, "ladder.csv"))
        ref = pq.read_table(LADDER_GRID)
        if lad.num_rows != ref.num_rows:
            raise SystemExit(f"{arm}/ladder: {lad.num_rows} extracted rows vs "
                             f"{ref.num_rows} in the registered grid")
        rid = [int(x) for x in lad.column("row_id").to_pylist()]
        if sorted(rid) != list(range(ref.num_rows)):
            raise SystemExit(f"{arm}/ladder: row_id is not a permutation of "
                             f"0..{ref.num_rows - 1}")
        gimg = ref.column("image_id").to_pylist()
        gparam = [float(x) for x in ref.column("codec_param").to_pylist()]
        mimg = [os.path.splitext(x)[0] for x in lad.column("ref_basename").to_pylist()]
        mparam = [float(x) for x in lad.column("human_score").to_pylist()]
        bad = [i for i, r in enumerate(rid)
               if gimg[r] != mimg[i] or abs(gparam[r] - mparam[i]) > 1e-9]
        if bad:
            i = bad[0]
            raise SystemExit(
                f"{arm}/ladder: grid[row_id] disagrees on {len(bad)} rows "
                f"(first at {i}: row_id {rid[i]} -> grid "
                f"({gimg[rid[i]]!r}, {gparam[rid[i]]}) vs extracted "
                f"({mimg[i]!r}, {mparam[i]}))")
        # scatter: extracted row i carries the features of grid row rid[i]
        order = [0] * ref.num_rows
        for i, r in enumerate(rid):
            order[r] = i
        lad = lad.take(pa.array(order))
        cols = {n: ref.column(n) for n in ("image_id", "codec", "q",
                                          "codec_param", "param_kind")}
        for i in range(372):
            cols[f"f{i}"] = lad.column(f"f{i}")
        p = os.path.join(out, f"{arm}_ladder.parquet")
        pq.write_table(pa.table(cols), p, compression="zstd")
        print(f"{arm}/ladder    rows={ref.num_rows:6d} sha256 {sha256(p)[:16]}… "
              f"(cell identity from the registered grid, features from the arm)")

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
