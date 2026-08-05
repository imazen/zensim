#!/usr/bin/env python3
"""Emit top-K input-index files from a `bake_contrib` TSV.

The ONE ranking step of appendix J (`benchmarks/sota944_campaign_2026-08-03.md`).
It writes index files for `zensim_mlp_train --keep-features` and a ranked
summary TSV. It computes no statistics: the contribution numbers are produced
by `bake_contrib` (the owner) and this script only sorts and slices them.

    topk_from_contrib.py --contrib <bake_contrib.tsv> --out-dir <dir> \
        --k 64,128,256,512,667 [--column mean_abs]

Ranking (frozen in J.2): `mean_abs` descending, ties by ascending index.
"""

import argparse
import csv
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contrib", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--k", required=True, help="comma-separated K values")
    ap.add_argument("--column", default="mean_abs")
    args = ap.parse_args()

    with open(args.contrib, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows:
        print(f"FATAL: {args.contrib} has no rows", file=sys.stderr)
        return 1
    for r in rows:
        r["_idx"] = int(r["idx"])
        r["_v"] = float(r[args.column])
    # Frozen order: value descending, index ascending on ties.
    rows.sort(key=lambda r: (-r["_v"], r["_idx"]))
    total = sum(r["_v"] for r in rows) or 1.0

    os.makedirs(args.out_dir, exist_ok=True)
    ranked = os.path.join(args.out_dir, "ranked.tsv")
    with open(ranked, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["rank", "idx", "family", args.column, "cum_share", "dead"])
        cum = 0.0
        for i, r in enumerate(rows):
            cum += r["_v"]
            w.writerow([i, r["_idx"], r["family"], f"{r['_v']:.6e}",
                        f"{cum / total:.6f}", r["dead"]])
    print(f"wrote {ranked} ({len(rows)} inputs, ranked by {args.column})")

    for k in (int(x) for x in args.k.split(",")):
        if k > len(rows):
            print(f"FATAL: K={k} > {len(rows)} inputs", file=sys.stderr)
            return 1
        keep = sorted(r["_idx"] for r in rows[:k])
        path = os.path.join(args.out_dir, f"top{k}.idx")
        share = sum(r["_v"] for r in rows[:k]) / total
        with open(path, "w") as fh:
            fh.write(f"# top-{k} of {len(rows)} by {args.column} (desc, ties by idx asc)\n")
            fh.write(f"# source: {os.path.abspath(args.contrib)}\n")
            fh.write(f"# contribution share captured: {share:.4f}\n")
            fh.write(",".join(str(i) for i in keep) + "\n")
        print(f"wrote {path} (share {share:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
