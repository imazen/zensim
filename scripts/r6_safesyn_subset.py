#!/usr/bin/env python3
"""R6: stratified safesyn training slice, as a `pairs-tsv` the extractor decodes.

WHY a new list rather than `--corpus safesyn`: `load_safesyn` reads the CSV's
`decoded_path`, and the file that carries `iwssim`
(`/mnt/v/zen/zensim-training/2026-05-16/safesyn_with_iwssim.csv`) points that
column at the `q<X>.png` DECODE CACHE, which was deleted 2026-06-22 (0/3000
sampled rows survive — `docs/DATASET_HISTORY.md` §3.32). The row-identical
`synthetic-v2/training_safe_synthetic.csv` points at the BITSTREAMS, which are
present, so the pixels are reachable — through the extractor's own zen-decode
owner, at a decoder era recorded in the manifest.

Stratification is (codec family x quality), 6 x 16 = 96 cells, because the
sweep discipline forbids a grid that is denser at high q than low q and safesyn
carries exactly 16 quality points per codec from 5 to 100. Quota is spread
evenly, then the shortfall of any small cell is redistributed over the cells
that still have rows, so the requested total is met without over-weighting the
large families.

Target column is `human_score` = `cpu_ssimulacra2`, falling back to
`gpu_ssimulacra2` when empty — byte-for-byte the rule
`extract_features_372col::load_safesyn` applies, so this list feeds the same
target the ADD156 / Profile-D lineage was fit on. (safesyn's target is an ssim2
SELF-TARGET; CID22 human MOS is never a training target here or anywhere.)

Deterministic: rows are ordered by their position in the CSV and drawn with a
fixed-seed shuffle per cell, so the SAME row set is used for every arm — which
is what makes the arms' fits comparable at all.
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import random
import sys

SRC = "/mnt/v/output/zensim/synthetic-v2/training_safe_synthetic.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=20260905)
    ap.add_argument("--out", required=True)
    ap.add_argument("--anchor-n", type=int, default=0,
                    help="also write <out>.anchor.tsv with this many rows, "
                         "disjoint from the training list")
    a = ap.parse_args()

    cells: dict[tuple[str, int], list[dict]] = collections.defaultdict(list)
    with open(a.src) as f:
        for i, r in enumerate(csv.DictReader(f)):
            r["_i"] = i
            cells[(r["codec"], int(float(r["quality"])))].append(r)

    keys = sorted(cells)
    want = a.n + a.anchor_n
    # even quota, then redistribute the shortfall of small cells
    quota = {k: want // len(keys) for k in keys}
    for _ in range(4):
        short = sum(max(0, quota[k] - len(cells[k])) for k in keys)
        if short == 0:
            break
        room = [k for k in keys if len(cells[k]) > quota[k]]
        if not room:
            break
        for j, k in enumerate(room):
            quota[k] += short // len(room) + (1 if j < short % len(room) else 0)
        for k in keys:
            quota[k] = min(quota[k], len(cells[k]))

    picked: list[dict] = []
    for k in keys:
        rows = sorted(cells[k], key=lambda r: r["_i"])
        rng = random.Random(f"{a.seed}:{k[0]}:{k[1]}")
        rng.shuffle(rows)
        picked.extend(rows[: quota[k]])
    picked.sort(key=lambda r: r["_i"])

    # anchor rows are taken from the TAIL of the deterministic order so the
    # training list and the anchor list are disjoint by construction.
    anchor = picked[len(picked) - a.anchor_n:] if a.anchor_n else []
    train = picked[: len(picked) - a.anchor_n] if a.anchor_n else picked

    def write(path: str, rows: list[dict]) -> None:
        with open(path, "w", newline="") as f:
            w = csv.writer(f, delimiter="\t", lineterminator="\n")
            w.writerow(["ref_path", "dist_path", "human_score", "codec", "quality"])
            for r in rows:
                cpu, gpu = r["cpu_ssimulacra2"], r["gpu_ssimulacra2"]
                y = cpu if cpu not in ("", None) else gpu
                w.writerow([r["source_path"], r["decoded_path"], y,
                            r["codec"], int(float(r["quality"]))])
        h = hashlib.sha256(open(path, "rb").read()).hexdigest()
        cc = collections.Counter(r["codec"] for r in rows)
        print(f"{path}: {len(rows)} rows sha256 {h}")
        for k in sorted(cc):
            print(f"    {k:28s} {cc[k]}")

    write(a.out, train)
    if anchor:
        write(a.out + ".anchor.tsv", anchor)
    return 0


if __name__ == "__main__":
    sys.exit(main())
