#!/usr/bin/env python3
"""x_lyb_read.py — APPENDIX X (X-L1): the LYB model-level read.

Forward a bake over the 960-pair LIVE-YT-Banding feature table of its OWN
GAIN definition (`bake_dial_refit predict` upstream), aggregate per distorted
video (mean over the 8 sampled frames), and report per-video SROCC vs the
official MOS through the canonical stats owner (`zen_stats.panel` — no
hand-rolled stats). Optionally a paired bootstrap of SROCC(a) − SROCC(b) over
videos via `panel --batch` indexed mode (the caller keeps the RNG).

    x_lyb_read.py --pred A.tsv [--pred-b B.tsv] [--b 2000] [--seed 20260806]

Inputs: pred TSVs from `bake_dial_refit predict` (row_idx\tpred, positional,
960 rows aligned with the P1.5 `pairs_manifest.csv`).
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.zen_stats import panel, panel_batch_indexed  # noqa: E402

MANIFEST = "/home/lilith/tmp/bandvis-dst/lyb-off/pairs_manifest.csv"
META = "/mnt/v/datasets/live-yt-banding/metadata/LIVE_Banding_metadata.csv"


def load_preds(p: str) -> list[float]:
    out = []
    for line in open(p):
        line = line.strip()
        if line and not line.startswith("row"):
            out.append(float(line.split("\t")[1]))
    if len(out) != 960:
        sys.exit(f"{p}: expected 960 rows, got {len(out)}")
    return out


def per_video(preds: list[float]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    man = list(csv.DictReader(open(MANIFEST)))
    mos = {r["Filename"]: float(r["MOS"]) for r in csv.DictReader(open(META))}
    agg: dict[str, list[float]] = {}
    for r, p in zip(man, preds):
        agg.setdefault(r["dist_file"], []).append(p)
    keys = list(agg)
    xs = np.array([np.mean(agg[k]) for k in keys])
    ys = np.array([mos[k] for k in keys])
    return xs, ys, keys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--pred-b")
    ap.add_argument("--b", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260806)
    a = ap.parse_args()

    xa, y, _ = per_video(load_preds(a.pred))
    st = panel(xa, y)
    print(f"{a.pred}: n_videos={int(st['n'])} srocc={st['srocc']:.4f} plcc={st['plcc']:.4f}")
    if not a.pred_b:
        return
    xb, y2, _ = per_video(load_preds(a.pred_b))
    assert np.array_equal(y, y2)
    stb = panel(xb, y)
    print(f"{a.pred_b}: srocc={stb['srocc']:.4f} plcc={stb['plcc']:.4f}")
    # paired bootstrap over videos, same index sets both arms
    rng = np.random.default_rng(a.seed)
    n = len(y)
    idx = [rng.integers(0, n, n).tolist() for _ in range(a.b)]
    bases = {"xa": xa, "xb": xb, "y": y}
    jobs_a = [(f"a{i}", "xa", "y", ix) for i, ix in enumerate(idx)]
    jobs_b = [(f"b{i}", "xb", "y", ix) for i, ix in enumerate(idx)]
    ra = panel_batch_indexed(bases, jobs_a, stats="srocc")
    rb = panel_batch_indexed(bases, jobs_b, stats="srocc")
    d = np.array([r2["srocc"] - r1["srocc"] for r1, r2 in zip(ra, rb)])
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(
        f"delta(b-a) srocc: median {np.median(d):+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
        f"P(d>0)={float((d > 0).mean()):.3f}  (B={a.b}, seed {a.seed})"
    )


if __name__ == "__main__":
    main()
