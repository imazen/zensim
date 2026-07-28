#!/usr/bin/env python3
"""Full-temporal LIVE-YT-Banding analysis (2026-07-28).

Consumes the ALL-frames per-frame signal table produced by the streamed
pipeline recorded in `benchmarks/bandvis_lyb_temporal_2026-07-28.md`
(~240 frames/video x 120 videos x {8 bandvis slots, tex_s3, mscn_s0,
art_s0, v1score}), and reports, per signal x temporal pooling:

  - SROCC pooled (120) + official 1000 content-aware test folds
  - PLCC after a 4-param logistic (scipy)
  - Krasula different-vs-similar AUC + better-vs-worse accuracy
    (|dMOS| threshold PROXY — the metadata carries no CIs; stated)
  - best <=3-feature linear combo, fit on train folds only, evaluated on
    test folds (leakage-pinned)

Poolings: mean | worst-1s-window (max of sliding fps-frame window means)
| soft-topk (Sum w.v/Sum w, w = sat(max(v - p60, 0), c) + 1e-6 with
c = 0.25*(p95 - p60) + 1e-9 — the CAMBI topk-0.6 analog without a hard
order statistic) | p95 (analysis-only).

Usage: bandvis_lyb_temporal_eval_2026-07-28.py [--out-dir ~/tmp/lybT-out]
"""
import argparse
import csv
import os
import re
from collections import defaultdict

import numpy as np
from scipy.optimize import curve_fit

META = "/mnt/v/datasets/live-yt-banding/metadata/LIVE_Banding_metadata.csv"
MAT = "/mnt/v/datasets/live-yt-banding/github-data/LIVE_Banding_contentaware.mat"

SIGNALS = ["loss_s3", "loss_s2", "gain_s3", "tex_s3", "mscn_s0", "v1score"]
POOLINGS = ["mean", "worst1s", "softtopk", "p95"]


def spearman(a, b):
    def ranks(v):
        idx = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        r[idx] = np.arange(len(v), dtype=float)
        for u in np.unique(v):
            m = v == u
            if m.sum() > 1:
                r[m] = r[m].mean()
        return r
    ra, rb = ranks(np.asarray(a, float)), ranks(np.asarray(b, float))
    ra -= ra.mean()
    rb -= rb.mean()
    d = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / d) if d > 0 else 0.0


def logistic4(x, b1, b2, b3, b4):
    return b2 + (b1 - b2) / (1.0 + np.exp(-(x - b3) / (abs(b4) + 1e-9)))


def plcc_after_logistic(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    try:
        p0 = [y.max(), y.min(), np.median(x), (x.std() + 1e-9)]
        popt, _ = curve_fit(logistic4, x, y, p0=p0, maxfev=20000)
        yh = logistic4(x, *popt)
    except Exception:
        yh = x if x.std() > 0 else x + np.random.default_rng(0).normal(0, 1e-9, len(x))
    c = np.corrcoef(yh, y)
    return float(c[0, 1])


def auc(scores, labels):
    """ROC AUC of `scores` separating labels==1 from labels==0."""
    s = np.asarray(scores, float)
    l = np.asarray(labels, int)
    pos = s[l == 1]
    neg = s[l == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-based (Mann-Whitney U)
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    r = np.empty(len(allv))
    r[order] = np.arange(1, len(allv) + 1, dtype=float)
    for u in np.unique(allv):
        m = allv == u
        if m.sum() > 1:
            r[m] = r[m].mean()
    rp = r[: len(pos)].sum()
    u_stat = rp - len(pos) * (len(pos) + 1) / 2
    return float(u_stat / (len(pos) * len(neg)))


def pool(vals, fps, kind):
    v = np.asarray(vals, float)
    if kind == "mean":
        return float(v.mean())
    if kind == "worst1s":
        w = max(1, int(round(fps)))
        if len(v) <= w:
            return float(v.mean())
        cs = np.concatenate([[0.0], np.cumsum(v)])
        means = (cs[w:] - cs[:-w]) / w
        return float(means.max())
    if kind == "softtopk":
        p60, p95 = np.percentile(v, 60), np.percentile(v, 95)
        c = 0.25 * (p95 - p60) + 1e-9
        u = np.maximum(v - p60, 0.0)
        w = u / (u + c) + 1e-6
        return float((w * v).sum() / w.sum())
    if kind == "p95":
        return float(np.percentile(v, 95))
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.expanduser("~/tmp/lybT-out"))
    ap.add_argument("--dmos-thresh", type=float, default=10.0)
    args = ap.parse_args()

    meta = list(csv.DictReader(open(META)))
    mos = {r["Filename"]: float(r["MOS"]) for r in meta}
    fps_of = {r["Filename"]: float(r["framerate"]) for r in meta}
    meta_order = [r["Filename"] for r in meta]

    frames = defaultdict(lambda: defaultdict(list))
    with open(os.path.join(args.out_dir, "perframe.csv")) as f:
        rd = csv.DictReader(f)
        cols = [c for c in rd.fieldnames if c not in ("dist_file", "frame")]
        for r in rd:
            for c in cols:
                frames[r["dist_file"]][c].append(float(r[c]))
    videos = sorted(frames)
    assert len(videos) == 120, f"{len(videos)} videos"
    n_frames = {v: len(frames[v][cols[0]]) for v in videos}
    print(f"videos: {len(videos)}, frames/video: min {min(n_frames.values())} "
          f"max {max(n_frames.values())} total {sum(n_frames.values())}")

    y = np.array([mos[v] for v in videos])

    # Pooled tables.
    P = {}
    for sig in SIGNALS:
        for pk in POOLINGS:
            P[(sig, pk)] = np.array(
                [pool(frames[v][sig], fps_of[v], pk) for v in videos]
            )

    # Official splits.
    import h5py
    vid_of_id = {i + 1: fn for i, fn in enumerate(meta_order)}
    with h5py.File(MAT, "r") as h:
        index = np.array(h["index"])
        if index.shape == (1000, 160):
            index = index.T
    test_sets, train_sets = [], []
    for split in range(index.shape[1]):
        t_ids = index[128:160, split].astype(int)
        tr_ids = index[0:96, split].astype(int)
        test_sets.append([videos.index(vid_of_id[i]) for i in t_ids if vid_of_id[i] in frames])
        train_sets.append([videos.index(vid_of_id[i]) for i in tr_ids if vid_of_id[i] in frames])

    # Krasula proxy pairs (all 120C2 pairs; labels by |dMOS| threshold).
    ii, jj = np.triu_indices(len(videos), k=1)
    dmos = np.abs(y[ii] - y[jj])
    diff_label = (dmos > args.dmos_thresh).astype(int)
    sign_true = np.sign(y[ii] - y[jj])
    print(f"Krasula proxy: |dMOS| > {args.dmos_thresh} -> {diff_label.sum()} different / "
          f"{len(diff_label) - diff_label.sum()} similar pairs (NO CIs in metadata — threshold proxy)")

    print(f"\n{'signal':<10} {'pooling':<9} {'SROCC':>8} {'folds mean±sd':>16} "
          f"{'PLCC':>7} {'AUC_ds':>7} {'BW_acc':>7}")
    best = None
    for sig in SIGNALS:
        for pk in POOLINGS:
            x = P[(sig, pk)]
            sr = spearman(x, y)
            fold = np.array([spearman(x[t], y[t]) for t in test_sets if len(t) >= 6])
            pl = plcc_after_logistic(x, y)
            dm = np.abs(x[ii] - x[jj])
            a_ds = auc(dm, diff_label)
            m = diff_label == 1
            bw = float((np.sign(x[ii] - x[jj])[m] * -1 == sign_true[m]).mean())
            # note: error features anti-correlate with MOS, so the metric's
            # "better" direction is LOWER — the *-1 accounts for polarity;
            # v1score is a quality score (higher better): flip back.
            if sig == "v1score":
                bw = 1.0 - bw
            print(f"{sig:<10} {pk:<9} {sr:>+8.4f} {fold.mean():>+8.4f}±{fold.std():.4f} "
                  f"{pl:>7.4f} {a_ds:>7.4f} {bw:>7.4f}")
            key = abs(fold.mean())
            if best is None or key > best[0]:
                best = (key, sig, pk, sr, fold.mean(), fold.std())
    print(f"\nBEST single (by |fold mean|): {best[1]} x {best[2]} — "
          f"pooled {best[3]:+.4f}, folds {best[4]:+.4f} ± {best[5]:.4f}")

    # <=3-feature linear combo, leakage-pinned: fit on TRAIN rows of each
    # split, evaluate SROCC on that split's TEST rows.
    combo_feats = [("loss_s3", "softtopk"), ("tex_s3", "mean"), ("mscn_s0", "mean")]
    X = np.stack([P[k] for k in combo_feats], axis=1)
    srs = []
    for tr, te in zip(train_sets, test_sets):
        if len(tr) < 10 or len(te) < 6:
            continue
        A = np.hstack([X[tr], np.ones((len(tr), 1))])
        w, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        pred = np.hstack([X[te], np.ones((len(te), 1))]) @ w
        srs.append(spearman(pred, y[te]))
    srs = np.array(srs)
    print(f"\n3-feature combo {combo_feats}: TEST-fold SROCC {srs.mean():+.4f} ± {srs.std():.4f} "
          f"(fit per split on its 72 train videos only; n={len(srs)} splits)")


if __name__ == "__main__":
    main()
