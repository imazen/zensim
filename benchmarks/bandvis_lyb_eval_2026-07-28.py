#!/usr/bin/env python3
"""BANDVIS external validation on LIVE-YT-Banding (2026-07-28).

Consumes the frame-sampled foldapp2 (944) extraction produced by the
pipeline recorded in `benchmarks/bandvis_lyb_validation_2026-07-28.md`
(8 timestamp-matched frames/video, FR pairs ref-vs-cq within content),
plus the official metadata CSV and the 1000 content-aware splits.

Outputs the correlation tables the benchmark doc commits:
  a. BANDVIS_GAIN per scale / best scale / mean-of-scales vs MOS
  b. metric-score baseline (separate driver run, v1stream column)
  c. best existing single feature (v2 art/det/mse + append MSCN/contrast)
  d. mean±sd SROCC over the official 1000 content-aware test folds
  e. false-positive check: high-BANDVIS / high-MOS videos

Usage: bandvis_lyb_eval_2026-07-28.py [--out-dir ~/tmp/lyb-out]
"""
import argparse
import csv
import os
import re
from collections import defaultdict

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

META = "/mnt/v/datasets/live-yt-banding/metadata/LIVE_Banding_metadata.csv"
MAT = "/mnt/v/datasets/live-yt-banding/github-data/LIVE_Banding_contentaware.mat"

# 944 layout (zensim feature_v2): v1 0..372 | v2 372..720 | append 720..924
# | append2 924..944 (scale*5 + local, Y-only).
V2_BASE, APP_BASE, APP2_BASE = 372, 720, 924
V2_PER_CH, APP_PER_CH = 29, 17


def v2_idx(scale, ch, local):
    return V2_BASE + scale * 3 * V2_PER_CH + ch * V2_PER_CH + local


def app_idx(scale, ch, local):
    return APP_BASE + scale * 3 * APP_PER_CH + ch * APP_PER_CH + local


def app2_idx(scale, local):
    return APP2_BASE + scale * 5 + local


def spearman(a, b):
    ar = np.argsort(np.argsort(a)).astype(float)
    br = np.argsort(np.argsort(b)).astype(float)
    # average ties
    for v in np.unique(a):
        m = a == v
        if m.sum() > 1:
            ar[m] = ar[m].mean()
    for v in np.unique(b):
        m = b == v
        if m.sum() > 1:
            br[m] = br[m].mean()
    ar -= ar.mean()
    br -= br.mean()
    d = np.sqrt((ar**2).sum() * (br**2).sum())
    return float((ar * br).sum() / d) if d > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=os.path.expanduser("~/tmp/lyb-out"))
    args = ap.parse_args()

    mos = {}
    for r in csv.DictReader(open(META)):
        mos[r["Filename"]] = float(r["MOS"])

    # manifest: row_id -> (content, dist_file, frame)
    man = {}
    for r in csv.DictReader(open(os.path.join(args.out_dir, "pairs_manifest.csv"))):
        man[int(r["row_id"])] = (r["content"], r["dist_file"], int(r["frame_idx"]))

    feats_by_video = defaultdict(list)
    with open(os.path.join(args.out_dir, "lyb_foldapp2_master.csv")) as f:
        rd = csv.reader(f)
        header = next(rd)
        nfeat = len(header) - 2
        assert nfeat == 944, f"expected 944 features, got {nfeat}"
        for row in rd:
            rid = int(float(row[1]))
            content, dist_file, _ = man[rid]
            feats_by_video[dist_file].append(np.array([float(v) for v in row[2:]]))

    videos = sorted(feats_by_video)
    assert len(videos) == 120, len(videos)
    F = np.stack([np.mean(feats_by_video[v], axis=0) for v in videos])
    y = np.array([mos[v] for v in videos])
    contents = [re.sub(r"_cq\d+\.webm$", "", v) for v in videos]

    print(f"videos: {len(videos)}, frames/video: {len(feats_by_video[videos[0]])}")

    # (a) BANDVIS
    print("\n== (a) BANDVIS_GAIN vs MOS (SROCC; MOS higher=better, expect negative) ==")
    bv = {}
    for s in range(4):
        g = F[:, app2_idx(s, 0)]
        bv[f"gain_s{s}"] = spearman(g, y)
        print(f"  gain scale {s}: {bv[f'gain_s{s}']:+.4f}  (nonzero on {np.count_nonzero(g)}/120)")
    gm = F[:, [app2_idx(s, 0) for s in range(4)]].mean(axis=1)
    bv["gain_mean"] = spearman(gm, y)
    print(f"  gain mean-of-scales: {bv['gain_mean']:+.4f}")
    for s in range(4):
        l = F[:, app2_idx(s, 1)]
        print(f"  loss scale {s}: {spearman(l, y):+.4f}")
    best_s = max(range(4), key=lambda s: abs(bv[f"gain_s{s}"]))
    best_gain = F[:, app2_idx(best_s, 0)]
    print(f"  BEST: gain scale {best_s} |SROCC| = {abs(bv[f'gain_s{best_s}']):.4f}")

    # (b) metric-score baseline, if present
    score_csv = os.path.join(args.out_dir, "lyb_v1stream_master.csv")
    if os.path.exists(score_csv):
        sc = defaultdict(list)
        with open(score_csv) as f:
            rd = csv.reader(f)
            next(rd)
            for row in rd:
                rid = int(float(row[1]))
                _, dist_file, _ = man[rid]
                sc[dist_file].append(float(row[2]))
        s_arr = np.array([np.mean(sc[v]) for v in videos])
        print(f"\n== (b) zensim codec_target v1-stream score baseline: SROCC {spearman(s_arr, y):+.4f} ==")

    # (c) existing single-feature battery (Y channel = ch 1)
    print("\n== (c) existing single features (Y) vs MOS ==")
    battery = {}
    for s in range(4):
        for name, local, blk in [
            ("art", 3, "v2"), ("det", 4, "v2"), ("mse", 5, "v2"),
            ("ssim_mean", 0, "v2"), ("banding_masked?ringing", 26, "v2"),
            ("mscn", 5, "app"), ("mscn2", 6, "app"),
            ("cgain", 7, "app"), ("closs", 8, "app"), ("tex", 9, "app"),
        ]:
            idx = v2_idx(s, 1, local) if blk == "v2" else app_idx(s, 1, local)
            battery[f"{name}_s{s}"] = spearman(F[:, idx], y)
    for probe in ["mse_s0", "mse_s3", "ssim_mean_s0", "det_s3", "closs_s3"]:
        print(f"  [sign-check] {probe}: {battery[probe]:+.4f}")
    top = sorted(battery.items(), key=lambda kv: -abs(kv[1]))[:8]
    for k, v in top:
        print(f"  {k}: {v:+.4f}")
    best_existing = top[0]
    print(f"  BEST EXISTING: {best_existing[0]} |SROCC| = {abs(best_existing[1]):.4f}")
    print(f"  ACCEPTANCE: BANDVIS best |{abs(bv[f'gain_s{best_s}']):.4f}| "
          f"{'BEATS' if abs(bv[f'gain_s{best_s}']) > abs(best_existing[1]) else 'DOES NOT BEAT'} "
          f"best existing |{abs(best_existing[1]):.4f}|")

    # (d) official 1000 content-aware splits — decoded per
    # data_info_maker.m: index[:, split] = 160 one-based video ids
    # (CSV-row order, source blocks of 4: ref, cq, cq, cq); positions
    # 129..160 are the TEST fold (8 sources x 4 videos).
    if h5py is not None and os.path.exists(MAT):
        meta_order = [r["Filename"] for r in csv.DictReader(open(META))]
        vid_of_id = {i + 1: fn for i, fn in enumerate(meta_order)}
        with h5py.File(MAT, "r") as h:
            index = np.array(h["index"])  # (160, 1000) in h5py layout
            if index.shape == (1000, 160):
                index = index.T
        nm, loc = best_existing[0].rsplit("_s", 1)
        loc_map = {"art": (3, "v2"), "det": (4, "v2"), "mse": (5, "v2"),
                   "ssim_mean": (0, "v2"), "banding_masked?ringing": (26, "v2"),
                   "mscn": (5, "app"), "mscn2": (6, "app"),
                   "cgain": (7, "app"), "closs": (8, "app"), "tex": (9, "app")}
        l, blk = loc_map[nm]
        ex_col = v2_idx(int(loc), 1, l) if blk == "v2" else app_idx(int(loc), 1, l)
        best_loss_s = max(range(4), key=lambda sc: abs(spearman(F[:, app2_idx(sc, 1)], y)))
        best_loss = F[:, app2_idx(best_loss_s, 1)]
        sr_bv, sr_ex, sr_lo = [], [], []
        for split in range(index.shape[1]):
            test_ids = index[128:160, split].astype(int)
            vis = [videos.index(vid_of_id[i]) for i in test_ids
                   if vid_of_id[i] in feats_by_video]  # distorted only (refs absent)
            if len(vis) < 6:
                continue
            vv = np.array(vis)
            sr_bv.append(spearman(best_gain[vv], y[vv]))
            sr_ex.append(spearman(F[vv, ex_col], y[vv]))
            sr_lo.append(spearman(best_loss[vv], y[vv]))
        sr_bv, sr_ex, sr_lo = np.array(sr_bv), np.array(sr_ex), np.array(sr_lo)
        print(f"\n== (d) official content-aware TEST folds (n={len(sr_bv)} splits, 24 distorted/fold) ==")
        print(f"  BANDVIS gain s{best_s}: {sr_bv.mean():+.4f} ± {sr_bv.std():.4f}")
        print(f"  BANDVIS loss s{best_loss_s}: {sr_lo.mean():+.4f} ± {sr_lo.std():.4f}")
        print(f"  best-existing ({best_existing[0]}): {sr_ex.mean():+.4f} ± {sr_ex.std():.4f}")
        print(f"  ACCEPTANCE (pair-wise): BANDVIS best-of-pair "
              f"{max(abs(sr_lo.mean()), abs(sr_bv.mean())):.4f} vs existing {abs(sr_ex.mean()):.4f}")

    # (d2) within-content CQ-ladder direction (content-confound-free):
    # mean SROCC over the 40 contents' 3-video ladders.
    print("\n== (d2) within-content 3-point ladder SROCC (mean over 40 contents) ==")
    from collections import defaultdict as dd2
    by_content = dd2(list)
    for i, v in enumerate(videos):
        by_content[contents[i]].append(i)
    def within(col):
        vals = []
        for c, idxs in by_content.items():
            if len(idxs) == 3:
                vals.append(spearman(col[np.array(idxs)], y[np.array(idxs)]))
        return np.mean(vals), np.std(vals)
    for label, col in [
        (f"bandvis_gain_s{best_s}", best_gain),
        ("bandvis_loss_s3", F[:, app2_idx(3, 1)]),
        ("mscn_s0", F[:, app_idx(0, 1, 5)]),
        ("mse_s0", F[:, v2_idx(0, 1, 5)]),
    ]:
        m, sd = within(col)
        print(f"  {label}: {m:+.4f} ± {sd:.4f}")

    # (e) false positives: high BANDVIS, high MOS
    print("\n== (e) limitation check: top BANDVIS(best scale) among HIGH-MOS (>60) videos ==")
    order = np.argsort(-best_gain)
    shown = 0
    for i in order:
        if y[i] > 60:
            print(f"  {videos[i]}: gain {best_gain[i]:.4f}, MOS {y[i]:.1f}")
            shown += 1
            if shown >= 6:
                break
    if shown == 0:
        print("  (no high-MOS video in the top BANDVIS ranks)")

    # persist per-video table for the doc
    with open(os.path.join(args.out_dir, "lyb_per_video.csv"), "w") as f:
        w = csv.writer(f)
        w.writerow(["video", "mos"] + [f"bv_gain_s{s}" for s in range(4)] + [f"bv_loss_s{s}" for s in range(4)])
        for i, v in enumerate(videos):
            w.writerow([v, y[i]]
                       + [F[i, app2_idx(s, 0)] for s in range(4)]
                       + [F[i, app2_idx(s, 1)] for s in range(4)])
    print("\nper-video table -> lyb_per_video.csv")


if __name__ == "__main__":
    main()
