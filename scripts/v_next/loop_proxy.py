#!/usr/bin/env python3
"""Loop-utility PROXY (design frozen in balance_campaign_2026-08-28.md).
Simulates the family k2/k3 bracketed q-bisection over the STORED 944 dial
grid, judged natively AND in translated ssim2/butter units (both
directions), with fairness floors (metric-steered oracle + k-inf ladder
quantization). Forward = predict_features_with_bake (owner); no encodes.

usage: loop_proxy.py name=bake.bin [name2=...] [--codecs jxl,avif] [--json OUT]
"""
import argparse, csv, json, os, struct, subprocess, sys, tempfile
from pathlib import Path
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
GRID = "/mnt/v/output/zensim/v2-eval-944-2026-08-01/dial_grid_944col_2026-08-01.parquet"
REFM = "/mnt/v/output/zensim/reports/refmetrics"
CODEC_MAP = {"jxl": "zenjxl", "avif": "zenavif", "jpeg": "zenjpeg", "webp": "zenwebp"}
TARGETS = [70.0, 80.0, 88.0]

def load_grid():
    t = pq.read_table(GRID)
    X = np.column_stack([np.asarray(t[f"f{i}"].to_pylist(), np.float32) for i in range(944)])
    # ladder PARAM: jxl grids by distance (param_kind='distance'; q col is display),
    # avif/jpeg/webp by q. Join + bisection both run on the param axis.
    # For distance, LOWER = higher quality — negate so "higher param = higher
    # quality" holds uniformly for the bisection bracket logic.
    kind = t["param_kind"].to_pylist()
    prm = [float(p) if k == "q" else -float(p) for p, k in zip(t["codec_param"].to_pylist(), kind)]
    return (X, np.array(t["image_id"].to_pylist()), np.array(t["codec"].to_pylist()),
            np.array(prm))

def load_peer(metric_file, col, neg=False):
    out = {}
    for r in csv.DictReader(open(f"{REFM}/{metric_file}"), delimiter="\t"):
        img = os.path.basename(r["ref_path"]).rsplit(".", 1)[0]
        knob = json.loads(r.get("knob_tuple_json") or "{}")
        p = -float(knob["distance"]) if "distance" in knob else float(r["q"])
        key = (img, r["codec"], round(p, 4))
        v = float(r[col])
        out[key] = -v if neg else v
    return out

def forward(bake, X):
    with tempfile.NamedTemporaryFile(suffix=".wire", delete=False) as f:
        f.write(struct.pack("<II", X.shape[1], X.shape[0])); f.write(X.astype("<f4").tobytes())
        wire = f.name
    try:
        r = subprocess.run([str(REPO / "target/release/predict_features_with_bake"),
                            "--bake", bake, "--features-file", wire],
                           capture_output=True, text=True, check=True)
    finally:
        os.unlink(wire)
    return np.array([float(v) for v in r.stdout.split()])

def _pava(y, w):
    """pool-adjacent-violators: weighted isotonic means (optimal monotone
    L2 fit on paired data). Returns fitted values, same length."""
    y = list(map(float, y)); w = list(map(float, w))
    vals, wts, cnt = [], [], []
    for yi, wi in zip(y, w):
        vals.append(yi); wts.append(wi); cnt.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            wv = wts[-2] + wts[-1]
            vals[-2] = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / wv
            wts[-2] = wv; cnt[-2] += cnt[-1]
            vals.pop(); wts.pop(); cnt.pop()
    out = []
    for v, c in zip(vals, cnt):
        out.extend([v] * c)
    return np.array(out)

def qmap(x, y, imgs=None, nb=None):
    """OPTIMAL-CLASS monotone translation x->y (upgraded 2026-08-28 after the
    user's mapping-optimality challenge): weighted PAVA isotonic regression on
    the paired cells (weights = 1/n_cells(image) so oversampled ladders don't
    dominate), evaluated by interpolation over the isotonic knots with LINEAR
    TAIL EXTRAPOLATION from the outer 10% of knots (the old flat clamp made
    near-lossless targets untranslatable above the top anchor)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    if imgs is not None:
        imgs = np.asarray(imgs)
        cnt = {im: (imgs == im).sum() for im in set(imgs.tolist())}
        w = np.array([1.0 / cnt[im] for im in imgs])
    else:
        w = np.ones(len(x))
    o = np.argsort(x, kind="stable")
    xs, ys, ws = x[o], y[o], w[o]
    fit = _pava(ys, ws)
    # unique-x knots (mean fit per x)
    ux, inv = np.unique(np.round(xs, 6), return_inverse=True)
    ky = np.array([fit[inv == i].mean() for i in range(len(ux))])
    ky = np.maximum.accumulate(ky)
    n = len(ux); k10 = max(2, n // 10)
    lo_s = (ky[k10] - ky[0]) / (ux[k10] - ux[0] + 1e-12)
    hi_s = (ky[-1] - ky[-1 - k10]) / (ux[-1] - ux[-1 - k10] + 1e-12)
    def f(v):
        v = np.asarray(v, float)
        out = np.interp(v, ux, ky)
        out = np.where(v < ux[0], ky[0] + (v - ux[0]) * max(lo_s, 0.0), out)
        out = np.where(v > ux[-1], ky[-1] + (v - ux[-1]) * max(hi_s, 0.0), out)
        return out if out.ndim else float(out)
    return lambda v: float(f(v)) if np.isscalar(v) or getattr(v, "ndim", 0) == 0 else f(v)

# Family-loop emulation (v2 after the registered validation gate FAILED on
# blind bisection): SEEDED start + SECANT steps — the shape the real loops
# use (jxl: fixed d-seed + secant on (param, score); censuses land 0.3-1.5).
SEEDS = {"jxl": -2.5, "avif": 78.0, "jpeg": 78.0, "webp": 78.0}

def secant_ladder(qs, scores, target, k, seed_param):
    seen = []
    def eval_at(p):
        cands = [j for j in range(len(qs)) if j not in [s[0] for s in seen]]
        if not cands: return None
        i = min(cands, key=lambda j: abs(qs[j] - p))
        seen.append((i, scores[i]))
        return i
    eval_at(seed_param)
    for _ in range(k - 1):
        if len(seen) == 1:
            i0, s0 = seen[-1]
            # first correction: local slope from neighbors, else fixed step
            j = min(max(i0, 1), len(qs) - 2)
            slope = (scores[j + 1] - scores[j - 1]) / (qs[j + 1] - qs[j - 1] + 1e-9)
            step = (target - s0) / slope if abs(slope) > 1e-6 else (qs[-1] - qs[0]) * 0.25 * np.sign(target - s0)
            nxt = qs[i0] + step
        else:
            (i1, s1), (i2, s2) = seen[-2], seen[-1]
            if abs(s2 - s1) < 1e-9:
                nxt = qs[i2] + (qs[-1] - qs[0]) * 0.1 * np.sign(target - s2)
            else:
                nxt = qs[i2] + (target - s2) * (qs[i2] - qs[i1]) / (s2 - s1)
        nxt = min(max(nxt, qs[0]), qs[-1])
        if eval_at(nxt) is None: break
    best = min(seen, key=lambda t2: abs(t2[1] - target))
    return best[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bakes", nargs="+", help="name=path.bin")
    ap.add_argument("--codecs", default="jxl,avif")
    ap.add_argument("--json")
    a = ap.parse_args()
    X, img, cod, qv = load_grid()
    peers = {"ssim2": load_peer("dialgrid_ssim2_gpu.tsv", "ssim2_gpu"),
             "butter": load_peer("dialgrid_butteraugli_gpu.tsv",
                                 [c for c in csv.DictReader(open(f"{REFM}/dialgrid_butteraugli_gpu.tsv"), delimiter="\t").fieldnames if "max" in c or "butter" in c][0], neg=True)}
    results = {}
    for spec in a.bakes:
        name, path = spec.split("=", 1)
        pred = forward(path, X)
        res = {}
        for codec in a.codecs.split(","):
            zc = CODEC_MAP[codec]
            mask = cod == codec
            # ladders: image -> sorted (q, idx)
            ladders = defaultdict(list)
            for i in np.where(mask)[0]:
                key = (img[i], zc, round(float(qv[i]), 4))
                if all(key in peers[m] for m in peers):
                    ladders[img[i]].append(i)
            ladders = {im: sorted(ix, key=lambda i: qv[i]) for im, ix in ladders.items() if len(ix) >= 25}
            # translation maps on the joined cells of THIS codec
            all_ix = [i for ix in ladders.values() for i in ix]
            b = pred[all_ix]
            mv = {m: np.array([peers[m][(img[i], zc, qv[i])] for i in all_ix]) for m in peers}
            im_of = np.array([img[i] for i in all_ix])
            fwd = {m: qmap(b, mv[m], im_of) for m in peers}   # bake -> metric (isotonic)
            rev = {m: qmap(mv[m], b, im_of) for m in peers}   # metric -> bake (isotonic)
            # round-trip drift diagnostic (directions are fit independently;
            # drift quantifies their non-inverseness in bake units)
            drift = {m: float(np.median(np.abs(np.asarray(rev[m](fwd[m](b))) - b))) for m in peers}
            cells = {}
            for k in (2, 3):
                nat, fwd_err, rev_err, oracle, qfloor = [], {m: [] for m in peers}, {m: [] for m in peers}, {m: [] for m in peers}, []
                for im, ix in ladders.items():
                    qs = qv[ix]; bs = pred[ix]
                    ms = {m: np.array([peers[m][(im, zc, round(float(q), 4))] for q in qs]) for m in peers}
                    for t in TARGETS:
                        li = secant_ladder(qs, bs, t, k, SEEDS[codec])
                        nat.append(abs(bs[li] - t))
                        qfloor.append(np.min(np.abs(bs - t)))
                        for m in peers:
                            tm = float(fwd[m](t))
                            fwd_err[m].append(abs(ms[m][li] - tm))
                            # oracle: steer ON the metric itself toward tm
                            oi = secant_ladder(qs, ms[m], tm, k, SEEDS[codec])
                            oracle[m].append(abs(ms[m][oi] - tm))
                            # reverse: metric-unit target -> bake units -> steer -> judge in metric
                            tb = float(rev[m](tm))
                            ri = secant_ladder(qs, bs, tb, k, SEEDS[codec])
                            rev_err[m].append(abs(ms[m][ri] - tm))
                nat = np.array(nat)
                cells[f"k{k}"] = {
                    "native_med": float(np.median(nat)), "native_pm2": int((nat <= 2).sum()), "n": len(nat),
                    "kinf_floor_med": float(np.median(qfloor)),
                    **{f"{m}_fwd_med": float(np.median(fwd_err[m])) for m in peers},
                    **{f"{m}_rev_med": float(np.median(rev_err[m])) for m in peers},
                    **{f"{m}_oracle_med": float(np.median(oracle[m])) for m in peers}}
            res[codec] = {"n_ladders": len(ladders), "cells": cells,
                          "roundtrip_drift_bake_units": {m: round(drift[m], 3) for m in peers}}
        results[name] = res
    for name, res in results.items():
        for codec, r in res.items():
            for kk, c in r["cells"].items():
                print(f"{name:<14}{codec:<6}{kk}  native {c['native_med']:.3f} (±2 {c['native_pm2']}/{c['n']}; k∞floor {c['kinf_floor_med']:.3f}) | "
                      f"ssim2 fwd {c['ssim2_fwd_med']:.2f} rev {c['ssim2_rev_med']:.2f} (oracle {c['ssim2_oracle_med']:.2f}) | "
                      f"butter fwd {c['butter_fwd_med']:.3f} rev {c['butter_rev_med']:.3f} (oracle {c['butter_oracle_med']:.3f})")
    if a.json:
        json.dump(results, open(a.json, "w"), indent=1)
        print("wrote", a.json)

if __name__ == "__main__":
    main()
