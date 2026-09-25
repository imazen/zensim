#!/usr/bin/env python3
"""Independent NumPy mirror of the restored-cut families `mapdev` (A1) and
`z1max` (B2), from XYB plane dumps written by the private
`restore_cuts_instrument` test (see `zensim/src/feature_v2/restore_cuts.rs`).

Usage: numpy_mirror.py <dump_dir> [--wrong-control]

The dump directory holds `index.tsv` (case, side, scale, channel, width,
height, file), the raw little-endian f32 planes, and `features.csv` (one row
per case, columns `mapdev0..59` and `z1max0..227`). Nothing here reads the
Rust maps: the V1 blur (reflect-101 box, radius 5, /11), the Rev3 direct SSIM
dissimilarity, art/det, the squared-error and HF maps, the per-map standard
deviations and the ungated 5x5 block-max pooling are recomputed in float64
from the planes. The kernel itself accumulates in f32, so the comparison is a
relative tolerance (`--tol`, default 2e-4), not bit equality.

`--wrong-control` perturbs the definitions (z1max block MAX -> block MEAN; gmsnative
stabilisers x16) and REQUIRES both comparisons to fail, proving the comparator
can reject. gmsnative uses the independent C8 reference (`gms_cell`, copied from
scripts/gmsbank/numpy_reference.py with its precision boundary) at tolerance 1e-9.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

# gmsbank_constants.rs (frozen TRAIN literals), X and B gradient stabilisers.
GMSBANK_X_C = [6.341568337543687e-05, 0.0002536627335017475, 0.00101465093400699,
               0.00405860373602796, 0.01623441494411184]
GMSBANK_B_C = [8.239404165903782e-05, 0.00032957616663615127, 0.001318304666544605,
               0.00527321866617842, 0.02109287466471368]

R = 5
DIAM = 2 * R + 1
C2 = 0.0009  # ssim_form::C2 (f32 literal 0.0009)


def box(a):
    """Reflect-101 box mean, radius 5, over both axes (H then V)."""
    a = a.astype(np.float64)
    for axis in (1, 0):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (R, R)
        p = np.pad(a, pad, mode="reflect")
        c = np.cumsum(p, axis=axis)
        zero_shape = list(c.shape)
        zero_shape[axis] = 1
        c = np.concatenate([np.zeros(zero_shape), c], axis=axis)
        n = a.shape[axis]
        hi = np.take(c, np.arange(DIAM, DIAM + n), axis=axis)
        lo = np.take(c, np.arange(0, n), axis=axis)
        a = (hi - lo) / DIAM
    return a


def maps(s, d):
    s = s.astype(np.float64)
    d = d.astype(np.float64)
    mu1, mu2 = box(s), box(d)
    ssq = box(s * s + d * d)
    err = box((s - d) ** 2)
    mean_error2 = (mu1 - mu2) ** 2
    var_sum = np.maximum(ssq - mu1 * mu1 - mu2 * mu2, 0.0)
    err_var = np.maximum(err - mean_error2, 0.0)
    luma_loss = np.minimum(mean_error2, 1.0)  # Rev2/Rev3 `Clamp`
    sd = np.maximum((1.0 - luma_loss) * (err_var / (var_sum + C2)) + luma_loss, 0.0)
    diff1 = np.abs(s - mu1)
    diff2 = np.abs(d - mu2)
    ed = (1.0 + diff2) / (1.0 + diff1) - 1.0
    vs, vd = s - mu1, d - mu2
    return {
        "sd": sd,
        "art": np.maximum(ed, 0.0),
        "det": np.maximum(-ed, 0.0),
        "mse": (s - d) ** 2,
        "hfss": vs * vs,
        "hfsd": vd * vd,
        "hfas": diff1,
        "hfad": diff2,
    }


def gms_cell(ref, dist, constants):
    """C8 gradient cell (loss, gain, population std per stabiliser), the
    independent reference of scripts/gmsbank/numpy_reference.py::one, copied
    with its documented precision boundary: central differences, reflect-101 in
    y, clamped x borders, complete interior V8 chunks in f32, borders/tails f64."""
    def magnitude(a):
        left = np.concatenate((a[:, :1], a[:, :-1]), axis=1)
        right = np.concatenate((a[:, 1:], a[:, -1:]), axis=1)
        up = np.concatenate((a[1:2], a[:-1]), axis=0)
        down = np.concatenate((a[1:], a[-2:-1]), axis=0)
        dx = right.astype(np.float64) - left
        dy = down.astype(np.float64) - up
        mag = np.sqrt(dx * dx + dy * dy)
        chunk_end = 1 + ((a.shape[1] - 2) // 8) * 8
        if chunk_end > 1:
            dx32 = right[:, 1:chunk_end] - left[:, 1:chunk_end]
            dy32 = down[:, 1:chunk_end] - up[:, 1:chunk_end]
            mag[:, 1:chunk_end] = np.sqrt(dx32 * dx32 + dy32 * dy32)
        return mag

    mr, md = magnitude(ref), magnitude(dist)
    out = []
    for c in constants:
        delta = (mr - md) ** 2 / (mr * mr + md * md + c)
        out.extend((float(np.mean(np.where(md < mr, delta, 0))),
                    float(np.mean(np.where(md >= mr, delta, 0))),
                    float(np.std(delta, ddof=0))))
    return out


def mapdev(m):
    return [float(np.std(m[k], ddof=0)) for k in ("mse", "hfss", "hfsd", "hfas", "hfad")]


def block_max(a, mean=False):
    h, w = a.shape
    nby, nbx = h // 5, w // 5
    b = a[: nby * 5, : nbx * 5].reshape(nby, 5, nbx, 5)
    return (np.mean if mean else np.max)(b, axis=(1, 3)).ravel()


def z1max(m, wrong=False):
    """19 slots: basic 13 then peaks 6 (BASIC/PEAKS order), pooled over block
    maxima with n = surviving block count. The 0.0-initialised max of the
    record only matters for negative maps; none of the eight maps is negative
    here except `sd`, which is clamped at 0."""
    bm = {k: block_max(v, mean=wrong) for k, v in m.items()}
    n = bm["sd"].size
    if n == 0:
        return [0.0] * 19
    sd, art, det = bm["sd"], bm["art"], bm["det"]

    def p(x, k):
        return float(np.mean(x ** k))

    out = [
        p(sd, 1), p(sd, 4) ** 0.25, p(sd, 2) ** 0.5,
        p(art, 1), p(art, 4) ** 0.25, p(art, 2) ** 0.5,
        p(det, 1), p(det, 4) ** 0.25, p(det, 2) ** 0.5,
        float(np.mean(bm["mse"])),
    ]
    var_s, var_d = float(np.mean(bm["hfss"])), float(np.mean(bm["hfsd"]))
    mad_s, mad_d = float(np.mean(bm["hfas"])), float(np.mean(bm["hfad"]))
    return out, (var_s, var_d, mad_s, mad_d), (
        float(sd.max()), float(art.max()), float(det.max()),
        p(sd, 8) ** 0.125, p(art, 8) ** 0.125, p(det, 8) ** 0.125,
    )


def main():
    root = Path(sys.argv[1])
    wrong = "--wrong-control" in sys.argv
    tol = 2e-4
    if "--tol" in sys.argv:
        tol = float(sys.argv[sys.argv.index("--tol") + 1])
    planes = {}
    with (root / "index.tsv").open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            w, h = int(row["width"]), int(row["height"])
            a = np.fromfile(root / row["file"], dtype="<f4")
            assert a.size == w * h, row
            planes[(row["case"], int(row["side"]), int(row["scale"]), int(row["channel"]))] = a.reshape(h, w)
    with (root / "features.csv").open() as f:
        rows = {r["case"]: r for r in csv.DictReader(f)}
    worst = (0.0, None)
    worst_gms = (0.0, None)
    cells = 0
    for case, row in rows.items():
        for scale in range(4):
            for ch in range(3):
                s = planes[(case, 0, scale, ch)]
                d = planes[(case, 1, scale, ch)]
                m = maps(s, d)
                cell = scale * 3 + ch
                want = list(mapdev(m))
                got = [float(row[f"mapdev{cell * 5 + i}"]) for i in range(5)]
                z = z1max(m, wrong=wrong)
                if isinstance(z, tuple):
                    basic, hf, peaks = z
                    var_s, var_d, mad_s, mad_d = hf
                    # hf_gain_form: loss/gain ratios of the two HF energies
                    # (the shipped forms are the registry-declared finalizers;
                    # they are pinned by the Rust tests against `V1BasicSums`),
                    # so this mirror pools the RAW block-max means and compares
                    # the slots that are direct moments.
                    want_z = basic[:10] + list(peaks)
                    idx = list(range(10)) + list(range(13, 19))
                else:
                    want_z, idx = z, []
                got_z = [float(row[f"z1max{cell * 19 + i}"]) for i in idx]
                for g, w_ in list(zip(got, want)) + list(zip(got_z, want_z)):
                    rel = abs(g - w_) / max(abs(w_), 1e-9)
                    cells += 1
                    if rel > worst[0]:
                        worst = (rel, (case, scale, ch, g, w_))
                if scale == 0 and ch in (0, 2):
                    consts = GMSBANK_X_C if ch == 0 else GMSBANK_B_C
                    if wrong:
                        consts = [c * 16 for c in consts]
                    want_g = gms_cell(s, d, consts)
                    base = (0 if ch == 0 else 15)
                    got_g = [float(row[f"gmsnative{base + i}"]) for i in range(15)]
                    for g, w_ in zip(got_g, want_g):
                        rel = abs(g - w_) / max(abs(w_), 1e-12)
                        cells += 1
                        if rel > worst_gms[0]:
                            worst_gms = (rel, (case, ch, g, w_))
    tol_gms = 1e-9
    print(json.dumps({"cells": cells, "max_relative_error": worst[0], "worst": worst[1],
                      "gmsnative_max_relative_error": worst_gms[0], "gmsnative_worst": worst_gms[1],
                      "tolerance": tol, "gmsnative_tolerance": tol_gms,
                      "wrong_control": wrong}, sort_keys=True))
    if wrong:
        assert worst[0] > tol, "wrong z1max definition was NOT rejected"
        assert worst_gms[0] > tol_gms, "wrong gmsnative constants were NOT rejected"
    else:
        assert worst[0] <= tol, f"NumPy mirror mismatch: {worst}"
        assert worst_gms[0] <= tol_gms, f"gmsnative NumPy mismatch: {worst_gms}"


if __name__ == "__main__":
    main()
