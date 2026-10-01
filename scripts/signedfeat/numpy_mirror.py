#!/usr/bin/env python3
"""Independent float64 mirror of the SIGNEDFEAT families `texgain` (f1825..1836) and `satsign`
(f1837..1852) from XYB plane dumps written by the private `signedfeat_plane_dump` instrument
(see `zensim/src/feature_v2/restore_cuts.rs`).

Usage: numpy_mirror.py <dump_dir> [--wrong-control]

Nothing here reads the Rust maps. The v1 band blur (reflect-101 box, radius 5, /11) is the restore-cuts
mirror's `box`, and `texgain` is recomputed in float64 from the planes:
    texgain = mean_px max(0, |d - mu2| - |s - mu1|) / (|d - mu2| + |s - mu1| + C_HF),   C_HF = 1e-4
`satsign` from the X and B planes (centring X-0.42, B-0.55 with the f32 literals' f64 values):
    m = sqrt(Xc^2/cx + Bc^2/cb), (cx, cb) = GMSBANK_CS_C[2]
    sat_gain/loss = mean_px max(0, +-(m_d - m_r)) / (m_d + m_r + 1);  gsat_* = same form on mean_px m.
The kernel computes the band maps in f32 (texgain) and the saturation in f64 from f32 planes, so the
comparison is a relative tolerance. `--wrong-control` perturbs the definitions (C_HF x100 and the
saturation stabiliser 1 -> 0.01) and REQUIRES the comparison to fail.
"""
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "restore_cuts"))
from numpy_mirror import box  # noqa: E402  (the restore-cuts mirror's reflect-101 box blur)

C_HF = 1e-4
CX, CB = 0.0014875777316638384, 0.7739603828506174
K_B0 = float(np.float32(0.0037930734))
X0 = float(np.float32(0.42))
B0 = float(np.float32(0.55)) + K_B0 ** (1.0 / 3.0)  # neutral-axis centring: -cbrt(K_B0) is the gray B offset


def texgain(s, d, wrong=False):
    s, d = s.astype(np.float64), d.astype(np.float64)
    a, b = np.abs(s - box(s)), np.abs(d - box(d))
    c = C_HF * (100 if wrong else 1)
    return float(np.mean(np.maximum(b - a, 0.0) / (b + a + c)))


def satsign(xs, bs, xd, bd, wrong=False):
    c = 0.01 if wrong else 1.0

    def m(x, b):
        xc, bc = x.astype(np.float64) - X0, b.astype(np.float64) - B0
        return np.sqrt(xc * xc / CX + bc * bc / CB)

    mr, md = m(xs, bs), m(xd, bd)
    den = md + mr + c
    gr, gd = mr.mean(), md.mean()
    gden = gd + gr + c
    return [float(np.mean(np.maximum(md - mr, 0) / den)), float(np.mean(np.maximum(mr - md, 0) / den)),
            float(max(gd - gr, 0) / gden), float(max(gr - gd, 0) / gden)]


def main():
    root = Path(sys.argv[1])
    wrong = "--wrong-control" in sys.argv
    planes = {}
    with (root / "index.tsv").open() as f:
        for row in csv.DictReader(f, delimiter="\t"):
            w, h = int(row["width"]), int(row["height"])
            a = np.fromfile(root / row["file"], dtype="<f4")
            assert a.size == w * h, row
            planes[(row["case"], int(row["side"]), int(row["scale"]), int(row["channel"]))] = a.reshape(h, w)
    with (root / "features.csv").open() as f:
        rows = {r["case"]: r for r in csv.DictReader(f)}
    errs = {"texgain": [], "satsign": []}
    worst = {"texgain": (0.0, None), "satsign": (0.0, None)}
    for case, row in rows.items():
        for scale in range(4):
            for ch in range(3):
                want = texgain(planes[(case, 0, scale, ch)], planes[(case, 1, scale, ch)], wrong)
                got = float(row[f"texgain{scale * 3 + ch}"])
                rel = abs(got - want) / max(abs(want), 1e-12)
                errs["texgain"].append(rel)
                if rel > worst["texgain"][0]:
                    worst["texgain"] = (rel, (case, scale, ch, got, want))
            want = satsign(planes[(case, 0, scale, 0)], planes[(case, 0, scale, 2)],
                           planes[(case, 1, scale, 0)], planes[(case, 1, scale, 2)], wrong)
            for k in range(4):
                got = float(row[f"satsign{scale * 4 + k}"])
                rel = abs(got - want[k]) / max(abs(want[k]), 1e-12)
                errs["satsign"].append(rel)
                if rel > worst["satsign"][0]:
                    worst["satsign"] = (rel, (case, scale, k, got, want[k]))
    tol = float(sys.argv[sys.argv.index("--tol") + 1]) if "--tol" in sys.argv else 1e-4
    ok = True
    for fam, v in errs.items():
        v = np.array(v)
        print(f"{fam}: cells={v.size} max_rel={v.max():.3e} median_rel={np.median(v):.3e} worst={worst[fam][1]}")
        ok &= bool(v.max() <= tol)
    if wrong:
        print("WRONG-CONTROL", "REJECTED (expected)" if not ok else "NOT REJECTED (comparator is vacuous)")
        sys.exit(0 if not ok else 1)
    print("MIRROR", "OK" if ok else f"FAIL (tol {tol:g})")
    sys.exit(0 if ok else 1)


main()
