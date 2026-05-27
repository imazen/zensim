#!/usr/bin/env python3
"""Falsification-first test of dial-fix candidate #3 (v47 methodology doc):
refit v47-strict's output calibration spline on a REAL-corpus prediction
distribution mapping pred -> [0,100] dial, instead of the degenerate
auto-spline (2 knots -> negative band -> G1=0.00).

Fit corpus: multiband_anchor_dial100 (2000 rows, per-row target_score in
[0,100], CID22-val-clean training anchor — V39's spline anchor).
Verify corpus: CID22 val (49-ref MCOS) — rank must stay ~0.855
(spline is monotone => rank-invariant) AND the calibrated range must pass
G1 (p5<=25 AND p95>=85).

Emits the spline payload (u32 n_knots + n*(f32 x, f32 y)) if the test
passes, so the production re-bake can inject it.

Usage: python3 scripts/v_next/fit_dial_spline_v47.py <bake.bin> <out_payload.bin>
"""
import sys, subprocess, struct, os, tempfile
import numpy as np
import pyarrow.parquet as pq
from scipy.interpolate import PchipInterpolator
from scipy.stats import spearmanr

ANCHOR = "/mnt/v/zen/zensim-training/canonical-2026-05-21/train/multiband_anchor_dial100.parquet"
CID22 = "/mnt/v/zen/zensim-training/2026-05-15-full-features/cid22_features_372col_2026-05-15.parquet"
PRED = "./target/release/predict_features_with_bake"


def raw_preds(bake, parquet):
    t = pq.read_table(parquet)
    fcols = [c for c in t.column_names if c.startswith('f') and c[1:].isdigit()]
    fcols.sort(key=lambda c: int(c[1:]))
    n = t.num_rows
    feats = np.zeros((n, len(fcols)), dtype=np.float32)
    for i, c in enumerate(fcols):
        feats[:, i] = t.column(c).to_numpy().astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix='.bin', delete=False) as f:
        tmp = f.name
        f.write(struct.pack('<II', len(fcols), n))
        f.write(feats.tobytes())
    r = subprocess.run([PRED, '--bake', bake, '--bake-post', 'raw', '--features-file', tmp],
                       capture_output=True, text=True, timeout=300)
    os.unlink(tmp)
    if r.returncode != 0:
        sys.exit(f"predict failed: {r.stderr[:500]}")
    preds = [float(x) for x in r.stdout.split() if x and not x.startswith('#')]
    return np.array(preds), t


def main():
    bake, out = sys.argv[1], sys.argv[2]

    # --- fit corpus ---
    ap, at = raw_preds(bake, ANCHOR)
    tgt = at.column('target_score').to_numpy().astype(float)
    print(f"anchor: {len(ap)} rows, raw pred range [{ap.min():.4f},{ap.max():.4f}] "
          f"(spread {ap.max()-ap.min():.4f}), target_score [{tgt.min():.1f},{tgt.max():.1f}]")
    print(f"  corr(raw_pred, target_score) = {np.corrcoef(ap, tgt)[0,1]:.4f}")

    # quantile-bin raw preds -> median target per bin (monotone knots)
    n_knots = 16
    qs = np.linspace(1, 99, n_knots)
    edges = np.percentile(ap, qs)
    kx, ky = [], []
    # below-range extrapolation knot
    lo = ap < edges[0]
    if lo.sum() >= 2:
        kx.append(float(np.median(ap[lo]))); ky.append(float(np.median(tgt[lo])))
    for i in range(len(edges) - 1):
        m = (ap >= edges[i]) & (ap < edges[i + 1])
        if m.sum() >= 2:
            kx.append(float(np.median(ap[m]))); ky.append(float(np.median(tgt[m])))
    hi = ap >= edges[-1]
    if hi.sum() >= 2:
        kx.append(float(np.median(ap[hi]))); ky.append(float(np.median(tgt[hi])))

    # enforce strictly increasing x AND y (monotone)
    cx, cy = [kx[0]], [ky[0]]
    for i in range(1, len(kx)):
        if kx[i] > cx[-1] + 1e-5 and ky[i] >= cy[-1]:
            cx.append(kx[i]); cy.append(ky[i])
    print(f"\nfit {len(cx)} monotone knots:")
    for x, y in zip(cx, cy):
        print(f"  pred={x:9.5f} -> dial={y:7.3f}")
    if len(cx) < 3:
        print("\n*** DEGENERATE: <3 usable knots — the pred band is too compressed. "
              "Candidate #3 FALSIFIED; sibling-ship v47-strict. ***")
        sys.exit(2)

    spline = PchipInterpolator(cx, cy, extrapolate=True)

    # --- verify on CID22 (rank-invariance + G1) ---
    cp, ct = raw_preds(bake, CID22)
    mcos = ct.column('human_score').to_numpy().astype(float) * 100.0
    raw_srocc = spearmanr(cp, mcos).statistic
    cal = spline(cp)
    cal_srocc = spearmanr(cal, mcos).statistic
    p = np.percentile(cal, [5, 25, 50, 75, 95])
    g1 = (p[0] <= 25) and (p[4] >= 85)
    print(f"\nCID22 (n={len(cp)}):")
    print(f"  raw pred SROCC vs MCOS      = {raw_srocc:.4f}")
    print(f"  CALIBRATED SROCC vs MCOS    = {cal_srocc:.4f}  (must equal raw — monotone)")
    print(f"  calibrated dial percentiles = p5={p[0]:.1f} p25={p[1]:.1f} p50={p[2]:.1f} "
          f"p75={p[3]:.1f} p95={p[4]:.1f}")
    print(f"  G1 dynamic range (p5<=25 & p95>=85): {'PASS' if g1 else 'FAIL'}")

    payload = struct.pack('<I', len(cx))
    for x, y in zip(cx, cy):
        payload += struct.pack('<ff', float(x), float(y))
    with open(out, 'wb') as f:
        f.write(payload)
    print(f"\nwrote spline payload ({len(payload)} bytes) -> {out}")
    print("VERDICT:", "CANDIDATE #3 VIABLE — proceed to re-bake" if g1 and abs(cal_srocc - raw_srocc) < 0.01
          else "MIXED — rank ok but G1 fail; inspect" if abs(cal_srocc - raw_srocc) < 0.01
          else "BROKEN — rank changed (spline non-monotone?)")


if __name__ == '__main__':
    main()
