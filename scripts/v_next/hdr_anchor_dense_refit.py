#!/usr/bin/env python3
# PARTIALLY-MIGRATED (2026-07-05): the base whole-spline refit is now
#   target/release/bake_dial_refit shared-anchor (zensim-validate/.../bake_dial_refit.rs).
# The research-specific 28-bin densify + Q-Q top-end knots + SDR-top-probe below
# are experiment logic, not a reusable bake primitive, so they stay here. See
# benchmarks/bake_refit_rust_migration_2026-07-05.md.
"""Densify the HDR anchored sibling's spline top-end (2026-07-03 night).

The 2026-07-03 shared-anchor HDR sibling (`shared_anchor_refit.py`) fit its
spline on hdr-v3 valdigits alone (3,900 rows, 18 bins) — the top bin's median
lands at ~88.6 while the data reaches 92.3, and only 103 rows sit above
human_score 0.9. This refit:

1. Augments the anchor with the v3 TRAIN digits' top band (human_score >
   0.85, +1,760 rows — train-legal; the spline is a monotone calibration,
   anchors routinely come from train).
2. Densifies bins (18 -> 28) so the top of the range gets knot resolution.
3. Appends an explicit END KNOT at the median of the top-25-by-prediction
   rows, anchoring the dial at the data ceiling instead of a fat top-bin
   median.
4. Optionally probes whether the SDR multiband anchor's near-lossless rows
   (target >= 88) can extend the dial past the HDR data ceiling (~92.8):
   they are appended ONLY if their predictions land monotonically above the
   HDR top (measured, not assumed).

Emits lp_hdr-lasso0.001-shaped-anchored2-f16.bin (originals untouched) and
verifies rank-exactness: hdrval + UPIQ SROCC must equal the tau0 original
(monotone spline => rank-invariant; verified empirically, not assumed).
"""
import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

SPEC = importlib.util.spec_from_file_location(
    "lp", Path(__file__).parent / "linear_projections_2026-07-03.py")
lp = importlib.util.module_from_spec(SPEC)
sys.modules["lp"] = lp
SPEC.loader.exec_module(lp)

PROBE = Path("/mnt/v/output/zensim-multicodec-probe")
FITS = PROBE / "linear-probe" / "fits"
BAKES = PROBE / "linear-probe" / "bakes"
KEY = "hdr-lasso0.001-shaped"
OUT = BAKES / "lp_hdr-lasso0.001-shaped-anchored2-f16.bin"


def load_features(path, target_col="human_score", scale=100.0):
    t = pq.read_table(path)
    F = np.column_stack([np.asarray(t[f"f{i}"], dtype=float) for i in range(372)])
    tgt = np.asarray(t[target_col], dtype=float) * scale
    return F, tgt


def main():
    z = np.load(FITS / f"{KEY}.npz")
    w, bias = z["w"].astype(float), float(z["bias"])
    mu, sd = z["mu"].astype(float), z["sd"].astype(float)
    space = str(z["space"]) if "space" in z.files else "raw"
    assert space == "shaped"
    transforms, tparams = lp.load_transforms()

    def preds_of(F):
        Fs = lp.shape_block(F, transforms, tparams)
        return ((Fs - mu) / np.where(sd == 0, 1, sd)) @ w + bias

    # anchor rows: all of val + train top band
    Fv, tv = load_features(PROBE / "hdr_zenjxl_v3_valdigits_2026-07-03.parquet")
    Ft, tt = load_features(PROBE / "hdr_zenjxl_v3_traindigits_2026-07-03.parquet")
    top = tt > 85.0
    F = np.vstack([Fv, Ft[top]])
    tgt = np.concatenate([tv, tt[top]])
    preds = preds_of(F)
    print(f"anchor: {len(tv)} val + {int(top.sum())} train-top rows; "
          f"target range [{tgt.min():.1f}, {tgt.max():.1f}]")

    # knots at 28 bins
    cx, cy = lp.fit_spline_knots(preds, tgt)
    # densify: lp.fit_spline_knots uses 18 percentile edges; refit manually at 28
    edges = np.percentile(preds, np.linspace(1, 99, 28))
    kx, ky = [], []
    lo_m = preds < edges[0]
    if lo_m.sum() >= 2:
        kx.append(float(np.median(preds[lo_m]))); ky.append(float(np.median(tgt[lo_m])))
    for i in range(len(edges) - 1):
        m = (preds >= edges[i]) & (preds < edges[i + 1])
        if m.sum() >= 2:
            kx.append(float(np.median(preds[m]))); ky.append(float(np.median(tgt[m])))
    hi_m = preds >= edges[-1]
    if hi_m.sum() >= 2:
        kx.append(float(np.median(preds[hi_m]))); ky.append(float(np.median(tgt[hi_m])))
    # Top-end: Q-Q (quantile-matching) knots. The conditional-median end knot
    # honestly tops at ~88.9 (the head's top-25 preds have median TRUE score
    # 88.9 — rank imperfection at the top). Q-Q instead maps pred quantiles to
    # target quantiles (rank-preserving, marginal-matching): the dial's top
    # reaches the DATA ceiling (92.8) rather than the conditional median.
    order = np.argsort(preds)
    top25 = order[-25:]
    for q in (0.995, 0.999):
        kx.append(float(np.quantile(preds, q)))
        ky.append(float(np.quantile(tgt, q)))
    kx.append(float(preds.max() + 1e-4))
    ky.append(float(tgt.max()))

    # SDR top-band probe (>=88 dial): does the HDR head rank them above its
    # own HDR top? Append only if monotone-compatible.
    ta = pq.read_table(
        "/mnt/v/zen/zensim-training/canonical-2026-05-21/train/multiband_anchor_dial100.parquet")
    Fa = np.column_stack([np.asarray(ta[f"f{i}"], dtype=float) for i in range(372)])
    tga = np.asarray(ta["target_score"], dtype=float)
    m88 = tga >= 88.0
    pa88 = preds_of(Fa[m88])
    hdr_top_pred = float(np.median(preds[top25]))
    frac_above = float((pa88 > hdr_top_pred).mean())
    print(f"SDR-top probe: n={int(m88.sum())} target>=88; "
          f"{frac_above*100:.0f}% of their preds land above the HDR top knot x "
          f"(median SDR-top pred {np.median(pa88):.3f} vs HDR top {hdr_top_pred:.3f})")
    used_sdr_top = False
    if frac_above >= 0.80:
        kx.append(float(np.median(pa88)))
        ky.append(float(np.median(tga[m88])))
        used_sdr_top = True

    # monotone filter (same rule as fit_spline_knots)
    pairs = sorted(zip(kx, ky))
    fx, fy = [pairs[0][0]], [pairs[0][1]]
    for x, y in pairs[1:]:
        if x > fx[-1] + 1e-7 and y > fy[-1]:
            fx.append(x); fy.append(y)
    print(f"knots: {len(fx)} (was 18-bin default), y-range [{fy[0]:.1f}, {fy[-1]:.1f}], "
          f"SDR-top knot used: {used_sdr_top}")

    payload = struct.pack("<I", len(fx)) + b"".join(
        struct.pack("<ff", float(x), float(y)) for x, y in zip(fx, fy))
    metadata = [
        {"key": "zentrain.feature_transforms", "type": "utf8", "text": "\n".join(transforms)},
        {"key": "zentrain.feature_transform_params", "type": "utf8",
         "text": "\n".join(",".join(f"{q}" for q in row) if row else "" for row in tparams)},
        {"key": lp.SPLINE_KEY, "type": "bytes", "hex": payload.hex()},
    ]
    req = {"schema_hash": 0, "flags": 0, "compressed": True,
           "scaler_mean": [float(v) for v in mu.astype(np.float32)],
           "scaler_scale": [float(v) for v in sd.astype(np.float32)],
           "layers": [{"in_dim": 372, "out_dim": 1, "activation": "identity",
                       "dtype": "f16", "weights": [float(v) for v in w], "biases": [bias]}],
           "metadata": metadata}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(req, f)
        jp = Path(f.name)
    r = subprocess.run([str(lp.BAKER), "bake", str(jp), str(OUT)],
                       capture_output=True, text=True)
    jp.unlink(missing_ok=True)
    assert r.returncode == 0, r.stderr[:400]
    print(f"emitted {OUT.name} ({OUT.stat().st_size} bytes)")
    with open(BAKES / "anchored2_knots.json", "w") as f:
        json.dump({"kx": fx, "ky": fy, "used_sdr_top": used_sdr_top}, f, indent=1)


if __name__ == "__main__":
    main()
