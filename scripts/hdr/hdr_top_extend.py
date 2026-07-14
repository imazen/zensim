#!/usr/bin/env python3
"""Generalized shaped-bake dial top-extension (rank-invariant).

The Rust `bake_dial_refit extend-top` handles only identity/winsor_p99
bakes; a SHAPED bake (winsor_p99 + quantile_bins + yeo_johnson + ... mixed
transforms, e.g. the HDR co-cal candidate) trips its f64 fit-forward guard.
This is the shaped-bake counterpart: it reads ALL of the input bake's
metadata (feature transforms + params + the existing output spline) VERBATIM
from `zenpredict inspect`, extends ONLY the spline's TOP above its top knot
by the concave saturation the anchor's near-lossless band shows, and re-emits
with the transforms/weights/scaler byte-preserved. Bottom + in-distribution
knots are kept exactly, so rank (SROCC / ramps) is unchanged — a monotone
spline never reorders — while the OOD-high dial can finally reach ~100.

Math is identical to `dense_dial_refit_b.py` (the SDR/B version) — robust
`log(100 − y) ≈ logA − k·raw` on the anchor's `y > band_min` rows, then
`score(r) = 100 − (100 − y0)·exp(−k·(r − x0))` from the top knot — but the
transform metadata is copied from the input, not hardcoded to winsor.

  usage: hdr_top_extend.py IN.bin OUT.bin
         [--anchor .../val/anchor.npz] [--space shaped] [--band-min 70]

The anchor npz must carry `<space>` (post-transform features, matching the
bake's forward) and `y` (the target dial), exactly as `bake_candidate`
consumes it.
"""
import argparse
import importlib.util
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "lp", REPO / "scripts/v_next/linear_projections_2026-07-03.py")
lp = importlib.util.module_from_spec(SPEC)
sys.modules["lp"] = lp
_argv = sys.argv
sys.argv = ["lp"]
SPEC.loader.exec_module(lp)
sys.argv = _argv

BAKER = lp.BAKER
SPLINE_KEY = lp.SPLINE_KEY
DEFAULT_ANCHOR = Path("/mnt/v/output/zensim-multicodec-probe/linear-probe/val/anchor.npz")


def parse_spline(hexstr):
    b = bytes.fromhex(hexstr)
    nk = struct.unpack("<I", b[:4])[0]
    kx, ky = [], []
    for i in range(nk):
        x, y = struct.unpack("<ff", b[4 + 8 * i:12 + 8 * i])
        kx.append(float(x))
        ky.append(float(y))
    return kx, ky


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inp")
    ap.add_argument("out")
    ap.add_argument("--anchor", default=str(DEFAULT_ANCHOR))
    ap.add_argument("--space", default="shaped", choices=["shaped", "raw"])
    ap.add_argument("--band-min", type=float, default=70.0)
    ap.add_argument("--n-knots", type=int, default=12)
    ap.add_argument("--mode", default="saturation",
                    choices=["saturation", "target-top", "full-target"],
                    help="saturation = log-OLS concave extrapolation above the top knot "
                         "(reaches 100 but under-shoots a sparse top); target-top = replace the "
                         "top knots (ky>keep-below) with knots placed on the anchor's own high-y "
                         "rows binned by TARGET (regime-safe, no new data, reaches the anchor's "
                         "real top density — not 100 if the anchor is sparse above ~95); "
                         "full-target = discard ALL existing knots and rebuild the WHOLE spline "
                         "from target-binned anchor rows across [0,100] (fixes a sagging/inflated "
                         "MID, not just the top — the dial tracks the anchor's B-dial in every "
                         "zone; rank-invariant since the map stays monotone in raw)")
    ap.add_argument("--keep-below", type=float, default=72.0,
                    help="target-top: keep existing spline knots with y<=this verbatim (bottom+"
                         "mid); rebuild knots above it from target-binned anchor rows")
    a = ap.parse_args()

    ins = json.loads(subprocess.run(
        [str(BAKER), "inspect", a.inp, "--weights"],
        capture_output=True, text=True).stdout)
    mu = np.array([float(v) for v in ins["scaler_mean"]], float)
    sd = np.array([float(v) for v in ins["scaler_scale"]], float)
    layer = ins["layers"][0]
    w = np.array([float(v) for v in layer["weights"]], float)
    bias = float(layer["biases"][0])
    dtype = layer.get("dtype", "f16")
    meta = {m["key"]: m for m in ins["metadata"]}
    if SPLINE_KEY not in meta:
        raise SystemExit("input bake has no output_calibration_spline to extend")
    sp = meta[SPLINE_KEY]
    kx, ky = parse_spline(sp.get("value_hex") or sp.get("hex"))
    x0, y0 = kx[-1], ky[-1]
    print(f"input spline: {len(kx)} knots, domain [{kx[0]:.3f},{x0:.3f}] "
          f"y [{ky[0]:.1f},{y0:.1f}] (bottom + mid kept VERBATIM)")

    # raw preds on the anchor's post-transform features (matches the runtime
    # forward: the bake's transforms already produced `shaped`).
    z = np.load(a.anchor)
    Xa = z[a.space].astype(np.float64)
    y = z["y"].astype(np.float64)
    raw_a = (Xa - mu) / sd @ w + bias

    if a.mode == "saturation":
        band = y > a.band_min
        if int(band.sum()) < 10:
            raise SystemExit(f"only {int(band.sum())} anchor rows with y>{a.band_min}; "
                             "lower --band-min or use a top-anchored npz")
        A = np.vstack([np.ones(int(band.sum())), raw_a[band]]).T
        coef, *_ = np.linalg.lstsq(A, np.log(np.clip(100.0 - y[band], 1e-3, None)), rcond=None)
        k = float(-coef[1])
        if k <= 0:
            raise SystemExit(f"saturation fit gave non-decaying k={k}; anchor's y>band raw "
                             "does not increase with quality — top-extend not applicable")
        r_far = x0 + (-np.log(1e-4) / k)
        added = 0
        for r in np.linspace(x0 + (r_far - x0) / a.n_knots, r_far, a.n_knots):
            yv = 100.0 - (100.0 - y0) * np.exp(-k * (r - x0))
            if r > kx[-1] + 1e-7 and yv > ky[-1]:
                kx.append(float(r))
                ky.append(float(yv))
                added += 1
        print(f"  top extension (saturation): k={k:.3f} (n={int(band.sum())}); "
              f"{x0:.3f}->{r_far:.2f}, +{added} knots, y-top {ky[-1]:.2f}; final {len(kx)} knots")
    elif a.mode == "full-target":
        # full-target: throw away the input spline entirely and rebuild it from
        # target-binned anchor rows across the WHOLE range. Bin by the target
        # dial y in 5-pt zones; each knot sits at (median raw, median y) of its
        # zone, so the baked dial reproduces the anchor's B-dial per zone —
        # meanΔ≈0 everywhere, not just the endpoints. The map is monotone in
        # raw (knots strictly increasing in raw), so rank/ramps are unchanged.
        edges = list(np.arange(0.0, 100.0 + 1e-6, 5.0)) + [100.01]
        nk_kept = len(kx)
        kx, ky = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (y >= lo) & (y < hi)
            if int(m.sum()) < 5:
                continue
            kxn = float(np.median(raw_a[m]))
            kyn = float(np.median(y[m]))
            # enforce strictly-increasing raw (a monotone spline needs a strictly
            # increasing domain); the target medians are already increasing by
            # construction of the y-bins, but rank noise can invert raw across
            # two adjacent thin bins — skip a bin whose median raw regressed.
            if kx and kxn <= kx[-1] + 1e-6:
                continue
            kx.append(kxn)
            ky.append(kyn)
        if len(kx) < 4:
            raise SystemExit(f"full-target produced only {len(kx)} monotone knots; "
                             "anchor too sparse or raw too noisy — widen bins")
        added = len(kx)
        print(f"  full respline (full-target): discarded {nk_kept} input knots; "
              f"rebuilt {added} target-binned knots across [0,100]; "
              f"raw [{kx[0]:.3f},{kx[-1]:.3f}] y [{ky[0]:.1f},{ky[-1]:.1f}]")
    else:
        # target-top: keep the bottom+mid knots (ky<=keep_below) VERBATIM, then
        # place the top knots on the anchor's OWN high-y rows binned by target.
        # Uses existing data only — regime-safe — so it reaches the anchor's real
        # top density (not 100 if the anchor is sparse above ~95).
        keep = [(x, yv) for x, yv in zip(kx, ky) if yv <= a.keep_below]
        if not keep:
            keep = [(kx[0], ky[0])]
        kx, ky = [p[0] for p in keep], [p[1] for p in keep]
        edges = [a.keep_below, 78, 84, 89, 93, 96, 100.01]
        added = 0
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (y >= lo) & (y < hi)
            if int(m.sum()) >= 5:
                kxn = float(np.median(raw_a[m]))
                kyn = float(np.median(y[m]))
                if kxn > kx[-1] + 1e-6 and kyn > ky[-1] + 1e-6:
                    kx.append(kxn)
                    ky.append(kyn)
                    added += 1
        print(f"  top rebuild (target-top): kept {len(keep)} bottom/mid knots (y<={a.keep_below}); "
              f"+{added} target-binned top knots; final {len(kx)} knots, y-top {ky[-1]:.2f} "
              f"(n(y>=95)={int((y>=95).sum())} — the top-density ceiling)")

    payload = struct.pack("<I", len(kx)) + b"".join(
        struct.pack("<ff", float(x), float(y)) for x, y in zip(kx, ky))

    # rebuild metadata in the SAME order as the input, transforms VERBATIM.
    metadata = []
    for key in ("zentrain.feature_transforms", "zentrain.feature_transform_params"):
        if key in meta:
            m = meta[key]
            metadata.append({"key": key, "type": "utf8",
                             "text": m.get("value_text", m.get("text", ""))})
    metadata.append({"key": SPLINE_KEY, "type": "bytes", "hex": payload.hex()})
    req = {
        "schema_hash": 0, "flags": 0, "compressed": True,
        "scaler_mean": [float(v) for v in mu],
        "scaler_scale": [float(v) for v in sd],
        "layers": [{"in_dim": len(w), "out_dim": 1, "activation": "identity",
                    "dtype": dtype, "weights": [float(v) for v in w], "biases": [bias]}],
        "metadata": metadata,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(req, f)
        jp = Path(f.name)
    r = subprocess.run([str(BAKER), "bake", str(jp), a.out], capture_output=True, text=True)
    jp.unlink(missing_ok=True)
    if r.returncode != 0:
        raise SystemExit(f"bake failed: {r.stderr[:400]}")
    print(f"emitted {a.out} ({Path(a.out).stat().st_size} B)")


if __name__ == "__main__":
    main()
