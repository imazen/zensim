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
    print(f"  top extension: k={k:.3f} (n={int(band.sum())}); {x0:.3f}->{r_far:.2f}, "
          f"+{added} knots, y-top {ky[-1]:.2f}; final {len(kx)} knots")

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
