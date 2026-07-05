#!/usr/bin/env python3
# DEPRECATED (2026-07-05): migrated to Rust. Use
#   target/release/bake_dial_refit extend-top --in <winsor.bin> --out <out.bin> \
#       --anchor <multiband_anchor.parquet> --target-col target_score
# (zensim-validate/src/bin/bake_dial_refit.rs). The Rust extend-top reproduces
# the shipped B bake BYTE-IDENTICALLY (sha b78adb15). Kept for provenance only;
# do not extend. See benchmarks/bake_refit_rust_migration_2026-07-05.md.
"""Best-of-both: give linear-B the MLP's clean dial WITHOUT touching rank.

The linear-B dial dead-zone (5.63%, the only dial gate it failed) is 100%
near-lossless pile-up at the TOP spline knot (y=95.9) — the linear's RAW output is
distinct on every near-lossless pair; only the spline cap flattens them. So it's a
pure calibration fix, rank-invariant (a monotone spline never changes rank).

Minimal + correct: take the winsor bake's EXISTING spline (whose bottom +
in-distribution knots are already correct — 0 below-knot on the outlier gate,
domain reaches the real-content raw floor ~-1.97) and ONLY extend its TOP above the
top knot, by the concave saturation the training data shows. Rebuilding the whole
spline from the balanced anchor instead lifts the bottom knot off the real-content
raw floor and makes low-quality content extrapolate DOWNWARD — the exact
wild-negative tail winsor exists to prevent (measured: 33% below-knot). So we
EXTEND the top, never rebuild.

  usage: dense_dial_refit_b.py OUT.bin [--in WINSOR_BAKE]

Weights / scaler / winsor transforms / bottom+mid spline are copied verbatim from
the input winsor bake, so CID22 + KonJND SROCC are IDENTICAL to it (the spline is
monotone => rank-invariant; measured downstream). Only the OOD-high dial changes.
Deterministic (fixed percentiles + lstsq) => byte-reproducible on a same-input run.
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
import pyarrow.parquet as pq

SPEC = importlib.util.spec_from_file_location(
    "lp", Path(__file__).parent / "linear_projections_2026-07-03.py")
lp = importlib.util.module_from_spec(SPEC)
sys.modules["lp"] = lp
SPEC.loader.exec_module(lp)

BAKER = lp.BAKER
REPO = Path(__file__).resolve().parents[2]
PROBE = Path("/mnt/v/output/zensim-multicodec-probe")
FIT_CORPUS = PROBE / "hdr_zenjxl_v3mix_traindigits_2026-07-03.parquet"
ANCHOR = Path("/mnt/v/zen/zensim-training/canonical-2026-05-21/train/multiband_anchor_dial100.parquet")
N = 372
DEFAULT_IN = REPO / "zensim/weights/archive/b_sdr_linear_cid80_winsor_2026-07-05.bin"


def winsor_bounds():
    """Per-feature [p0.1, p99.9] on the fit corpus — IDENTICAL to winsorize_bake.py
    (same zero-constant floor), so the emitted transforms match the winsor bake byte
    for byte."""
    t = pq.read_table(FIT_CORPUS, columns=[f"f{i}" for i in range(N)])
    F = np.column_stack([np.asarray(t[f"f{i}"], dtype=float) for i in range(N)])
    lo = np.percentile(F, 0.1, axis=0)
    hi = np.percentile(F, 99.9, axis=0)
    hi = np.where((lo == 0.0) & (hi == 0.0), 1e-9, hi)
    return lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--in", dest="inp", default=str(DEFAULT_IN),
                    help="winsor bake whose spline TOP is extended (bottom kept verbatim)")
    a = ap.parse_args()

    # read the winsor bake: scaler, weights, and its (correct) spline
    ins = json.loads(subprocess.run(
        [BAKER, "inspect", a.inp, "--weights"], capture_output=True, text=True).stdout)
    mu = [float(v) for v in ins["scaler_mean"]]
    sd = [float(v) for v in ins["scaler_scale"]]
    layer = ins["layers"][0]
    w = np.array([float(v) for v in layer["weights"]], float)
    bias = float(layer["biases"][0])
    sp = {m["key"]: m for m in ins["metadata"]}[lp.SPLINE_KEY]
    sb = bytes.fromhex(sp.get("value_hex") or sp.get("hex"))
    nk = struct.unpack("<I", sb[:4])[0]
    kx = [struct.unpack("<ff", sb[4 + 8 * i:12 + 8 * i])[0] for i in range(nk)]
    ky = [struct.unpack("<ff", sb[4 + 8 * i:12 + 8 * i])[1] for i in range(nk)]
    x0, y0 = float(kx[-1]), float(ky[-1])
    print(f"input winsor spline: {nk} knots, domain [{kx[0]:.3f},{x0:.3f}] "
          f"y [{ky[0]:.1f},{y0:.1f}] (bottom + in-distribution kept VERBATIM)")

    # winsorized anchor preds for the saturation fit (match runtime: winsor, scale, dot)
    lo, hi = winsor_bounds()
    ta = pq.read_table(ANCHOR)
    F = np.column_stack([np.asarray(ta[f"f{i}"], dtype=float) for i in range(N)])
    tgt = np.asarray(ta["target_score"], dtype=float)
    preds = ((np.clip(F, lo, hi) - np.array(mu)) / np.array(sd)) @ w + bias

    # PRINCIPLED top extension above the top knot x0: continue the concave saturation
    # the training data shows. Fit decay k by ROBUST regression log(100-target) ≈
    # logA − k·raw on the anchor's near-lossless band, then append
    # score(r)=100−(100−y0)·exp(−k·(r−x0)) from (x0,y0). Monotone, concave, →100 —
    # derived only from training saturation (no hand values, no eval-grid dependence;
    # real content tops at raw ~1.12 so the endpoint CAN'T come from data — the
    # saturation shape IS the principle). identity=100 is held by the runtime
    # is_identical short-circuit, not this top knot.
    band = tgt > 70.0  # 600 multiband rows — robust fit
    A = np.vstack([np.ones(int(band.sum())), preds[band]]).T
    coef, *_ = np.linalg.lstsq(A, np.log(np.clip(100.0 - tgt[band], 1e-3, None)), rcond=None)
    k = float(-coef[1])
    assert k > 0, f"saturation fit gave non-decaying k={k}"
    r_far = x0 + (-np.log(1e-4) / k)                    # where score reaches ~99.99
    added = 0
    for r in np.linspace(x0 + (r_far - x0) / 12, r_far, 12):
        y = 100.0 - (100.0 - y0) * np.exp(-k * (r - x0))
        if r > kx[-1] + 1e-7 and y > ky[-1]:
            kx.append(float(r)); ky.append(float(y)); added += 1
    print(f"  top extension: k={k:.2f} (n={int(band.sum())}); {x0:.3f}->{r_far:.2f}, "
          f"+{added} knots, y-top {ky[-1]:.2f}; final {len(kx)} knots")

    payload = struct.pack("<I", len(kx)) + b"".join(
        struct.pack("<ff", float(x), float(y)) for x, y in zip(kx, ky))
    # metadata order matches the winsor bake exactly: transforms, params, spline
    metadata = [
        {"key": "zentrain.feature_transforms", "type": "utf8",
         "text": "\n".join(["winsor_p99"] * N)},
        {"key": "zentrain.feature_transform_params", "type": "utf8",
         "text": "\n".join(f"{lo[i]},{hi[i]}" for i in range(N))},
        {"key": lp.SPLINE_KEY, "type": "bytes", "hex": payload.hex()},
    ]
    req = {"schema_hash": 0, "flags": 0, "compressed": True,
           "scaler_mean": mu, "scaler_scale": sd,
           "layers": [{"in_dim": N, "out_dim": 1, "activation": "identity",
                       "dtype": "f16", "weights": [float(v) for v in w], "biases": [bias]}],
           "metadata": metadata}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(req, f); jp = Path(f.name)
    r = subprocess.run([BAKER, "bake", str(jp), a.out], capture_output=True, text=True)
    jp.unlink(missing_ok=True)
    assert r.returncode == 0, r.stderr[:400]
    print(f"emitted {a.out} ({Path(a.out).stat().st_size} B)")


if __name__ == "__main__":
    main()
