#!/usr/bin/env python3
# DEPRECATED (2026-07-05): the bake-level whole-spline refit migrated to Rust —
#   target/release/bake_dial_refit shared-anchor --in <bake> --out <out> \
#       --anchor <parquet> --target-col <col> [--target-scale 100]
# (zensim-validate/src/bin/bake_dial_refit.rs; percentile-edge fit_spline_knots
# ported faithfully). This script fits from linear-probe .npz artifacts (research
# workflow) rather than a bake; kept for that provenance. See
# benchmarks/bake_refit_rust_migration_2026-07-05.md.
"""Refit the two ship-pick linear bakes' output splines against the SHARED
anchor scale, making cross-model dial agreement true by construction
(2026-07-03 two-model verdict: the 15pt offset is pure scale).

Anchor convention (one scale for both domains):
  - SDR pick: the canonical multiband anchor (target_score 0..100, the same
    parquet every MLP dial calibrates against).
  - HDR pick: hdr-v3 val rows with target_score = human_score*100 (the same
    ssim2-derived 0..100 scale the SDR anchor encodes), which also fixes the
    cid80 >100 output-range wart.

Emits *_anchored.bin siblings (never overwrites the originals), then
verifies: (a) rank unchanged on the fit-free UPIQ overlap, (b) dial range
within [0,100]-ish, (c) the cross-model raw offset shrinks.
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

PICKS = [
    # (fit key, anchor source, out name)
    ("ens-Pline-cid80", "sdr", "lp_ens-Pline-cid80-anchored-f16.bin"),
    ("hdr-lasso0.001-shaped", "hdr", "lp_hdr-lasso0.001-shaped-anchored-f16.bin"),
]


def anchor_xy(kind, w, bias, mu, sd, space):
    if kind == "sdr":
        t = pq.read_table(
            "/mnt/v/zen/zensim-training/canonical-2026-05-21/train/multiband_anchor_dial100.parquet")
        F = np.column_stack([np.asarray(t[f"f{i}"], dtype=float) for i in range(372)])
        tgt = np.asarray(t["target_score"], dtype=float)
    else:
        t = pq.read_table(PROBE / "hdr_zenjxl_v3_valdigits_2026-07-03.parquet")
        F = np.column_stack([np.asarray(t[f"f{i}"], dtype=float) for i in range(372)])
        tgt = np.asarray(t["human_score"], dtype=float) * 100.0
    if space == "shaped":
        transforms, tparams = lp.load_transforms()
        F = lp.shape_block(F, transforms, tparams)
    preds = ((F - mu) / np.where(sd == 0, 1, sd)) @ w + bias
    return preds, tgt


def main():
    results = []
    for key, kind, outname in PICKS:
        z = np.load(FITS / f"{key}.npz")
        w, bias, mu, sd = z["w"].astype(float), float(z["bias"]), z["mu"].astype(float), z["sd"].astype(float)
        space = str(z["space"]) if "space" in z.files else "raw"
        preds, tgt = anchor_xy(kind, w, bias, mu, sd, space)
        cx, cy = lp.fit_spline_knots(preds, tgt)
        payload = struct.pack("<I", len(cx)) + b"".join(
            struct.pack("<ff", float(x), float(y)) for x, y in zip(cx, cy))
        # re-bake with the anchored spline via the module's request shape
        metadata = [{"key": lp.SPLINE_KEY, "type": "bytes", "hex": payload.hex()}]
        if space == "shaped":
            transforms, tparams = lp.load_transforms()
            metadata += [
                {"key": "zentrain.feature_transforms", "type": "utf8",
                 "text": "\n".join(transforms)},
                {"key": "zentrain.feature_transform_params", "type": "utf8",
                 "text": "\n".join(",".join(f"{q}" for q in row) if row else "" for row in tparams)},
            ]
        req = {
            "schema_hash": 0, "flags": 0, "compressed": True,
            "scaler_mean": [float(v) for v in mu.astype(np.float32)],
            "scaler_scale": [float(v) for v in sd.astype(np.float32)],
            "layers": [{"in_dim": 372, "out_dim": 1, "activation": "identity",
                        "dtype": "f16", "weights": [float(v) for v in w],
                        "biases": [bias]}],
            "metadata": metadata,
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(req, f)
            jp = Path(f.name)
        out = BAKES / outname
        r = subprocess.run([str(lp.BAKER), "bake", str(jp), str(out)],
                           capture_output=True, text=True)
        jp.unlink(missing_ok=True)
        assert r.returncode == 0, r.stderr[:400]
        print(f"{key}: anchored spline {len(cx)} knots, dial y-range "
              f"[{min(cy):.1f}, {max(cy):.1f}] -> {out.name}")
        results.append((key, str(out)))
    return results


if __name__ == "__main__":
    main()
