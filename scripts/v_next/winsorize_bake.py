#!/usr/bin/env python3
"""Add a winsorizing feature-transform guard to a raw-space linear ZNPR bake.

This is the FINAL step that produced the shipped Profile-B SDR bake
(`b_sdr_linear_cid80_winsor_2026-07-05.bin`) — it had been an ad-hoc inline
command; this script makes it reproducible (see `scripts/reproduce_b.sh`).

A raw-space linear bake feeds features straight into a scaler+weights dot
product. A heavy-tailed feature (f155 on tiny screen-content renditions: fit-
corpus p99.9 = 0.479 vs a val max of 14,532) then drives the raw prediction far
outside the output spline's knot domain, where the dial extrapolates to absurd
values. `winsor_p99` (zenpredict's existing transform op — "clip to [lo,hi],
preserves rank within bounds") clips each feature to its fit-corpus [p_lo,p_hi].
It is IDENTITY within the bounds, so rank is preserved on the ~99.8% in-
distribution rows and the ~0.2% tail is clipped — provably bounding the raw
output inside the knot domain. `predict_transformed` is auto-dispatched by every
runtime consumer via `has_nontrivial_feature_transforms()` — no code change.

The transforms apply BEFORE the scaler, so weights/scaler/spline are copied
verbatim from the input bake; only the transform metadata is added. Ranking is
therefore unchanged on in-distribution data (SROCC is invariant to the identity
region of the clip); only the extrapolating tail moves.

  usage: winsorize_bake.py --in RAW.bin --fit-corpus FIT.parquet --out OUT.bin \
             [--lo-pct 0.1] [--hi-pct 99.9] [--expect-sha256 <hex>]
"""
import argparse
import hashlib
import importlib.util
import json
import os
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

BAKER = os.path.expanduser("~/work/zen/zenanalyze/target/release/zenpredict")
N_FEAT = 372


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="raw-space linear bake (no transforms)")
    ap.add_argument("--fit-corpus", required=True,
                    help="parquet with f0..f371 the winsor bounds are computed from "
                         "(the corpus the weights were fit against)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--lo-pct", type=float, default=0.1)
    ap.add_argument("--hi-pct", type=float, default=99.9)
    ap.add_argument("--expect-sha256", default=None,
                    help="if given, assert the output matches this sha256 (byte-repro gate)")
    a = ap.parse_args()

    # 1. read the input bake's scaler / weights / spline verbatim
    ins = json.loads(subprocess.run(
        [BAKER, "inspect", a.inp, "--weights"], capture_output=True, text=True).stdout)
    layer = ins["layers"][0]
    md = {m["key"]: m for m in ins["metadata"]}
    assert not any("transform" in k for k in md), \
        "input already has feature_transforms — winsorize a RAW bake, not a shaped one"
    sp = md[lp.SPLINE_KEY]
    spline_hex = sp.get("value_hex") or sp.get("hex")

    # 2. per-feature winsor bounds = [lo-pct, hi-pct] on the fit corpus
    t = pq.read_table(a.fit_corpus, columns=[f"f{i}" for i in range(N_FEAT)])
    F = np.column_stack([np.asarray(t[f"f{i}"], dtype=float) for i in range(N_FEAT)])
    lo = np.percentile(F, a.lo_pct, axis=0)
    hi = np.percentile(F, a.hi_pct, axis=0)
    # a FULLY-zero-constant feature (the PU21-constant f25/f64, lo==hi==0) gets a
    # [0,1e-9] range rather than a degenerate [0,0]. Condition is exactly
    # lo==0 AND hi==0 so signed features with a negative p99.9 upper bound are
    # left untouched (a blanket hi<=0 floor would corrupt those).
    hi = np.where((lo == 0.0) & (hi == 0.0), 1e-9, hi)

    # 3. emit BakeRequestJson: same scaler/weights/spline + 372 winsor_p99 transforms
    req = {
        "schema_hash": 0, "flags": 0, "compressed": True,
        "scaler_mean": [float(v) for v in ins["scaler_mean"]],
        "scaler_scale": [float(v) for v in ins["scaler_scale"]],
        "layers": [{"in_dim": N_FEAT, "out_dim": 1, "activation": "identity",
                    "dtype": "f16",
                    "weights": [float(v) for v in layer["weights"]],
                    "biases": [float(layer["biases"][0])]}],
        # order matches the shipped bake: transforms, params, then spline last
        "metadata": [
            {"key": "zentrain.feature_transforms", "type": "utf8",
             "text": "\n".join(["winsor_p99"] * N_FEAT)},
            {"key": "zentrain.feature_transform_params", "type": "utf8",
             "text": "\n".join(f"{lo[i]},{hi[i]}" for i in range(N_FEAT))},
            {"key": lp.SPLINE_KEY, "type": "bytes", "hex": spline_hex},
        ],
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(req, f)
        jp = Path(f.name)
    r = subprocess.run([BAKER, "bake", str(jp), a.out], capture_output=True, text=True)
    jp.unlink(missing_ok=True)
    assert r.returncode == 0, r.stderr[:400]

    got = hashlib.sha256(Path(a.out).read_bytes()).hexdigest()
    print(f"winsorized {os.path.basename(a.inp)} -> {os.path.basename(a.out)} "
          f"({Path(a.out).stat().st_size} B)")
    print(f"  372 winsor_p99 transforms, fit-corpus [p{a.lo_pct},p{a.hi_pct}] "
          f"from {os.path.basename(a.fit_corpus)}")
    print(f"  sha256 {got}")
    if a.expect_sha256:
        if got.startswith(a.expect_sha256):
            print(f"  BYTE-REPRODUCED (matches expected {a.expect_sha256}) ✓")
        else:
            print(f"  !! sha mismatch: expected {a.expect_sha256}", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
