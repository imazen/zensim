#!/usr/bin/env python3
# DEPRECATED (2026-07-05): migrated to Rust. Use
#   target/release/bake_dial_refit bottom-extend --in <bake> --out <out> --floor-raw 0.0
# (zensim-validate/src/bin/bake_dial_refit.rs) — reproduces this script's output
# BYTE-IDENTICALLY. Kept for provenance only. See
# benchmarks/bake_refit_rust_migration_2026-07-05.md.
"""Extend BHdr's output-spline BOTTOM to cover the HDR raw floor (rank-invariant).

DIAGNOSIS (2026-07-05, measured via bake_outlier_gate on hdr_v3mix, 7,410 rows):
`bhdr_linear_shaped_anchored2` FAILS the HARD G-RANGE gate at the BOTTOM —
`below-knot 2.132%`: 157 pairs have raw (floor 0.034) below the bottom spline knot
(raw 0.297 -> score 25.9). The anchored2 Q-Q fit calibrated the TOP to the data
ceiling but left the bottom as a STEEP linear extrapolation that goes NEGATIVE
(raw 0.034 -> dial -1.97). (The top is fine: above-knot 0.000% — y=92.8 is the
honest HDR ceiling, NOT a dead-zone.)

FIX: prepend a bottom knot at (0.0, 0.0). This covers the raw floor (G-RANGE PASS)
and remaps the bottom sub-range MONOTONICALLY from the negative-going extrapolation
to a gentle approach to 0 — so:
  - rank byte-identical: same raw preds (shaping+weights untouched), monotone
    sub-range remap => HDR SROCC unchanged (verified 0.914897 == 0.914897, Δ=0);
  - no negative dial scores (min -1.97 -> +2.98);
  - identity=100 still delivered by the runtime is_identical short-circuit.

CAVEAT (deeper, separate from this narrow fix): the 157 pairs have targets 0.3-0.6
([0,1] scale) yet very low raw — BHdr under-RANKS them relative to target on the
bottom tail. That is a model/calibration question (bottom re-anchor), NOT fixed
here; this only removes the negatives + passes the gate. Left for a deliberate
decision. Staged, NOT auto-rotated (BHdr is a second shipped profile).

  usage: bhdr_bottom_extend.py OUT.bin [--in BHDR_BAKE] [--floor-raw 0.0]
"""
import argparse
import importlib.util
import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

BAKER = os.path.expanduser("~/work/zen/zenanalyze/target/release/zenpredict")
REPO = Path(__file__).resolve().parents[2]
DEFAULT_IN = REPO / "zensim/weights/bhdr_linear_shaped_anchored2_2026-07-04.bin"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--in", dest="inp", default=str(DEFAULT_IN))
    ap.add_argument("--floor-raw", type=float, default=0.0,
                    help="raw value of the new bottom knot (score 0); must be < current bottom knot")
    a = ap.parse_args()

    ins = json.loads(subprocess.run(
        [BAKER, "inspect", a.inp, "--weights"], capture_output=True, text=True).stdout)
    md = {m["key"]: m for m in ins["metadata"]}
    spk = next(k for k in md if k.endswith("output_calibration_spline"))
    sb = bytes.fromhex(md[spk].get("value_hex") or md[spk].get("hex"))
    nk = struct.unpack("<I", sb[:4])[0]
    kx = [struct.unpack("<ff", sb[4 + 8 * i:12 + 8 * i])[0] for i in range(nk)]
    ky = [struct.unpack("<ff", sb[4 + 8 * i:12 + 8 * i])[1] for i in range(nk)]
    assert a.floor_raw < kx[0], f"floor-raw {a.floor_raw} must be < bottom knot {kx[0]}"
    kx = [a.floor_raw] + kx
    ky = [0.0] + ky
    payload = struct.pack("<I", len(kx)) + b"".join(
        struct.pack("<ff", float(x), float(y)) for x, y in zip(kx, ky))

    # copy every metadata entry verbatim except the spline (weights/scaler/shaping untouched)
    meta = []
    for m in ins["metadata"]:
        if m["key"] == spk:
            meta.append({"key": spk, "type": "bytes", "hex": payload.hex()})
        elif m.get("value_text") is not None or m.get("type") == "utf8":
            meta.append({"key": m["key"], "type": "utf8",
                         "text": m.get("value_text", m.get("text", ""))})
        else:
            meta.append({"key": m["key"], "type": "bytes",
                         "hex": m.get("value_hex") or m.get("hex")})
    layer = ins["layers"][0]
    req = {"schema_hash": 0, "flags": 0, "compressed": True,
           "scaler_mean": [float(v) for v in ins["scaler_mean"]],
           "scaler_scale": [float(v) for v in ins["scaler_scale"]],
           "layers": [{"in_dim": layer["in_dim"], "out_dim": 1,
                       "activation": layer.get("activation", "identity"), "dtype": "f16",
                       "weights": [float(v) for v in layer["weights"]],
                       "biases": [float(layer["biases"][0])]}],
           "metadata": meta}
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(req, f); jp = f.name
    r = subprocess.run([BAKER, "bake", jp, a.out], capture_output=True, text=True)
    os.unlink(jp)
    assert r.returncode == 0, r.stderr[:400]
    print(f"BHdr bottom-extended: {nk}->{len(kx)} knots, new bottom ({a.floor_raw},0.0); "
          f"{Path(a.out).stat().st_size} B -> {a.out}")


if __name__ == "__main__":
    main()
