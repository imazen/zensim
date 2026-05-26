#!/usr/bin/env python3
"""Bidirectional PCHIP spline calibrator for zensim-b-phone (2026-05-26).

The in-trainer spline fitter (mlp_train/mod.rs:7553) and calibrate_v9_spline.py
both assume a SCORE-shaped bake (raw pred increases with target_score). The
zensim-b-phone bake came out DISTANCE-shaped (raw pred DECREASES as the
phone-CVVDP dial target increases — anchor SROCC(raw, target) ≈ -0.93), so
those fitters drop nearly every band and produce a degenerate 3-5-knot
spline -> broken dial (G1=0.00).

The runtime PCHIP `apply` only requires xs (pred) strictly increasing; ys
(target) may DECREASE (Fritsch-Carlson preserves monotone-decreasing). So
this calibrator builds knots sorted by pred ascending, auto-detects the
dominant target direction, and keeps bands monotone in that direction.
Then it reuses the proven JSON-pipeline injection (zenpredict inspect ->
append zentrain.output_calibration_spline -> zenpredict bake).
"""
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

PREDICT_BIN = Path("/home/lilith/work/zen/zensim/target/release/predict_features_with_bake")
ZENPREDICT_BIN = Path("/home/lilith/work/zen/zenanalyze/target/release/zenpredict")


def load_anchor(parquet_path: Path):
    tbl = pq.read_table(str(parquet_path))
    fcols = sorted([c for c in tbl.column_names if c.startswith("f") and c[1:].isdigit()],
                   key=lambda c: int(c[1:]))
    feats = np.column_stack([
        np.asarray(tbl.column(c).to_numpy(zero_copy_only=False), dtype=np.float32)
        for c in fcols
    ])
    targets = np.asarray(tbl.column("target_score").to_numpy(zero_copy_only=False),
                         dtype=np.float64)
    return feats, targets


def predict_raw(bake: Path, feats: np.ndarray) -> np.ndarray:
    n_rows, n_features = feats.shape
    tmp = Path("/tmp/phone_calibrate_feats.bin")
    with tmp.open("wb") as f:
        f.write(struct.pack("<II", n_features, n_rows))
        f.write(feats.astype("<f4", copy=False).tobytes())
    proc = subprocess.run(
        [str(PREDICT_BIN), "--bake", str(bake), "--features-file", str(tmp),
         "--bake-post", "raw"],
        check=True, capture_output=True, text=True,
    )
    lines = proc.stdout.strip().splitlines()
    if len(lines) != n_rows:
        raise RuntimeError(f"predict emitted {len(lines)} lines for {n_rows} rows")
    return np.array([float(s) for s in lines], dtype=np.float64)


def build_knots_bidir(preds: np.ndarray, targets: np.ndarray, n_bins: int):
    """Bin by target into n_bins quantile-or-uniform bands, take median
    pred per band, sort knots by pred (x) ascending, keep them monotone
    in the dominant target (y) direction.

    Returns (xs_ascending, ys, direction_str, dropped).
    """
    # Detect direction.
    cov = np.cov(preds, targets)[0, 1]
    decreasing = cov < 0.0
    # Uniform target bins over the realized target range.
    edges = np.linspace(targets.min(), targets.max(), n_bins + 1)
    bands = []  # (median_pred, median_target)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (targets >= lo) & (targets < hi if i < n_bins - 1 else targets <= hi)
        if m.sum() == 0:
            continue
        bands.append((float(np.median(preds[m])), float(np.median(targets[m]))))
    # Sort by x (median_pred) ascending — required by runtime binary search.
    bands.sort(key=lambda p: p[0])
    kept = []
    dropped = []
    for x, y in bands:
        if not kept:
            kept.append((x, y)); continue
        if x <= kept[-1][0] + 1e-4:
            dropped.append((x, y)); continue  # x must strictly increase
        # y must move in the dominant direction.
        y_ok = (y < kept[-1][1] - 1e-9) if decreasing else (y > kept[-1][1] + 1e-9)
        if y_ok:
            kept.append((x, y))
        else:
            dropped.append((x, y))
    if len(kept) < 2:
        raise RuntimeError(f"only {len(kept)} monotone knots; cannot build spline")
    xs = np.array([k[0] for k in kept])
    ys = np.array([k[1] for k in kept])
    return xs, ys, ("decreasing" if decreasing else "increasing"), dropped


def encode_payload(xs: np.ndarray, ys: np.ndarray) -> bytes:
    payload = struct.pack("<I", len(xs))
    for x, y in zip(xs, ys):
        payload += struct.pack("<ff", float(x), float(y))
    return payload


def inject_spline(bake: Path, xs: np.ndarray, ys: np.ndarray, out: Path):
    insp = subprocess.run(
        [str(ZENPREDICT_BIN), "inspect", str(bake), "--weights"],
        capture_output=True, text=True, check=True,
    )
    w = json.loads(insp.stdout)
    out_layers = [{
        "in_dim": l["in_dim"], "out_dim": l["out_dim"], "activation": l["activation"],
        "dtype": l["dtype"], "weights": l["weights"], "biases": l["biases"],
    } for l in w["layers"]]
    md_list = []
    for e in w.get("metadata", []):
        # Drop any existing spline so we replace, not duplicate.
        if e["key"] == "zentrain.output_calibration_spline":
            continue
        if "value_hex" in e:
            md_list.append({"key": e["key"], "type": e["kind"], "hex": e["value_hex"]})
        elif "value_text" in e:
            md_list.append({"key": e["key"], "type": e["kind"], "text": e["value_text"]})
        elif "value_f32_array" in e:
            md_list.append({"key": e["key"], "type": e["kind"], "f32": e["value_f32_array"]})
        else:
            raise RuntimeError(f"cannot re-encode metadata {e['key']}")
    md_list.append({
        "key": "zentrain.output_calibration_spline",
        "type": "numeric",
        "hex": encode_payload(xs, ys).hex(),
    })
    sh = w.get("schema_hash", 0)
    schema_hash = int(sh, 16) if isinstance(sh, str) and sh.startswith("0x") else int(sh)
    req = {
        "schema_hash": schema_hash, "flags": 0, "compressed": True,
        "scaler_mean": w["scaler_mean"], "scaler_scale": w["scaler_scale"],
        "layers": out_layers, "metadata": md_list,
    }
    jp = out.with_suffix(".tmp.json")
    jp.write_text(json.dumps(req))
    subprocess.run([str(ZENPREDICT_BIN), "bake", str(jp), str(out)], check=True)
    jp.unlink()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bake", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--anchor-parquet", type=Path, required=True)
    ap.add_argument("--n-bins", type=int, default=12)
    args = ap.parse_args()

    feats, targets = load_anchor(args.anchor_parquet)
    print(f"anchor: {len(feats)} rows")
    preds = predict_raw(args.bake, feats)
    finite = np.isfinite(preds)
    preds, targets = preds[finite], targets[finite]
    from scipy.stats import spearmanr
    print(f"raw SROCC(pred, target) = {spearmanr(preds, targets).correlation:+.4f} "
          f"(determines spline direction)")
    xs, ys, direction, dropped = build_knots_bidir(preds, targets, args.n_bins)
    print(f"direction={direction}  {len(xs)} knots (dropped {len(dropped)}):")
    for x, y in zip(xs, ys):
        print(f"  pred={x:9.3f} -> target={y:6.1f}")
    inject_spline(args.bake, xs, ys, args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
