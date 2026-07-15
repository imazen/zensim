#!/usr/bin/env python3
"""EXP-CROSS-CODEC-V11 Balanced PCHIP spline calibrator (task #189, 2026-05-20).

Refits the V10 BalancedV3 spline mechanism on the new V11 ssim2-anchored
substrate. Two paths:

(a) Plain Balanced bake (trained on canonical-2026-05-21, no anchor data
    used during training because plain MLP doesn't wire anchor loss).
(b) V10 BalancedV3 ship bake (V_22-mix-LARGE+iwssim, the prior shipping
    weight without ssim2-anchor calibration).

For each bake, we:
1. Score the V11 anchor parquet rows.
2. Group by target_score (10 bands per the V11 design).
3. Compute median raw_pred per band.
4. Sort knots by raw_pred ascending → fit PCHIP monotone spline → inject as
   `zentrain.output_calibration_spline` metadata.

Sibling of `calibrate_balanced_v9_spline.py` but parameterized for the new
V11 ssim2-anchor parquet and 300-feature LARGE schema.
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

V11_ANCHOR_PARQUET_DEFAULT = Path(
    "/mnt/v/zen/zensim-training/2026-05-20-ssim2-anchors/anchors_ssim2_300col.parquet"
)
PREDICT_BIN_DEFAULT = Path(
    "/home/lilith/work/zen/zensim/target/release/"
    "predict_features_with_bake"
)
ZENPREDICT_BIN_DEFAULT = Path(
    "/home/lilith/work/zen/zenanalyze/target/release/zenpredict"
)


def load_anchor_features(parquet_path: Path, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    tbl = pq.read_table(parquet_path)
    df = tbl.to_pandas()
    feat_cols = [f"f{i}" for i in range(n_features)]
    feats = df[feat_cols].to_numpy(dtype=np.float32, copy=False)
    targets = df["target_score"].to_numpy(dtype=np.float64, copy=False)
    return feats, targets


def predict_via_binary(bake_path: Path, feats: np.ndarray, predict_bin: Path,
                       post_mode: str = "raw") -> np.ndarray:
    n_rows, n_features = feats.shape
    tmp = Path("/tmp/v11_balanced_calibrate_feats.bin")
    with tmp.open("wb") as f:
        f.write(struct.pack("<II", n_features, n_rows))
        f.write(feats.astype("<f4", copy=False).tobytes())
    cmd = [
        str(predict_bin),
        "--bake", str(bake_path),
        "--features-file", str(tmp),
        "--bake-post", post_mode,
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    lines = proc.stdout.strip().splitlines()
    if len(lines) != n_rows:
        raise RuntimeError(
            f"predict_features_with_bake emitted {len(lines)} lines for {n_rows} rows"
        )
    return np.array([float(s) for s in lines], dtype=np.float64)


def pchip_derivs(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    n = len(xs)
    if n == 2:
        s = (ys[1] - ys[0]) / (xs[1] - xs[0])
        return np.array([s, s])
    h = np.diff(xs)
    s = np.diff(ys) / h
    d = np.zeros(n)
    for k in range(1, n - 1):
        if s[k - 1] * s[k] <= 0:
            d[k] = 0.0
        else:
            w1 = 2 * h[k] + h[k - 1]
            w2 = h[k] + 2 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / s[k - 1] + w2 / s[k])
    d[0] = pchip_endpoint(h[0], h[1], s[0], s[1])
    d[n - 1] = pchip_endpoint(h[n - 2], h[n - 3], s[n - 2], s[n - 3])
    return d


def pchip_endpoint(h0: float, h1: float, s0: float, s1: float) -> float:
    d = ((2 * h0 + h1) * s0 - h0 * s1) / (h0 + h1)
    if d * s0 <= 0:
        return 0.0
    if s0 * s1 <= 0 and abs(d) > 3 * abs(s0):
        return 3 * s0
    return d


def pchip_eval(x: float, xs: np.ndarray, ys: np.ndarray, derivs: np.ndarray) -> float:
    n = len(xs)
    if not np.isfinite(x):
        return x
    if x <= xs[0]:
        return ys[0] + derivs[0] * (x - xs[0])
    if x >= xs[-1]:
        return ys[-1] + derivs[-1] * (x - xs[-1])
    lo, hi = 0, n - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if xs[mid] <= x:
            lo = mid
        else:
            hi = mid
    h = xs[hi] - xs[lo]
    t = (x - xs[lo]) / h
    h00 = (1 + 2 * t) * (1 - t) ** 2
    h10 = t * (1 - t) ** 2
    h01 = t * t * (3 - 2 * t)
    h11 = t * t * (t - 1)
    return h00 * ys[lo] + h10 * h * derivs[lo] + h01 * ys[hi] + h11 * h * derivs[hi]


def build_spline_knots(
    preds: np.ndarray, targets: np.ndarray
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """Per target_score band, compute MEDIAN raw_pred. Drop bands that
    don't maintain monotone y across the chosen direction.
    """
    bands: list[tuple[float, float, int]] = []
    for t in sorted(set(targets.tolist())):
        mask = targets == t
        if mask.sum() == 0:
            continue
        median_pred = float(np.median(preds[mask]))
        bands.append((float(t), median_pred, int(mask.sum())))

    bands_sorted = sorted(bands, key=lambda b: b[1])

    kept: list[tuple[float, float]] = []
    dropped: list[tuple[float, float]] = []
    direction: int = 0
    for t, x, n in bands_sorted:
        if not kept:
            kept.append((t, x))
            continue
        prev_t, prev_x = kept[-1]
        if x <= prev_x + 1e-4:
            dropped.append((t, x))
            continue
        if direction == 0:
            direction = 1 if t > prev_t else -1
            kept.append((t, x))
            continue
        if direction == 1 and t <= prev_t:
            dropped.append((t, x))
            continue
        if direction == -1 and t >= prev_t:
            dropped.append((t, x))
            continue
        kept.append((t, x))

    if dropped:
        print(f"  WARN: dropped {len(dropped)} bands: {dropped}")
    if len(kept) < 2:
        raise RuntimeError(f"only {len(kept)} knots; cannot build spline")

    xs_out = np.array([x for (t, x) in kept])
    ys_out = np.array([t for (t, x) in kept])
    return xs_out, ys_out, dropped


def encode_spline_payload(xs: np.ndarray, ys: np.ndarray) -> bytes:
    n = len(xs)
    payload = struct.pack("<I", n)
    for x, y in zip(xs, ys):
        payload += struct.pack("<ff", float(x), float(y))
    return payload


def add_spline_metadata(
    bake_path: Path, xs: np.ndarray, ys: np.ndarray, out_path: Path,
    zenpredict_bin: Path,
) -> None:
    inspect_w = subprocess.run(
        [str(zenpredict_bin), "inspect", str(bake_path), "--weights"],
        capture_output=True, text=True, check=True,
    )
    if not inspect_w.stdout.strip():
        raise RuntimeError(f"zenpredict inspect: empty stdout; stderr={inspect_w.stderr!r}")
    inspected_w = json.loads(inspect_w.stdout)
    scaler_mean = inspected_w["scaler_mean"]
    scaler_scale = inspected_w["scaler_scale"]
    layers_w = inspected_w["layers"]
    out_layers = []
    for l in layers_w:
        out_layers.append({
            "in_dim": l["in_dim"], "out_dim": l["out_dim"],
            "activation": l["activation"], "dtype": l["dtype"],
            "weights": l["weights"], "biases": l["biases"],
        })

    md_list = []
    for entry in inspected_w.get("metadata", []):
        key = entry["key"]
        kind = entry["kind"]
        # Skip any existing spline metadata — we replace it.
        if key == "zentrain.output_calibration_spline":
            continue
        if "value_hex" in entry:
            md_list.append({"key": key, "type": kind, "hex": entry["value_hex"]})
        elif "value_text" in entry:
            md_list.append({"key": key, "type": kind, "text": entry["value_text"]})
        elif "value_f32_array" in entry:
            md_list.append({"key": key, "type": kind, "f32": entry["value_f32_array"]})
        else:
            raise RuntimeError(f"don't know how to re-encode {key}: keys={list(entry.keys())}")

    payload = encode_spline_payload(xs, ys)
    md_list.append({
        "key": "zentrain.output_calibration_spline",
        "type": "numeric", "hex": payload.hex(),
    })

    schema_hash_raw = inspected_w.get("schema_hash", 0)
    if isinstance(schema_hash_raw, str):
        schema_hash = int(schema_hash_raw, 16) if schema_hash_raw.startswith("0x") else int(schema_hash_raw)
    else:
        schema_hash = int(schema_hash_raw)

    req = {
        "schema_hash": schema_hash,
        "flags": 0, "compressed": True,
        "scaler_mean": scaler_mean, "scaler_scale": scaler_scale,
        "layers": out_layers, "metadata": md_list,
    }
    json_path = out_path.with_suffix(".tmp.json")
    json_path.write_text(json.dumps(req))
    subprocess.run([str(zenpredict_bin), "bake", str(json_path), str(out_path)], check=True)
    json_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bake", type=Path, required=True,
                        help="Input bake (ZNPR v3, no spline metadata required)")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output bake with V11 ssim2-anchored spline")
    parser.add_argument("--anchor-parquet", type=Path,
                        default=V11_ANCHOR_PARQUET_DEFAULT)
    parser.add_argument("--predict-bin", type=Path, default=PREDICT_BIN_DEFAULT)
    parser.add_argument("--zenpredict-bin", type=Path, default=ZENPREDICT_BIN_DEFAULT)
    parser.add_argument("--n-features", type=int, default=300)
    parser.add_argument("--spline-csv", type=Path, default=None)
    args = parser.parse_args()

    print(f"=== V11 ssim2-anchored spline refit ===")
    print(f"  bake: {args.bake}")
    print(f"  anchor: {args.anchor_parquet}")
    print(f"  out: {args.out}")

    feats, targets = load_anchor_features(args.anchor_parquet, args.n_features)
    print(f"  anchor rows: {len(feats)} x {feats.shape[1]} features")

    print(f"=== scoring bake on anchor features ===")
    preds = predict_via_binary(args.bake, feats, args.predict_bin, post_mode="raw")
    finite = np.isfinite(preds)
    if not finite.all():
        print(f"  WARN: {(~finite).sum()} NaN; dropping")
    valid = finite
    preds_v = preds[valid]
    targets_v = targets[valid]

    print(f"=== per-band median predicted ===")
    band_summary = []
    for t in sorted(set(targets_v.tolist())):
        mask = targets_v == t
        sub = preds_v[mask]
        if len(sub) == 0:
            continue
        band_summary.append({
            "target": float(t), "n": int(len(sub)),
            "med": float(np.median(sub)),
            "p25": float(np.percentile(sub, 25)),
            "p75": float(np.percentile(sub, 75)),
            "min": float(sub.min()), "max": float(sub.max()),
        })
    for b in band_summary:
        print(f"  target={b['target']:6.1f} n={b['n']:6d} med={b['med']:8.3f} "
              f"p25={b['p25']:8.3f} p75={b['p75']:8.3f} min={b['min']:8.3f} max={b['max']:8.3f}")

    if args.spline_csv:
        with args.spline_csv.open("w") as f:
            f.write("target_score,n,med_raw_pred,p25,p75,min,max\n")
            for b in band_summary:
                f.write(f"{b['target']},{b['n']},{b['med']},{b['p25']},{b['p75']},"
                        f"{b['min']},{b['max']}\n")
        print(f"  wrote per-band CSV {args.spline_csv}")

    print(f"=== building spline knots ===")
    xs, ys, dropped = build_spline_knots(preds_v, targets_v)
    print(f"  kept {len(xs)} knots:")
    for x, y in zip(xs, ys):
        print(f"    raw={x:9.4f} → score={y:6.1f}")

    print(f"=== injecting spline metadata ===")
    add_spline_metadata(args.bake, xs, ys, args.out, args.zenpredict_bin)
    print(f"DONE: wrote {args.out} ({args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
