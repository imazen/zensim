#!/usr/bin/env python3
"""EXP-CROSS-CODEC-V9 PCHIP spline calibrator (2026-05-20).

For each V9 trained bake:

1. Run inference on the V9 anchor parquet rows -> collect
   `(predicted_raw_after_tanh_pin, target_score)` pairs.
2. Group by `target_score` (8 bands × ~thousands of rows each).
3. Per band, compute MEDIAN predicted -> the spline knot
   `(median_pred, target_score)`.
4. Verify knots are strictly increasing in x. If two adjacent
   knots collide in x (the network puts two bands at the same
   predicted score), shift one by ε to enforce monotonicity.
5. Build a PCHIP from the 8 (sometimes fewer if collapsed) knots.
6. Verify monotonicity on a dense grid.
7. Read the trained bake, rebuild as JSON via `zenpredict inspect`
   converted to a `BakeRequestJson`, append the
   `zentrain.output_calibration_spline` metadata entry, then call
   `zenpredict bake` to produce the calibrated bake.

Output: `<bake>_calibrated.bin`, and a per-bake spline knot CSV at
`<bake>.spline.csv` for inspection.

The spline metadata payload is `[u32 n_knots, n_knots × (f32 x, f32 y)]`,
hex-encoded in the JSON metadata's "hex" field with type "numeric".
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

V9_ANCHOR_PARQUET = Path(
    "/mnt/v/zen/zensim-training/2026-05-20-v9-anchors/anchors_v9_372col.parquet"
)
PREDICT_BIN = Path(
    "/home/lilith/work/zen/zensim--v10/target/release/"
    "predict_features_with_bake"
)
ZENPREDICT_BIN = Path("/home/lilith/work/zen/zenanalyze/target/release/zenpredict")

# Band targets in V9 (must match build_v9_anchor_parquet.py).
V9_TARGET_SCORES = [0.0, 10.0, 30.0, 50.0, 60.0, 80.0, 90.0, 100.0]


def load_anchor_features(parquet_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Returns (features [N, 372] float32, target_scores [N] float64)."""
    tbl = pq.read_table(parquet_path)
    df = tbl.to_pandas()
    feat_cols = [f"f{i}" for i in range(372)]
    feats = df[feat_cols].to_numpy(dtype=np.float32, copy=False)
    targets = df["target_score"].to_numpy(dtype=np.float64, copy=False)
    return feats, targets


def predict_all(bake_path: Path, feats: np.ndarray) -> np.ndarray:
    """Run predict_features_with_bake over all rows. Returns float64
    predictions (post-tanh-pin, no spline yet)."""
    n_rows, n_features = feats.shape
    # The CLI expects a single text arg "row1 row2 ... rowN flat" but
    # for 22k rows × 372 floats that's huge. Instead, use stdin with
    # the expected format. Let me check the binary first.
    # Actually simplest: write features to a tmp binary file and use
    # the `--features-bin` flag if it exists.
    # Let me inspect the binary's help.
    raise NotImplementedError(
        "Use the binary's actual interface - need to check it."
    )


def predict_via_binary(
    bake_path: Path, feats: np.ndarray, post_mode: str = "raw"
) -> np.ndarray:
    """Run predict_features_with_bake; returns predictions (post-tanh-pin,
    pre-spline since the input bake doesn't yet carry the spline metadata).

    Wire format: [u32 LE n_features, u32 LE n_rows, f32 LE row-major].
    """
    n_rows, n_features = feats.shape
    tmp = Path("/tmp/v9_calibrate_feats.bin")
    with tmp.open("wb") as f:
        f.write(struct.pack("<II", n_features, n_rows))
        f.write(feats.astype("<f4", copy=False).tobytes())
    cmd = [
        str(PREDICT_BIN),
        "--bake",
        str(bake_path),
        "--features-file",
        str(tmp),
        "--bake-post",
        post_mode,
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    out_lines = proc.stdout.strip().splitlines()
    if len(out_lines) != n_rows:
        raise RuntimeError(
            f"predict_features_with_bake emitted {len(out_lines)} lines "
            f"for {n_rows} rows"
        )
    return np.array([float(s) for s in out_lines], dtype=np.float64)


def pchip_derivs(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Reimplement scipy.interpolate.PchipInterpolator's derivative rule
    so we don't take a scipy dep (and to match the Rust runtime
    bit-exactly). Fritsch-Carlson monotone preserving."""
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
    # Endpoints
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
    # Binary search
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
) -> tuple[np.ndarray, np.ndarray]:
    """For each unique target_score, compute the median predicted-raw,
    then build (xs, ys) sorted by target_score (y) ascending. Enforce
    strict monotonicity in BOTH x and y by:
      1. Sorting by target_score (y), low to high — the user-facing
         band ordering is canonical.
      2. If consecutive medians have x_{i+1} <= x_i (the network
         inverts the band ordering at the median), drop the lower-y
         band whose x is OUT-OF-ORDER vs its neighbors — this means
         the network couldn't tell low-y from the band on either side,
         and the spline can't fix that.
      3. After dropping, ensure n_knots >= 2; else fail.

    Returns (xs, ys, n_dropped).
    """
    bands: list[tuple[float, float]] = []  # (target_score, median_pred)
    for t in sorted(set(targets.tolist())):
        mask = targets == t
        if mask.sum() == 0:
            continue
        median_pred = float(np.median(preds[mask]))
        bands.append((t, median_pred))
    # Sort by y (target_score) ascending — canonical band order.
    bands.sort(key=lambda p: p[0])
    # Drop bands that violate strict monotonicity in x.
    kept: list[tuple[float, float]] = []
    dropped: list[tuple[float, float]] = []
    for y, x in bands:
        if not kept or x > kept[-1][1] + 1e-4:
            kept.append((y, x))
        else:
            # Network's median for this band is not greater than the
            # already-kept previous band's median. The local ordering
            # contradicts the canonical y order. Drop this band; the
            # spline will linearly interpolate between the surviving
            # neighbors.
            dropped.append((y, x))
    if dropped:
        print(
            f"  WARN: dropping {len(dropped)} band(s) due to x-order "
            f"violation: {dropped}"
        )
    if len(kept) < 2:
        raise RuntimeError(
            f"after monotonicity filter only {len(kept)} knots remain; "
            "cannot build spline"
        )
    xs_out = np.array([x for (y, x) in kept])
    ys_out = np.array([y for (y, x) in kept])
    return xs_out, ys_out


def verify_pchip_monotone(
    xs: np.ndarray, ys: np.ndarray, derivs: np.ndarray, n_grid: int = 1000
) -> tuple[bool, float]:
    """Check the PCHIP output is monotone non-decreasing across a
    dense grid. Returns (is_monotone, max_drop)."""
    x_grid = np.linspace(xs[0] - 5, xs[-1] + 5, n_grid)
    y_grid = np.array([pchip_eval(x, xs, ys, derivs) for x in x_grid])
    diffs = np.diff(y_grid)
    max_drop = float(-diffs.min()) if diffs.min() < 0 else 0.0
    return max_drop <= 1e-6, max_drop


def encode_spline_payload(xs: np.ndarray, ys: np.ndarray) -> bytes:
    """Encode (xs, ys) as `[u32 n_knots, n × (f32 x, f32 y)]` LE."""
    n = len(xs)
    payload = struct.pack("<I", n)
    for x, y in zip(xs, ys):
        payload += struct.pack("<ff", float(x), float(y))
    return payload


def add_spline_metadata(bake_path: Path, xs: np.ndarray, ys: np.ndarray, out_path: Path) -> None:
    """Inspect the bake, build a BakeRequestJson, append the spline
    metadata entry, bake via the zenpredict CLI.

    The `zenpredict inspect --weights` output uses top-level
    `scaler_mean` / `scaler_scale` (not nested) and layer entries with
    `weights` / `biases` as float arrays. Metadata entries carry
    `value_hex` (universal) plus optionally `value_f32_array` and
    `value_text`.
    """
    inspect_w = subprocess.run(
        [str(ZENPREDICT_BIN), "inspect", str(bake_path), "--weights"],
        capture_output=True,
        text=True,
        check=True,
    )
    if not inspect_w.stdout.strip():
        raise RuntimeError(f"zenpredict inspect emitted no output; stderr={inspect_w.stderr!r}")
    inspected_w = json.loads(inspect_w.stdout)

    scaler_mean = inspected_w["scaler_mean"]
    scaler_scale = inspected_w["scaler_scale"]
    layers_w = inspected_w["layers"]
    out_layers = []
    for l in layers_w:
        out_layers.append(
            {
                "in_dim": l["in_dim"],
                "out_dim": l["out_dim"],
                "activation": l["activation"],
                "dtype": l["dtype"],
                "weights": l["weights"],
                "biases": l["biases"],
            }
        )

    md_list = []
    for entry in inspected_w.get("metadata", []):
        key = entry["key"]
        kind = entry["kind"]
        # Prefer hex (universal). `value_text` / `value_f32_array`
        # are convenience views — for round-tripping numeric payloads
        # the hex is canonical.
        if "value_hex" in entry:
            md_list.append({"key": key, "type": kind, "hex": entry["value_hex"]})
        elif "value_text" in entry:
            md_list.append(
                {"key": key, "type": kind, "text": entry["value_text"]}
            )
        elif "value_f32_array" in entry:
            md_list.append(
                {"key": key, "type": kind, "f32": entry["value_f32_array"]}
            )
        else:
            raise RuntimeError(
                f"don't know how to re-encode metadata entry {key}; "
                f"inspected keys: {list(entry.keys())}"
            )

    # Append spline metadata.
    payload = encode_spline_payload(xs, ys)
    md_list.append(
        {
            "key": "zentrain.output_calibration_spline",
            "type": "numeric",
            "hex": payload.hex(),
        }
    )

    # schema_hash from inspect is a "0x..." hex string; the BakeRequestJson
    # consumes a u64 integer. Convert.
    schema_hash_raw = inspected_w.get("schema_hash", 0)
    if isinstance(schema_hash_raw, str):
        schema_hash = int(schema_hash_raw, 16) if schema_hash_raw.startswith("0x") else int(schema_hash_raw)
    else:
        schema_hash = int(schema_hash_raw)

    req = {
        "schema_hash": schema_hash,
        "flags": 0,
        "compressed": True,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "layers": out_layers,
        "metadata": md_list,
    }
    json_path = out_path.with_suffix(".tmp.json")
    json_path.write_text(json.dumps(req))
    subprocess.run(
        [str(ZENPREDICT_BIN), "bake", str(json_path), str(out_path)],
        check=True,
    )
    json_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bake", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--anchor-parquet", type=Path, default=V9_ANCHOR_PARQUET
    )
    parser.add_argument(
        "--spline-csv",
        type=Path,
        default=None,
        help="If set, write the per-band median + spline knots to this CSV.",
    )
    args = parser.parse_args()

    print(f"=== loading anchor parquet {args.anchor_parquet} ===")
    feats, targets = load_anchor_features(args.anchor_parquet)
    print(f"  {len(feats)} rows × {feats.shape[1]} features")

    print(f"=== scoring {args.bake} on anchor features ===")
    preds = predict_via_binary(args.bake, feats, post_mode="raw")
    # post_mode="raw" returns the bake's raw output (post-tanh-pin
    # since the bake metadata applies it; spline absent at this stage).
    finite = np.isfinite(preds)
    if not finite.all():
        nan_count = int((~finite).sum())
        print(f"  WARN: {nan_count} NaN predictions; dropping from band analysis")
    valid = finite
    preds_v = preds[valid]
    targets_v = targets[valid]

    print("=== per-band median predicted ===")
    band_summary: list[dict] = []
    for t in sorted(set(targets_v.tolist())):
        mask = targets_v == t
        if mask.sum() == 0:
            continue
        sub = preds_v[mask]
        band_summary.append(
            {
                "target": t,
                "n": int(mask.sum()),
                "med": float(np.median(sub)),
                "p25": float(np.percentile(sub, 25)),
                "p75": float(np.percentile(sub, 75)),
                "min": float(sub.min()),
                "max": float(sub.max()),
            }
        )
    for b in band_summary:
        print(
            f"  target={b['target']:6.1f} n={b['n']:6d} med={b['med']:7.3f} "
            f"p25={b['p25']:7.3f} p75={b['p75']:7.3f} "
            f"min={b['min']:7.3f} max={b['max']:7.3f}"
        )

    print("=== building spline knots ===")
    xs, ys = build_spline_knots(preds_v, targets_v)
    print(f"  {len(xs)} knots:")
    for x, y in zip(xs, ys):
        print(f"    x={x:8.4f} → y={y:6.1f}")

    derivs = pchip_derivs(xs, ys)
    print(f"  derivatives: {derivs}")
    ok, max_drop = verify_pchip_monotone(xs, ys, derivs)
    print(f"  monotone={ok}  max_drop={max_drop:.2e}")
    if not ok:
        print("ERROR: spline is not monotone; refusing to write bake.")
        sys.exit(1)

    if args.spline_csv:
        with args.spline_csv.open("w") as f:
            f.write("target,n,median_pred,p25,p75,min,max\n")
            for b in band_summary:
                f.write(
                    f"{b['target']},{b['n']},{b['med']},{b['p25']},"
                    f"{b['p75']},{b['min']},{b['max']}\n"
                )
        print(f"  wrote summary to {args.spline_csv}")

    print(f"=== writing calibrated bake to {args.out} ===")
    add_spline_metadata(args.bake, xs, ys, args.out)
    bytes_in = args.bake.stat().st_size
    bytes_out = args.out.stat().st_size
    print(f"  bake size: {bytes_in} → {bytes_out} bytes (+{bytes_out - bytes_in})")


if __name__ == "__main__":
    main()
