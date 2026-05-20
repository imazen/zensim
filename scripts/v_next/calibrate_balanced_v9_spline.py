#!/usr/bin/env python3
"""V9 PCHIP spline calibrator for the BALANCED bake (task #176, 2026-05-20).

Sibling of `calibrate_v9_spline.py` but adapted for the
V_22-mix-LARGE+iwssim Balanced bake:

- 300-input network (LARGE schema, no IW-pool block).
- Standard MLP — NO per_sample_alpha_head, NO hybrid_head, NO tanh
  output pin metadata.
- DISTANCE-SHAPED raw output: high raw value = low quality (positive
  for worst codec, negative for highest quality). This is the
  opposite of V9's score-shaped raw output, so we order knots by x
  ASCENDING (not by target_score ascending) — the spline is monotone
  but DECREASING in y as x increases.

For each anchor row, score it with the Balanced bake to get raw_pred.
Group by `target_score` (8 bands), take MEDIAN raw_pred per band.
Sort knots by raw_pred ASCENDING; this gives target_score DESCENDING
as a monotone function. Verify monotonicity, inject spline as
`zentrain.output_calibration_spline` metadata, round-trip through
zenpredict.

Output: calibrated bake at the requested path; per-band CSV at
the requested CSV path.
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

V9_ANCHOR_PARQUET_DEFAULT = Path(
    "/mnt/v/zen/zensim-training/2026-05-20-v9-anchors/anchors_v9_372col.parquet"
)
PREDICT_BIN_DEFAULT = Path(
    "/home/lilith/work/zen/zensim--cross-codec-v9/target/release/"
    "predict_features_with_bake"
)
ZENPREDICT_BIN_DEFAULT = Path(
    "/home/lilith/work/zen/zenanalyze/target/release/zenpredict"
)


def load_anchor_features(parquet_path: Path, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """Returns (features [N, n_features] float32, target_scores [N] float64)."""
    tbl = pq.read_table(parquet_path)
    df = tbl.to_pandas()
    feat_cols = [f"f{i}" for i in range(n_features)]
    feats = df[feat_cols].to_numpy(dtype=np.float32, copy=False)
    targets = df["target_score"].to_numpy(dtype=np.float64, copy=False)
    return feats, targets


def predict_via_binary(
    bake_path: Path, feats: np.ndarray, predict_bin: Path, post_mode: str = "raw"
) -> np.ndarray:
    """Run predict_features_with_bake; returns raw float64 predictions."""
    n_rows, n_features = feats.shape
    tmp = Path("/tmp/v9_balanced_calibrate_feats.bin")
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
            f"predict_features_with_bake emitted {len(lines)} lines "
            f"for {n_rows} rows"
        )
    return np.array([float(s) for s in lines], dtype=np.float64)


def pchip_derivs(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Fritsch-Carlson monotone-preserving PCHIP derivative rule."""
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
    preds: np.ndarray, targets: np.ndarray, distance_shaped: bool
) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float]]]:
    """Per target_score band, compute MEDIAN raw_pred. Sort knots by
    raw_pred (x) ascending — strictly required by the PCHIP parser.

    When `distance_shaped=True` the network's raw output is a distance
    (high = low quality), so sorting by x ascending produces target_score
    DECREASING — still a valid monotone PCHIP (the parser only requires
    strict monotonicity in x).

    When `distance_shaped=False` (score-shaped), sorting by x ascending
    produces target_score INCREASING — V9's original behavior.

    Drops bands whose median x value, after sorting, would violate
    strict monotonicity vs the previously-kept band (network failed to
    distinguish two adjacent bands). Returns (xs, ys, dropped).
    """
    bands: list[tuple[float, float, int]] = []  # (target_score, median_pred, n)
    for t in sorted(set(targets.tolist())):
        mask = targets == t
        if mask.sum() == 0:
            continue
        median_pred = float(np.median(preds[mask]))
        bands.append((float(t), median_pred, int(mask.sum())))

    # Sort by x (median_pred) ascending.
    bands_sorted = sorted(bands, key=lambda b: b[1])

    # The y direction must be monotone (either strictly increasing or
    # strictly decreasing) — PCHIP can shape either. Determine direction
    # from the first two non-collapsed knots.
    kept: list[tuple[float, float]] = []
    dropped: list[tuple[float, float]] = []
    direction: int = 0  # 0 = unknown, +1 = y increasing, -1 = y decreasing
    for t, x, n in bands_sorted:
        if not kept:
            kept.append((t, x))
            continue
        prev_t, prev_x = kept[-1]
        if x <= prev_x + 1e-4:
            # x collision — drop this band (network couldn't tell it
            # from the previous one).
            dropped.append((t, x))
            continue
        if direction == 0:
            direction = 1 if t > prev_t else -1
            kept.append((t, x))
            continue
        # Validate the band's y agrees with the chosen direction.
        if direction == 1 and t <= prev_t:
            dropped.append((t, x))
            continue
        if direction == -1 and t >= prev_t:
            dropped.append((t, x))
            continue
        kept.append((t, x))

    if dropped:
        print(
            f"  WARN: dropped {len(dropped)} band(s) (x-collision or "
            f"direction-violation): {dropped}"
        )

    if len(kept) < 2:
        raise RuntimeError(
            f"after monotonicity filter only {len(kept)} knots remain; "
            "cannot build spline"
        )

    xs_out = np.array([x for (t, x) in kept])
    ys_out = np.array([t for (t, x) in kept])
    return xs_out, ys_out, dropped


def verify_pchip_monotone(
    xs: np.ndarray, ys: np.ndarray, derivs: np.ndarray, n_grid: int = 1000
) -> tuple[bool, float, float]:
    """Verify monotonicity on a dense grid. Returns (is_monotone,
    max_drop, direction_sign).

    `direction_sign` = +1 if ys are increasing, -1 if decreasing.
    """
    x_grid = np.linspace(xs[0] - 5, xs[-1] + 5, n_grid)
    y_grid = np.array([pchip_eval(x, xs, ys, derivs) for x in x_grid])
    direction = 1.0 if ys[-1] > ys[0] else -1.0
    # Multiply by direction so that "monotone" becomes "non-decreasing"
    # regardless of which way the spline points.
    y_signed = direction * y_grid
    diffs = np.diff(y_signed)
    max_drop = float(-diffs.min()) if diffs.min() < 0 else 0.0
    return max_drop <= 1e-6, max_drop, direction


def encode_spline_payload(xs: np.ndarray, ys: np.ndarray) -> bytes:
    n = len(xs)
    payload = struct.pack("<I", n)
    for x, y in zip(xs, ys):
        payload += struct.pack("<ff", float(x), float(y))
    return payload


def add_spline_metadata(
    bake_path: Path,
    xs: np.ndarray,
    ys: np.ndarray,
    out_path: Path,
    zenpredict_bin: Path,
) -> None:
    """Inspect → rebuild JSON → append spline metadata → bake."""
    inspect_w = subprocess.run(
        [str(zenpredict_bin), "inspect", str(bake_path), "--weights"],
        capture_output=True,
        text=True,
        check=True,
    )
    if not inspect_w.stdout.strip():
        raise RuntimeError(
            f"zenpredict inspect emitted no output; stderr={inspect_w.stderr!r}"
        )
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
        if "value_hex" in entry:
            md_list.append({"key": key, "type": kind, "hex": entry["value_hex"]})
        elif "value_text" in entry:
            md_list.append({"key": key, "type": kind, "text": entry["value_text"]})
        elif "value_f32_array" in entry:
            md_list.append({"key": key, "type": kind, "f32": entry["value_f32_array"]})
        else:
            raise RuntimeError(
                f"don't know how to re-encode metadata entry {key}; "
                f"inspected keys: {list(entry.keys())}"
            )

    payload = encode_spline_payload(xs, ys)
    md_list.append(
        {
            "key": "zentrain.output_calibration_spline",
            "type": "numeric",
            "hex": payload.hex(),
        }
    )

    schema_hash_raw = inspected_w.get("schema_hash", 0)
    if isinstance(schema_hash_raw, str):
        schema_hash = int(schema_hash_raw, 16) if schema_hash_raw.startswith("0x") else int(schema_hash_raw)
    else:
        schema_hash = int(schema_hash_raw)

    # NOTE: the inspect output's `flags` field is the on-disk header
    # bitfield (compression flags + dtype hints), NOT the same as the
    # JSON `flags` field. Setting `compressed: true` is the canonical
    # way to ask the baker to LZ4-pack the payload; let the baker
    # decide the on-disk header flags from that switch.
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
        [str(zenpredict_bin), "bake", str(json_path), str(out_path)],
        check=True,
    )
    json_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bake", type=Path, required=True,
                        help="Input bake (ZNPR v3, no spline metadata)")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output bake with spline metadata appended")
    parser.add_argument("--anchor-parquet", type=Path,
                        default=V9_ANCHOR_PARQUET_DEFAULT,
                        help="V9 anchor parquet (8 bands × 22k rows)")
    parser.add_argument("--predict-bin", type=Path,
                        default=PREDICT_BIN_DEFAULT)
    parser.add_argument("--zenpredict-bin", type=Path,
                        default=ZENPREDICT_BIN_DEFAULT)
    parser.add_argument("--n-features", type=int, default=300,
                        help="Number of features the bake reads from the "
                             "anchor parquet (300 for LARGE-schema bakes)")
    parser.add_argument("--spline-csv", type=Path, default=None,
                        help="Per-band summary CSV path")
    args = parser.parse_args()

    print(f"=== loading anchor parquet {args.anchor_parquet} ===")
    feats, targets = load_anchor_features(args.anchor_parquet, args.n_features)
    print(f"  {len(feats)} rows × {feats.shape[1]} features")

    print(f"=== scoring {args.bake} on anchor features ===")
    preds = predict_via_binary(args.bake, feats, args.predict_bin, post_mode="raw")
    finite = np.isfinite(preds)
    if not finite.all():
        nan_count = int((~finite).sum())
        print(f"  WARN: {nan_count} NaN predictions; dropping")
    valid = finite
    preds_v = preds[valid]
    targets_v = targets[valid]

    print("=== per-band median predicted (raw, pre-spline) ===")
    band_summary: list[dict] = []
    for t in sorted(set(targets_v.tolist())):
        mask = targets_v == t
        if mask.sum() == 0:
            continue
        sub = preds_v[mask]
        band_summary.append(
            {
                "target": float(t),
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

    # Determine bake shape (distance vs score) from the trend across
    # bands. If median at target=100 < median at target=0 → distance-shaped.
    target_low = [b for b in band_summary if b["target"] <= 10][0]["med"]
    target_high = [b for b in band_summary if b["target"] >= 90][-1]["med"]
    distance_shaped = target_high < target_low
    shape_label = "distance" if distance_shaped else "score"
    print(f"  inferred bake shape: {shape_label}-shaped "
          f"(target=high raw_med={target_high:.2f}, target=low raw_med={target_low:.2f})")

    print("=== building spline knots ===")
    xs, ys, dropped = build_spline_knots(preds_v, targets_v, distance_shaped)
    print(f"  {len(xs)} knots (xs ascending, ys following bake shape):")
    for x, y in zip(xs, ys):
        print(f"    x={x:8.4f} → y={y:6.1f}")

    derivs = pchip_derivs(xs, ys)
    print(f"  derivatives: {derivs}")
    ok, max_drop, direction = verify_pchip_monotone(xs, ys, derivs)
    print(f"  monotone={ok}  max_drop={max_drop:.2e}  direction={direction:+.0f}")
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
            f.write("\n# spline knots (xs, ys):\n")
            for x, y in zip(xs, ys):
                f.write(f"# x={x},y={y}\n")
            f.write(f"# dropped_bands={len(dropped)}\n")
            for d in dropped:
                f.write(f"# dropped target={d[0]} x={d[1]}\n")
        print(f"  wrote summary to {args.spline_csv}")

    print(f"=== writing calibrated bake to {args.out} ===")
    add_spline_metadata(args.bake, xs, ys, args.out, args.zenpredict_bin)
    bytes_in = args.bake.stat().st_size
    bytes_out = args.out.stat().st_size
    print(f"  bake size: {bytes_in} → {bytes_out} bytes (+{bytes_out - bytes_in})")


if __name__ == "__main__":
    main()
