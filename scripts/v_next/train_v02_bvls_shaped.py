#!/usr/bin/env python3
"""V0_2-style + new features + BVLS + input shaping hybrid (#44, 2026-05-28).

User pivot (2026-05-28): "look at what V0_2 trainer did and consider that with
the new features and data sets and shaping etc, and BVLS."

V0_2 was a 228-feature LINEAR weights table (127/228 active under Nelder-Mead
on 218k concordant pairs), shipped at CID22 raw-distance SROCC 0.8676. The
v47_linear MVP (372 raw features, BVLS, NO shaping) shipped at CID22 0.824.
Hypothesis: combining V0_2's BVLS-style global optimum with the 372 feature
set AND the per-feature shaping (Yeo-Johnson, Winsor, SignedCbrt, QuantileBins)
the MLP path already uses should close the gap to v47-strict-QAT-native's 0.866.

Architecture:
  1. Load `benchmarks/yeo_johnson_screen_widest_2026-05-25/screen_results_cross_corpus_safe.tsv`
     → per-feature transform + params, applied at feature ingestion (matches
     the Rust runtime exactly).
  2. Apply per-feature transforms in Python (replicate `apply_with_params`).
  3. Standardize the SHAPED features (z-score; same as v47 MLP path).
  4. BVLS bounded LS with per-feature sign mask:
       300 sign-safe features: w_i ≥ 0
       72 sign-flip features:  w_i ∈ (-∞, +∞)
  5. Forward the multiband anchor through the projected linear net → raw
     distances.
  6. Fit PCHIP monotone spline (same algorithm zensim-validate uses) on
     (raw_distance, anchor_target_score).
  7. Emit ZNPR v3 bake with:
       - `zentrain.feature_transforms`        (utf8, 372 tokens)
       - `zentrain.feature_transform_params`  (utf8, 372 CSV rows)
       - `zentrain.output_calibration_spline` (bytes, u32+f32×N PCHIP knots)
       - 1 layer: 372→1, Identity, weights = BVLS coefs, biases = [bias]
       - scaler_mean / scaler_scale = computed on SHAPED features
  8. Shell to `zenpredict-bake` (the canonical serializer; never raw struct.pack).
  9. Evaluate via `bake_verdict` on the 6 held-out corpora.

Output: a ZNPR v3 bake that's mathematically a linear function of input-shaped
features, with a monotone-by-construction sign-masked weight vector, plus a
PCHIP dial calibration. Bake size estimate: ~16-20 KB depending on sparsity
and spline knot count.
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import lsq_linear
from scipy.stats import spearmanr

REPO = Path(__file__).resolve().parent.parent.parent
TRAIN_DIR = Path("/mnt/v/zen/zensim-training/canonical-2026-05-21/train")
VAL_DIR = Path("/mnt/v/zen/zensim-training/canonical-2026-05-18/val")
MASK_TSV = REPO / "benchmarks/feature_sign_mask_2026-05-26.tsv"
TRANSFORM_TSV = REPO / "benchmarks/yeo_johnson_screen_widest_2026-05-25/screen_results_cross_corpus_safe.tsv"
ANCHOR_PQ = TRAIN_DIR / "multiband_anchor_dial100.parquet"

GROUPS_DEFAULT = [
    # (name, parquet basename, train_w, target_column)
    ("safesyn",      "safesyn",            1.0, "human_score"),
    ("cid22_train",  "cid22_train_norm",   1.5, "human_score"),
    ("kadid",        "kadid",              0.5, "human_score"),
    ("tid",          "tid",                0.5, "human_score"),
]

GROUPS_WITH_KONJND = GROUPS_DEFAULT + [
    # konjnd-dense-norm's `human_score` is the active mix output normalized
    # to [0, 1] on a MOS-equivalent scale (verified 2026-05-28: min=0.0,
    # max=1.0, mean=0.73, n=20,160). Earlier scripts excluded the older
    # konjnd_dense parquet (raw PJND) for scale-mismatch reasons; the
    # normalized variant is safe to include. Low weight (0.3) so it doesn't
    # dominate but adds the visually-lossless boundary signal.
    ("konjnd_dense", "konjnd-dense-norm",  0.3, "human_score"),
]

HOLDOUTS = ["cid22", "kadid", "tid", "konjnd", "aic3", "aic4"]

# Reference: v47-strict-QAT-native (shipped Profile::A)
V47_REF = {
    "cid22":  (0.8657, 0.512),
    "kadid":  (0.7933, 0.613),
    "tid":    (0.7927, 0.577),
    "konjnd": (0.4185, 0.932),
    "aic3":   (0.7680, 0.620),
    "aic4":   (0.8854, 0.481),
}

N_FEATURES = 372


# -----------------------------------------------------------------------------
# Per-feature transforms — replicate ../zenanalyze/zenpredict/src/feature_transform.rs

def yeo_johnson(x: np.ndarray, lam: float) -> np.ndarray:
    """Mirrors zenpredict::feature_transform::yeo_johnson exactly (f32 math).

    x >= 0:  λ == 0 → log1p(x);  else → ((x+1)^λ - 1) / λ
    x  < 0:  λ == 2 → -log1p(-x); else → -(((-x)+1)^(2-λ) - 1) / (2-λ)
    """
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    neg = ~pos
    if abs(lam) < 1e-9:
        out[pos] = np.log1p(x[pos])
    else:
        out[pos] = (np.power(x[pos] + 1.0, lam) - 1.0) / lam
    if abs(lam - 2.0) < 1e-9:
        out[neg] = -np.log1p(-x[neg])
    else:
        exp = 2.0 - lam
        out[neg] = -((np.power(-x[neg] + 1.0, exp)) - 1.0) / exp
    return out


def apply_transform(token: str, params: list[float], x: np.ndarray) -> np.ndarray:
    """Replicate Rust runtime's `apply_with_params` for the variants the
    auto_transforms TSV produces. Returns f64 (we standardize after; cast
    to f32 only at scaler_mean/scaler_scale serialization time)."""
    x = x.astype(np.float64)
    if token == "identity":
        return x
    if token == "log1p":
        return np.log1p(x)
    if token == "signed_log1p":
        return np.sign(x) * np.log1p(np.abs(x))
    if token == "signed_cbrt":
        return np.sign(x) * np.cbrt(np.abs(x))
    if token == "clip_then_log1p":
        eps = params[0] if params else 0.0
        return np.log1p(np.maximum(0.0, x - eps))
    if token == "winsor_p99":
        if len(params) >= 2:
            lo, hi = params[0], params[1]
            return np.clip(x, lo, hi)
        return x
    if token == "quantile_bins":
        if not params:
            return x
        edges = np.asarray(params, dtype=np.float64)
        # idx = count of edges <= x  /  n_edges
        # broadcast-safe; small N so it's cheap
        # Rust: idx = (x >= edge).sum() / n
        # Numpy: searchsorted gives the count of edges <= x.
        # Because Rust uses `x >= edge`, equal cases count. Use side='right'.
        return np.searchsorted(edges, x, side="right").astype(np.float64) / float(
            len(edges)
        )
    if token == "yeo_johnson":
        lam = params[0] if params else -50.0  # default sentinel (won't match)
        if lam <= -49.0:  # caller error fallback in runtime → Identity
            return x
        return yeo_johnson(x, lam)
    # Stacked variants we don't need yet — keep as identity fallback.
    return x


def load_transforms() -> tuple[list[str], list[list[float]]]:
    """Parse the auto_transforms TSV → (tokens[372], params[372])."""
    transforms = ["identity"] * N_FEATURES
    params = [[] for _ in range(N_FEATURES)]
    with open(TRANSFORM_TSV) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            idx = int(row["feat_idx"])
            if idx >= N_FEATURES:
                continue
            transforms[idx] = row["best_transform"]
            csv_params = row.get("params_csv", "") or ""
            if csv_params.strip():
                params[idx] = [float(v) for v in csv_params.split(",")]
    return transforms, params


# -----------------------------------------------------------------------------

def load_mask() -> np.ndarray:
    """Per-feature bool array: True = pin_geq0 (w ≥ 0), False = free."""
    pin = np.zeros(N_FEATURES, dtype=bool)
    with open(MASK_TSV) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            idx = int(row["feat_idx"])
            if idx < N_FEATURES:
                pin[idx] = row["sign_mask"] == "pin_geq0"
    return pin


def load_group(basename: str, target_col: str = "human_score") -> tuple[np.ndarray, np.ndarray]:
    p = TRAIN_DIR / f"{basename}.parquet"
    t = pq.read_table(
        p, columns=[f"f{i}" for i in range(N_FEATURES)] + [target_col]
    )
    cols = [
        np.asarray(t[f"f{i}"].combine_chunks().to_numpy(), dtype=np.float64)
        for i in range(N_FEATURES)
    ]
    X = np.column_stack(cols)
    y = np.asarray(t[target_col].combine_chunks().to_numpy(), dtype=np.float64)
    keep = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    return X[keep], y[keep]


def minmax01(y: np.ndarray) -> np.ndarray:
    lo, hi = np.quantile(y, 0.001), np.quantile(y, 0.999)
    return np.clip((y - lo) / max(hi - lo, 1e-9), 0.0, 1.0)


def apply_per_feature(
    X: np.ndarray, transforms: list[str], params: list[list[float]]
) -> np.ndarray:
    """Apply each feature's transform column-wise. Drops rows that become
    non-finite after shaping (rare; usually a log on a zero feature)."""
    Y = np.empty_like(X, dtype=np.float64)
    for i in range(N_FEATURES):
        Y[:, i] = apply_transform(transforms[i], params[i], X[:, i]).astype(np.float64)
    return Y


def fit_monotone_pchip_payload(
    preds: np.ndarray, targets: np.ndarray, n_bins: int = 18
) -> bytes | None:
    """Mirror zensim-validate's fit_monotone_spline:
    Bin predictions into n_bins quantile bins → median (pred, target) per bin
    → keep strictly-monotone subset → emit `u32 n_knots + (f32 x, f32 y) × n_knots`.
    """
    n = min(len(preds), len(targets))
    if n < 4 or n_bins < 2:
        return None
    # Direction: positive cov → increasing; negative cov → decreasing.
    mean_p = float(preds[:n].mean())
    mean_t = float(targets[:n].mean())
    cov = float(((preds[:n] - mean_p) * (targets[:n] - mean_t)).sum())
    decreasing = cov < 0.0

    # Sort by predicted value, take quantile bins, take median per bin.
    order = np.argsort(preds[:n], kind="stable")
    bin_size = (n + n_bins - 1) // n_bins
    raw_knots: list[tuple[float, float]] = []
    for start in range(0, n, bin_size):
        end = min(start + bin_size, n)
        bin_idx = order[start:end]
        if len(bin_idx) == 0:
            continue
        p_med = float(np.median(preds[bin_idx]))
        t_med = float(np.median(targets[bin_idx]))
        raw_knots.append((p_med, t_med))
    if not raw_knots:
        return None

    knots: list[tuple[float, float]] = [raw_knots[0]]
    for x, y in raw_knots[1:]:
        last_x, last_y = knots[-1]
        if x <= last_x + 1e-6:
            continue
        y_ok = (y < last_y) if decreasing else (y > last_y)
        if y_ok:
            knots.append((x, y))
    if len(knots) < 2:
        return None

    payload = struct.pack("<I", len(knots))
    for x, y in knots:
        payload += struct.pack("<ff", float(x), float(y))
    return payload


def find_baker() -> Path:
    candidates = [
        Path.home() / "work/zen/zenanalyze/target/release/zenpredict-bake",
        Path.home() / "work/zen/zenanalyze/target/release/zenpredict",
        Path.cwd() / "target/release/zenpredict-bake",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(
        f"zenpredict-bake binary not found in any of {candidates}. "
        "Build with: cd ~/work/zen/zenanalyze && cargo build --release --bin zenpredict-bake -p zenpredict-bake"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("/mnt/v/output/zensim/bakes/v02_bvls_shaped_2026-05-28.bin"),
        help="Output ZNPR v3 bake path.",
    )
    ap.add_argument(
        "--max-iter",
        type=int,
        default=4000,
        help="BVLS max iterations (default 4000; 2000 was the MVP default).",
    )
    ap.add_argument(
        "--no-shaping",
        action="store_true",
        help="Skip per-feature transforms (control — should reproduce MVP).",
    )
    ap.add_argument(
        "--with-konjnd",
        action="store_true",
        help="Include konjnd-dense-norm training group at weight 0.3. "
        "Earlier scripts excluded it; the normalized variant has a "
        "MOS-scaled `human_score` column that's safe to mix.",
    )
    ap.add_argument(
        "--cid22-weight",
        type=float,
        default=None,
        help="Override cid22_train group weight (default 1.5). Raising to "
        "3.0+ trades held-out KADID/TID for CID22 rank.",
    )
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("== v02-bvls-shaped (V0_2 simplicity + 372 features + Yeo-Johnson/etc. + BVLS) ==")
    pin = load_mask()
    n_pin = int(pin.sum())
    print(f"sign mask: {n_pin} pin_geq0, {N_FEATURES - n_pin} free")

    if args.no_shaping:
        transforms = ["identity"] * N_FEATURES
        tparams: list[list[float]] = [[] for _ in range(N_FEATURES)]
        print("transforms: ALL identity (--no-shaping)")
    else:
        transforms, tparams = load_transforms()
        token_counts: dict[str, int] = {}
        for tok in transforms:
            token_counts[tok] = token_counts.get(tok, 0) + 1
        print(
            "transforms loaded from auto_transforms TSV: "
            + ", ".join(f"{k}={v}" for k, v in sorted(token_counts.items()))
        )

    # ----- Load training groups, apply shaping, weight by group -----
    groups = GROUPS_WITH_KONJND if args.with_konjnd else GROUPS_DEFAULT
    if args.cid22_weight is not None:
        groups = [
            (n, b, args.cid22_weight if n == "cid22_train" else w, tc)
            for (n, b, w, tc) in groups
        ]
    print()
    Xs, ys, ws = [], [], []
    for name, base, gw, tcol in groups:
        X, y = load_group(base, tcol)
        # Standardize MOS-style targets to [0,1] per group for the LS fit.
        y01 = minmax01(y)
        Xs.append(X)
        ys.append(y01)
        ws.append(np.full(len(y), gw))
        print(
            f"  {name:14s} {len(y):>7d} rows  target={tcol:14s} "
            f"raw=[{y.min():+.3f},{y.max():+.3f}] → [0,1]  train_w={gw}"
        )
    X = np.vstack(Xs)
    y = np.concatenate(ys)
    w = np.concatenate(ws)
    print(f"total: {len(y)} rows")

    # Apply per-feature transforms.
    print()
    print(f"applying per-feature transforms ({len(transforms)} features)...")
    Xs_shaped = apply_per_feature(X, transforms, tparams)

    # Standardize the SHAPED features.
    mu = Xs_shaped.mean(axis=0)
    sd = Xs_shaped.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Xs_z = (Xs_shaped - mu) / sd

    # ----- BVLS with sign mask + bias column -----
    A = np.hstack([Xs_z, np.ones((len(y), 1))])
    sw = np.sqrt(w)
    A_w = A * sw[:, None]
    y_w = y * sw

    lo = np.full(N_FEATURES + 1, -np.inf)
    hi = np.full(N_FEATURES + 1, np.inf)
    lo[:N_FEATURES] = np.where(pin, 0.0, -np.inf)

    print(f"\nfitting bounded LS (BVLS, max_iter={args.max_iter})...")
    res = lsq_linear(
        A_w, y_w, bounds=(lo, hi), method="bvls", verbose=1, max_iter=args.max_iter
    )
    coef = res.x[:N_FEATURES]
    bias = float(res.x[N_FEATURES])
    n_active = int(np.sum(np.abs(coef) > 1e-6))
    print(
        f"  cost={res.cost:.4f}  n_iter={res.nit}  "
        f"active={n_active}/{N_FEATURES} (~{n_active/N_FEATURES*100:.0f}%)"
    )

    # ----- Evaluate raw fit (before spline) on held-out corpora -----
    print(f"\n{'corpus':10s} {'n':>5s}  {'rawSROCC':>9s}  | v47:  {'SROCC':>8s}  | Δ-SROCC")
    print("-" * 70)
    holdout_raw_preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for v in HOLDOUTS:
        p = VAL_DIR / f"{v}.parquet"
        t = pq.read_table(
            p, columns=[f"f{i}" for i in range(N_FEATURES)] + ["human_score"]
        )
        cols = [
            np.asarray(t[f"f{i}"].combine_chunks().to_numpy(), dtype=np.float64)
            for i in range(N_FEATURES)
        ]
        Xv = np.column_stack(cols)
        yv = np.asarray(t["human_score"].combine_chunks().to_numpy(), dtype=np.float64)
        keep = np.isfinite(yv) & np.all(np.isfinite(Xv), axis=1)
        Xv, yv = Xv[keep], yv[keep]
        Xv_shaped = apply_per_feature(Xv, transforms, tparams)
        Xv_z = (Xv_shaped - mu) / sd
        pred = Xv_z @ coef + bias  # SCORE-shaped (target was [0,1] → high = better)
        srocc = float(spearmanr(pred, yv).statistic)
        ref_s, _ = V47_REF[v]
        delta = srocc - ref_s
        print(f"{v:10s} {len(yv):5d}  {srocc:9.4f}  | v47:  {ref_s:8.4f}  | {delta:+.4f}")
        holdout_raw_preds[v] = (pred, yv)

    # ----- Fit PCHIP spline on multiband_anchor -----
    print(f"\nfitting PCHIP dial spline on multiband_anchor_dial100.parquet...")
    t = pq.read_table(
        ANCHOR_PQ,
        columns=[f"f{i}" for i in range(N_FEATURES)] + ["human_score"],
    )
    cols = [
        np.asarray(t[f"f{i}"].combine_chunks().to_numpy(), dtype=np.float64)
        for i in range(N_FEATURES)
    ]
    Xa = np.column_stack(cols)
    ya = np.asarray(t["human_score"].combine_chunks().to_numpy(), dtype=np.float64)
    keep = np.isfinite(ya) & np.all(np.isfinite(Xa), axis=1)
    Xa, ya = Xa[keep], ya[keep]
    Xa_shaped = apply_per_feature(Xa, transforms, tparams)
    Xa_z = (Xa_shaped - mu) / sd
    raw_a = Xa_z @ coef + bias
    print(
        f"  anchor n={len(raw_a)}  raw pred [{raw_a.min():.4f}, {raw_a.max():.4f}]  "
        f"target [{ya.min():.2f}, {ya.max():.2f}]"
    )
    payload = fit_monotone_pchip_payload(raw_a, ya, n_bins=18)
    if payload is None:
        print(
            "  WARNING: fit_monotone_spline returned None — bake will ship WITHOUT a dial spline."
        )
    else:
        n_knots = struct.unpack("<I", payload[:4])[0]
        print(f"  PCHIP spline fit: {n_knots} knots, payload {len(payload)} bytes")

    # ----- Emit ZNPR v3 JSON bake -----
    print(f"\nemitting ZNPR v3 bake → {args.out}")

    metadata = [
        {
            "key": "zentrain.feature_transforms",
            "type": "utf8",
            "text": "\n".join(transforms),
        },
        {
            "key": "zentrain.feature_transform_params",
            "type": "utf8",
            "text": "\n".join(
                ",".join(f"{p}" for p in row) if row else ""
                for row in tparams
            ),
        },
    ]
    if payload is not None:
        metadata.append(
            {
                "key": "zentrain.output_calibration_spline",
                "type": "bytes",
                "hex": payload.hex(),
            }
        )

    # 1-layer bake: 372 → 1, Identity activation. Weights are coefs (row-major
    # input-major: W[i] is contribution of feature i to the single output).
    req = {
        "schema_hash": 0,
        "flags": 0,
        "scaler_mean": [float(v) for v in mu.astype(np.float32)],
        "scaler_scale": [float(v) for v in sd.astype(np.float32)],
        "layers": [
            {
                "in_dim": N_FEATURES,
                "out_dim": 1,
                "activation": "identity",
                "dtype": "f32",
                "weights": [float(v) for v in coef.astype(np.float32)],
                "biases": [float(bias)],
            }
        ],
        "metadata": metadata,
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(req, f)
        json_path = Path(f.name)
    baker = find_baker()
    try:
        # zenpredict-bake binary is the legacy alias; the new CLI is `zenpredict bake`.
        if baker.name == "zenpredict":
            cmd = [str(baker), "bake", str(json_path), str(args.out)]
        else:
            cmd = [str(baker), str(json_path), str(args.out)]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.returncode != 0:
            print(f"zenpredict-bake failed (exit {result.returncode})", file=sys.stderr)
            return result.returncode
    finally:
        json_path.unlink(missing_ok=True)

    size = args.out.stat().st_size
    print(f"baked {args.out} ({size:,} bytes)  t={time.time()-t0:.1f}s")
    print(f"  n_active = {n_active}/{N_FEATURES} ({n_active/N_FEATURES*100:.1f}%)")
    print(f"  per-feature transforms in {sum(1 for t in transforms if t != 'identity')} features (rest identity)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
