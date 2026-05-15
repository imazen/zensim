#!/usr/bin/env python3
"""V_20 input-shaping greedy screen.

For each of the 228 (or 300 with masked) features × 6 FeatureTransform
candidates, computes |Pearson(transform(feat), human_score)| and ranks
by lift vs the Identity baseline.

Per the V_20 design doc (`benchmarks/v0_20_v0_21_design_2026-05-14.md`),
training MLP variants for all 1600 (feature, transform) cells costs
~640 GPU-hours. This screen filters to ~50-200 candidates by Pearson
gap > 0.005, taking ~minutes of CPU.

## Inputs

Tabular features CSV (or parquet) with columns `ref_basename`,
`human_score`, `f0..f<N-1>`. Defaults to the 218k+e1=340k clean
safe-synthetic features at the 2026-05-07 path.

## Output

TSV with one row per (feature, best_transform) candidate where the
lift clears `--min-lift` (default 0.005). Sorted by lift desc.

Columns:
- feat_idx
- best_transform_token
- params_csv          (empty for non-parameterized variants)
- baseline_pearson    (|Pearson(feat, mos)|)
- transformed_pearson (best across the variant's param sweep)
- lift                (transformed − baseline)
- baseline_spearman   (rank-correlation sanity check)
- n_samples           (count of finite feat values)

## Param sweeps per variant

| Variant | Param schedule |
|---|---|
| clip_then_log1p | ε = [p5, p10, p25, p50, p75] of the feature distribution |
| winsor_p99 | pairs: (p1, p99), (p5, p95), (p10, p90), (p25, p75) |
| quantile_bins | one 8-bin config from feature distribution percentiles |

Non-parameterized variants (identity / log / log1p / signed_log1p /
signed_sqrt / signed_cbrt) have a single config each.

Identity baseline is always included so per-feature winners can stay
Identity if no transform helps.

## Usage

  python3 scripts/v_next/v0_20_feature_transform_greedy_screen.py \\
    --features-parquet /mnt/v/zen/zensim-training/2026-05-07/v06-features/safe_synth_ssim2_features.parquet \\
    --out benchmarks/v0_20_greedy_screen_2026-05-15.tsv \\
    --min-lift 0.005
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np


# -- transforms -----------------------------------------------------------------


def t_identity(x: np.ndarray, _params: list[float]) -> np.ndarray:
    return x


def t_log(x: np.ndarray, _params: list[float]) -> np.ndarray:
    # ln(x). Only valid for strictly positive; NaN below 0.
    out = np.full_like(x, np.nan)
    pos = x > 0
    out[pos] = np.log(x[pos])
    return out


def t_log1p(x: np.ndarray, _params: list[float]) -> np.ndarray:
    out = np.full_like(x, np.nan)
    valid = x >= 0
    out[valid] = np.log1p(x[valid])
    return out


def t_signed_log1p(x: np.ndarray, _params: list[float]) -> np.ndarray:
    return np.sign(x) * np.log1p(np.abs(x))


def t_signed_sqrt(x: np.ndarray, _params: list[float]) -> np.ndarray:
    return np.sign(x) * np.sqrt(np.abs(x))


def t_signed_cbrt(x: np.ndarray, _params: list[float]) -> np.ndarray:
    return np.sign(x) * np.cbrt(np.abs(x))


def t_clip_then_log1p(x: np.ndarray, params: list[float]) -> np.ndarray:
    eps = params[0] if params else 0.0
    return np.log1p(np.maximum(0.0, x - eps))


def t_winsor_p99(x: np.ndarray, params: list[float]) -> np.ndarray:
    if len(params) < 2:
        return x
    return np.clip(x, params[0], params[1])


def t_quantile_bins(x: np.ndarray, params: list[float]) -> np.ndarray:
    if len(params) < 2:
        return x
    # Number of edges = N; outputs are bin_idx/N
    edges = np.asarray(params, dtype=np.float64)
    idx = np.zeros_like(x, dtype=np.float64)
    for edge in edges:
        idx += (x >= edge).astype(np.float64)
    return idx / len(edges)


TRANSFORMS: dict[str, Callable[[np.ndarray, list[float]], np.ndarray]] = {
    "identity": t_identity,
    "log": t_log,
    "log1p": t_log1p,
    "signed_log1p": t_signed_log1p,
    "signed_sqrt": t_signed_sqrt,
    "signed_cbrt": t_signed_cbrt,
    "clip_then_log1p": t_clip_then_log1p,
    "winsor_p99": t_winsor_p99,
    "quantile_bins": t_quantile_bins,
}


# -- param sweeps ---------------------------------------------------------------


def sweep_for(name: str, col: np.ndarray) -> list[list[float]]:
    """List of param-vector candidates for a given transform name +
    feature column. Single empty list for non-parameterized variants."""
    valid = col[np.isfinite(col)]
    if valid.size == 0:
        return [[]]
    if name == "clip_then_log1p":
        # ε = percentiles of the feature's positive distribution. We
        # want ε at the noise-floor — try p5..p75 spread to find where
        # subtracting noise yields the cleanest log-shape.
        pcts = [5, 10, 25, 50, 75]
        return [[float(np.percentile(valid, p))] for p in pcts]
    if name == "winsor_p99":
        # (lo, hi) clamp pairs from feature percentiles.
        bounds = [(1, 99), (5, 95), (10, 90), (25, 75)]
        return [
            [float(np.percentile(valid, lo)), float(np.percentile(valid, hi))]
            for (lo, hi) in bounds
        ]
    if name == "quantile_bins":
        # 8-bin edges from percentiles 12.5, 25, 37.5, ..., 87.5
        edges = [float(np.percentile(valid, p)) for p in [12.5, 25, 37.5, 50, 62.5, 75, 87.5]]
        return [edges]
    # Non-parameterized
    return [[]]


# -- I/O ------------------------------------------------------------------------


def load_features_parquet(path: Path):
    """Load a parquet file with columns `human_score, f0..f<N-1>`.

    Returns (mos: np.ndarray[n], features: np.ndarray[n, N], feat_names).
    """
    try:
        import pyarrow.parquet as pq  # noqa: F401
        import pyarrow as pa  # noqa: F401
    except ImportError:
        print("ERROR: pyarrow not installed. pip install pyarrow", file=sys.stderr)
        sys.exit(2)
    import pyarrow.parquet as pq

    t0 = time.perf_counter()
    tbl = pq.read_table(str(path))
    n = tbl.num_rows
    cols = tbl.column_names
    if "human_score" not in cols:
        raise RuntimeError(f"missing human_score column in {path}")
    feat_cols = [c for c in cols if c.startswith("f") and c[1:].isdigit()]
    feat_cols.sort(key=lambda c: int(c[1:]))
    mos = tbl["human_score"].to_numpy().astype(np.float64)
    arrs = []
    for c in feat_cols:
        arrs.append(tbl[c].to_numpy().astype(np.float64))
    features = np.stack(arrs, axis=1)
    print(
        f"loaded {path.name}: n={n} rows × {len(feat_cols)} features "
        f"in {time.perf_counter() - t0:.1f}s",
        file=sys.stderr,
    )
    return mos, features, feat_cols


def load_features_csv(path: Path):
    """Slower CSV path. Reads everything into memory."""
    t0 = time.perf_counter()
    feat_cols = None
    mos_list = []
    feat_rows = []
    with path.open() as f:
        r = csv.reader(f)
        header = next(r)
        feat_cols = [c for c in header if c.startswith("f") and c[1:].isdigit()]
        feat_idx = [header.index(c) for c in feat_cols]
        mos_idx = header.index("human_score")
        for row in r:
            try:
                mos = float(row[mos_idx])
                feats = [float(row[i]) for i in feat_idx]
            except (ValueError, IndexError):
                continue
            mos_list.append(mos)
            feat_rows.append(feats)
    mos = np.array(mos_list, dtype=np.float64)
    features = np.array(feat_rows, dtype=np.float64)
    print(
        f"loaded {path.name}: n={len(mos)} rows × {len(feat_cols)} features "
        f"in {time.perf_counter() - t0:.1f}s",
        file=sys.stderr,
    )
    return mos, features, feat_cols


# -- core analysis --------------------------------------------------------------


def safe_pearson(a: np.ndarray, b: np.ndarray) -> tuple[float, int]:
    """|Pearson(a, b)| dropping non-finite rows. Returns (corr, n)."""
    mask = np.isfinite(a) & np.isfinite(b)
    n = int(mask.sum())
    if n < 3:
        return (float("nan"), n)
    aa = a[mask]
    bb = b[mask]
    sa = aa.std()
    sb = bb.std()
    if sa < 1e-12 or sb < 1e-12:
        return (0.0, n)
    return (float(abs(np.corrcoef(aa, bb)[0, 1])), n)


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    """|Spearman| via ranks then Pearson. Drops non-finite rows."""
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return float("nan")
    aa = a[mask]
    bb = b[mask]
    # ranks via argsort-argsort
    ra = np.argsort(np.argsort(aa)).astype(np.float64)
    rb = np.argsort(np.argsort(bb)).astype(np.float64)
    if ra.std() < 1e-12 or rb.std() < 1e-12:
        return 0.0
    return float(abs(np.corrcoef(ra, rb)[0, 1]))


def screen_one_feature(
    i: int,
    feat_col: np.ndarray,
    mos: np.ndarray,
) -> dict:
    """For one feature column, sweep all transforms × param configs and
    return the best transform's (lift, transform_token, params, etc.).

    Training-safety gates (added 2026-05-15 after first run hit NaN
    loss): some transforms produce NaN on real training data when the
    Pearson screen accepts them via its NaN-dropping rule. The gates:

    - `log` requires `min(feat) > 0` (strictly positive). Mathematically
      `log(x ≤ 0) = NaN`, and the trainer's apply path doesn't drop
      bad rows.
    - `log1p` requires `min(feat) > -1` (anything > -1, since
      `log1p(0) = 0`).

    Other transforms (`signed_*`, `clip_then_log1p`, `winsor_p99`,
    `quantile_bins`) are safe across the full real line."""
    finite = feat_col[np.isfinite(feat_col)]
    feat_min = float(finite.min()) if finite.size else 0.0
    baseline, n = safe_pearson(feat_col, mos)
    baseline_spearman = safe_spearman(feat_col, mos)
    best_lift = 0.0
    best_token = "identity"
    best_params: list[float] = []
    best_corr = baseline
    for token, fn in TRANSFORMS.items():
        # Training-safety gates
        if token == "log" and feat_min <= 0.0:
            continue
        if token == "log1p" and feat_min <= -1.0:
            continue
        for params in sweep_for(token, feat_col):
            tx = fn(feat_col, params)
            corr, _ = safe_pearson(tx, mos)
            if math.isnan(corr):
                continue
            lift = corr - baseline
            if lift > best_lift:
                best_lift = lift
                best_token = token
                best_params = params
                best_corr = corr
    return {
        "feat_idx": i,
        "best_transform": best_token,
        "params_csv": ",".join(f"{v:.6g}" for v in best_params) if best_params else "",
        "baseline_pearson": baseline,
        "transformed_pearson": best_corr,
        "lift": best_lift,
        "baseline_spearman": baseline_spearman,
        "n_samples": n,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-parquet", type=Path)
    ap.add_argument("--features-csv", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--min-lift",
        type=float,
        default=0.005,
        help="emit rows where transformed_pearson - baseline_pearson >= this. Default 0.005.",
    )
    args = ap.parse_args()
    if not args.features_parquet and not args.features_csv:
        print("ERROR: pass --features-parquet or --features-csv", file=sys.stderr)
        return 2

    if args.features_parquet:
        mos, features, feat_names = load_features_parquet(args.features_parquet)
    else:
        mos, features, feat_names = load_features_csv(args.features_csv)

    n_features = features.shape[1]
    rows: list[dict] = []
    t0 = time.perf_counter()
    for i in range(n_features):
        rec = screen_one_feature(i, features[:, i], mos)
        rows.append(rec)
    print(
        f"screened {n_features} features in {time.perf_counter() - t0:.1f}s",
        file=sys.stderr,
    )

    # Sort by lift desc
    rows.sort(key=lambda r: -r["lift"])

    # Emit all rows (informational) and a filtered "winners" subset.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(
            [
                "feat_idx",
                "best_transform",
                "params_csv",
                "baseline_pearson",
                "transformed_pearson",
                "lift",
                "baseline_spearman",
                "n_samples",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r["feat_idx"],
                    r["best_transform"],
                    r["params_csv"],
                    f"{r['baseline_pearson']:.6f}",
                    f"{r['transformed_pearson']:.6f}",
                    f"{r['lift']:.6f}",
                    f"{r['baseline_spearman']:.6f}",
                    r["n_samples"],
                ]
            )
    winners = [r for r in rows if r["lift"] >= args.min_lift]
    print(
        f"wrote {len(rows)} rows to {args.out}; "
        f"{len(winners)} features clear --min-lift {args.min_lift}",
        file=sys.stderr,
    )
    # Print top 20 to stderr for quick eyeball
    print("\ntop 20 by lift:", file=sys.stderr)
    print("feat | transform        | params                       | base_p | tx_p   | lift", file=sys.stderr)
    for r in rows[:20]:
        print(
            f"f{r['feat_idx']:<3} | {r['best_transform']:<16} | {r['params_csv']:<28} | "
            f"{r['baseline_pearson']:.4f} | {r['transformed_pearson']:.4f} | {r['lift']:+.4f}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
