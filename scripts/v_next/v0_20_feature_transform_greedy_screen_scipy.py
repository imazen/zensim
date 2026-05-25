#!/usr/bin/env python3
"""V_20 input-shaping greedy screen — scipy-based Yeo-Johnson MLE.

Mirrors `v0_20_feature_transform_greedy_screen.py` but replaces the
in-process golden-section λ search (clamped to [_YJ_GRID_LO,
_YJ_GRID_HI]) with `scipy.stats.yeojohnson_normmax`, which uses
data-driven bounds that avoid over/underflow and effectively performs
unconstrained MLE on the marginal feature distribution.

Why this exists (task #214 follow-up, 2026-05-25):

The wider-grid screen at ±5 still pinned 361/372 features at the
λ = −5 boundary. The widest-grid screen at ±10 still pinned 349/372
features at λ = −10. The features want even more extreme negative λ
than the in-process golden-section search permits, even with the
widest reasonable hard clamp. Switching to scipy's data-driven bounds
(`yeojohnson_normmax`) reports the true MLE λ — for our heavy-positive-
tail features that's commonly λ ≈ −150 to −300.

The output schema matches the prior screen so `--auto-transforms` and
all downstream tooling continue to work. The only change is that the
`params_csv` column may now contain λ values far outside [-10, 10].

The runtime side (`zenpredict::FeatureTransform::YeoJohnson` /
`Predictor::predict_transformed`) handles arbitrary λ correctly — it
applies the same closed-form formula. Already verified bit-identical
at λ = -212 to scipy.stats.yeojohnson.

Usage:
  python3 scripts/v_next/v0_20_feature_transform_greedy_screen_scipy.py \\
    --features-parquet /mnt/v/zen/zensim-training/canonical-2026-05-21/train/safesyn.parquet \\
    --target-column mix_cv40_iw60 \\
    --out benchmarks/yeo_johnson_screen_widest_2026-05-25/screen_results.tsv \\
    --per-transform-out benchmarks/yeo_johnson_screen_widest_2026-05-25/per_transform.tsv \\
    --min-lift 0.005
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path

import numpy as np

# Import building blocks from the canonical screen.
sys.path.insert(0, str(Path(__file__).parent))
from v0_20_feature_transform_greedy_screen import (  # noqa: E402
    TRANSFORMS,
    _block_for_feat_idx,
    load_features_csv,
    load_features_parquet,
    safe_pearson,
    safe_spearman,
    sweep_for,
    t_yeo_johnson,
)


def fit_yj_lambda_scipy(col: np.ndarray) -> float:
    """Data-driven unconstrained MLE Yeo-Johnson λ via scipy.

    Returns 1.0 (identity) when the column has insufficient signal
    (matches the in-process fit's degenerate path).
    """
    from scipy.stats import yeojohnson_normmax

    finite = col[np.isfinite(col)]
    if finite.size < 2:
        return 1.0
    var = float(finite.var())
    if var < 1e-10:
        return 1.0
    # scipy's normmax computes data-driven bounds that avoid
    # over/underflow on the transformed variance. Effectively
    # unconstrained MLE.
    try:
        lam = float(yeojohnson_normmax(finite))
    except Exception:
        return 1.0
    if not math.isfinite(lam):
        return 1.0
    return lam


def sweep_for_scipy(name: str, col: np.ndarray) -> list[list[float]]:
    """Like the canonical `sweep_for`, but YJ uses scipy's MLE."""
    if name == "yeo_johnson":
        valid = col[np.isfinite(col)]
        if valid.size == 0:
            return [[]]
        lam = fit_yj_lambda_scipy(valid)
        return [[lam]]
    return sweep_for(name, col)


def screen_one_feature_all_transforms_scipy(
    i: int,
    feat_col: np.ndarray,
    mos: np.ndarray,
) -> tuple[dict, dict[str, tuple[float, float, list[float]]]]:
    """Per-feature screen that uses scipy's unconstrained YJ MLE.

    Layout matches the canonical
    `screen_one_feature_all_transforms` so the emitted TSVs are
    schema-compatible with `--auto-transforms`."""
    finite = feat_col[np.isfinite(feat_col)]
    feat_min = float(finite.min()) if finite.size else 0.0
    baseline, n = safe_pearson(feat_col, mos)
    baseline_spearman = safe_spearman(feat_col, mos)
    best_lift = 0.0
    best_token = "identity"
    best_params: list[float] = []
    best_corr = baseline
    per_transform: dict[str, tuple[float, float, list[float]]] = {
        "identity": (baseline, 0.0, []),
    }
    for token, fn in TRANSFORMS.items():
        if token == "identity":
            continue
        if token == "log" and feat_min <= 0.0:
            per_transform[token] = (float("nan"), float("nan"), [])
            continue
        if token == "log1p" and feat_min <= -1.0:
            per_transform[token] = (float("nan"), float("nan"), [])
            continue
        best_corr_this = float("-inf")
        best_params_this: list[float] = []
        for params in sweep_for_scipy(token, feat_col):
            tx = fn(feat_col, params)
            corr, _ = safe_pearson(tx, mos)
            if math.isnan(corr):
                continue
            if corr > best_corr_this:
                best_corr_this = corr
                best_params_this = params
        if math.isfinite(best_corr_this):
            lift_this = best_corr_this - baseline
            per_transform[token] = (best_corr_this, lift_this, best_params_this)
            if lift_this > best_lift:
                best_lift = lift_this
                best_token = token
                best_params = best_params_this
                best_corr = best_corr_this
        else:
            per_transform[token] = (float("nan"), float("nan"), [])
    rec = {
        "feat_idx": i,
        "best_transform": best_token,
        "params_csv": ",".join(f"{v:.6g}" for v in best_params) if best_params else "",
        "baseline_pearson": baseline,
        "transformed_pearson": best_corr,
        "lift": best_lift,
        "baseline_spearman": baseline_spearman,
        "n_samples": n,
    }
    return rec, per_transform


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features-parquet", type=Path)
    ap.add_argument("--features-csv", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--target-column", type=str, default="human_score")
    ap.add_argument("--min-lift", type=float, default=0.005)
    ap.add_argument("--per-transform-out", type=Path, default=None)
    args = ap.parse_args()

    if not args.features_parquet and not args.features_csv:
        print("ERROR: pass --features-parquet or --features-csv", file=sys.stderr)
        return 2

    print(
        "YJ MLE via scipy.stats.yeojohnson_normmax (data-driven bounds, "
        "effectively unconstrained)",
        file=sys.stderr,
    )

    if args.features_parquet:
        mos, features, feat_names = load_features_parquet(
            args.features_parquet, target_column=args.target_column
        )
    else:
        mos, features, feat_names = load_features_csv(
            args.features_csv, target_column=args.target_column
        )

    n_features = features.shape[1]
    rows: list[dict] = []
    per_transform_rows: list[tuple[int, str, float, float, str]] = []
    t0 = time.perf_counter()
    for i in range(n_features):
        rec, per_tx = screen_one_feature_all_transforms_scipy(i, features[:, i], mos)
        rows.append(rec)
        for tok, (corr, lift, params) in per_tx.items():
            params_csv = ",".join(f"{v:.6g}" for v in params) if params else ""
            per_transform_rows.append(
                (
                    i,
                    tok,
                    corr if math.isfinite(corr) else float("nan"),
                    lift if math.isfinite(lift) else float("nan"),
                    params_csv,
                )
            )
    print(
        f"screened {n_features} features in {time.perf_counter() - t0:.1f}s",
        file=sys.stderr,
    )

    rows.sort(key=lambda r: -r["lift"])

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
    if args.per_transform_out:
        args.per_transform_out.parent.mkdir(parents=True, exist_ok=True)
        with args.per_transform_out.open("w") as f:
            w = csv.writer(f, delimiter="\t")
            w.writerow(["feat_idx", "transform", "pearson", "lift", "params_csv"])
            for (idx, tok, corr, lift, params_csv) in per_transform_rows:
                w.writerow([
                    idx, tok,
                    f"{corr:.6f}" if math.isfinite(corr) else "nan",
                    f"{lift:.6f}" if math.isfinite(lift) else "nan",
                    params_csv,
                ])
        print(
            f"wrote {len(per_transform_rows)} per-transform rows to {args.per_transform_out}",
            file=sys.stderr,
        )
    winners = [r for r in rows if r["lift"] >= args.min_lift]
    print(
        f"wrote {len(rows)} rows to {args.out}; "
        f"{len(winners)} features clear --min-lift {args.min_lift}",
        file=sys.stderr,
    )

    # Per-block summary
    print("\nper-block: feature counts where best_transform != identity:", file=sys.stderr)
    block_totals: dict[str, int] = {}
    block_wins: dict[str, int] = {}
    block_yj_wins: dict[str, int] = {}
    for r in rows:
        b = _block_for_feat_idx(r["feat_idx"])
        block_totals[b] = block_totals.get(b, 0) + 1
        if r["best_transform"] != "identity":
            block_wins[b] = block_wins.get(b, 0) + 1
        if r["best_transform"] == "yeo_johnson":
            block_yj_wins[b] = block_yj_wins.get(b, 0) + 1
    print("block     | total | any_transform_wins | yj_wins", file=sys.stderr)
    for b in ("basic", "peak", "masked", "iw_pool"):
        tot = block_totals.get(b, 0)
        any_w = block_wins.get(b, 0)
        yj_w = block_yj_wins.get(b, 0)
        print(f"{b:<9} | {tot:>5} | {any_w:>18} | {yj_w:>7}", file=sys.stderr)

    # YJ-specific summary
    yj_lifts: list[float] = []
    yj_lambdas: list[float] = []
    for (_idx, tok, _corr, lift, params_csv) in per_transform_rows:
        if tok == "yeo_johnson" and math.isfinite(lift):
            yj_lifts.append(lift)
            if params_csv:
                try:
                    yj_lambdas.append(float(params_csv.split(",")[0]))
                except ValueError:
                    pass
    yj_arr = np.array(yj_lifts, dtype=np.float64)
    if yj_arr.size:
        print(
            f"\nyeo_johnson lift across {yj_arr.size} features: "
            f"mean={yj_arr.mean():.4f}, median={float(np.median(yj_arr)):.4f}, "
            f"max={yj_arr.max():.4f}, "
            f"n>=0.005={(yj_arr >= 0.005).sum()}, "
            f"n>=0.02={(yj_arr >= 0.02).sum()}, "
            f"n>=0.05={(yj_arr >= 0.05).sum()}",
            file=sys.stderr,
        )
    lam_arr = np.array(yj_lambdas, dtype=np.float64)
    if lam_arr.size:
        print(
            f"\nλ distribution (scipy MLE): "
            f"min={lam_arr.min():.2f}, median={float(np.median(lam_arr)):.2f}, "
            f"max={lam_arr.max():.2f}",
            file=sys.stderr,
        )
        # buckets
        print("λ histogram:", file=sys.stderr)
        buckets = [(-np.inf, -1000), (-1000, -300), (-300, -100), (-100, -30),
                   (-30, -10), (-10, -5), (-5, -2), (-2, 0), (0, 2), (2, 5),
                   (5, np.inf)]
        for lo, hi in buckets:
            c = int(((lam_arr >= lo) & (lam_arr < hi)).sum())
            print(f"  [{lo:+g}, {hi:+g}): {c}", file=sys.stderr)

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
