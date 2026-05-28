#!/usr/bin/env python3
"""V0_2 methodology applied to the 372-feature set (#44 follow-up, 2026-05-28).

User question: "apply v02 methodology to 372 features then"

V0_2's recipe (2026-03):
  1. SAFESYN ONLY training (218k concordance-filtered, NO other corpora)
  2. Target = ssim2_gpu (a stable single-source metric, NOT a mix of MOS)
  3. All weights >= 0 (positive bounds; NM happened to find a positive optimum)
  4. Score-map: 100 * exp(-(a/100) * d^b), a=18, b=0.7 (fixed, NOT a learned spline)
  5. Optimizer: Nelder-Mead with 10 restarts maximizing SROCC (or Pearson proxy)

Today's stack adds:
  - 372 features (was 228) — basic + peak + masked + IW-pool blocks
  - The canonical-2026-05-21 safesyn has 196k rows after CID22 contamination purge
    (slightly smaller than V0_2's 218k pre-purge dataset)

This script applies V0_2's recipe to the 372-feature set in 4 cells:
  - safesyn / ssim2_gpu / pos-bounds / BVLS                  ← closest to V0_2
  - safesyn / ssim2_gpu / pos-bounds / BVLS / concordance-filter
  - safesyn / human_score / pos-bounds / BVLS                ← v47/v02-bvls's target
  - safesyn / ssim2_gpu / mask-bounds (300 ≥0, 72 free) / BVLS

Validates each on the 6 canonical val corpora with V0_2's score-map.

Uses BVLS instead of Nelder-Mead — BVLS finds the GLOBAL bounded-LS
optimum deterministically in seconds. Nelder-Mead at 372 dims would
take hours and converge to the same optimum (the loss landscape for
linear LS is convex). The V0_2 paper recipe used NM because the
infrastructure of 2026-03 didn't have BVLS wired in.
"""
from __future__ import annotations
import argparse, csv, json, os, struct, subprocess, sys, tempfile, time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import lsq_linear
from scipy.stats import spearmanr, pearsonr

REPO = Path(__file__).resolve().parent.parent.parent
TRAIN_DIR = Path("/mnt/v/zen/zensim-training/canonical-2026-05-21/train")
VAL_DIR = Path("/mnt/v/zen/zensim-training/canonical-2026-05-18/val")
MASK_TSV = REPO / "benchmarks/feature_sign_mask_2026-05-26.tsv"
ANCHOR_PQ = TRAIN_DIR / "multiband_anchor_dial100.parquet"

N_FEATURES = 372

V47_REF = {
    "cid22":  (0.8657, 0.512), "kadid":  (0.7933, 0.613), "tid": (0.7927, 0.577),
    "konjnd": (0.4185, 0.932), "aic3":   (0.7680, 0.620), "aic4": (0.8854, 0.481),
}
V02_REF_SROCC = {  # V0_2 measured today on canonical-2026-05-18 val/
    "cid22":  0.8676, "kadid":  0.8192, "tid":    0.8427,
    "konjnd": 0.4695, "aic3":   0.7962, "aic4":   0.9107,
}


def load_mask() -> np.ndarray:
    pin = np.zeros(N_FEATURES, dtype=bool)
    with open(MASK_TSV) as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            idx = int(row["feat_idx"])
            if idx < N_FEATURES:
                pin[idx] = row["sign_mask"] == "pin_geq0"
    return pin


def load_safesyn(target_col: str = "ssim2_gpu"):
    """Load safesyn + return (X, y_distance_form, ssim2_gpu, iwssim, cvvdp_score).
    The target `y` is converted to DISTANCE form: high = bad quality (matches
    the score-map and the positive-coef constraint direction).

    `ssim2_gpu` (range -739..+98, high=quality)  → `y = max(0, 100 - ssim2_gpu)` (distance, ≥0)
    `human_score` (range -7.39..+0.99, high=quality)  → `y = (max - human_score)` (still anti-corr w/ quality)
    """
    p = TRAIN_DIR / "safesyn.parquet"
    cols = [f"f{i}" for i in range(N_FEATURES)] + [target_col, "ssim2_gpu", "iwssim", "cvvdp_score"]
    cols = [c for c in cols if c]
    t = pq.read_table(p, columns=list(set(cols)))
    X = np.column_stack([np.asarray(t[f"f{i}"].combine_chunks().to_numpy(), dtype=np.float64) for i in range(N_FEATURES)])
    raw_y = np.asarray(t[target_col].combine_chunks().to_numpy(), dtype=np.float64)
    s = np.asarray(t["ssim2_gpu"].combine_chunks().to_numpy(), dtype=np.float64)
    iw = np.asarray(t["iwssim"].combine_chunks().to_numpy(), dtype=np.float64)
    cv = np.asarray(t["cvvdp_score"].combine_chunks().to_numpy(), dtype=np.float64)
    keep = np.isfinite(raw_y) & np.all(np.isfinite(X), axis=1) & np.isfinite(s) & np.isfinite(iw) & np.isfinite(cv)
    raw_y, X, s, iw, cv = raw_y[keep], X[keep], s[keep], iw[keep], cv[keep]
    # Convert to distance form so positive-coef BVLS direction matches.
    # SSIMULACRA2 caps at 100 (identity). 100 - ssim2_gpu is the standard
    # DSSIM-shaped distance. Clamp at 0 to avoid negative distances on
    # outlier scores > 100 (shouldn't happen but defensive).
    if target_col == "ssim2_gpu":
        y_dist = np.maximum(100.0 - raw_y, 0.0)
    elif target_col == "human_score":
        # safesyn human_score range [-7.39, 0.99], high = quality. Flip + offset:
        # y_dist = (max - human_score), >= 0 with high = bad.
        y_dist = float(raw_y.max()) - raw_y
    else:
        # Generic flip: y_dist = max - y so high = bad
        y_dist = float(raw_y.max()) - raw_y
    return X, y_dist, s, iw, cv


def concordance_filter(s: np.ndarray, m2: np.ndarray, n_buckets: int = 20) -> np.ndarray:
    """V0_2's concordance filter: keep pairs where ssim2 AND a second metric
    rank-agree. V0_2 used butteraugli; we approximate using `iwssim` (a
    structural metric of similar family) or `cvvdp_score` (perceptual,
    DIFFERENT family — closer to V0_2's butter intent).

    Simple proxy: bucket by ssim2 rank-percentile, bucket by m2 rank-pct,
    keep rows where the two bucket indices are within ±2 (i.e., the two
    metrics agree on coarse rank).
    """
    n = len(s)
    s_rank = np.argsort(np.argsort(s)) / n  # rank percentile
    m_rank = np.argsort(np.argsort(m2)) / n
    s_bucket = np.clip((s_rank * n_buckets).astype(int), 0, n_buckets - 1)
    m_bucket = np.clip((m_rank * n_buckets).astype(int), 0, n_buckets - 1)
    return np.abs(s_bucket - m_bucket) <= 2


def fit_bvls(X: np.ndarray, y: np.ndarray, pin: np.ndarray | None = None) -> tuple[np.ndarray, float, dict]:
    """Bounded LS: optionally apply per-feature ≥0 mask. Returns (coef, bias, info)."""
    n_feat = X.shape[1]
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    Xs = (X - mu) / sd
    A = np.hstack([Xs, np.ones((len(y), 1))])
    lo = np.full(n_feat + 1, -np.inf)
    hi = np.full(n_feat + 1, np.inf)
    if pin is not None:
        lo[:n_feat] = np.where(pin, 0.0, -np.inf)
    else:
        lo[:n_feat] = 0.0  # all features pinned ≥0 (V0_2-style)
    print(f"    fitting BVLS (pin_geq0={int(np.sum(np.isfinite(lo[:n_feat]) & (lo[:n_feat] == 0.0)))}/{n_feat})...")
    res = lsq_linear(A, y, bounds=(lo, hi), method="bvls", max_iter=4000, verbose=0)
    coef = res.x[:n_feat]
    bias = float(res.x[n_feat])
    n_active = int(np.sum(np.abs(coef) > 1e-6))
    info = {
        "cost": float(res.cost),
        "n_iter": int(res.nit),
        "n_active": n_active,
        "n_feat": n_feat,
        "scaler_mean": mu,
        "scaler_scale": sd,
        "bias": bias,
    }
    return coef, bias, info


def eval_corpora(coef: np.ndarray, bias: float, mu: np.ndarray, sd: np.ndarray, label: str):
    """Evaluate on the 6 canonical val corpora. Returns dict of {corpus: srocc}.
    Forward pass: distance = (Xs @ coef + bias); score = 100*exp(-0.18*max(0, d)^0.7).
    """
    out = {}
    print(f"\n  === {label} ===")
    print(f"  {'corpus':10s} {'n':>5s}  {'|SROCC|':>9s}  {'v47':>7s}  {'V0_2(228)':>9s}  Δ-v47 / Δ-V0_2")
    print("  " + "-" * 75)
    for v in ["cid22", "kadid", "tid", "konjnd", "aic3", "aic4"]:
        p = VAL_DIR / f"{v}.parquet"
        t = pq.read_table(p, columns=[f"f{i}" for i in range(N_FEATURES)] + ["human_score"])
        X = np.column_stack([np.asarray(t[f"f{i}"].combine_chunks().to_numpy(), dtype=np.float64) for i in range(N_FEATURES)])
        y = np.asarray(t["human_score"].combine_chunks().to_numpy(), dtype=np.float64)
        keep = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        X, y = X[keep], y[keep]
        Xs = (X - mu) / sd
        dist = Xs @ coef + bias
        # V0_2-style score-map: 100 * exp(-(a/100) * d^b), a=18, b=0.7. Distance-shaped.
        score = 100.0 * np.exp(-0.18 * np.power(np.maximum(dist, 0.0), 0.7))
        srocc = abs(spearmanr(score, y).statistic)
        v47_s, _ = V47_REF[v]
        v02_s = V02_REF_SROCC[v]
        d47 = srocc - v47_s
        d02 = srocc - v02_s
        flag = "✓" if d02 >= 0 else " "
        print(f"  {v:10s} {len(y):5d}  {srocc:9.4f}  {v47_s:7.4f}  {v02_s:9.4f}  {d47:+.4f} / {d02:+.4f} {flag}")
        out[v] = srocc
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=["all", "v02-target", "v02-concordant", "human-target", "mask", "bake"],
                    default="all", help="Which experiment cell(s) to run.")
    args = ap.parse_args()

    print("=" * 80)
    print("V0_2 methodology + 372 features (#44 follow-up, 2026-05-28)")
    print("=" * 80)

    # Always load with ssim2_gpu as primary; also returns iwssim + cvvdp for concordance.
    print("\nloading safesyn (196k × 372 features + ssim2_gpu/iwssim/cvvdp_score)...")
    X, y_ssim2, s, iw, cv = load_safesyn("ssim2_gpu")
    print(f"  safesyn: n={len(y_ssim2)}, ssim2_gpu range=[{y_ssim2.min():.2f}, {y_ssim2.max():.2f}]")
    # human_score for parallel comparison
    print("\nalso loading safesyn human_score for the comparison cell...")
    X_h, y_h, _, _, _ = load_safesyn("human_score")
    assert len(y_h) == len(y_ssim2), "human_score row count mismatch"
    print(f"  safesyn: human_score range=[{y_h.min():.3f}, {y_h.max():.3f}]")

    # ---- Cell 1: V0_2-target (safesyn / ssim2_gpu / all ≥0 / BVLS / no filter) ----
    if args.cell in ("all", "v02-target"):
        print("\n--- Cell 1: V0_2-target ---")
        print("  safesyn × 196k × 372 features × ssim2_gpu target × all-weights-≥0 × BVLS × no filter")
        coef, bias, info = fit_bvls(X, y_ssim2, pin=None)
        print(f"  n_active = {info['n_active']}/{N_FEATURES}  cost={info['cost']:.4f}")
        eval_corpora(coef, bias, info["scaler_mean"], info["scaler_scale"],
                     "Cell 1 (ssim2_gpu target, all ≥0, no concordance)")

    # ---- Cell 2: V0_2-concordant (concordance-filtered ssim2 + cvvdp) ----
    if args.cell in ("all", "v02-concordant"):
        print("\n--- Cell 2: V0_2-target + concordance filter ---")
        # V0_2 used ssim2 + butter; we use ssim2 + cvvdp (perceptual, similar family).
        mask = concordance_filter(s, cv, n_buckets=20)
        n_keep = int(mask.sum())
        print(f"  concordance (ssim2 vs cvvdp, ±2 buckets of 20): kept {n_keep}/{len(mask)} ({100*n_keep/len(mask):.1f}%)")
        coef, bias, info = fit_bvls(X[mask], y_ssim2[mask], pin=None)
        print(f"  n_active = {info['n_active']}/{N_FEATURES}  cost={info['cost']:.4f}")
        eval_corpora(coef, bias, info["scaler_mean"], info["scaler_scale"],
                     "Cell 2 (ssim2_gpu target, all ≥0, +concordance filter)")

    # ---- Cell 3: V0_2-target with iwssim (closer to V0_2's structural-family pair) ----
    if args.cell in ("all", "v02-concordant"):
        print("\n--- Cell 2b: V0_2-target + concordance filter (ssim2 vs iwssim) ---")
        mask = concordance_filter(s, iw, n_buckets=20)
        n_keep = int(mask.sum())
        print(f"  concordance (ssim2 vs iwssim, ±2 buckets): kept {n_keep}/{len(mask)} ({100*n_keep/len(mask):.1f}%)")
        coef, bias, info = fit_bvls(X[mask], y_ssim2[mask], pin=None)
        print(f"  n_active = {info['n_active']}/{N_FEATURES}  cost={info['cost']:.4f}")
        eval_corpora(coef, bias, info["scaler_mean"], info["scaler_scale"],
                     "Cell 2b (ssim2_gpu target, all ≥0, +concordance ssim2∩iwssim)")

    # ---- Cell 4: human_score target — to compare to v02-bvls ----
    if args.cell in ("all", "human-target"):
        print("\n--- Cell 3: human_score target (vs ssim2_gpu in Cell 1) ---")
        coef, bias, info = fit_bvls(X_h, y_h, pin=None)
        print(f"  n_active = {info['n_active']}/{N_FEATURES}  cost={info['cost']:.4f}")
        eval_corpora(coef, bias, info["scaler_mean"], info["scaler_scale"],
                     "Cell 3 (human_score target, all ≥0)")

    # ---- Cell 5: per-feature sign mask ----
    if args.cell in ("all", "mask"):
        print("\n--- Cell 4: V0_2-target + per-feature sign mask (300 ≥0, 72 free) ---")
        pin = load_mask()
        coef, bias, info = fit_bvls(X, y_ssim2, pin=pin)
        print(f"  n_active = {info['n_active']}/{N_FEATURES}  cost={info['cost']:.4f}")
        eval_corpora(coef, bias, info["scaler_mean"], info["scaler_scale"],
                     "Cell 4 (ssim2_gpu target, sign-masked 300/72)")

    # ---- Cell 5: mask + concordance filter (the winner combo to try) ----
    if args.cell in ("all", "mask"):
        print("\n--- Cell 5: V0_2-target + mask + concordance filter (ssim2 vs cvvdp) ---")
        pin = load_mask()
        mask_conc = concordance_filter(s, cv, n_buckets=20)
        n_keep = int(mask_conc.sum())
        print(f"  concordance (ssim2 vs cvvdp): kept {n_keep}/{len(mask_conc)} ({100*n_keep/len(mask_conc):.1f}%)")
        coef, bias, info = fit_bvls(X[mask_conc], y_ssim2[mask_conc], pin=pin)
        print(f"  n_active = {info['n_active']}/{N_FEATURES}  cost={info['cost']:.4f}")
        eval_corpora(coef, bias, info["scaler_mean"], info["scaler_scale"],
                     "Cell 5 (ssim2_gpu target, mask 300/72, +concordance)")

    # ---- Cell 5-bake: emit Cell 5 as a ZNPR v3 bake for end-to-end testing ----
    if args.cell in ("all", "bake"):
        print("\n--- Cell 5-bake: emitting ZNPR v3 bake for the Cell 5 winner ---")
        pin = load_mask()
        mask_conc = concordance_filter(s, cv, n_buckets=20)
        n_keep = int(mask_conc.sum())
        print(f"  training set: {n_keep}/{len(mask_conc)} rows after concordance filter")
        coef, bias, info = fit_bvls(X[mask_conc], y_ssim2[mask_conc], pin=pin)
        mu, sd = info["scaler_mean"], info["scaler_scale"]
        print(f"  n_active = {info['n_active']}/{N_FEATURES}")

        # Build a PCHIP dial spline on the multiband anchor (instead of V0_2's
        # fixed exp-decay score-map) so the bake outputs in [0, 100].
        ap_t = pq.read_table(ANCHOR_PQ,
                             columns=[f"f{i}" for i in range(N_FEATURES)] + ["human_score"])
        Xa = np.column_stack([np.asarray(ap_t[f"f{i}"].combine_chunks().to_numpy(), dtype=np.float64) for i in range(N_FEATURES)])
        ya = np.asarray(ap_t["human_score"].combine_chunks().to_numpy(), dtype=np.float64)
        keepa = np.isfinite(ya) & np.all(np.isfinite(Xa), axis=1)
        Xa, ya = Xa[keepa], ya[keepa]
        Xa_z = (Xa - mu) / sd
        raw_a = Xa_z @ coef + bias  # distance-form predictions on anchor
        print(f"  anchor: n={len(raw_a)}, raw pred [{raw_a.min():.3f}, {raw_a.max():.3f}], target [{ya.min():.2f}, {ya.max():.2f}]")

        # Fit PCHIP spline. Anchor's `human_score` is the per-row dial target.
        n_bins = 18
        order = np.argsort(raw_a, kind="stable")
        bin_size = (len(raw_a) + n_bins - 1) // n_bins
        raw_knots = []
        for start in range(0, len(raw_a), bin_size):
            end = min(start + bin_size, len(raw_a))
            bi = order[start:end]
            if len(bi) == 0:
                continue
            raw_knots.append((float(np.median(raw_a[bi])), float(np.median(ya[bi]))))
        # Direction: positive cov → increasing; negative cov → decreasing.
        # raw_a is DISTANCE (high=bad); ya is quality (high=good). Negative corr expected.
        from scipy.stats import pearsonr
        rho = float(pearsonr(raw_a, ya).statistic)
        decreasing = rho < 0
        print(f"  spline direction: {'decreasing' if decreasing else 'increasing'} (Pearson={rho:+.3f})")
        knots = [raw_knots[0]]
        for x, y in raw_knots[1:]:
            lx, ly = knots[-1]
            if x <= lx + 1e-6:
                continue
            if (decreasing and y < ly) or (not decreasing and y > ly):
                knots.append((x, y))
        print(f"  spline: {len(knots)} monotone knots")
        # Emit payload
        payload = struct.pack("<I", len(knots))
        for x, y in knots:
            payload += struct.pack("<ff", float(x), float(y))

        # Emit JSON for zenpredict-bake
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
            "metadata": [
                {
                    "key": "zentrain.output_calibration_spline",
                    "type": "bytes",
                    "hex": payload.hex(),
                }
            ],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(req, f)
            jp = Path(f.name)
        out_path = Path("/mnt/v/output/zensim/bakes/v02_372feat_cell5_2026-05-28.bin")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        baker = Path.home() / "work/zen/zenanalyze/target/release/zenpredict-bake"
        if not baker.exists():
            baker = Path.home() / "work/zen/zenanalyze/target/release/zenpredict"
        cmd = [str(baker)]
        if baker.name == "zenpredict":
            cmd.append("bake")
        cmd += [str(jp), str(out_path)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        jp.unlink()
        if r.returncode != 0:
            print(f"  zenpredict-bake failed: {r.stderr[:200]}")
            return 1
        size = out_path.stat().st_size
        print(f"  baked {out_path} ({size:,} bytes)")
        print(f"  → run: ./target/release/bake_verdict --bake {out_path}")

    # ---- Cell 6: STRICT concordance filter — see if tighter helps ----
    if args.cell in ("all", "mask"):
        print("\n--- Cell 6: V0_2-target + mask + STRICT concordance (ssim2 vs cvvdp, ±1 bucket) ---")
        pin = load_mask()
        # tighter filter
        s_rank = np.argsort(np.argsort(s)) / len(s)
        c_rank = np.argsort(np.argsort(cv)) / len(s)
        N_B = 20
        s_b = np.clip((s_rank * N_B).astype(int), 0, N_B - 1)
        c_b = np.clip((c_rank * N_B).astype(int), 0, N_B - 1)
        mask_strict = np.abs(s_b - c_b) <= 1
        n_keep = int(mask_strict.sum())
        print(f"  concordance strict (±1 bucket of 20): kept {n_keep}/{len(mask_strict)} ({100*n_keep/len(mask_strict):.1f}%)")
        coef, bias, info = fit_bvls(X[mask_strict], y_ssim2[mask_strict], pin=pin)
        print(f"  n_active = {info['n_active']}/{N_FEATURES}  cost={info['cost']:.4f}")
        eval_corpora(coef, bias, info["scaler_mean"], info["scaler_scale"],
                     "Cell 6 (ssim2_gpu target, mask 300/72, +STRICT concordance ±1)")

    print()
    print("=" * 80)
    print("References: V0_2 (228 features) — CID22=0.8676, KADID=0.8192, TID=0.8427,")
    print("                                  KonJND=0.4695, AIC-3=0.7962, AIC-4=0.9107")
    print("             v47 (current Profile::A, 27 KB MLP) — see V47_REF dict")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
