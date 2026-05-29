#!/usr/bin/env python3
"""Search for a bake that beats Cell 5 on the full Mohammadi panel (#48).

Cell 5 = V0_2 methodology on 372 features: safesyn-only, ssim2_gpu
distance-form target, 300/72 sign mask, ssim2∩cvvdp concordance filter,
BVLS, PCHIP dial spline. CID22 SROCC 0.8703, mean geomean3 0.815.

This sweeps the recipe's degrees of freedom and bakes + full-panel-scores
each candidate via bake_verdict:
  - TARGET metric: ssim2_gpu / cvvdp_score / iwssim / 50-50 mixes
  - CONCORDANCE filter: none / ssim2∩cvvdp / ssim2∩iwssim / 3-way
  - TRAINING groups: safesyn-only / +cid22_train+kadid+tid

Each candidate emits a ZNPR v3 bake (via zenpredict-bake) + the full panel
on the 6 canonical val corpora. Keep any candidate that beats Cell 5's
mean geomean3 (0.815) AND doesn't crater CID22 below 0.86.

Output: /mnt/v/output/zensim/bakes/sweep_v02_372feat/<cell>.bin + a
summary TSV at /tmp/v02_372feat_sweep_summary.tsv.
"""
from __future__ import annotations
import csv, json, os, struct, subprocess, sys, tempfile, time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from scipy.optimize import lsq_linear

REPO = Path(__file__).resolve().parent.parent.parent
TRAIN_DIR = Path("/mnt/v/zen/zensim-training/canonical-2026-05-21/train")
VAL_DIR = Path("/mnt/v/zen/zensim-training/canonical-2026-05-18/val")
MASK_TSV = REPO / "benchmarks/feature_sign_mask_2026-05-26.tsv"
ANCHOR_PQ = TRAIN_DIR / "multiband_anchor_dial100.parquet"
OUT_DIR = Path("/mnt/v/output/zensim/bakes/sweep_v02_372feat")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY = Path("/tmp/v02_372feat_sweep_summary.tsv")
_BAKER_CANDIDATES = [
    Path.home() / "work/zen/zenanalyze/target/release/zenpredict-bake",
    Path.home() / "work/zen/zenanalyze/target/release/zenpredict",
]
BAKER = next((b for b in _BAKER_CANDIDATES if b.exists()), _BAKER_CANDIDATES[-1])
BAKE_VERDICT = REPO / "target/release/bake_verdict"

N_FEATURES = 372
CORPORA = ["cid22", "kadid", "tid", "konjnd", "aic3", "aic4"]

# Cell 5 reference (the bar to beat)
CELL5 = {"cid22": 0.8703, "kadid": 0.7842, "tid": 0.8039,
         "konjnd": 0.4904, "aic3": 0.7732, "aic4": 0.8899, "mean_g3": 0.815}


def load_mask():
    pin = np.zeros(N_FEATURES, dtype=bool)
    with open(MASK_TSV) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            i = int(row["feat_idx"])
            if i < N_FEATURES:
                pin[i] = row["sign_mask"] == "pin_geq0"
    return pin


def to_distance(raw, col):
    """Convert a quality-shaped metric to distance form (high = bad, ≥0)."""
    if col == "ssim2_gpu":
        return np.maximum(100.0 - raw, 0.0)
    if col == "cvvdp_score":
        # cvvdp JOD 0..10, 10 = imperceptible. distance = 10 - cvvdp, scaled ×10.
        return np.maximum(10.0 - raw, 0.0) * 10.0
    if col == "iwssim":
        # iwssim 0..1, 1 = identical. distance = (1 - iwssim) × 100.
        return np.maximum(1.0 - raw, 0.0) * 100.0
    # default: flip around max
    return float(np.nanmax(raw)) - raw


def load_safesyn_cols(cols):
    p = TRAIN_DIR / "safesyn.parquet"
    want = [f"f{i}" for i in range(N_FEATURES)] + list(cols)
    t = pq.read_table(p, columns=list(dict.fromkeys(want)))
    X = np.column_stack([np.asarray(t[f"f{i}"].combine_chunks().to_numpy(), dtype=np.float64) for i in range(N_FEATURES)])
    out = {c: np.asarray(t[c].combine_chunks().to_numpy(), dtype=np.float64) for c in cols}
    return X, out


def load_group_features(basename, target_col):
    p = TRAIN_DIR / f"{basename}.parquet"
    t = pq.read_table(p, columns=[f"f{i}" for i in range(N_FEATURES)] + [target_col])
    X = np.column_stack([np.asarray(t[f"f{i}"].combine_chunks().to_numpy(), dtype=np.float64) for i in range(N_FEATURES)])
    y = np.asarray(t[target_col].combine_chunks().to_numpy(), dtype=np.float64)
    return X, y


def concordance_mask(a, b, n_buckets=20, tol=2):
    n = len(a)
    ar = np.argsort(np.argsort(a)) / n
    br = np.argsort(np.argsort(b)) / n
    ab = np.clip((ar * n_buckets).astype(int), 0, n_buckets - 1)
    bb = np.clip((br * n_buckets).astype(int), 0, n_buckets - 1)
    return np.abs(ab - bb) <= tol


def fit_bvls(X, y_dist, pin, sw=None):
    mu = X.mean(axis=0); sd = X.std(axis=0); sd[sd < 1e-9] = 1.0
    Xs = (X - mu) / sd
    A = np.hstack([Xs, np.ones((len(y_dist), 1))])
    if sw is not None:
        rt = np.sqrt(sw)
        A = A * rt[:, None]; y_fit = y_dist * rt
    else:
        y_fit = y_dist
    lo = np.full(N_FEATURES + 1, -np.inf); hi = np.full(N_FEATURES + 1, np.inf)
    lo[:N_FEATURES] = np.where(pin, 0.0, -np.inf)
    res = lsq_linear(A, y_fit, bounds=(lo, hi), method="bvls", max_iter=4000)
    return res.x[:N_FEATURES], float(res.x[N_FEATURES]), mu, sd, int(np.sum(np.abs(res.x[:N_FEATURES]) > 1e-6))


def fit_spline(raw_a, ya, n_bins=18):
    order = np.argsort(raw_a, kind="stable")
    bin_size = (len(raw_a) + n_bins - 1) // n_bins
    knots_raw = []
    for s in range(0, len(raw_a), bin_size):
        e = min(s + bin_size, len(raw_a)); bi = order[s:e]
        if len(bi): knots_raw.append((float(np.median(raw_a[bi])), float(np.median(ya[bi]))))
    from scipy.stats import pearsonr
    dec = pearsonr(raw_a, ya).statistic < 0
    knots = [knots_raw[0]]
    for x, y in knots_raw[1:]:
        lx, ly = knots[-1]
        if x <= lx + 1e-6: continue
        if (dec and y < ly) or (not dec and y > ly): knots.append((x, y))
    if len(knots) < 2: return None
    payload = struct.pack("<I", len(knots))
    for x, y in knots: payload += struct.pack("<ff", float(x), float(y))
    return payload


def emit_bake(coef, bias, mu, sd, spline, out_path):
    req = {"schema_hash": 0, "flags": 0,
           "scaler_mean": [float(v) for v in mu.astype(np.float32)],
           "scaler_scale": [float(v) for v in sd.astype(np.float32)],
           "layers": [{"in_dim": N_FEATURES, "out_dim": 1, "activation": "identity",
                       "dtype": "f32", "weights": [float(v) for v in coef.astype(np.float32)],
                       "biases": [float(bias)]}],
           "metadata": ([{"key": "zentrain.output_calibration_spline", "type": "bytes", "hex": spline.hex()}]
                        if spline else [])}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(req, f); jp = Path(f.name)
    try:
        cmd = [str(BAKER)]
        if BAKER.name == "zenpredict":
            cmd.append("bake")
        cmd += [str(jp), str(out_path)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.returncode == 0
    finally:
        jp.unlink(missing_ok=True)


def verdict(bake_path):
    """Run bake_verdict, parse per-corpus SROCC + geomean3."""
    out_md = tempfile.NamedTemporaryFile("w", suffix=".md", delete=False).name
    r = subprocess.run([str(BAKE_VERDICT), "--bake", str(bake_path), "--output", out_md],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    res = {}
    name_map = {"CID22": "cid22", "KADIK10k": "kadid", "TID2013": "tid",
                "KonJND-1k (full)": "konjnd", "AIC-3 CTC": "aic3", "AIC-4 sample": "aic4"}
    for line in open(out_md):
        if not line.startswith("| "): continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 10: continue
        corpus = cells[0]
        if corpus in name_map:
            try:
                res[name_map[corpus]] = {"srocc": float(cells[2]), "g3": float(cells[9])}
            except ValueError:
                pass
    os.unlink(out_md)
    return res


def main():
    pin = load_mask()
    # Preload safesyn metrics for concordance + targets
    print("loading safesyn...", file=sys.stderr)
    Xss, ss = load_safesyn_cols(["ssim2_gpu", "cvvdp_score", "iwssim"])
    finite = (np.all(np.isfinite(Xss), axis=1)
              & np.isfinite(ss["ssim2_gpu"]) & np.isfinite(ss["cvvdp_score"]) & np.isfinite(ss["iwssim"]))
    Xss = Xss[finite]
    for k in ss: ss[k] = ss[k][finite]
    print(f"safesyn: {len(Xss)} rows", file=sys.stderr)

    # Anchor for spline
    at = pq.read_table(ANCHOR_PQ, columns=[f"f{i}" for i in range(N_FEATURES)] + ["human_score"])
    Xa = np.column_stack([np.asarray(at[f"f{i}"].combine_chunks().to_numpy(), dtype=np.float64) for i in range(N_FEATURES)])
    ya = np.asarray(at["human_score"].combine_chunks().to_numpy(), dtype=np.float64)
    keepa = np.isfinite(ya) & np.all(np.isfinite(Xa), axis=1)
    Xa, ya = Xa[keepa], ya[keepa]

    # Candidate cells: (label, target_col, concordance, multi_group)
    cells = [
        ("cell5_repro",     "ssim2_gpu",   "cvvdp",  False),  # baseline Cell 5
        ("cvvdp_target",    "cvvdp_score", "cvvdp",  False),
        ("iwssim_target",   "iwssim",      "iwssim", False),
        ("ssim2_3way",      "ssim2_gpu",   "3way",   False),
        ("ssim2_iwconc",    "ssim2_gpu",   "iwssim", False),
        ("ssim2_noconc",    "ssim2_gpu",   "none",   False),
        ("mix_ss_cv",       "mix_ss_cv",   "cvvdp",  False),
        ("mix_ss_iw",       "mix_ss_iw",   "iwssim", False),
        ("ssim2_multigroup","ssim2_gpu",   "cvvdp",  True),
    ]

    rows = []
    print(f"\n{'cell':18s} {'active':>7s} {'CID22':>7s} {'KADID':>7s} {'TID':>7s} {'KonJND':>7s} {'AIC3':>7s} {'AIC4':>7s} {'meanG3':>7s} {'vsCell5':>8s}", file=sys.stderr)
    print("-" * 100, file=sys.stderr)
    for label, tcol, conc, multi in cells:
        # Build target (distance form)
        if tcol == "mix_ss_cv":
            yd = 0.5 * to_distance(ss["ssim2_gpu"], "ssim2_gpu") + 0.5 * to_distance(ss["cvvdp_score"], "cvvdp_score")
        elif tcol == "mix_ss_iw":
            yd = 0.5 * to_distance(ss["ssim2_gpu"], "ssim2_gpu") + 0.5 * to_distance(ss["iwssim"], "iwssim")
        else:
            yd = to_distance(ss[tcol], tcol)
        # Concordance mask
        if conc == "cvvdp":
            cm = concordance_mask(ss["ssim2_gpu"], ss["cvvdp_score"])
        elif conc == "iwssim":
            cm = concordance_mask(ss["ssim2_gpu"], ss["iwssim"])
        elif conc == "3way":
            cm = concordance_mask(ss["ssim2_gpu"], ss["cvvdp_score"]) & concordance_mask(ss["ssim2_gpu"], ss["iwssim"])
        else:
            cm = np.ones(len(Xss), dtype=bool)
        Xtr, ytr = Xss[cm], yd[cm]
        # Multi-group: append cid22_train + kadid + tid (human_score, normalized to distance via flip)
        if multi:
            extra_X, extra_y = [], []
            for g, w in [("cid22_train_norm", 1.5), ("kadid", 0.5), ("tid", 0.5)]:
                gX, gy = load_group_features(g, "human_score")
                gk = np.all(np.isfinite(gX), axis=1) & np.isfinite(gy)
                gX, gy = gX[gk], gy[gk]
                gyd = float(np.nanmax(gy)) - gy  # quality→distance flip
                # min-max each group's distance to ssim2's ~[0,100] scale for LS comparability
                lo, hi = np.quantile(gyd, 0.001), np.quantile(gyd, 0.999)
                gyd = np.clip((gyd - lo) / max(hi - lo, 1e-9), 0, 1) * float(np.median(ytr) * 2)
                extra_X.append(gX); extra_y.append(gyd)
            Xtr = np.vstack([Xtr] + extra_X)
            ytr = np.concatenate([ytr] + extra_y)
        # Fit
        coef, bias, mu, sd, nact = fit_bvls(Xtr, ytr, pin)
        # Spline on anchor
        Xa_z = (Xa - mu) / sd
        raw_a = Xa_z @ coef + bias
        spline = fit_spline(raw_a, ya)
        # Emit + verdict
        bake_path = OUT_DIR / f"{label}.bin"
        if not emit_bake(coef, bias, mu, sd, spline, bake_path):
            print(f"{label:18s}  BAKE FAILED", file=sys.stderr); continue
        v = verdict(bake_path)
        if not v:
            print(f"{label:18s}  VERDICT FAILED", file=sys.stderr); continue
        sroccs = {c: v.get(c, {}).get("srocc", float("nan")) for c in CORPORA}
        g3s = [v.get(c, {}).get("g3", float("nan")) for c in CORPORA]
        mean_g3 = float(np.nanmean(g3s))
        delta = mean_g3 - CELL5["mean_g3"]
        flag = "  WIN" if (delta > 0 and sroccs["cid22"] >= 0.86) else ""
        print(f"{label:18s} {nact:7d} {sroccs['cid22']:7.4f} {sroccs['kadid']:7.4f} {sroccs['tid']:7.4f} "
              f"{sroccs['konjnd']:7.4f} {sroccs['aic3']:7.4f} {sroccs['aic4']:7.4f} {mean_g3:7.4f} {delta:+8.4f}{flag}", file=sys.stderr)
        rows.append({"cell": label, "n_active": nact, "mean_g3": mean_g3, "delta_vs_cell5": delta,
                     **{f"srocc_{c}": sroccs[c] for c in CORPORA}})

    # Write summary TSV
    with open(SUMMARY, "w") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter="\t")
            w.writeheader(); w.writerows(rows)
    print(f"\nsummary → {SUMMARY}", file=sys.stderr)


if __name__ == "__main__":
    main()
