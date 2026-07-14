#!/usr/bin/env python3
"""Linear-projection probe (SDR + HDR) over the 372 zensim features — 2026-07-03.

Mission: find the best modern LINEAR heads for SDR and HDR, exploiting the
per-weight postprocessing toolkit linear bakes make cheap (sign/bound
projections via BVLS, zerobias pruning, f16 quantization, monotone PCHIP
output spline refit AFTER quantization per the standard pack-then-calibrate
order). Linear fits are closed-form / active-set deterministic — no seed axis,
so the MLP recipe collapse mode (43.75% across 16 seeds on 2026-07-03's wide
fan) is structurally impossible.

Design: one streaming pass per training group accumulates raw MOMENTS
(S = X^T X, s = sum X, q = X^T y per target, Y1, Y2, n) in BOTH feature
spaces (raw + shaped via the yeo-johnson screen TSV). Every fit family then
solves from the (weighted, additive) Grams:

  ridge:  (G_z + lam*W*I) w = c_z          (closed form, lambda swept free)
  bvls:   min ||L^T w - z0||^2 s.t. sign mask   (Cholesky trick: G_z = L L^T,
          z0 = L^{-1} c_z — BVLS on a 372x372 system == BVLS on ALL rows)
  lasso:  coordinate descent on (G_z/W, c_z/W), fixed sweep order

Standardization is derived algebraically from the mix's weighted moments, so
ANY data-mix weighting is solvable without re-reading parquets.

Selection is on train-legal axes ONLY: bigcodec valdigits SROCC (SDR),
hdr valdigits SROCC (HDR), konjnd-dense-norm train pjnd_target |SROCC| (guard).
CID22 / AIC-3 / AIC-4 / KonJND-val are reporting-only via bake_verdict.

Subcommands:
  gram      accumulate per-group moments + cache val/guard/anchor matrices
  fit       solve all families x shapings x mixes, print val table
  finalize  tau-sweep + f16 + spline-on-packed for named candidates, emit
            ZNPR v3 via `zenpredict bake` (JSON pipeline; never raw bytes)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
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
SCRATCH = Path("/mnt/v/output/zensim-multicodec-probe/linear-probe")
PROBE = Path("/mnt/v/output/zensim-multicodec-probe")
CANON = Path("/mnt/v/zen/zensim-training/canonical-2026-05-21/train")
ANCHOR_PQ = CANON / "multiband_anchor_dial100.parquet"
N_FEAT = 372
BAKER = Path.home() / "work/zen/zenanalyze/target/release/zenpredict"
SPLINE_KEY = "zentrain.output_calibration_spline"

# Reuse the transform/mask code from the 2026-05-28 BVLS script verbatim
# (it mirrors zenpredict::feature_transform's f32 math).
_spec = importlib.util.spec_from_file_location(
    "v02bvls", REPO / "scripts/v_next/train_v02_bvls_shaped.py"
)
_v02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v02)
apply_transform = _v02.apply_transform
load_transforms = _v02.load_transforms
load_mask = _v02.load_mask

# ---------------------------------------------------------------------------
# Group registry: (name, path, target columns to accumulate q for)
GROUPS = {
    "bigcodec": (PROBE / "bigcodec_traindigits_2026-07-02.parquet", ["human_score"]),
    "safesyn": (CANON / "safesyn.parquet", ["human_score"]),
    "cid22_train": (CANON / "cid22_train_norm.parquet", ["human_score"]),
    "kadid": (CANON / "kadid.parquet", ["human_score"]),
    "tid": (CANON / "tid.parquet", ["human_score"]),
    "konjnd_dense": (CANON / "konjnd-dense-norm.parquet", ["human_score", "pjnd_target"]),
    "hdr_v3": (PROBE / "hdr_zenjxl_v3_traindigits_2026-07-03.parquet", ["human_score"]),
    "hdr_v3mix": (PROBE / "hdr_zenjxl_v3mix_traindigits_2026-07-03.parquet", ["human_score"]),
    # 2026-07-12 teacher-ceiling probe (§8.3): iwssim-teacher targets. NOTE:
    # iwssim NaNs on ALL tiny scales (5-scale pyramid min size) → these corpora
    # carry NO tiny renditions (5,928/3,107 vs v3mix's 7,410/3,900).
    "hdr_v3iwmix": (PROBE / "hdr_zenjxl_v3iwmix_traindigits_2026-07-12.parquet", ["human_score"]),
    "hdr_v3iw": (PROBE / "hdr_zenjxl_v3iw_traindigits_2026-07-12.parquet", ["human_score"]),
    # 2026-07-13 kadis-hdr synthetic-distortion family: 25 KADIS types in PQ
    # code-value domain, 11,387 joined cells over 1,140 imazen-26 HDR
    # renditions, fleet-scored (bhdr_improvement_split_lineage §8.9-8.10).
    # A SECOND distortion family disjoint from jxl-encode artifacts.
    # mix target = 0.5·ssim2norm + 0.5·cvvdp-JOD-norm (the shipped-BHdr lever).
    "hdr_kadis_mix": (PROBE / "hdr_kadis_mix_traindigits_2026-07-13.parquet", ["human_score"]),
    "hdr_kadis": (PROBE / "hdr_kadis_traindigits_2026-07-13.parquet", ["human_score"]),
}
VAL_SETS = {
    "bigcodec_val": (PROBE / "bigcodec_valdigits_2026-07-02.parquet", "human_score"),
    "hdr_val": (PROBE / "hdr_zenjxl_v3_valdigits_2026-07-03.parquet", "human_score"),
    "hdr_valmix": (PROBE / "hdr_zenjxl_v3mix_valdigits_2026-07-03.parquet", "human_score"),
    "hdr_valiwmix": (PROBE / "hdr_zenjxl_v3iwmix_valdigits_2026-07-12.parquet", "human_score"),
    "hdr_valiw": (PROBE / "hdr_zenjxl_v3iw_valdigits_2026-07-12.parquet", "human_score"),
    "konjnd_guard": (CANON / "konjnd-dense-norm.parquet", "pjnd_target"),
    # cid22_train_norm is a TRAIN group (ssim2-anchored, NOT MOS) — using it
    # as a selection axis is train-legal (it is already a fit input).
    "cid22tr_sel": (CANON / "cid22_train_norm.parquet", "human_score"),
    # kadis-hdr held-out-origin val (2,994 rows, human-free, never-burned) —
    # registered selection axis of §8.11 alongside hdr_valmix.
    "hdr_kadis_valmix": (PROBE / "hdr_kadis_mix_valdigits_2026-07-13.parquet", "human_score"),
}

FCOLS = [f"f{i}" for i in range(N_FEAT)]


def minmax01_bounds(y: np.ndarray) -> tuple[float, float]:
    lo, hi = np.quantile(y, 0.001), np.quantile(y, 0.999)
    return float(lo), float(max(hi - lo, 1e-9))


def shape_block(X: np.ndarray, transforms, tparams) -> np.ndarray:
    Y = np.empty_like(X, dtype=np.float64)
    for i in range(N_FEAT):
        Y[:, i] = apply_transform(transforms[i], tparams[i], X[:, i])
    return Y


# ---------------------------------------------------------------------------
def cmd_gram(args) -> int:
    transforms, tparams = load_transforms()
    (SCRATCH / "grams").mkdir(parents=True, exist_ok=True)
    (SCRATCH / "val").mkdir(parents=True, exist_ok=True)

    for name, (path, tcols) in GROUPS.items():
        out = SCRATCH / "grams" / f"{name}.npz"
        if out.exists() and not args.force:
            print(f"[gram] {name}: cached, skip")
            continue
        t0 = time.time()
        pf = pq.ParquetFile(path)
        # target minmax bounds from a single-column read
        tb = {}
        for tc in tcols:
            ycol = pq.read_table(path, columns=[tc])[tc].to_numpy(zero_copy_only=False).astype(np.float64)
            ycol = ycol[np.isfinite(ycol)]
            tb[tc] = minmax01_bounds(ycol)
        acc = {}
        for space in ("raw", "shaped"):
            acc[space] = dict(
                S=np.zeros((N_FEAT, N_FEAT)), s=np.zeros(N_FEAT), n=0.0,
                dropped=0,
                **{f"q_{tc}": np.zeros(N_FEAT) for tc in tcols},
                **{f"Y1_{tc}": 0.0 for tc in tcols},
                **{f"Y2_{tc}": 0.0 for tc in tcols},
            )
        for batch in pf.iter_batches(batch_size=131072, columns=FCOLS + tcols):
            X = np.column_stack(
                [batch[c].to_numpy(zero_copy_only=False).astype(np.float64) for c in FCOLS]
            )
            ys = {}
            keep = np.all(np.isfinite(X), axis=1)
            for tc in tcols:
                y = batch[tc].to_numpy(zero_copy_only=False).astype(np.float64)
                lo, span = tb[tc]
                ys[tc] = np.clip((y - lo) / span, 0.0, 1.0)
                keep &= np.isfinite(y)
            n_drop0 = int((~keep).sum())
            X = X[keep]
            ys = {tc: v[keep] for tc, v in ys.items()}
            for space in ("raw", "shaped"):
                Xs = X if space == "raw" else shape_block(X, transforms, tparams)
                k2 = np.all(np.isfinite(Xs), axis=1)
                a = acc[space]
                a["dropped"] += n_drop0 + int((~k2).sum())
                Xk = Xs[k2]
                a["S"] += Xk.T @ Xk
                a["s"] += Xk.sum(axis=0)
                a["n"] += len(Xk)
                for tc in tcols:
                    yk = ys[tc][k2]
                    a[f"q_{tc}"] += Xk.T @ yk
                    a[f"Y1_{tc}"] += float(yk.sum())
                    a[f"Y2_{tc}"] += float((yk * yk).sum())
        save = {}
        for space, a in acc.items():
            for k, v in a.items():
                save[f"{space}__{k}"] = v
        np.savez_compressed(out, **save)
        print(f"[gram] {name}: n={acc['raw']['n']:.0f} dropped_raw={acc['raw']['dropped']} "
              f"dropped_shaped={acc['shaped']['dropped']} t={time.time()-t0:.1f}s")

    # Val / guard / anchor matrices, cached raw+shaped as f32
    for name, (path, tcol) in VAL_SETS.items():
        out = SCRATCH / "val" / f"{name}.npz"
        if out.exists() and not args.force:
            print(f"[val] {name}: cached, skip")
            continue
        t = pq.read_table(path, columns=FCOLS + [tcol, "ref_basename"])
        X = np.column_stack([t[c].to_numpy(zero_copy_only=False).astype(np.float64) for c in FCOLS])
        y = t[tcol].to_numpy(zero_copy_only=False).astype(np.float64)
        keep = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        X, y = X[keep], y[keep]
        Xs = shape_block(X, transforms, tparams)
        ks = np.all(np.isfinite(Xs), axis=1)
        np.savez_compressed(out, raw=X[ks].astype(np.float32), shaped=Xs[ks].astype(np.float32),
                            y=y[ks], n_drop=int((~keep).sum() + (~ks).sum()))
        print(f"[val] {name}: n={ks.sum()} dropped={int((~keep).sum()+(~ks).sum())}")

    out = SCRATCH / "val" / "anchor.npz"
    if not out.exists() or args.force:
        t = pq.read_table(ANCHOR_PQ, columns=FCOLS + ["target_score"])
        X = np.column_stack([t[c].to_numpy(zero_copy_only=False).astype(np.float64) for c in FCOLS])
        y = t["target_score"].to_numpy(zero_copy_only=False).astype(np.float64)
        keep = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        X, y = X[keep], y[keep]
        Xs = shape_block(X, transforms, tparams)
        np.savez_compressed(out, raw=X.astype(np.float32), shaped=Xs.astype(np.float32), y=y)
        print(f"[anchor] n={len(y)}")
    return 0


# ---------------------------------------------------------------------------
# Mix algebra: weighted moments -> standardized Gram + rhs
class MixGram:
    def __init__(self, space: str, mix: list[tuple[str, float, str]]):
        """mix: list of (group_name, weight, target_col)."""
        S = np.zeros((N_FEAT, N_FEAT)); s = np.zeros(N_FEAT)
        q = np.zeros(N_FEAT); Y1 = 0.0; Y2 = 0.0; W = 0.0
        self.desc = []
        for g, w, tc in mix:
            z = np.load(SCRATCH / "grams" / f"{g}.npz")
            S += w * z[f"{space}__S"]; s += w * z[f"{space}__s"]
            q += w * z[f"{space}__q_{tc}"]; Y1 += w * float(z[f"{space}__Y1_{tc}"])
            Y2 += w * float(z[f"{space}__Y2_{tc}"]); W += w * float(z[f"{space}__n"])
            self.desc.append(f"{g}:{w}:{tc}(n={float(z[f'{space}__n']):.0f})")
        mu = s / W
        var = np.maximum(S.diagonal() / W - mu * mu, 0.0)
        sd = np.sqrt(var); sd[sd < 1e-9] = 1.0
        Sc = S - W * np.outer(mu, mu)                      # sum (x-mu)(x-mu)^T
        self.G = Sc / np.outer(sd, sd)                     # sum z z^T
        self.ybar = Y1 / W
        self.c = (q - mu * Y1) / sd                        # sum z*(y-ybar)
        self.W = W; self.mu = mu; self.sd = sd
        self.yss = Y2 - W * self.ybar ** 2

    def ridge(self, lam: float) -> tuple[np.ndarray, float]:
        A = self.G + lam * self.W * np.eye(N_FEAT)
        w = np.linalg.solve(A, self.c)
        return w, self.ybar

    def _chol(self):
        jit = 1e-10 * float(np.trace(self.G)) / N_FEAT
        for _ in range(8):
            try:
                return np.linalg.cholesky(self.G + jit * np.eye(N_FEAT))
            except np.linalg.LinAlgError:
                jit *= 10.0
        raise RuntimeError("cholesky failed")

    def bvls(self, pin: np.ndarray, max_iter: int = 4000) -> tuple[np.ndarray, float]:
        L = self._chol()
        z0 = np.linalg.solve(L, self.c)
        lo = np.where(pin, 0.0, -np.inf)
        hi = np.full(N_FEAT, np.inf)
        res = lsq_linear(L.T, z0, bounds=(lo, hi), method="bvls", max_iter=max_iter)
        return res.x, self.ybar

    def lasso(self, lam: float, n_sweeps: int = 200, tol: float = 1e-10) -> tuple[np.ndarray, float]:
        """Coordinate descent on (G/W, c/W); lam on the mean-loss scale."""
        Gn = self.G / self.W
        cn = self.c / self.W
        d = Gn.diagonal().copy()
        d[d < 1e-12] = 1e-12
        w = np.zeros(N_FEAT)
        Gw = np.zeros(N_FEAT)  # Gn @ w
        for _ in range(n_sweeps):
            delta = 0.0
            for j in range(N_FEAT):
                rho = cn[j] - Gw[j] + d[j] * w[j]
                nw = np.sign(rho) * max(abs(rho) - lam, 0.0) / d[j]
                if nw != w[j]:
                    Gw += Gn[:, j] * (nw - w[j])
                    delta = max(delta, abs(nw - w[j]))
                    w[j] = nw
            if delta < tol:
                break
        return w, self.ybar


_VAL_CACHE: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}


def _val_xy(name: str, space: str) -> tuple[np.ndarray, np.ndarray]:
    k = (name, space)
    if k not in _VAL_CACHE:
        z = np.load(SCRATCH / "val" / f"{name}.npz")
        _VAL_CACHE[k] = (z[space].astype(np.float64), z["y"].astype(np.float64))
    return _VAL_CACHE[k]


def val_metrics(w: np.ndarray, bias: float, mu, sd, space: str) -> dict[str, float]:
    out = {}
    for name in VAL_SETS:
        X, y = _val_xy(name, space)
        pred = (X - mu) / sd @ w + bias
        r = spearmanr(pred, y).statistic
        out[name] = float(r)
    return out


# ---------------------------------------------------------------------------
# Data mixes.  Weights for *w7lin* mirror w7_guard_s101.toml's train_w.
MIXES_SDR = {
    "big": [("bigcodec", 1.0, "human_score")],
    "canon": [("safesyn", 1.0, "human_score"), ("cid22_train", 1.5, "human_score"),
              ("kadid", 0.5, "human_score"), ("tid", 0.5, "human_score")],
    "w7sdr": [("safesyn", 1.0, "human_score"), ("cid22_train", 1.5, "human_score"),
              ("kadid", 0.5, "human_score"), ("tid", 0.5, "human_score"),
              ("konjnd_dense", 1.2, "human_score"), ("bigcodec", 0.25, "human_score")],
}
MIXES_HDR = {
    "hdr": [("hdr_v3", 1.0, "human_score")],
    "hdrmix": [("hdr_v3mix", 1.0, "human_score")],
    # §8.11 pre-registered broadened-corpus mixes (2026-07-14): jxl-encode
    # family + kadis-hdr synthetic-distortion family, three weightings.
    # Selection axis registered as mean(hdr_valmix, hdr_kadis_valmix); ONE
    # UPIQ look for the single selected candidate.
    "hdrbroad11": [("hdr_v3mix", 1.0, "human_score"),
                   ("hdr_kadis_mix", 1.0, "human_score")],
    "hdrbroad1h": [("hdr_v3mix", 1.0, "human_score"),
                   ("hdr_kadis_mix", 0.5, "human_score")],
    "hdrbroadh1": [("hdr_v3mix", 0.5, "human_score"),
                   ("hdr_kadis_mix", 1.0, "human_score")],
    # 2026-07-12 teacher-ceiling probe (§8.3/§8.4): iwssim-teacher families.
    "hdriwmix": [("hdr_v3iwmix", 1.0, "human_score")],
    "hdriw": [("hdr_v3iw", 1.0, "human_score")],
    "w7lin": [("safesyn", 1.0, "human_score"), ("cid22_train", 1.5, "human_score"),
              ("kadid", 0.5, "human_score"), ("tid", 0.5, "human_score"),
              ("konjnd_dense", 1.2, "human_score"), ("bigcodec", 0.25, "human_score"),
              ("hdr_v3", 1.0, "human_score")],
    "w8lin": [("safesyn", 1.0, "human_score"), ("cid22_train", 1.5, "human_score"),
              ("kadid", 0.5, "human_score"), ("tid", 0.5, "human_score"),
              ("konjnd_dense", 1.2, "human_score"), ("bigcodec", 0.25, "human_score"),
              ("hdr_v3mix", 1.0, "human_score")],
    # Round 2 (2026-07-03, after round-1 panel): bigcodec mass drags linear
    # CID22 down (w8lin-bvls 0.6526 vs canon-bvls 0.8280) while hdr_v3mix alone
    # gave the best linear CID22 (0.8689). Blend canon + up-weighted hdr_v3mix,
    # NO bigcodec. All round-2 fits are panel'd and reported — no cherry-pick.
    "canonhdr15": [("safesyn", 1.0, "human_score"), ("cid22_train", 1.5, "human_score"),
                   ("kadid", 0.5, "human_score"), ("tid", 0.5, "human_score"),
                   ("hdr_v3mix", 15.0, "human_score")],
    "canonhdr40": [("safesyn", 1.0, "human_score"), ("cid22_train", 1.5, "human_score"),
                   ("kadid", 0.5, "human_score"), ("tid", 0.5, "human_score"),
                   ("hdr_v3mix", 40.0, "human_score")],
    "canonkjhdr15": [("safesyn", 1.0, "human_score"), ("cid22_train", 1.5, "human_score"),
                     ("kadid", 0.5, "human_score"), ("tid", 0.5, "human_score"),
                     ("konjnd_dense", 1.2, "human_score"),
                     ("hdr_v3mix", 15.0, "human_score")],
}
RIDGE_LAMS = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
LASSO_LAMS = [3e-5, 1e-4, 3e-4, 5e-4, 1e-3, 2e-3]


def cmd_fit(args) -> int:
    pin = load_mask()
    rows = []
    mixes = dict(**MIXES_SDR, **MIXES_HDR)
    if args.only:
        mixes = {k: v for k, v in mixes.items() if k in args.only.split(",")}
    (SCRATCH / "fits").mkdir(parents=True, exist_ok=True)
    for mname, mix in mixes.items():
        for space in ("raw", "shaped"):
            mg = MixGram(space, mix)
            fits = []
            for lam in RIDGE_LAMS:
                w, b = mg.ridge(lam)
                fits.append((f"ridge{lam:g}", w, b))
            w, b = mg.bvls(pin)
            fits.append(("bvls", w, b))
            for lam in LASSO_LAMS:
                w, b = mg.lasso(lam)
                fits.append((f"lasso{lam:g}", w, b))
            for fam, w, b in fits:
                vm = val_metrics(w, b, mg.mu, mg.sd, space)
                nact = int((np.abs(w) > 1e-7).sum())
                key = f"{mname}-{fam}-{space}"
                rows.append((key, nact, vm))
                np.savez_compressed(SCRATCH / "fits" / f"{key}.npz",
                                    w=w, bias=b, mu=mg.mu, sd=mg.sd,
                                    space=space, desc="|".join(mg.desc))
                print(f"{key:38s} act={nact:3d}  bigval={vm['bigcodec_val']:+.4f}  "
                      f"hdrval={vm['hdr_val']:+.4f}  hdrmixval={vm['hdr_valmix']:+.4f}  "
                      f"konjnd_guard={abs(vm['konjnd_guard']):.4f}", flush=True)
    with open(SCRATCH / "fits" / "table.json", "w") as f:
        json.dump([{"key": k, "n_active": n, **vm} for k, n, vm in rows], f, indent=1)
    return 0


# ---------------------------------------------------------------------------
def fit_spline_knots(tp: np.ndarray, tgt: np.ndarray, neg_tail: bool = True):
    """Same knot logic as pack_and_calibrate.py."""
    edges = np.percentile(tp, np.linspace(1, 99, 18))
    kx, ky = [], []
    lo = tp < edges[0]
    if lo.sum() >= 2:
        kx.append(float(np.median(tp[lo]))); ky.append(float(np.median(tgt[lo])))
    for i in range(len(edges) - 1):
        m = (tp >= edges[i]) & (tp < edges[i + 1])
        if m.sum() >= 2:
            kx.append(float(np.median(tp[m]))); ky.append(float(np.median(tgt[m])))
    hi = tp >= edges[-1]
    if hi.sum() >= 2:
        kx.append(float(np.median(tp[hi]))); ky.append(float(np.median(tgt[hi])))
    cx, cy = [kx[0]], [ky[0]]
    for i in range(1, len(kx)):
        if kx[i] > cx[-1] + 1e-7 and ky[i] >= cy[-1]:
            cx.append(kx[i]); cy.append(ky[i])
    if neg_tail:
        zeros = [i for i, y in enumerate(cy) if y <= 1e-6]
        if len(zeros) > 1:
            drop = set(zeros[:-1])
            cx = [x for i, x in enumerate(cx) if i not in drop]
            cy = [y for i, y in enumerate(cy) if i not in drop]
    return cx, cy


def bake_candidate(key: str, tau: float, out_path: Path, use_f16: bool = True) -> dict:
    """Zerobias -> f16 round -> spline refit on the PACKED weights -> one bake."""
    z = np.load(SCRATCH / "fits" / f"{key}.npz")
    w = z["w"].astype(np.float64).copy()
    bias = float(z["bias"]); mu = z["mu"]; sd = z["sd"]; space = str(z["space"])
    n0 = int((np.abs(w) > 1e-7).sum())
    if tau > 0:
        w[np.abs(w) < tau] = 0.0
    if use_f16:
        w = w.astype(np.float16).astype(np.float64)   # exact packed weights
    n1 = int((np.abs(w) > 0).sum())

    # spline on the PACKED forward over the anchor
    az = np.load(SCRATCH / "val" / "anchor.npz")
    Xa = az[space].astype(np.float64)
    raw_a = (Xa - mu) / sd @ w + bias
    cx, cy = fit_spline_knots(raw_a, az["y"].astype(np.float64))
    payload = struct.pack("<I", len(cx)) + b"".join(
        struct.pack("<ff", x, y) for x, y in zip(cx, cy))

    metadata = []
    if space == "shaped":
        transforms, tparams = load_transforms()
        metadata += [
            {"key": "zentrain.feature_transforms", "type": "utf8",
             "text": "\n".join(transforms)},
            {"key": "zentrain.feature_transform_params", "type": "utf8",
             "text": "\n".join(",".join(f"{p}" for p in row) if row else "" for row in tparams)},
        ]
    metadata.append({"key": SPLINE_KEY, "type": "bytes", "hex": payload.hex()})
    req = {
        "schema_hash": 0, "flags": 0, "compressed": True,
        "scaler_mean": [float(v) for v in mu.astype(np.float32)],
        "scaler_scale": [float(v) for v in sd.astype(np.float32)],
        "layers": [{
            "in_dim": N_FEAT, "out_dim": 1, "activation": "identity",
            "dtype": "f16" if use_f16 else "f32",
            "weights": [float(v) for v in w], "biases": [bias],
        }],
        "metadata": metadata,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(req, f)
        jp = Path(f.name)
    try:
        r = subprocess.run([str(BAKER), "bake", str(jp), str(out_path)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"zenpredict bake failed: {r.stderr[:400]}")
    finally:
        jp.unlink(missing_ok=True)
    vm = val_metrics(w, bias, mu, sd, space)
    return {"key": key, "tau": tau, "f16": use_f16, "n_active": n1, "n_active_pre": n0,
            "size": out_path.stat().st_size, "knots": len(cx),
            "sha256": hashlib.sha256(out_path.read_bytes()).hexdigest()[:12], **vm}


def cmd_finalize(args) -> int:
    (SCRATCH / "bakes").mkdir(parents=True, exist_ok=True)
    results = []
    for key in args.keys.split(","):
        for tau in [float(t) for t in args.taus.split(",")]:
            tag = f"{key}-tau{tau:g}-f16"
            out = SCRATCH / "bakes" / f"lp_{tag}.bin"
            info = bake_candidate(key, tau, out)
            results.append(info)
            print(f"{tag:44s} act={info['n_active']:3d} size={info['size']:6d}B "
                  f"knots={info['knots']:2d} bigval={info['bigcodec_val']:+.4f} "
                  f"hdrval={info['hdr_val']:+.4f} guard={abs(info['konjnd_guard']):.4f}",
                  flush=True)
    with open(SCRATCH / "bakes" / "finalize.json", "a") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    return 0


# ---------------------------------------------------------------------------
# Phase 2 (2026-07-03 evening): ensembles + cascade + residual-stack prep.

# Diverse head pool for convex ensembles. RAW feature space only — a convex
# blend of raw-space linear heads collapses to a SINGLE 372->1 linear layer
# (v_k = w_k/sd_k in raw space; scaler folds away), so the ensemble bakes as
# one tiny layer. Shaped heads are excluded (incompatible input transform).
HEAD_POOL = [
    ("cid", "hdrmix-lasso0.002-raw", 0.0),        # CID22 axis (0.8740)
    ("kon", "canonhdr15-bvls-raw", 0.005),        # KonJND axis (0.6696)
    ("upq", "canonhdr15-lasso0.0005-raw", 0.005), # UPIQ-raw axis (0.7148)
    ("kad", "canon-ridge1e-05-raw", 0.0),         # KADID/TID axis (0.88/0.86)
    ("cbv", "canon-bvls-raw", 0.005),             # 2026-05-28-recipe BVLS
    ("hds", "hdr-lasso0.001-raw", 0.0),           # pure-HDR ssim2 target
    ("pjt", "pjnd-ridge0.001-raw", 0.0),          # NEW: pjnd_target head
    # NNLS-all-pinned head dropped: it fit to all-zero weights (w>=0 for every
    # feature is infeasible for score prediction here) — negative result.
    ("s3h", "hdrmix300-lasso0.002-raw", 0.0),     # NEW: f0..f299 subset head
]

SEL_AXES = ["cid22tr_sel", "bigcodec_val", "hdr_val", "hdr_valmix", "konjnd_guard"]


def cmd_poolfits(args) -> int:
    """Three bespoke diversity fits for the ensemble pool."""
    out = SCRATCH / "fits"

    # 1. pjnd_target head: konjnd-dense-norm, target = per-ref PJND threshold.
    mg = MixGram("raw", [("konjnd_dense", 1.0, "pjnd_target")])
    w, b = mg.ridge(1e-3)
    np.savez_compressed(out / "pjnd-ridge0.001-raw.npz", w=w, bias=b, mu=mg.mu,
                        sd=mg.sd, space="raw", desc="konjnd_dense:1.0:pjnd_target")
    vm = val_metrics(w, b, mg.mu, mg.sd, "raw")
    print(f"pjnd-ridge0.001-raw         bigval={vm['bigcodec_val']:+.4f} "
          f"hdrval={vm['hdr_val']:+.4f} guard={abs(vm['konjnd_guard']):.4f}")

    # 2. NNLS (ALL weights pinned >= 0) on the canonhdr15 mix.
    mg = MixGram("raw", MIXES_HDR["canonhdr15"])
    w, b = mg.bvls(np.ones(N_FEAT, dtype=bool))
    np.savez_compressed(out / "canonhdr15nnls-bvls_allpin-raw.npz", w=w, bias=b,
                        mu=mg.mu, sd=mg.sd, space="raw", desc="canonhdr15 allpin")
    vm = val_metrics(w, b, mg.mu, mg.sd, "raw")
    nact = int((np.abs(w) > 1e-7).sum())
    print(f"canonhdr15nnls (act={nact})   bigval={vm['bigcodec_val']:+.4f} "
          f"hdrval={vm['hdr_val']:+.4f} guard={abs(vm['konjnd_guard']):.4f}")

    # 3. 300-feature subset head (f0..f299, no IW-pool block) on hdrmix.
    mg = MixGram("raw", MIXES_HDR["hdrmix"])
    sub = np.arange(300)
    Gs = mg.G[np.ix_(sub, sub)]; cs = mg.c[sub]
    Gn = Gs / mg.W; cn = cs / mg.W
    d = Gn.diagonal().copy(); d[d < 1e-12] = 1e-12
    ws = np.zeros(300); Gw = np.zeros(300); lam = 0.002
    for _ in range(200):
        delta = 0.0
        for j in range(300):
            rho = cn[j] - Gw[j] + d[j] * ws[j]
            nw = np.sign(rho) * max(abs(rho) - lam, 0.0) / d[j]
            if nw != ws[j]:
                Gw += Gn[:, j] * (nw - ws[j]); delta = max(delta, abs(nw - ws[j])); ws[j] = nw
        if delta < 1e-10:
            break
    w = np.zeros(N_FEAT); w[sub] = ws
    b = mg.ybar
    np.savez_compressed(out / "hdrmix300-lasso0.002-raw.npz", w=w, bias=b,
                        mu=mg.mu, sd=mg.sd, space="raw", desc="hdrmix f0..f299")
    vm = val_metrics(w, b, mg.mu, mg.sd, "raw")
    nact = int((np.abs(w) > 1e-7).sum())
    print(f"hdrmix300-lasso0.002 (act={nact}) bigval={vm['bigcodec_val']:+.4f} "
          f"hdrval={vm['hdr_val']:+.4f} guard={abs(vm['konjnd_guard']):.4f}")
    return 0


def _load_head_rawspace(key: str, tau: float):
    """Head -> raw-feature-space (v, c): pred = x·v + c."""
    z = np.load(SCRATCH / "fits" / f"{key}.npz")
    w = z["w"].astype(np.float64).copy()
    if tau > 0:
        w[np.abs(w) < tau] = 0.0
    mu = z["mu"].astype(np.float64); sd = z["sd"].astype(np.float64)
    v = w / sd
    c = float(z["bias"]) - float(mu @ v)
    return v, c


def _srocc_fast(q: np.ndarray, ry: np.ndarray) -> float:
    rq = np.empty(len(q)); rq[np.argsort(q, kind="stable")] = np.arange(len(q))
    rq -= rq.mean(); d = float(np.sqrt((rq * rq).sum()))
    return float((rq * ry).sum() / (d * np.linalg.norm(ry)))


def cmd_ensemble(args) -> int:
    """Convex blends over HEAD_POOL. alpha chosen ONLY on train-legal axes
    (SEL_AXES); 4 pre-registered scalarizations; every winner is baked and
    panel'd (report all)."""
    K = len(HEAD_POOL)
    heads = [_load_head_rawspace(k, t) for _, k, t in HEAD_POOL]
    aliases = [a for a, _, _ in HEAD_POOL]

    # Normalize each head's output scale on the anchor set (fixed reference).
    az = np.load(SCRATCH / "val" / "anchor.npz")
    A = az["raw"].astype(np.float64)
    norm = []
    for v, c in heads:
        p = A @ v + c
        m, s = float(p.mean()), float(p.std())
        s = s if s > 1e-9 else 1.0
        norm.append((m, s))
    V = np.column_stack([v / s for (v, c), (m, s) in zip(heads, norm)])   # 372 x K
    C = np.array([(c - m) / s for (v, c), (m, s) in zip(heads, norm)])    # K

    # Selection predictions (deterministic stride subsample on bigcodec_val).
    P, RY, ABS = {}, {}, {}
    for name in SEL_AXES:
        z = np.load(SCRATCH / "val" / f"{name}.npz")
        X = z["raw"].astype(np.float64); y = z["y"].astype(np.float64)
        if name == "bigcodec_val":
            X, y = X[::7], y[::7]
        P[name] = X @ V + C[None, :]
        ry = np.empty(len(y)); ry[np.argsort(y, kind="stable")] = np.arange(len(y))
        ry -= ry.mean()
        RY[name] = ry
        ABS[name] = name == "konjnd_guard"
        print(f"  sel axis {name}: n={len(y)}")

    def axes_of(alpha_idx, alpha):
        out = {}
        for name in SEL_AXES:
            q = P[name][:, alpha_idx] @ alpha
            r = _srocc_fast(q, RY[name])
            out[name] = abs(r) if ABS[name] else r
        return out

    # Corner (pure-head) axis values -> per-axis [0,1] normalization.
    corners = [axes_of([k], np.array([1.0])) for k in range(K)]
    lo = {n: min(c[n] for c in corners) for n in SEL_AXES}
    hi = {n: max(c[n] for c in corners) for n in SEL_AXES}

    def nrm(vals):
        return {n: (vals[n] - lo[n]) / max(hi[n] - lo[n], 1e-9) for n in SEL_AXES}

    SCALARIZATIONS = {
        "S1maximin": lambda z: min(z["cid22tr_sel"], z["bigcodec_val"], z["hdr_val"], z["konjnd_guard"]),
        "S2triple":  lambda z: 0.35 * z["cid22tr_sel"] + 0.35 * z["konjnd_guard"] + 0.30 * z["hdr_val"],
        "S3balance": lambda z: sum(z[n] for n in SEL_AXES) / len(SEL_AXES),
        "S4cidlean": lambda z: 0.55 * z["cid22tr_sel"] + 0.25 * z["konjnd_guard"] + 0.20 * z["hdr_val"],
    }
    best = {s: (-1e9, None, None, None) for s in SCALARIZATIONS}

    def consider(idx, alpha):
        vals = axes_of(idx, alpha)
        z = nrm(vals)
        for s, f in SCALARIZATIONS.items():
            sc = f(z)
            if sc > best[s][0]:
                best[s] = (sc, list(idx), alpha.copy(), vals)

    import itertools
    t0 = time.time()
    n_comb = 0
    for k in range(K):
        consider([k], np.array([1.0])); n_comb += 1
    for i, j in itertools.combinations(range(K), 2):
        for a in np.arange(0.05, 0.951, 0.05):
            consider([i, j], np.array([a, 1 - a])); n_comb += 1
    for tri in itertools.combinations(range(K), 3):
        for c1 in range(1, 9):
            for c2 in range(1, 10 - c1):
                a = np.array([c1, c2, 10 - c1 - c2]) / 10.0
                consider(list(tri), a); n_comb += 1
    for quad in itertools.combinations(range(K), 4):
        for c1 in range(1, 8):
            for c2 in range(1, 9 - c1):
                for c3 in range(1, 10 - c1 - c2):
                    a = np.array([c1, c2, c3, 10 - c1 - c2 - c3]) / 10.0
                    consider(list(quad), a); n_comb += 1
    print(f"searched {n_comb} alpha combos in {time.time()-t0:.1f}s")

    # Pre-registered extras (added before ANY ensemble panel was run):
    # S5: quality-axes-only blend over the pool WITHOUT pjt (the pjt head owns
    #     the guard axis by construction, and the guard is a known weak
    #     selector — S5 tests blending without it).
    # P-line: fixed cid<->kon 2-head line at alpha {0.3, 0.5, 0.7} — a direct
    #     probe of the CID22<->KonJND Pareto trade the triple gate asks about.
    pjt_i = aliases.index("pjt")
    best["S5noguard"] = (-1e9, None, None, None)
    s5 = lambda z: 0.45 * z["cid22tr_sel"] + 0.20 * z["bigcodec_val"] + 0.20 * z["hdr_val"] + 0.15 * z["hdr_valmix"]
    def consider5(idx, alpha):
        if pjt_i in idx:
            return
        vals = axes_of(idx, alpha)
        z = nrm(vals)
        sc = s5(z)
        if sc > best["S5noguard"][0]:
            best["S5noguard"] = (sc, list(idx), alpha.copy(), vals)
    for k in range(K):
        consider5([k], np.array([1.0]))
    for i, j in itertools.combinations(range(K), 2):
        for a in np.arange(0.05, 0.951, 0.05):
            consider5([i, j], np.array([a, 1 - a]))
    for tri in itertools.combinations(range(K), 3):
        for c1 in range(1, 9):
            for c2 in range(1, 10 - c1):
                consider5(list(tri), np.array([c1, c2, 10 - c1 - c2]) / 10.0)
    ci, ki = aliases.index("cid"), aliases.index("kon")
    for a in (0.3, 0.5, 0.7):
        alpha = np.array([a, 1 - a])
        best[f"Pline-cid{int(a*100)}"] = (0.0, [ci, ki], alpha, axes_of([ci, ki], alpha))

    (SCRATCH / "fits").mkdir(exist_ok=True)
    report = {}
    for s, (sc, idx, alpha, vals) in best.items():
        names = [aliases[i] for i in idx]
        w_ens = V[:, idx] @ alpha
        b_ens = float(C[idx] @ alpha)
        key = f"ens-{s}"
        np.savez_compressed(SCRATCH / "fits" / f"{key}.npz",
                            w=w_ens, bias=b_ens, mu=np.zeros(N_FEAT),
                            sd=np.ones(N_FEAT), space="raw",
                            desc=f"{s}: " + "+".join(f"{n}:{a:.2f}" for n, a in zip(names, alpha)))
        report[key] = {"score": sc, "heads": names, "alpha": [round(float(a), 3) for a in alpha],
                       "axes": {k: round(v, 4) for k, v in vals.items()}}
        print(f"{key:14s} -> {'+'.join(f'{n}:{a:.2f}' for n, a in zip(names, alpha)):40s} "
              f"axes: " + " ".join(f"{n.split('_')[0]}={vals[n]:+.4f}" for n in SEL_AXES))
    with open(SCRATCH / "fits" / "ensemble_report.json", "w") as f:
        json.dump(report, f, indent=1)
    return 0


def cmd_cascade(args) -> int:
    """2-stage linear cascade: BVLS base (KonJND-preserving) + sparse lasso
    correction fit on the TRAIN residual (still closed-form/deterministic;
    the sum is still one linear layer)."""
    mg = MixGram("raw", MIXES_HDR["canonhdr15"])
    z = np.load(SCRATCH / "fits" / "canonhdr15-bvls-raw.npz")
    w1 = z["w"].astype(np.float64).copy()
    w1[np.abs(w1) < 0.005] = 0.0     # the panel'd tau
    c_res = mg.c - mg.G @ w1
    Gn = mg.G / mg.W
    d = Gn.diagonal().copy(); d[d < 1e-12] = 1e-12
    for lam in (5e-4, 1e-3):
        cn = c_res / mg.W
        w2 = np.zeros(N_FEAT); Gw = np.zeros(N_FEAT)
        for _ in range(200):
            delta = 0.0
            for j in range(N_FEAT):
                rho = cn[j] - Gw[j] + d[j] * w2[j]
                nw = np.sign(rho) * max(abs(rho) - lam, 0.0) / d[j]
                if nw != w2[j]:
                    Gw += Gn[:, j] * (nw - w2[j]); delta = max(delta, abs(nw - w2[j])); w2[j] = nw
            if delta < 1e-10:
                break
        w = w1 + w2
        b = float(z["bias"])
        key = f"casc-bvlsbase-lasso{lam:g}"
        np.savez_compressed(SCRATCH / "fits" / f"{key}.npz", w=w, bias=b,
                            mu=z["mu"], sd=z["sd"], space="raw",
                            desc=f"canonhdr15-bvls(tau.005) + lasso{lam} residual correction")
        vm = val_metrics(w, b, z["mu"].astype(np.float64), z["sd"].astype(np.float64), "raw")
        n2 = int((np.abs(w2) > 1e-9).sum())
        print(f"{key:28s} corr_act={n2:3d} bigval={vm['bigcodec_val']:+.4f} "
              f"hdrval={vm['hdr_val']:+.4f} guard={abs(vm['konjnd_guard']):.4f}")
    return 0


RESIDUAL_SETS = [
    # (out_name, source_path, affine_from)  — val files reuse the TRAIN affine
    ("bigcodec_traindigits_residual", PROBE / "bigcodec_traindigits_2026-07-02.parquet", None),
    ("bigcodec_valdigits_residual", PROBE / "bigcodec_valdigits_2026-07-02.parquet", "bigcodec_traindigits_residual"),
    ("hdr_zenjxl_v3_traindigits_residual", PROBE / "hdr_zenjxl_v3_traindigits_2026-07-03.parquet", None),
    ("hdr_zenjxl_v3_valdigits_residual", PROBE / "hdr_zenjxl_v3_valdigits_2026-07-03.parquet", "hdr_zenjxl_v3_traindigits_residual"),
]
BASE_KEY, BASE_TAU = "hdrmix-lasso0.002-raw", 0.0


def cmd_residual(args) -> int:
    """Residual-stack corpora: target = human_score - (a*base_pred + b), with
    (a, b) OLS-fit on the TRAIN file and reused for its val file. base = the
    SDR pick (hdrmix-lasso0.002-raw tau0), applied EXACTLY as the f16 bake
    stores it (f16-rounded weights, f32 scaler, f32 math)."""
    import pyarrow as pa
    outdir = SCRATCH / "residual"
    outdir.mkdir(exist_ok=True)
    z = np.load(SCRATCH / "fits" / f"{BASE_KEY}.npz")
    w16 = z["w"].astype(np.float16).astype(np.float32)
    mu32 = z["mu"].astype(np.float32); sd32 = z["sd"].astype(np.float32)
    b32 = np.float32(z["bias"])
    base_bake = SCRATCH / "bakes" / f"lp_{BASE_KEY}-tau0-f16.bin"
    base_sha = hashlib.sha256(base_bake.read_bytes()).hexdigest()
    print(f"base bake: {base_bake.name} sha256={base_sha}")

    # Clamp the base pred to the anchor-observed domain (the dial spline's
    # trusted raw range). Without this, the sparse HDR-fit head extrapolates
    # wildly on OOD bigcodec rows (raw residuals reached +158 on valdigits and
    # the outliers squashed the OLS slope to 0.17). The clamp is exactly
    # replicable at runtime (two constants, recorded in the manifest).
    az = np.load(SCRATCH / "val" / "anchor.npz")
    pa_anchor = ((az["raw"].astype(np.float32) - mu32) / sd32 @ w16 + b32).astype(np.float64)
    clamp_lo, clamp_hi = float(pa_anchor.min()), float(pa_anchor.max())
    print(f"pred clamp domain (anchor min/max): [{clamp_lo:.4f}, {clamp_hi:.4f}]")

    affines: dict[str, tuple[float, float]] = {}
    manifest = {"base_bake": str(base_bake), "base_bake_sha256": base_sha,
                "base_fit_key": BASE_KEY,
                "pred_clamp": [clamp_lo, clamp_hi],
                "definition":
                "residual_target = human_score - clip01(a*clip(linear_pred, clamp_lo, clamp_hi) + b); "
                "(a,b) OLS on the TRAIN file's clipped preds, reused for its val file; "
                "clip01 bounds the composed base to [0,1] so residual_target is in [-1,1] "
                "by construction (runtime composition: final = clip01(a*base+b) + residual_mlp)",
                "files": {}}
    for out_name, src, affine_from in RESIDUAL_SETS:
        pf = pq.ParquetFile(src)
        # pass 1: accumulate pred/target moments for the affine (train files)
        if affine_from is None:
            sp = sy = spp = spy = n = 0.0
            for batch in pf.iter_batches(batch_size=131072, columns=FCOLS + ["human_score"]):
                X = np.column_stack([batch[c].to_numpy(zero_copy_only=False).astype(np.float32) for c in FCOLS])
                p = np.clip(((X - mu32) / sd32 @ w16 + b32).astype(np.float64), clamp_lo, clamp_hi)
                y = batch["human_score"].to_numpy(zero_copy_only=False).astype(np.float64)
                sp += p.sum(); sy += y.sum(); spp += (p * p).sum(); spy += (p * y).sum(); n += len(y)
            varp = spp / n - (sp / n) ** 2
            a = (spy / n - sp / n * sy / n) / max(varp, 1e-12)
            b = sy / n - a * sp / n
            affines[out_name] = (float(a), float(b))
        else:
            a, b = affines[affine_from]
        # pass 2: stream-write with linear_pred + residual_target
        pf = pq.ParquetFile(src)
        out_path = outdir / f"{out_name}_2026-07-03.parquet"
        writer = None
        rmin, rmax, rows = np.inf, -np.inf, 0
        for batch in pf.iter_batches(batch_size=131072):
            t = pa.Table.from_batches([batch])
            X = np.column_stack([t[c].to_numpy(zero_copy_only=False).astype(np.float32) for c in FCOLS])
            p = np.clip(((X - mu32) / sd32 @ w16 + b32).astype(np.float64), clamp_lo, clamp_hi)
            y = t["human_score"].to_numpy(zero_copy_only=False).astype(np.float64)
            r = y - np.clip(a * p + b, 0.0, 1.0)
            rmin = min(rmin, float(r.min())); rmax = max(rmax, float(r.max())); rows += len(r)
            cols = {"ref_basename": t["ref_basename"], "human_score": t["human_score"],
                    "linear_pred": pa.array(p), "residual_target": pa.array(r)}
            for c in FCOLS:
                cols[c] = t[c]
            ot = pa.table(cols)
            if writer is None:
                meta = {b"residual_base_bake_sha256": base_sha.encode(),
                        b"residual_affine_a": f"{a!r}".encode(),
                        b"residual_affine_b": f"{b!r}".encode(),
                        b"residual_source": str(src).encode()}
                writer = pq.ParquetWriter(out_path, ot.schema.with_metadata(meta), compression="zstd")
                ot = ot.replace_schema_metadata(meta)
            writer.write_table(ot)
        writer.close()
        sha = hashlib.sha256()
        with open(out_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                sha.update(chunk)
        manifest["files"][out_path.name] = {
            "rows": rows, "sha256": sha.hexdigest(), "affine_a": a, "affine_b": b,
            "residual_range": [rmin, rmax], "source": str(src)}
        print(f"{out_path.name}: rows={rows} a={a:.4f} b={b:+.4f} "
              f"residual range [{rmin:+.4f}, {rmax:+.4f}] sha256={sha.hexdigest()}")
    with open(outdir / "_MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"manifest: {outdir / '_MANIFEST.json'}")
    return 0


# ---------------------------------------------------------------------------
# w10b (2026-07-03 night): SDR residual corpora v2 with the MIX target.
# w10 v1's SDR residual was falsified at every lambda (ssim2-target defect:
# composed CID22 0.874 -> 0.847) while the HDR residual with the mix-derived
# target GAINED +0.041 UPIQ. v2 rebuilds the SDR residual against
#   target_mix = 0.5*clip01(ssim2_norm) + 0.5*clip01((cvvdp_jod - 6)/4)
# (the hdr_v3mix convention), same base + a,b OLS + clip01 recipe.
MM6_TRAIN = PROBE / "bigcodec_mm6_traindigits_2026-07-02.parquet"
FILL4_SIDECAR = Path("/mnt/v/datasets/fill4-6codec-2026-07-01/fill4metrics_sidecar_patched_2026-07-02.parquet")
CANON_PICKER = Path("/mnt/v/output/canonical-picker-2026-06-27")
MM6_DATASETS = ["zenjpeg_lossy", "zenjxl_lossless", "zenjxl_lossy",
                "zenpng_lossless", "zenwebp_lossless", "zenwebp_lossy"]


def _mix_target(hs: np.ndarray, cvvdp: np.ndarray) -> np.ndarray:
    return 0.5 * np.clip(hs, 0.0, 1.0) + 0.5 * np.clip((cvvdp - 6.0) / 4.0, 0.0, 1.0)


def cmd_residual2(args) -> int:
    """v2 SDR residual corpora (mix target). Train side streams the mm6 join;
    val side is built by the SAME sidecar join over the canonical picker
    validate splits (sidecar coverage verified 100%)."""
    import pyarrow as pa
    sys.path.insert(0, "/home/lilith/work/zen/zenmetrics/scripts/picker")
    from origin_split import split_of

    outdir = SCRATCH / "residual"
    outdir.mkdir(exist_ok=True)
    z = np.load(SCRATCH / "fits" / f"{BASE_KEY}.npz")
    w16 = z["w"].astype(np.float16).astype(np.float32)
    mu32 = z["mu"].astype(np.float32); sd32 = z["sd"].astype(np.float32)
    b32 = np.float32(z["bias"])
    base_bake = SCRATCH / "bakes" / f"lp_{BASE_KEY}-tau0-f16.bin"
    base_sha = hashlib.sha256(base_bake.read_bytes()).hexdigest()
    az = np.load(SCRATCH / "val" / "anchor.npz")
    pa_anchor = ((az["raw"].astype(np.float32) - mu32) / sd32 @ w16 + b32).astype(np.float64)
    clamp_lo, clamp_hi = float(pa_anchor.min()), float(pa_anchor.max())
    print(f"base sha256={base_sha}  pred clamp [{clamp_lo:.4f}, {clamp_hi:.4f}]", flush=True)

    def base_pred(X32: np.ndarray) -> np.ndarray:
        return np.clip(((X32 - mu32) / sd32 @ w16 + b32).astype(np.float64), clamp_lo, clamp_hi)

    # ---- pass 1 over mm6 TRAIN: affine moments on the mix target ----
    pf = pq.ParquetFile(MM6_TRAIN)
    sp = sy = spp = spy = n = 0.0
    n_nan = 0
    for batch in pf.iter_batches(batch_size=131072, columns=FCOLS + ["human_score", "score_cvvdp"]):
        X = np.column_stack([batch[c].to_numpy(zero_copy_only=False).astype(np.float32) for c in FCOLS])
        hs = batch["human_score"].to_numpy(zero_copy_only=False).astype(np.float64)
        cv = batch["score_cvvdp"].to_numpy(zero_copy_only=False).astype(np.float64)
        keep = np.isfinite(cv) & np.isfinite(hs)
        n_nan += int((~keep).sum())
        y = _mix_target(hs[keep], cv[keep])
        p = base_pred(X[keep])
        sp += p.sum(); sy += y.sum(); spp += (p * p).sum(); spy += (p * y).sum(); n += len(y)
    varp = spp / n - (sp / n) ** 2
    a = (spy / n - sp / n * sy / n) / max(varp, 1e-12)
    b = sy / n - a * sp / n
    print(f"mm6 train: n={n:.0f} nan_cvvdp_dropped={n_nan}  affine a={a:.4f} b={b:+.4f}", flush=True)

    manifest_path = outdir / "_MANIFEST.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"files": {}}
    manifest["v2_mix"] = {
        "base_bake_sha256": base_sha, "pred_clamp": [clamp_lo, clamp_hi],
        "target": "target_mix = 0.5*clip01(ssim2_norm) + 0.5*clip01((score_cvvdp-6)/4)  (hdr_v3mix convention)",
        "definition": "residual_target = target_mix - clip01(a*clip(linear_pred) + b); "
                      "(a,b) OLS on mm6 TRAIN mix rows, reused for the val join",
        "affine_a": float(a), "affine_b": float(b), "nan_cvvdp_dropped_train": n_nan,
    }

    def write_stream(out_path: Path, batches, src_desc: str):
        writer = None
        rmin, rmax, rows = np.inf, -np.inf, 0
        for names, hs, cv, X in batches:
            keep = np.isfinite(cv) & np.isfinite(hs)
            if not keep.any():
                continue
            names = [nm for nm, k in zip(names, keep) if k]
            hs, cv, X = hs[keep], cv[keep], X[keep]
            y = _mix_target(hs, cv)
            p = base_pred(X.astype(np.float32))
            r = y - np.clip(a * p + b, 0.0, 1.0)
            rmin = min(rmin, float(r.min())); rmax = max(rmax, float(r.max())); rows += len(r)
            cols = {"ref_basename": pa.array(names), "human_score": pa.array(hs),
                    "score_cvvdp": pa.array(cv), "target_mix": pa.array(y),
                    "linear_pred": pa.array(p), "residual_target": pa.array(r)}
            for i, c in enumerate(FCOLS):
                cols[c] = pa.array(X[:, i])
            ot = pa.table(cols)
            if writer is None:
                meta = {b"residual_base_bake_sha256": base_sha.encode(),
                        b"residual_affine_a": f"{a!r}".encode(),
                        b"residual_affine_b": f"{b!r}".encode(),
                        b"residual_target_def": b"mix(ssim2,cvvdp) - clip01(a*clip(pred)+b)",
                        b"residual_source": src_desc.encode()}
                writer = pq.ParquetWriter(out_path, ot.schema.with_metadata(meta), compression="zstd")
                ot = ot.replace_schema_metadata(meta)
            writer.write_table(ot)
        writer.close()
        sha = hashlib.sha256()
        with open(out_path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                sha.update(chunk)
        manifest["files"][out_path.name] = {
            "rows": rows, "sha256": sha.hexdigest(), "affine_a": float(a), "affine_b": float(b),
            "residual_range": [rmin, rmax], "source": src_desc, "target": "mix(ssim2,cvvdp)"}
        print(f"{out_path.name}: rows={rows} residual range [{rmin:+.4f}, {rmax:+.4f}] "
              f"sha256={sha.hexdigest()}", flush=True)

    # ---- TRAIN v2: stream mm6 ----
    def train_batches():
        pf = pq.ParquetFile(MM6_TRAIN)
        for batch in pf.iter_batches(batch_size=131072,
                                     columns=["ref_basename", "human_score", "score_cvvdp"] + FCOLS):
            t = pa.Table.from_batches([batch])
            yield (t["ref_basename"].to_pylist(),
                   t["human_score"].to_numpy(zero_copy_only=False).astype(np.float64),
                   t["score_cvvdp"].to_numpy(zero_copy_only=False).astype(np.float64),
                   np.column_stack([t[c].to_numpy(zero_copy_only=False).astype(np.float64) for c in FCOLS]))
    write_stream(outdir / "bigcodec_mm6mix_traindigits_residual_v2_2026-07-03.parquet",
                 train_batches(), str(MM6_TRAIN))

    # ---- VAL v2: sidecar join over canonical validate splits ----
    side = pq.read_table(FILL4_SIDECAR, columns=["encoded_filename", "score_cvvdp"])
    lut = dict(zip(side["encoded_filename"].to_pylist(),
                   np.asarray(side["score_cvvdp"], dtype=np.float64)))
    del side
    print(f"sidecar cvvdp lut: {len(lut):,}", flush=True)
    seen: set = set()
    stats = {"dup": 0, "nonval": 0, "nomatch": 0}

    def val_batches():
        import os as _os
        for ds in MM6_DATASETS:
            p = CANON_PICKER / ds / "validate.parquet"
            pf = pq.ParquetFile(p)
            featcols = sorted([c for c in pf.schema_arrow.names
                               if c.startswith("feat_") and c[5:].isdigit()],
                              key=lambda c: int(c[5:]))
            for batch in pf.iter_batches(batch_size=65536,
                                         columns=["image_path", "encoded_filename", "score_ssim2"] + featcols):
                t = pa.Table.from_batches([batch])
                names = [_os.path.basename(x) for x in t["image_path"].to_pylist()]
                hs = np.clip(np.asarray(t["score_ssim2"], dtype=np.float64) / 100.0, 0.0, 1.0)
                F = np.column_stack([np.asarray(t[c], dtype=np.float64) for c in featcols])
                cv = np.array([lut.get(fn, np.nan) for fn in t["encoded_filename"].to_pylist()])
                stats["nomatch"] += int(np.isnan(cv).sum())
                keep = []
                for i, (nm, h) in enumerate(zip(names, hs)):
                    k = (nm, round(float(h), 9), round(float(F[i][0]), 9),
                         round(float(F[i][1]), 9), round(float(F[i][2]), 9))
                    if k in seen:
                        stats["dup"] += 1
                    elif split_of(nm) == "val":
                        seen.add(k); keep.append(i)
                    else:
                        seen.add(k); stats["nonval"] += 1
                if keep:
                    yield ([names[i] for i in keep], hs[keep], cv[keep], F[keep])
        # hqfill val-origin rows (19,082 of 62,173 — near-lossless JXL band;
        # joined exactly as build_mm6_join.py does for train: key =
        # (rendition basename minus sha, knob_tuple_json) -> cvvdp).
        hq_side = pq.read_table(PROBE / "hqfill_7metric_sidecar_2026-07-02.parquet",
                                columns=["encoded_filename", "knob_tuple_json",
                                         "score_cvvdp_imazen_v0_0_1"])
        hq2 = {}
        for fn, kj, cvv in zip(hq_side["encoded_filename"].to_pylist(),
                               hq_side["knob_tuple_json"].to_pylist(),
                               np.asarray(hq_side["score_cvvdp_imazen_v0_0_1"], dtype=np.float64)):
            rb = "_".join(fn.split("_zenjxl_")[0].split("_")[:-1])
            hq2[(rb, kj)] = cvv
        pf = pq.ParquetFile("/mnt/v/datasets/jxl-lossy-hqfill-A/zenjxl_lossy_hqfill_A_features_2026-07-02.parquet")
        featcols = sorted([c for c in pf.schema_arrow.names
                           if c.startswith("feat_") and c[5:].isdigit()],
                          key=lambda c: int(c[5:]))
        for batch in pf.iter_batches(batch_size=65536,
                                     columns=["image_path", "knob_tuple_json", "score_ssim2"] + featcols):
            t = pa.Table.from_batches([batch])
            names = [_os.path.basename(x) for x in t["image_path"].to_pylist()]
            kjs = t["knob_tuple_json"].to_pylist()
            hs = np.clip(np.asarray(t["score_ssim2"], dtype=np.float64) / 100.0, 0.0, 1.0)
            F = np.column_stack([np.asarray(t[c], dtype=np.float64) for c in featcols])
            cv = np.array([hq2.get((_os.path.splitext(nm)[0] if nm.endswith(".png") else nm, k),
                                   hq2.get((nm, k), np.nan))
                           for nm, k in zip(names, kjs)], dtype=np.float64)
            stats["nomatch"] += int(np.isnan(cv).sum())
            keep = []
            for i, (nm, h) in enumerate(zip(names, hs)):
                k = (nm, round(float(h), 9), round(float(F[i][0]), 9),
                     round(float(F[i][1]), 9), round(float(F[i][2]), 9))
                if k in seen:
                    stats["dup"] += 1
                elif split_of(nm) == "val":
                    seen.add(k); keep.append(i)
                else:
                    seen.add(k); stats["nonval"] += 1
            if keep:
                yield ([names[i] for i in keep], hs[keep], cv[keep], F[keep])
    write_stream(outdir / "bigcodec_mm6mix_valdigits_residual_v2_2026-07-03.parquet",
                 val_batches(),
                 f"{CANON_PICKER}/<ds>/validate.parquet x {len(MM6_DATASETS)} + fill4 sidecar "
                 "+ hqfill-A val-origin rows w/ hqfill 7-metric sidecar (val-split join, "
                 "same key/dedup semantics as build_mm6_join.py)")
    manifest["v2_mix"]["val_join_stats"] = stats
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"manifest updated: {manifest_path}  val-join stats {stats}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gram"); g.add_argument("--force", action="store_true")
    f = sub.add_parser("fit"); f.add_argument("--only", default=None)
    z = sub.add_parser("finalize")
    z.add_argument("--keys", required=True)
    z.add_argument("--taus", default="0")
    sub.add_parser("poolfits")
    sub.add_parser("ensemble")
    sub.add_parser("cascade")
    sub.add_parser("residual")
    sub.add_parser("residual2")
    args = ap.parse_args()
    return {"gram": cmd_gram, "fit": cmd_fit, "finalize": cmd_finalize,
            "poolfits": cmd_poolfits, "ensemble": cmd_ensemble,
            "cascade": cmd_cascade, "residual": cmd_residual,
            "residual2": cmd_residual2}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
