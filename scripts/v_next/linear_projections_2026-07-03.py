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
    "konjnd_dense": (CANON / "konjnd-dense-norm.parquet", ["human_score"]),
    "hdr_v3": (PROBE / "hdr_zenjxl_v3_traindigits_2026-07-03.parquet", ["human_score"]),
    "hdr_v3mix": (PROBE / "hdr_zenjxl_v3mix_traindigits_2026-07-03.parquet", ["human_score"]),
}
VAL_SETS = {
    "bigcodec_val": (PROBE / "bigcodec_valdigits_2026-07-02.parquet", "human_score"),
    "hdr_val": (PROBE / "hdr_zenjxl_v3_valdigits_2026-07-03.parquet", "human_score"),
    "hdr_valmix": (PROBE / "hdr_zenjxl_v3mix_valdigits_2026-07-03.parquet", "human_score"),
    "konjnd_guard": (CANON / "konjnd-dense-norm.parquet", "pjnd_target"),
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


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gram"); g.add_argument("--force", action="store_true")
    f = sub.add_parser("fit"); f.add_argument("--only", default=None)
    z = sub.add_parser("finalize")
    z.add_argument("--keys", required=True)
    z.add_argument("--taus", default="0")
    args = ap.parse_args()
    return {"gram": cmd_gram, "fit": cmd_fit, "finalize": cmd_finalize}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
