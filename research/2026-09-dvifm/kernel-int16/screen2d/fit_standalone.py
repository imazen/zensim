#!/usr/bin/env python3
"""Phase-2d Part D — standalone DVIFM constants fitter (convex head).

Model (prereg benchmarks/dvifm_screen2d_prereg_2026-09-19.md §5.1 +
Amendment 1):
  per (plane p, level l):  s_{p,l} = mean_b( v_b * m_b^P )
    v_b = max(vis(C_s), vis(C_d))   — the pool_block visibility,
        vis(c) = exp(-softplus(beta*sharp*(ln c - ln c0))/sharp), c>0
    e_b = m_b^P                     — block max |ref-dist|
  E_{p,l} = s^{1/P}
  E_p     = sum_l w_{p,l} E_{p,l}   — w_{p,.} on the simplex (softmax)
  E       = cY*E_Y' + cC*(E_Cb + E_Cr)  — (cY, cC, cC) on the simplex
  yhat    = A*exp(-lam*E) + B,  A > 0   — output map REFIT per evaluation

Convex by construction: every mixing weight >= 0, each mix sums to 1,
E is non-decreasing in every s_{p,l} and every m_b, E=0 at identity, and
A > 0 keeps yhat non-increasing in E with identity at the top of scale.

Amendment-1 fitting (replaces the fixed-lambda objective — the first run
ranked grid cells by how well they rescale E to a stale lambda, every
optimum landed on the C0=0.3 / beta=1.4 bounds, and beta=1.4 was a logit
infinity that crashed native3):

  * Every loss evaluation — sanity gate, grid cell, Adam objective,
    accept/reject, reported MSE — first refits the output map: for a
    given lam, (A,B) by constrained least squares on exp(-lam*E) (A<=0
    -> boundary A->0+, i.e. the constant predictor, mse=var(y)); lam by
    golden-section on log lam over [LAM_LO, LAM_HI], <= MAP_MAX_EVALS
    evaluations. Gradients hold the map at its argmin (envelope form).
  * Sanity gate: init params must beat the constant predictor
    (refit MSE < var(y)) on each fit set before any grid; else STOP.
  * Grid: C0 in [1e-4, 3] log-spaced x18, beta in [0.05, 3] log-spaced
    x18; each cell records refit-MSE + SROCC + KROCC of -E vs y;
    selection by refit MSE; all three surfaces committed. Edge optima
    extend the edged axis once (GRID_EXTEND pts); a still-edged optimum
    is reported, never silently clamped.
  * Parameterisation bounds strictly contain the grid: g in (0.2, 2.0)
    logit, beta in (0.01, 100) logit, c0 = exp(raw) in [1e-8, 1e4],
    sharp = 1+softplus. Grid endpoints encode interiorly.
  * >=8 multi-start Adam refinements per (p,l) from the best cells;
    head refit (level/channel logits) after each sweep; alternate until
    relative loss gain < 1e-4 or 3 sweeps.

Estimator note (recorded): Adam steps evaluate a FIXED uniform block
subsample (<= ADAM_BLOCK_CAP blocks per row per level, deterministic
stride) — an unbiased estimate of the per-row mean. Grids and all
reported losses use ALL blocks (the subsampled grid ranking is verified
by the full-block accept check before any level commits).

Loss: refit-map MSE(yhat, target_0_100) where target_0_100 =
human_score*100 in the lineage pairs-TSV convention (CID22 MCOS/100,
TID MOS/9, KADID (dmos-1)/4 — quality-oriented source column, corrected
per Amendment 1 — imazen26 score_ssim2/100, konfig 1-q/3.2).

Usage:
  fit_standalone.py fit --variant {luma,native3,xyb-y} \
      --cache plane=bin ... --pairs tsv [--pairs tsv2 ...] \
      --init-spec k1.json --out fit.json --surfaces-dir dir
  fit_standalone.py score --fit fit.json --cache plane=bin ... \
      --pairs tsv --out scores.csv
  fit_standalone.py eval-consts --spec spec.json --variant xyb-y \
      --cache plane=bin --pairs tsv
      (K1 control: constants as given, uniform level weights,
       lambda by 1D golden search — the only free scalar)
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
import multiprocessing as _mp
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# fork-pool safety: pin BLAS to one thread BEFORE numpy loads it, so
# forked workers never inherit a threaded BLAS pool. The fit workload is
# elementwise, not BLAS-bound — no numeric effect on the estimator.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import scipy.stats

F32 = np.dtype("<f4")
REC = 18
LEVELS = 5
ADAM_BLOCK_CAP = 512
# Amendment-1 widened grid; parameterisation bounds strictly contain it.
GRID_C0 = np.exp(np.linspace(math.log(1e-4), math.log(3.0), 18))
GRID_BETA = np.exp(np.linspace(math.log(0.05), math.log(3.0), 18))
GRID_EXTEND = 8          # extra points per edged axis, one extension max
N_STARTS = 8
ADAM_STEPS_LEVEL = 150
ADAM_STEPS_HEAD = 250
ADAM_LR = 0.03
MAX_SWEEPS = 3
REL_STOP = 1e-4
G_LO, G_HI = 0.2, 2.0
B_LO, B_HI = 0.01, 100.0   # contains the base grid AND one extension
C0_LO, C0_HI = 1e-8, 1e4
LAM_LO, LAM_HI = 1e-9, 1e9
MAP_MAX_EVALS = 40
# memory bounds (implementation detail — identical arithmetic): a full
# level's records are gathered in <= FULL_CHUNK_RECS chunks (~288MB f64
# transient; imazen26 l0 alone is 109M recs = 15.8GB f64) and at most
# SUB_RECS_CACHE subsample recs arrays stay resident at once
FULL_CHUNK_RECS = 2_000_000
SUB_RECS_CACHE = 2

PLANES_3 = ("ycbcr_y", "ycbcr_cb", "ycbcr_cr")
PLANES_LUMA = ("ycbcr_y",)
PLANES_XYB = ("xyb_y",)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


# ---------- fork-pool parallelism ----------
# The Adam chains and the grid cells are PURE deterministic functions of
# their argument plus read-only shared state (record arrays, Ssub, params,
# y). Threads only parallelize where one long numpy op holds the GIL — the
# many-small-op steps on small levels serialize. Forked children inherit
# the parent's whole address space copy-on-write — including the warmed
# subsample record arrays — so they see that state for free and only pay
# for their own temporaries. Results are bit-identical to serial execution.
# Callers MUST warm caches[p].level_arrays(l, sub=True) in the parent
# before par_map so children inherit the records instead of re-reading.
_FORK_CTX = _mp.get_context("fork")
_FORK_FUNCS = {}
PAR_WORKERS = 8
PAR_TIMEOUT_S = 7200.0     # safety bound; a legit level's Adam is << this


def _pool_apply(key_arg):
    key, arg = key_arg
    return _FORK_FUNCS[key](arg)


def par_map(key, func, args, workers=PAR_WORKERS):
    """map(func, args) over forked workers; falls back to threads on any
    pool failure. `func` takes a single picklable arg."""
    args = list(args)
    if not args:
        return []
    _FORK_FUNCS[key] = func
    try:
        with _FORK_CTX.Pool(min(workers, len(args))) as pool:
            return pool.map_async(
                _pool_apply, [(key, a) for a in args]).get(
                    timeout=PAR_TIMEOUT_S)
    except Exception as exc:
        log(f"par_map {key}: process pool failed ({exc!r}) — "
            f"retrying on threads (identical results)")
        with ThreadPoolExecutor(max_workers=min(8, len(args))) as ex:
            return list(ex.map(func, args))


def softplus(x):
    return np.logaddexp(0.0, x)


def phi_g(x, g):
    return np.sign(x) * np.abs(x) ** g


# ---------- output map (Amendment 1) ----------

def _lsq_exp(E, y, lam):
    """Constrained LS of y ~ A*exp(-lam*E) + B with A >= 0.

    A <= 0 unconstrained -> the constrained optimum is the boundary
    A -> 0+, i.e. yhat = mean(y); returns mse = var(y) so the cell is
    honestly recorded as 'no better than constant'."""
    x = np.exp(-lam * E)
    xm, ym = float(x.mean()), float(y.mean())
    vy = float(np.mean((y - ym) ** 2))
    vx = float(np.mean((x - xm) ** 2))
    if vx < 1e-30:
        return 0.0, ym, vy
    a = float(np.mean((x - xm) * (y - ym)) / vx)
    if a <= 0.0:
        return 0.0, ym, vy
    b = ym - a * xm
    r = a * x + b - y
    return a, b, float(np.mean(r * r))


def fit_map(E, y, lo=LAM_LO, hi=LAM_HI, max_evals=MAP_MAX_EVALS):
    """Fit yhat = A*exp(-lam*E) + B (A > 0): (A,B) by constrained LS per
    lambda; lambda by golden-section on log lambda, <= max_evals evals."""
    llo, lhi = math.log(lo), math.log(hi)
    gr = (math.sqrt(5.0) - 1.0) / 2.0

    def f(ll):
        return _lsq_exp(E, y, math.exp(ll))[2]

    x1, x2 = lhi - gr * (lhi - llo), llo + gr * (lhi - llo)
    f1, f2 = f(x1), f(x2)
    ne = 2
    while ne < max_evals:
        if f1 > f2:
            llo = x1
            x1, f1 = x2, f2
            x2 = llo + gr * (lhi - llo)
            f2 = f(x2)
        else:
            lhi = x2
            x2, f2 = x1, f1
            x1 = lhi - gr * (lhi - llo)
            f1 = f(x1)
        ne += 1
    ll = 0.5 * (llo + lhi)
    a, b, mse = _lsq_exp(E, y, math.exp(ll))
    return {"A": a, "B": b, "lambda": math.exp(ll), "mse": mse}


def map_resid(E, y, fixed_map=None):
    """(mse, yhat, dL/dE) under the refit (or caller-fixed) output map.

    dL/dE holds (A, B, lam) at their argmin — exact to first order for the
    interior (A, B) and the golden-sectioned lambda (envelope form)."""
    mp = fixed_map if fixed_map is not None else fit_map(E, y)
    lam, a, b = mp["lambda"], mp["A"], mp["B"]
    x = np.exp(-lam * E)
    yhat = a * x + b
    r = yhat - y
    dle = (2.0 / len(y)) * r * (-lam * a * x)
    return float(np.mean(r * r)), yhat, dle


def _rank_stats(E, y):
    """Scale-free rank criteria of -E vs y (SROCC, KROCC)."""
    sro = scipy.stats.spearmanr(-E, y).statistic
    kro = scipy.stats.kendalltau(-E, y).statistic
    return float(sro), float(kro)


# ---------- cache ----------

def load_index(bin_path):
    idx = [json.loads(x) for x in
           Path(str(bin_path) + ".index.jsonl").read_text().splitlines()
           if x.strip()]
    idx.sort(key=lambda e: e["row_index"])
    return idx


class PlaneCache:
    """Memmap'd DVIFM block cache for one (domain, plane).

    level_arrays(l, sub=True) -> (recs (N,18) f64, bounds, counts) CSR over
    rows that have >=1 block; nz_rows[l] maps CSR row -> source row.
    Full-block access is via iter_level(l, sub=False): a whole level's recs
    can exceed RAM (imazen26 l0 = 109M recs = 15.8GB f64) so it is gathered
    in <= FULL_CHUNK_RECS chunks split at row boundaries — the chunked row
    sums are bitwise identical to the unchunked reduceat.
    """

    def __init__(self, bin_path):
        self.bin_path = str(bin_path)
        self.index = load_index(bin_path)
        self.data = np.memmap(bin_path, dtype=F32, mode="r")
        self.n_rows = len(self.index)
        self.nb = np.zeros((self.n_rows, LEVELS), np.int64)
        self.row_elem = np.zeros((self.n_rows, LEVELS), np.int64)
        for i, e in enumerate(self.index):
            base = e["offset"] // 4
            cnt = np.asarray(e["level_records"], np.int64)
            self.nb[i] = cnt
            self.row_elem[i] = base + np.cumsum(
                np.concatenate([[0], cnt]))[:5] * REC
        self.nz_rows = [np.nonzero(self.nb[:, l])[0] for l in range(LEVELS)]
        # deterministic per-row block subsample for Adam steps
        self._sub = [None] * LEVELS
        self._full = [None] * LEVELS
        # f64 record cache: the gather+astype dominates s_level; the SAME recs
        # are reused by every optimizer call on a level, so keep the sub copy
        # — LRU-bounded: 15 levels of recs can exceed the mem cap on large
        # domains (imazen26 sub ~0.9GB/level)
        self._recs_sub = OrderedDict()

    def _build(self, l, sub):
        ix, bounds, counts = [], [], []
        pos = 0
        for r in self.nz_rows[l]:
            n = int(self.nb[r, l])
            start = self.row_elem[r, l] // REC
            if sub:
                stride = max(1, -(-n // ADAM_BLOCK_CAP))
                k = -(-n // stride)
                bidx = start + np.arange(k) * stride
            else:
                k = n
                bidx = np.arange(start, start + n)
            bounds.append(pos)
            counts.append(k)
            ix.append(bidx)
            pos += k
        flat = np.concatenate(ix) if ix else np.zeros(0, np.int64)
        if flat.size and flat.max() < np.iinfo(np.int32).max:
            flat = flat.astype(np.int32)   # record ids fit in int32 —
            # halves the resident index (imazen26 l0: 109M ids = 436MB)
        return flat, np.asarray(bounds), np.asarray(counts)

    def level_arrays(self, l, sub=True):
        assert sub, "full blocks via iter_level(l, False) — a whole " \
                    "level can exceed RAM"
        if self._sub[l] is None:
            self._sub[l] = self._build(l, sub=True)
        flat, bounds, counts = self._sub[l]
        if l in self._recs_sub:
            self._recs_sub.move_to_end(l)
            return self._recs_sub[l], bounds, counts
        # flat holds RECORD indices; memmap is f32-element-indexed
        recs = self.data.reshape(-1, REC)[flat].astype(np.float64)
        self._recs_sub[l] = recs
        self._recs_sub.move_to_end(l)
        while len(self._recs_sub) > SUB_RECS_CACHE:
            self._recs_sub.popitem(last=False)
        return recs, bounds, counts

    def iter_level(self, l, sub, chunk_recs=FULL_CHUNK_RECS):
        """Yield (recs f64 (n,18), rows, bounds_local, counts) covering each
        nz row exactly once, in <= chunk_recs groups split at row ends —
        bitwise identical to the single-shot reduceat over the same rows.
        sub=True slices the cached subsample recs (views, no copy) so the
        per-call temporaries are bounded; sub=False gathers chunks from
        the memmap so a giant level never materializes whole."""
        key = self._sub if sub else self._full
        if key[l] is None:
            key[l] = self._build(l, sub)
        flat, bounds, counts = key[l]
        nz = self.nz_rows[l]
        n_l = len(counts)
        if n_l == 0:
            return
        recs = self.level_arrays(l)[0] if sub else None
        data = self.data.reshape(-1, REC)
        i0 = 0
        while i0 < n_l:
            # rows i0..i1-1 with <= chunk_recs records (>=1 row always)
            i1 = int(np.searchsorted(
                bounds, bounds[i0] + chunk_recs, "right")) - 1
            i1 = min(max(i1, i0 + 1), n_l)
            lo = int(bounds[i0])
            hi = int(bounds[i1]) if i1 < n_l else int(flat.size)
            if sub:
                yield (recs[lo:hi], nz[i0:i1],
                       bounds[i0:i1] - lo, counts[i0:i1])
            else:
                yield (data[flat[lo:hi]].astype(np.float64),
                       nz[i0:i1], bounds[i0:i1] - lo, counts[i0:i1])
            i0 = i1


class CatCache:
    """Virtual concatenation of PlaneCaches — row i of bin k maps to the
    concatenated row range in order. Avoids materialising pooled .bin files
    (the bytes are identical; only the row index space widens)."""

    def __init__(self, bins):
        self.caches = [PlaneCache(b) for b in bins]
        self.bin_path = "+".join(str(b) for b in bins)
        self.n_rows = sum(c.n_rows for c in self.caches)
        self.nb = np.concatenate([c.nb for c in self.caches])
        self.nz_rows = [np.nonzero(self.nb[:, l])[0] for l in range(LEVELS)]
        self._offsets = np.concatenate(
            [[0], np.cumsum([c.n_rows for c in self.caches])[:-1]])
        self._sub = [None] * LEVELS
        # bounded concat'd sub recs (same LRU discipline as PlaneCache)
        self._recs_cat = OrderedDict()

    def level_arrays(self, l, sub=True):
        assert sub, "full blocks via iter_level(l, False)"
        if self._sub[l] is None:
            bounds_l, counts_l = [], []
            pos = 0
            for c in self.caches:
                _, b, ct = c.level_arrays(l, sub=True)
                bounds_l.append(b + pos)
                counts_l.append(ct)
                pos += int(ct.sum())
            self._sub[l] = (np.concatenate(bounds_l),
                            np.concatenate(counts_l))
        bounds, counts = self._sub[l]
        if l in self._recs_cat:
            self._recs_cat.move_to_end(l)
            return self._recs_cat[l], bounds, counts
        recs = np.concatenate(
            [c.level_arrays(l, sub=True)[0] for c in self.caches]) \
            if self.caches else np.zeros((0, REC))
        self._recs_cat[l] = recs
        self._recs_cat.move_to_end(l)
        while len(self._recs_cat) > SUB_RECS_CACHE:
            self._recs_cat.popitem(last=False)
        return recs, bounds, counts

    def iter_level(self, l, sub, chunk_recs=FULL_CHUNK_RECS):
        for ci, c in enumerate(self.caches):
            off = self._offsets[ci]
            for recs, rows, bounds_l, counts_l in c.iter_level(
                    l, sub, chunk_recs):
                yield recs, rows + off, bounds_l, counts_l


# ---------- forward ----------

def contrast_grad(recs, g):
    """Per-block (cs, cd, dcs/dg, dcd/dg): min over quadrants (edge rule)."""
    cq_s = phi_g(recs[:, 2:6], g) - phi_g(recs[:, 6:10], g)
    cq_d = phi_g(recs[:, 10:14], g) - phi_g(recs[:, 14:18], g)
    rows = np.arange(recs.shape[0])
    js, jd = cq_s.argmin(axis=1), cq_d.argmin(axis=1)
    cs, cd = cq_s[rows, js], cq_d[rows, jd]
    mx_s = recs[rows, 2 + js]
    mn_s = recs[rows, 6 + js]
    mx_d = recs[rows, 10 + jd]
    mn_d = recs[rows, 14 + jd]

    def dphi(x):
        out = np.zeros_like(x)
        nz = x != 0.0
        out[nz] = phi_g(x[nz], g) * np.log(np.abs(x[nz]))
        return out

    return cs, cd, dphi(mx_s) - dphi(mn_s), dphi(mx_d) - dphi(mn_d)


def visibility(c, c0, beta, sharp):
    c = np.asarray(c, np.float64)
    v = np.ones_like(c)
    dc0 = np.zeros_like(c)
    db = np.zeros_like(c)
    ds = np.zeros_like(c)
    dc = np.zeros_like(c)
    pos = c > 0.0
    if np.any(pos):
        cp = c[pos]
        lc = np.log(cp)
        u = beta * sharp * (lc - math.log(c0))
        su = sig(u)
        lower = np.logaddexp(0.0, u)
        vv = np.exp(-lower / sharp)
        v[pos] = vv
        dc0[pos] = vv * su * beta / c0
        db[pos] = -vv * su * (lc - math.log(c0))
        ds[pos] = vv * (lower - su * u) / (sharp * sharp)
        dc[pos] = -vv * su * beta / cp
    return v, dc0, db, ds, dc


def s_rows_like(recs, g, P, c0, beta, sharp):
    """mean_b(v_b * m_b^P) over a plain (N,18) record array (test helper)."""
    cs, cd, _, _ = contrast_grad(recs, g)
    vs = visibility(cs, c0, beta, sharp)[0]
    vd = visibility(cd, c0, beta, sharp)[0]
    return float((np.maximum(vs, vd) * recs[:, 0] ** P).mean())


def s_level(cache, l, phys, sub=False, with_grad=False):
    """Per-row s_{p,l} over all rows (zeros where a row has no blocks).

    iter_level() yields <= FULL_CHUNK_RECS record chunks split at row ends
    (bitwise identical row sums), so a giant level never materializes its
    whole f64 recs array and per-call temporaries are bounded."""
    g, P, c0, beta, sharp = phys
    n_rows = cache.n_rows
    s = np.zeros(n_rows)
    d = np.zeros((n_rows, 5))
    for recs, nz, bounds, counts in cache.iter_level(l, sub):
        if recs.shape[0] == 0:
            continue
        cs, cd, dcs, dcd = contrast_grad(recs, g)
        vs, vs_c0, vs_b, vs_s, vs_c = visibility(cs, c0, beta, sharp)
        vd, vd_c0, vd_b, vd_s, vd_c = visibility(cd, c0, beta, sharp)
        pick = vd > vs
        v = np.where(pick, vd, vs)
        e = np.power(recs[:, 0], P)
        den = np.maximum(counts, 1).astype(np.float64)
        s[nz] = np.add.reduceat(v * e, bounds) / den
        if with_grad:
            de = np.where(recs[:, 0] > 0,
                          e * np.log(np.maximum(recs[:, 0], 1e-300)), 0.0)
            d[nz, 0] = np.add.reduceat(
                np.where(pick, vd_c * dcd, vs_c * dcs) * e, bounds) / den
            d[nz, 1] = np.add.reduceat(v * de, bounds) / den
            d[nz, 2] = np.add.reduceat(np.where(pick, vd_c0, vs_c0) * e,
                                       bounds) / den
            d[nz, 3] = np.add.reduceat(np.where(pick, vd_b, vs_b) * e,
                                       bounds) / den
            d[nz, 4] = np.add.reduceat(np.where(pick, vd_s, vs_s) * e,
                                       bounds) / den
    return (s, d) if with_grad else s


class Params:
    def __init__(self, planes, levels_init=None):
        self.planes = tuple(planes)
        self.raw = {p: np.zeros((LEVELS, 5)) for p in self.planes}
        if levels_init is not None:
            for p in self.planes:
                for l, lv in enumerate(levels_init):
                    self.raw[p][l] = self.encode(
                        lv["g"], lv["p"], lv["c0"], lv["beta"], lv["sharp"])
        self.wl = {p: np.zeros(LEVELS) for p in self.planes}
        self.wc = np.zeros(2)
        # output map yhat = A*exp(-lam*E)+B — refit during training, read
        # from the artefact for scoring; default = 100*exp(-E).
        self.map_abl = (100.0, 0.0, 1.0)

    @staticmethod
    def encode(g, P, c0, beta, sharp):
        # interior clamps: grid endpoints are strictly inside the
        # parameterisation bounds, so no grid value is a logit infinity
        gg = min(max(g, G_LO + 1e-9), G_HI - 1e-9)
        bb = min(max(beta, B_LO + 1e-9), B_HI - 1e-9)
        cc = min(max(c0, C0_LO), C0_HI)
        return np.array([
            math.log((gg - G_LO) / (G_HI - gg)),
            math.log(max(P, 1e-9)),
            math.log(cc),
            math.log((bb - B_LO) / (B_HI - bb)),
            math.log(math.expm1(max(sharp, 1.0 + 1e-6) - 1.0)),
        ])

    @staticmethod
    def decode(raw):
        # clamp the exp/logit inputs — an Adam step can push a raw coordinate
        # far past the point where exp() stays finite; the physical parameter
        # is saturated long before that anyway
        r0 = min(max(float(raw[0]), -60.0), 60.0)
        r1 = min(max(float(raw[1]), -20.0), 20.0)
        r2 = min(max(float(raw[2]), math.log(C0_LO)), math.log(C0_HI))
        r3 = min(max(float(raw[3]), -60.0), 60.0)
        r4 = min(max(float(raw[4]), -30.0), 30.0)
        return (G_LO + (G_HI - G_LO) * sig(r0), math.exp(r1), math.exp(r2),
                B_LO + (B_HI - B_LO) * sig(r3), 1.0 + softplus(r4))

    def phys(self, p, l):
        return self.decode(self.raw[p][l])

    def level_weights(self, p):
        w = np.exp(self.wl[p] - self.wl[p].max())
        return w / w.sum()

    def channel_weights(self):
        zy, zc = self.wc
        ey, ec = math.exp(zy), math.exp(zc)
        d = ey + 2.0 * ec
        return ey / d, ec / d

    def head_vector(self):
        # the output map (A,B,lam) is NOT a head parameter — it is refit
        # in closed form inside every objective call
        parts = [self.wl[p] for p in self.planes]
        parts.append(self.wc.copy())
        return np.concatenate(parts)

    def set_head(self, x):
        k = 0
        for p in self.planes:
            self.wl[p] = x[k:k + LEVELS].copy()
            k += LEVELS
        self.wc = x[k:k + 2].copy()


def forward_E(params, S, raw_override=None):
    """S: plane -> (n,5) s matrix. Returns (E, El dict).

    raw_override: optional {(p,l): raw5} — evaluate that (p,l)'s P from a
    candidate raw vector instead of params.raw (thread-pure candidate eval).
    """
    cY, cC = params.channel_weights()
    if len(params.planes) == 1:
        cY, cC = 1.0, 0.0          # single-plane: channel mix degenerates
    n = next(iter(S.values())).shape[0]
    E = np.zeros(n)
    El = {}
    for p in params.planes:
        w = params.level_weights(p)
        cp = cY if p == params.planes[0] else cC
        for l in range(LEVELS):
            if raw_override and (p, l) in raw_override:
                P = Params.decode(np.asarray(raw_override[(p, l)]))[1]
            else:
                P = params.phys(p, l)[1]
            s = np.maximum(S[p][:, l], 0.0)
            # E_{p,l} = s^{1/P}; exactly 0 at s=0 (identity -> E=0)
            El[(p, l)] = np.where(s > 0, np.power(np.maximum(s, 1e-300),
                                                1.0 / P), 0.0)
            E += cp * w[l] * El[(p, l)]
    return E, El


def score_of(params, E):
    a, b, lam = params.map_abl
    return a * np.exp(-lam * E) + b


def mse_of(params, S, y):
    """Refit-map MSE — the single loss used for every comparison."""
    E, _ = forward_E(params, S)
    return fit_map(E, y)["mse"]


def build_S(params, caches, sub=False, want_D=False):
    S, D = {}, {}
    for p in params.planes:
        c = caches[p]
        S[p] = np.zeros((c.n_rows, LEVELS))
        if want_D:
            D[p] = np.zeros((c.n_rows, LEVELS, 5))
        for l in range(LEVELS):
            if want_D:
                s, d = s_level(c, l, params.phys(p, l), sub=sub,
                               with_grad=True)
                D[p][:, l, :] = d
            else:
                s = s_level(c, l, params.phys(p, l), sub=sub)
            S[p][:, l] = s
    return (S, D) if want_D else S


# ---------- losses / grads ----------

def level_obj_grads(params, S, y, p, l, s_col, d_col, raw=None,
                    fixed_map=None):
    """Refit-map loss + dL/d(5 raw level params) for (p,l) on a given S.

    `raw` overrides params.raw[p][l] — lets a candidate x be evaluated
    without mutating shared params (thread-parallel multi-start).
    `fixed_map` pins (A,B,lam) — used by the finite-difference test so the
    analytic gradient is compared against the identical map."""
    E, El = forward_E(params, S, raw_override={(p, l): raw} if raw is not None
                      else None)
    loss, sc, dL_dE = map_resid(E, y, fixed_map)
    cY, cC = params.channel_weights()
    if len(params.planes) == 1:
        cY, cC = 1.0, 0.0          # match forward_E's single-plane mix
    cp = cY if p == params.planes[0] else cC
    wl = params.level_weights(p)[l]
    if raw is None:
        raw = params.raw[p][l]
    P = Params.decode(np.asarray(raw))[1]
    s = np.maximum(s_col, 0.0)
    Elv = np.power(np.maximum(s, 1e-300), 1.0 / P)
    coef = dL_dE * cp * wl
    # clip the division operand: P*s can underflow to 0 at denormal s, which
    # manufactures inf -> NaN grads; rows that small carry no signal anyway
    sdiv = np.where(s > 1e-120, s, 1e-120)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        lns = np.where(s > 0, np.log(np.maximum(s, 1e-300)), 0.0)
        dEl_dP = np.where(s > 0, Elv * (-lns / (P * P)), 0.0)
        dEl_ds = np.where(s > 0, Elv / (P * sdiv), 0.0)
    # physical grads then chain through the raw-parameter maps
    g_phys = np.zeros(5)
    g_phys[0] = np.sum(coef * dEl_ds * d_col[:, 0])
    g_phys[1] = np.sum(coef * (dEl_dP + dEl_ds * d_col[:, 1]))
    g_phys[2] = np.sum(coef * dEl_ds * d_col[:, 2])
    g_phys[3] = np.sum(coef * dEl_ds * d_col[:, 3])
    g_phys[4] = np.sum(coef * dEl_ds * d_col[:, 4])
    np.nan_to_num(g_phys, copy=False)
    dg = g_phys[0] * (G_HI - G_LO) * sig(raw[0]) * (1 - sig(raw[0]))
    dP = g_phys[1] * math.exp(raw[1])
    dc0 = g_phys[2] * math.exp(raw[2])
    dbeta = g_phys[3] * (B_HI - B_LO) * sig(raw[3]) * (1 - sig(raw[3]))
    dsharp = g_phys[4] * sig(raw[4])
    return loss, np.array([dg, dP, dc0, dbeta, dsharp])


def head_obj_grads(params, S, y, fixed_map=None):
    E, El = forward_E(params, S)
    loss, sc, dL_dE = map_resid(E, y, fixed_map)
    cY, cC = params.channel_weights()
    if len(params.planes) == 1:
        cY, cC = 1.0, 0.0
    Ep = {}
    for p in params.planes:
        w = params.level_weights(p)
        Ep[p] = np.zeros(len(y))
        for l in range(LEVELS):
            Ep[p] += w[l] * El[(p, l)]
    parts = []
    for p in params.planes:
        w = params.level_weights(p)
        cp = cY if p == params.planes[0] else cC
        g = np.zeros(LEVELS)
        for k in range(LEVELS):
            g[k] = np.sum(dL_dE * cp * w[k] * (El[(p, k)] - Ep[p]))
        parts.append(g)
    EY = Ep[params.planes[0]]
    EC = (Ep[params.planes[1]] + Ep[params.planes[2]]
          if len(params.planes) == 3 else np.zeros(len(y)))
    dwY = np.array([cY * (1 - cY), -2 * cY * cC])
    dwC = np.array([-cC * cY, cC - 2 * cC * cC])
    gwc = np.array([
        np.sum(dL_dE * (EY * dwY[0] + EC * dwC[0])),
        np.sum(dL_dE * (EY * dwY[1] + EC * dwC[1])),
    ])
    out = np.concatenate(parts + [gwc])
    np.nan_to_num(out, copy=False)
    return loss, out


def adam(obj, x0, steps, lr=ADAM_LR, patience=25, tol=1e-7):
    """Adam with early stop: quit after `patience` steps without improving the
    best loss by more than `tol` (relative). Deterministic given obj."""
    x = np.asarray(x0, np.float64).copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    best = (np.inf, x.copy())
    stall = 0
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, steps + 1):
        loss, g = obj(x)          # loss is L(x) BEFORE this step's update
        if not (np.isfinite(loss) and np.all(np.isfinite(g))):
            stall += 1            # non-finite eval: count as stall, no update
            if stall >= patience:
                break
            continue
        if loss < best[0] - tol * max(abs(best[0]), 1e-12):
            best = (loss, x.copy())
            stall = 0
        elif loss < best[0]:
            best = (loss, x.copy())
            stall += 1
        else:
            stall += 1
        g = np.asarray(g, np.float64)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        x = x - lr * (m / (1 - b1 ** t)) / (np.sqrt(v / (1 - b2 ** t)) + eps)
        if stall >= patience:
            break
    return best


# ---------- driver ----------

def fit_variant(name, planes, caches, y, init_levels, out_path,
                surfaces_dir, prov):
    t0 = time.time()
    params = Params(planes, levels_init=init_levels)
    S = build_S(params, caches)

    # --- sanity gate (Amendment 1): with the map refit the init params
    # must beat the constant predictor on this fit set, else STOP and
    # diagnose (units, orientation, row join) before fitting anything.
    var_y = float(np.var(y))
    E0, _ = forward_E(params, S)
    mp0 = fit_map(E0, y)
    sro0, kro0 = _rank_stats(E0, y)
    log(f"{name}: SANITY init refit-MSE {mp0['mse']:.5f} vs var(y) "
        f"{var_y:.5f} | map A={mp0['A']:.4g} B={mp0['B']:.4g} "
        f"lam={mp0['lambda']:.4g} | SROCC(-E,y)={sro0:.4f} "
        f"KROCC={kro0:.4f} | mean(y)={float(np.mean(y)):.3f}")
    if not mp0["mse"] < var_y:
        log(f"{name}: SANITY FAIL — init does not beat the constant "
            f"predictor (mse {mp0['mse']:.5f} >= var {var_y:.5f}). "
            f"Check units/orientation/row join. STOP.")
        raise SystemExit(2)
    best_loss = mp0["mse"]
    params.map_abl = (mp0["A"], mp0["B"], mp0["lambda"])
    hist = [{"sweep": 0, "mse": best_loss}]
    level_report = []

    for sweep in range(1, MAX_SWEEPS + 1):
        # subsampled S reused inside per-level Adam objectives
        Ssub = build_S(params, caches, sub=True)
        for p in params.planes:
            for l in range(LEVELS):
                cells, diag = grid_sweep_level(
                    params, caches, Ssub, y, p, l, surfaces_dir,
                    f"{name}_s{sweep}")
                diag.update({"sweep": sweep, "plane": p, "level": l})
                level_report.append(diag)
                log(f"{name} s{sweep} {p} l{l}: grid best "
                    f"{cells[0][0]:.5f} c0={cells[0][1]:.4g} "
                    f"beta={cells[0][2]:.3f} srocc={cells[0][3]:.4f}"
                    + (f" EDGE:{diag['grid_edge']}"
                       if diag["grid_edge"] else ""))
                cur_loss = mse_of(params, S, y)
                best_cand = (cur_loss, params.raw[p][l].copy())

                def obj(x, p=p, l=l):
                    # pure: no params mutation -> safe across threads
                    ph = Params.decode(np.asarray(x))
                    s_col, d_col = s_level(
                        caches[p], l, ph, sub=True, with_grad=True)
                    Ss = dict(Ssub)
                    Ss[p] = Ssub[p].copy()
                    Ss[p][:, l] = s_col
                    return level_obj_grads(params, Ss, y, p, l,
                                           s_col, d_col, raw=x)

                def run_start(raw0):
                    return adam(obj, raw0, ADAM_STEPS_LEVEL)

                starts = []
                for _, c0, beta, _, _ in cells[:N_STARTS]:
                    raw0 = params.raw[p][l].copy()
                    ph = params.phys(p, l)
                    # encode() clamps interior — no grid value is a logit
                    # infinity under the widened (0.01, 100) beta bounds
                    raw0[2:4] = Params.encode(ph[0], ph[1], c0, beta,
                                              ph[4])[2:4]
                    starts.append(raw0)
                # 8 Adam chains in forked workers: each chain is pure and
                # deterministic, and recs_sub[l] is warmed in the parent so
                # children inherit the records COW instead of re-reading.
                # (The level_arrays call below IS that warm — keep it.)
                caches[p].level_arrays(l, sub=True)
                results = par_map(f"adam_{name}_{p}_l{l}",
                                  run_start, starts)
                # accept on the subsampled loss (consistent estimator), then
                # verify on ALL blocks before committing the level
                bx = min(results, key=lambda r: r[0])[1]
                save = params.raw[p][l].copy()
                params.raw[p][l] = bx
                s_full = s_level(caches[p], l, params.phys(p, l))
                Sf = dict(S)
                Sf[p] = S[p].copy()
                Sf[p][:, l] = s_full
                fl = mse_of(params, Sf, y)
                params.raw[p][l] = save
                if fl < best_cand[0]:
                    best_cand = (fl, bx)
                params.raw[p][l] = best_cand[1]
                S[p][:, l] = s_level(caches[p], l, params.phys(p, l))
                Ssub[p][:, l] = s_level(caches[p], l, params.phys(p, l),
                                        sub=True)
        # head refit on full S (map is refit inside every objective call)
        def hobj(x):
            params.set_head(x)
            return head_obj_grads(params, S, y)
        _, bx = adam(hobj, params.head_vector(), ADAM_STEPS_HEAD)
        params.set_head(bx)
        S = build_S(params, caches)
        E_new, _ = forward_E(params, S)
        mp = fit_map(E_new, y)
        params.map_abl = (mp["A"], mp["B"], mp["lambda"])
        new_loss = mp["mse"]
        hist.append({"sweep": sweep, "mse": new_loss})
        log(f"{name} sweep {sweep}: MSE {new_loss:.5f} "
            f"(map A={mp['A']:.4g} B={mp['B']:.4g} lam={mp['lambda']:.4g})")
        if best_loss - new_loss < REL_STOP * max(best_loss, 1e-12):
            best_loss = new_loss
            break
        best_loss = new_loss

    E_fin, _ = forward_E(params, S)
    mp = fit_map(E_fin, y)
    params.map_abl = (mp["A"], mp["B"], mp["lambda"])
    sro, kro = _rank_stats(E_fin, y)
    artefact = {
        "schema": "dvifm-standalone-fit-v2",
        "variant": name,
        "planes": list(params.planes),
        "levels": {p: [dict(zip(("g", "p", "c0", "beta", "sharp"),
                               params.phys(p, l))) for l in range(LEVELS)]
                   for p in params.planes},
        "level_weights": {p: params.level_weights(p).tolist()
                          for p in params.planes},
        "channel_weights": params.channel_weights(),
        "map": {"A": mp["A"], "B": mp["B"], "lambda": mp["lambda"]},
        "fit_mse": mp["mse"],
        "var_y": var_y,
        "srocc_negE_fit": sro,
        "krocc_negE_fit": kro,
        "sanity": {"init_mse": mp0["mse"], "var_y": var_y,
                   "init_srocc_negE": sro0, "init_krocc_negE": kro0,
                   "init_map": mp0},
        "level_report": level_report,
        "history": hist,
        "wall_s": time.time() - t0,
        "provenance": prov,
    }
    Path(out_path).write_text(json.dumps(artefact, indent=1) + "\n")
    log(f"{name}: DONE MSE {mp['mse']:.5f} (var {var_y:.5f}, "
        f"SROCC {sro:.4f}) -> {out_path}")
    return artefact


def grid_sweep_level(params, caches, Ssub, y, p, l, surfaces_dir, tag):
    """(C0, beta) grid for (p,l) on the deterministic block subsample.

    Amendment-1: EVERY cell refits the output map A*exp(-lam*E)+B (A>0;
    (A,B) closed-form LS, lam golden-sectioned on log lam) before scoring —
    the surface ranks cells by masking quality, not by how well they
    rescale E to a stale lambda. Each cell also records the scale-free
    rank criteria SROCC/KROCC of -E vs y; selection is by refit MSE and
    all three surfaces are committed.

    If the best cell sits on a grid edge, the edged axis is extended once
    (GRID_EXTEND points continuing the log spacing); a still-edged optimum
    is reported via the returned diagnostics — never silently clamped.

    `Ssub` carries the subsampled per-row s for every (plane,level) at the
    current parameters so the E_const term stays coherent with the
    candidate's subsampled s. Every accepted candidate is re-scored on ALL
    blocks before commit.

    Returns (rows, diag): rows sorted by refit MSE —
    (mse, c0, beta, srocc, krocc); diag has best/edge/sharpness info.
    """
    g, P, _, _, sharp = params.phys(p, l)
    cache = caches[p]
    recs, bounds, counts = cache.level_arrays(l, sub=True)
    cs, cd, _, _ = contrast_grad(recs, g)
    e = recs[:, 0] ** P
    nz = cache.nz_rows[l]
    den = np.maximum(counts, 1).astype(np.float64)
    cY, cC = params.channel_weights()
    if len(params.planes) == 1:
        cY, cC = 1.0, 0.0          # match forward_E's single-plane mix
    cp = cY if p == params.planes[0] else cC
    wl = params.level_weights(p)[l]
    E_const = np.zeros(cache.n_rows)
    for pp in params.planes:
        w = params.level_weights(pp)
        cpp = cY if pp == params.planes[0] else cC
        for ll in range(LEVELS):
            if pp == p and ll == l:
                continue
            P2 = params.phys(pp, ll)[1]
            E_const += cpp * w[ll] * np.power(
                np.maximum(Ssub[pp][:, ll], 1e-300), 1.0 / P2)

    def eval_cell(c0, beta):
        vs = visibility(cs, c0, beta, sharp)[0]
        vd = visibility(cd, c0, beta, sharp)[0]
        contrib = np.maximum(vs, vd) * e
        sr = np.zeros(cache.n_rows)
        sr[nz] = np.add.reduceat(contrib, bounds) / den
        El = np.power(np.maximum(sr, 1e-300), 1.0 / P)
        E = E_const + cp * wl * El
        mp = fit_map(E, y)
        sro, kro = _rank_stats(E, y)
        return (mp["mse"], float(c0), float(beta), sro, kro)

    c0_vals = list(float(c) for c in GRID_C0)
    beta_vals = list(float(b) for b in GRID_BETA)
    results = {}

    def eval_cells(cell_list):
        # cells are pure functions of (c0,beta) over read-only state;
        # recs_sub[l] was warmed above so forked children inherit it COW
        for out in par_map(f"grid_{tag}_{p}_l{l}",
                           lambda cb: eval_cell(*cb), cell_list):
            results[(out[1], out[2])] = out

    eval_cells([(c, b) for c in c0_vals for b in beta_vals])

    edge = None
    extended = False
    best = min(results.values())
    if (best[1] in (c0_vals[0], c0_vals[-1])
            or best[2] in (beta_vals[0], beta_vals[-1])):
        extended = True
        c0_ext, b_ext = [], []
        if best[1] == c0_vals[-1]:
            r = c0_vals[-1] / c0_vals[-2]
            c0_ext = [c0_vals[-1] * r ** k for k in range(1, GRID_EXTEND + 1)]
        elif best[1] == c0_vals[0]:
            r = c0_vals[0] / c0_vals[1]
            c0_ext = [c0_vals[0] * r ** k for k in range(1, GRID_EXTEND + 1)]
        if best[2] == beta_vals[-1]:
            r = beta_vals[-1] / beta_vals[-2]
            b_ext = [beta_vals[-1] * r ** k
                     for k in range(1, GRID_EXTEND + 1)]
        elif best[2] == beta_vals[0]:
            r = beta_vals[0] / beta_vals[1]
            b_ext = [beta_vals[0] * r ** k
                     for k in range(1, GRID_EXTEND + 1)]
        all_c = c0_vals + c0_ext
        all_b = beta_vals + b_ext
        eval_cells([(c, b) for c in all_c for b in all_b
                    if (c, b) not in results])
        c0_vals, beta_vals = sorted(all_c), sorted(all_b)
        best = min(results.values())
        edge = {}
        if best[1] in (c0_vals[0], c0_vals[-1]):
            edge["c0"] = "lo" if best[1] == c0_vals[0] else "hi"
        if best[2] in (beta_vals[0], beta_vals[-1]):
            edge["beta"] = "lo" if best[2] == beta_vals[0] else "hi"
        if not edge:
            edge = None

    # surface file: c0,beta,mse_refit,srocc,krocc — all three surfaces in
    # one CSV (mse selects; srocc/krocc are the scale-free criteria)
    c0_sorted = sorted(results.keys())
    surf = np.array([[c, b, *results[(c, b)][:1], *results[(c, b)][3:]]
                     for c, b in c0_sorted])
    np.savetxt(Path(surfaces_dir) / f"grid_{tag}_{p}_l{l}.csv", surf,
               delimiter=",",
               header="c0,beta,mse_refit,srocc,krocc", comments="")

    # sharpness: refit-MSE increase at +/-1 grid step in each axis around
    # the best cell (None on the unexplored side of an edge optimum)
    bi = c0_vals.index(best[1])
    bj = beta_vals.index(best[2])
    m = {(c, b): r[0] for (c, b), r in results.items()}
    sharp_d = {
        "c0_minus": (m[(c0_vals[bi - 1], best[2])] - best[0]
                     if bi > 0 else None),
        "c0_plus": (m[(c0_vals[bi + 1], best[2])] - best[0]
                    if bi < len(c0_vals) - 1 else None),
        "beta_minus": (m[(best[1], beta_vals[bj - 1])] - best[0]
                       if bj > 0 else None),
        "beta_plus": (m[(best[1], beta_vals[bj + 1])] - best[0]
                      if bj < len(beta_vals) - 1 else None),
    }
    diag = {"best_mse": best[0], "best_c0": best[1], "best_beta": best[2],
            "best_srocc": best[3], "best_krocc": best[4],
            "grid_edge": edge, "grid_extended": extended,
            "sharpness": sharp_d,
            "surface_csv": f"grid_{tag}_{p}_l{l}.csv"}
    rows = sorted(results.values())
    return rows, diag


def params_from_fit(fit):
    params = Params(fit["planes"])
    for p in fit["planes"]:
        for l, lv in enumerate(fit["levels"][p]):
            params.raw[p][l] = Params.encode(
                lv["g"], lv["p"], lv["c0"], lv["beta"], lv["sharp"])
        w = np.maximum(np.asarray(fit["level_weights"][p]), 1e-12)
        params.wl[p] = np.log(w)
    cY, cC = fit["channel_weights"]
    params.wc = np.log(np.maximum(np.asarray([cY, cC]), 1e-12))
    if "map" in fit:                       # v2: recorded refit map
        mp = fit["map"]
        params.map_abl = (mp["A"], mp["B"], mp["lambda"])
    else:                                  # v1 compat: 100*exp(-lam*E)
        params.map_abl = (100.0, 0.0, fit["lambda"])
    return params


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    fp = sub.add_parser("fit")
    fp.add_argument("--variant", required=True,
                    choices=["luma", "native3", "xyb-y"])
    fp.add_argument("--cache", action="append", required=True)
    fp.add_argument("--pairs", action="append", required=True)
    fp.add_argument("--init-spec", required=True)
    fp.add_argument("--out", required=True)
    fp.add_argument("--surfaces-dir", required=True)
    sp = sub.add_parser("score")
    sp.add_argument("--fit", required=True)
    sp.add_argument("--cache", action="append", required=True)
    sp.add_argument("--pairs", required=True)
    sp.add_argument("--out", required=True)
    kp = sub.add_parser("eval-consts")
    kp.add_argument("--spec", required=True)
    kp.add_argument("--variant", required=True, choices=["luma", "xyb-y"])
    kp.add_argument("--cache", required=True)
    kp.add_argument("--pairs", required=True)
    a = ap.parse_args()

    import csv

    def load_y(tsvs):
        y = []
        for tsv in tsvs:
            with open(tsv) as f:
                y += [float(r["human_score"]) * 100.0
                      for r in csv.DictReader(f, delimiter="\t")]
        return np.asarray(y)

    def load_caches(specs):
        out = {}
        for s in specs:
            p, _, path = s.partition("=")
            bins = path.split(",")
            out[p] = PlaneCache(bins[0]) if len(bins) == 1 else CatCache(bins)
        return out

    if a.cmd == "fit":
        planes = {"luma": PLANES_LUMA, "native3": PLANES_3,
                  "xyb-y": PLANES_XYB}[a.variant]
        caches = load_caches(a.cache)
        y = load_y(a.pairs)
        n = next(iter(caches.values())).n_rows
        assert len(y) == n, (len(y), n)
        k1 = json.loads(Path(a.init_spec).read_text())
        Path(a.surfaces_dir).mkdir(parents=True, exist_ok=True)
        prov = {
            "caches": {p: {"bin": c.bin_path,
                           "sha256": [sha(b) for b in
                                      c.bin_path.split("+")]}
                       for p, c in caches.items()},
            "pairs": [{"path": t, "sha256": sha(t)} for t in a.pairs],
            "init_spec": {"path": a.init_spec, "sha256": sha(a.init_spec)},
            "estimator": f"Adam on deterministic <={ADAM_BLOCK_CAP}-block "
                         "per-row stride subsample; grids and reported "
                         "losses on ALL blocks",
            "objective": "Amendment-1 refit-map MSE: yhat=A*exp(-lam*E)+B, "
                         "A>0; (A,B) constrained LS per lam, lam golden-"
                         "sectioned on log lam <= 40 evals; selection by "
                         "refit MSE; SROCC/KROCC of -E recorded per cell",
            "grid": {"c0": GRID_C0.tolist(), "beta": GRID_BETA.tolist(),
                     "spacing": "log", "edge_extend_points": GRID_EXTEND},
            "param_bounds": {"g": [G_LO, G_HI], "beta": [B_LO, B_HI],
                             "c0": [C0_LO, C0_HI], "lam": [LAM_LO, LAM_HI]},
            "adam": {"level_steps": ADAM_STEPS_LEVEL,
                     "head_steps": ADAM_STEPS_HEAD, "lr": ADAM_LR,
                     "starts": N_STARTS, "max_sweeps": MAX_SWEEPS,
                     "rel_stop": REL_STOP},
        }
        fit_variant(a.variant, planes, caches, y, k1["levels"], a.out,
                    a.surfaces_dir, prov)
        return 0

    if a.cmd == "score":
        fit = json.loads(Path(a.fit).read_text())
        caches = load_caches(a.cache)
        params = params_from_fit(fit)
        with open(a.pairs) as f:
            rows = list(csv.DictReader(f, delimiter="\t"))
        n = next(iter(caches.values())).n_rows
        assert len(rows) == n, (len(rows), n)
        S = build_S(params, caches)
        E, _ = forward_E(params, S)
        sc = score_of(params, E)
        with open(a.out, "w") as f:
            f.write("ref_path,dist_path,target,score,E\n")
            for r, s_, e_ in zip(rows, sc, E):
                f.write(f"{r['ref_path']},{r['dist_path']},"
                        f"{float(r['human_score']) * 100.0},{s_},{e_}\n")
        return 0

    if a.cmd == "eval-consts":
        # K1 control (Amendment-1 map family): constants + uniform level
        # weights as given; the ONLY free scalars are the output map
        # (A, B, lam) fitted by the same closed-form+golden procedure.
        spec = json.loads(Path(a.spec).read_text())
        plane = {"luma": "ycbcr_y", "xyb-y": "xyb_y"}[a.variant]
        bins = a.cache.split(",")
        cache = PlaneCache(bins[0]) if len(bins) == 1 else CatCache(bins)
        y = load_y([a.pairs])
        params = Params([plane], levels_init=spec["levels"])
        S = build_S(params, {plane: cache})
        E, _ = forward_E(params, S)
        mp = fit_map(E, y)
        sro, kro = _rank_stats(E, y)
        out = {"schema": "dvifm-standalone-eval-consts-v2",
               "spec": a.spec, "spec_sha256": sha(a.spec),
               "variant": a.variant,
               "map": {"A": mp["A"], "B": mp["B"], "lambda": mp["lambda"]},
               "fit_mse": mp["mse"], "var_y": float(np.var(y)),
               "srocc_negE": sro, "krocc_negE": kro,
               "head": "uniform level weights; output map "
                       "A*exp(-lam*E)+B refit (only free scalars)"}
        print(json.dumps(out, indent=1))
        return 0


if __name__ == "__main__":
    sys.exit(main())
