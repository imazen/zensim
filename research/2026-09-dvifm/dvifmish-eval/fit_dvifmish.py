#!/usr/bin/env python3
"""Standalone DVIFM constants fitter for the dvifmish presets.

Lineage: the faithful lane's `fit_faithful.py` (preserved beside this file in
`faithful-lane-tools/`) generalised from one plane to one-or-three planes
with the phase-2d channel mix, plus a prior-constants mode. The numerics are
the shared phase-2d primitives (`../kernel-int16/screen2d/fit_standalone.py`: visibility and its
gradients, the corner-min contrast, the Amendment-1 refit-map MSE objective,
Adam) — nothing statistical is reimplemented here.

Model (per row; planes p, levels l = 0..4):
    C_s, C_d  = block contrast per side: plain range φ_g(max) − φ_g(min), or
                the min over the four 3×3 corners (edge discount)
    v_b       = max(v(C_s), v(C_d))   v = curve | gate | off
    s_{p,l}   = mean_b(v_b · m_b^P)
    E_{p,l}   = s^L (pooling "free") or s^(1/P) (pooling "tied")
    E         = Σ_p c_p Σ_l w_{p,l} E_{p,l}   (w, c on simplexes; with three
                planes the two chroma planes share one channel weight)
    yhat      = A·exp(−λE) + B, refit inside every objective evaluation
Loss: refit-map MSE on the segment labels (orientation: higher = better).

Constants protocols:
    prior   g = 1, P = 1, L = 1, ς = 4, β = --beta (shared), knee = the fit
            rows' 10th percentile of min(C_s, C_d) per (plane, level) at g = 1
            (the gate knee too). Only the head (w, c) and the map are fitted.
    fit     curve: per (plane, level) knee (grid then Adam), g, P, ς, L
            (Adam); β shared across all levels and planes (golden search), or with
            --beta-mode level one β per plane and level, fitted by Adam with the rest.
            gate: per (plane, level) knee (grid + golden, or "off"), P, L.
            off: P, L. Then the head. ≤ 3 sweeps.

Usage:
  fit_dvifmish.py fit --segs seg.json --planes ycbcr_y,ycbcr_cb,ycbcr_cr \
      --vis gate --contrast edge --pooling tied --constants prior --beta 0.693 \
      --out fit.json [--name NAME]
  seg.json: [{"name": .., "pairs": tsv, "bins": {plane: bin}, "label_scale": 0.01}]
"""
import argparse
import csv
import json
import math
import sys
import time
from collections import OrderedDict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "kernel-int16" / "screen2d"))
import numpy as np  # noqa: E402
import scipy.stats  # noqa: E402
import fit_standalone as fs  # noqa: E402

LEVELS = 5
L_LO, L_HI = 0.02, 50.0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# caches (f16 capped records, dvifmish `records` / zensim extractor layout)
# ---------------------------------------------------------------------------

class PlaneCacheF16(fs.PlaneCache):
    """`fit_core.PlaneCacheF16`: the record element is IEEE f16."""

    def __init__(self, bin_path):
        self.bin_path = str(bin_path)
        self.index = fs.load_index(bin_path)
        self.data = np.memmap(bin_path, dtype=np.dtype("<f2"), mode="r")
        self.n_rows = len(self.index)
        self.nb = np.zeros((self.n_rows, LEVELS), np.int64)
        self.row_elem = np.zeros((self.n_rows, LEVELS), np.int64)
        for i, e in enumerate(self.index):
            base = e["offset"] // 2
            cnt = np.asarray(e["level_records"], np.int64)
            self.nb[i] = cnt
            self.row_elem[i] = base + np.cumsum(np.concatenate([[0], cnt]))[:5] * fs.REC
        self.nz_rows = [np.nonzero(self.nb[:, l])[0] for l in range(LEVELS)]
        self._sub = [None] * LEVELS
        self._full = [None] * LEVELS
        self._recs_sub = OrderedDict()


class ConcatCache:
    """Several segment caches as one row space (per plane)."""

    def __init__(self, caches):
        self.caches = caches
        self.n_rows = sum(c.n_rows for c in caches)
        self.offsets = np.cumsum([0] + [c.n_rows for c in caches])[:-1]

    def iter_level(self, l, sub):
        for c, off in zip(self.caches, self.offsets):
            for recs, nz, bounds, counts in c.iter_level(l, sub):
                yield recs, nz + off, bounds, counts


# ---------------------------------------------------------------------------
# per-level parameters (the faithful lane's raw6 encoding)
# ---------------------------------------------------------------------------

def encode6(g, P, c0, beta, sharp, L):
    gg = min(max(g, fs.G_LO + 1e-9), fs.G_HI - 1e-9)
    bb = min(max(beta, fs.B_LO + 1e-9), fs.B_HI - 1e-9)
    cc = min(max(c0, fs.C0_LO), fs.C0_HI)
    ll = min(max(L, L_LO), L_HI)
    return np.array([math.log((gg - fs.G_LO) / (fs.G_HI - gg)), math.log(max(P, 1e-9)),
                     math.log(cc), math.log((bb - fs.B_LO) / (fs.B_HI - bb)),
                     math.log(math.expm1(max(sharp, 1.0 + 1e-6) - 1.0)), math.log(ll)])


def decode6(raw):
    r = [float(x) for x in raw]
    r0 = min(max(r[0], -60.0), 60.0)
    r1 = min(max(r[1], -20.0), 20.0)
    r2 = min(max(r[2], math.log(fs.C0_LO)), math.log(fs.C0_HI))
    r3 = min(max(r[3], -60.0), 60.0)
    r4 = min(max(r[4], -30.0), 30.0)
    r5 = min(max(r[5], math.log(L_LO)), math.log(L_HI))
    return (fs.G_LO + (fs.G_HI - fs.G_LO) * fs.sig(r0), math.exp(r1), math.exp(r2),
            fs.B_LO + (fs.B_HI - fs.B_LO) * fs.sig(r3), 1.0 + fs.softplus(r4), math.exp(r5))


def decode6_jac(raw):
    r = [float(x) for x in raw]
    s0 = fs.sig(min(max(r[0], -60.0), 60.0))
    s3 = fs.sig(min(max(r[3], -60.0), 60.0))
    return np.array([(fs.G_HI - fs.G_LO) * s0 * (1 - s0),
                     math.exp(min(max(r[1], -20.0), 20.0)),
                     math.exp(min(max(r[2], math.log(fs.C0_LO)), math.log(fs.C0_HI))),
                     (fs.B_HI - fs.B_LO) * s3 * (1 - s3),
                     fs.sig(min(max(r[4], -30.0), 30.0)),
                     math.exp(min(max(r[5], math.log(L_LO)), math.log(L_HI)))])


# ---------------------------------------------------------------------------
# contrast, visibility, level means
# ---------------------------------------------------------------------------

def contrast_any(recs, g, edge):
    """(cs, cd, dcs/dg, dcd/dg) — fit_faithful.contrast_any verbatim."""
    if edge:
        return fs.contrast_grad(recs, g)
    mx_s = recs[:, 2:6].max(axis=1)
    mn_s = recs[:, 6:10].min(axis=1)
    mx_d = recs[:, 10:14].max(axis=1)
    mn_d = recs[:, 14:18].min(axis=1)
    cs = fs.phi_g(mx_s, g) - fs.phi_g(mn_s, g)
    cd = fs.phi_g(mx_d, g) - fs.phi_g(mn_d, g)

    def dphi(x):
        out = np.zeros_like(x)
        nz = x != 0.0
        out[nz] = fs.phi_g(x[nz], g) * np.log(np.abs(x[nz]))
        return out

    return cs, cd, dphi(mx_s) - dphi(mn_s), dphi(mx_d) - dphi(mn_d)


def vis_of(c, cell):
    if cell["vis"] == "off":
        return np.ones_like(c)
    if cell["vis"] == "gate":
        return (c <= cell["knee"]).astype(np.float64)
    return fs.visibility(c, cell["knee"], cell["beta"], cell["sharp"])[0]


def s_level_rows(cache, l, cell, edge, sub=False):
    s = np.zeros(cache.n_rows)
    for recs, nz, bounds, counts in cache.iter_level(l, sub):
        if recs.shape[0] == 0:
            continue
        cs, cd, _, _ = contrast_any(recs, cell["g"], edge)
        v = np.maximum(vis_of(cs, cell), vis_of(cd, cell))
        e = np.power(recs[:, 0], cell["P"])
        den = np.maximum(counts, 1).astype(np.float64)
        s[nz] = np.add.reduceat(v * e, bounds) / den
    return s


def s_level_grad(cache, l, cell, edge, sub=True):
    """(s, ds/d(g, P, knee, β, ς)) on the block subsample."""
    s = np.zeros(cache.n_rows)
    d = np.zeros((cache.n_rows, 5))
    curve = cell["vis"] == "curve"
    for recs, nz, bounds, counts in cache.iter_level(l, sub):
        if recs.shape[0] == 0:
            continue
        cs, cd, dcs, dcd = contrast_any(recs, cell["g"], edge)
        if curve:
            vs, vs_c0, vs_b, vs_s, vs_c = fs.visibility(cs, cell["knee"], cell["beta"], cell["sharp"])
            vd, vd_c0, vd_b, vd_s, vd_c = fs.visibility(cd, cell["knee"], cell["beta"], cell["sharp"])
            pick = vd > vs
            v = np.where(pick, vd, vs)
        else:
            v = np.maximum(vis_of(cs, cell), vis_of(cd, cell))
        e = np.power(recs[:, 0], cell["P"])
        den = np.maximum(counts, 1).astype(np.float64)
        s[nz] = np.add.reduceat(v * e, bounds) / den
        de = np.where(recs[:, 0] > 0, e * np.log(np.maximum(recs[:, 0], 1e-300)), 0.0)
        d[nz, 1] = np.add.reduceat(v * de, bounds) / den
        if curve:
            d[nz, 0] = np.add.reduceat(np.where(pick, vd_c * dcd, vs_c * dcs) * e, bounds) / den
            d[nz, 2] = np.add.reduceat(np.where(pick, vd_c0, vs_c0) * e, bounds) / den
            d[nz, 3] = np.add.reduceat(np.where(pick, vd_b, vs_b) * e, bounds) / den
            d[nz, 4] = np.add.reduceat(np.where(pick, vd_s, vs_s) * e, bounds) / den
    return s, d


# ---------------------------------------------------------------------------
# the model over planes
# ---------------------------------------------------------------------------

class Model:
    def __init__(self, planes, cells, pooling, edge):
        self.planes = list(planes)
        self.cells = cells              # {plane: [cell × 5]}
        self.pooling = pooling
        self.edge = edge
        self.wl = {p: np.zeros(LEVELS) for p in self.planes}
        self.wc = np.zeros(2)

    def level_weights(self, p):
        w = np.exp(self.wl[p] - self.wl[p].max())
        return w / w.sum()

    def channel_weights(self):
        if len(self.planes) == 1:
            return {self.planes[0]: 1.0}
        ey, ec = math.exp(self.wc[0]), math.exp(self.wc[1])
        d = ey + (len(self.planes) - 1) * ec
        return {p: (ey / d if i == 0 else ec / d) for i, p in enumerate(self.planes)}

    def el(self, p, l, s):
        c = self.cells[p][l]
        ex = c["L"] if self.pooling == "free" else 1.0 / c["P"]
        sp = np.maximum(s, 0.0)
        return np.where(sp > 0, np.power(np.maximum(sp, 1e-300), ex), 0.0)

    def forward(self, S):
        cw = self.channel_weights()
        E = np.zeros(next(iter(S.values())).shape[0])
        El = {}
        for p in self.planes:
            w = self.level_weights(p)
            for l in range(LEVELS):
                El[(p, l)] = self.el(p, l, S[p][:, l])
                E += cw[p] * w[l] * El[(p, l)]
        return E, El

    def head_vector(self):
        return np.concatenate([self.wl[p] for p in self.planes] + [self.wc])

    def set_head(self, x):
        k = 0
        for p in self.planes:
            self.wl[p] = np.array(x[k:k + LEVELS])
            k += LEVELS
        self.wc = np.array(x[k:k + 2])


def build_S(model, caches, sub=False):
    return {p: np.stack([s_level_rows(caches[p], l, model.cells[p][l], model.edge, sub)
                         for l in range(LEVELS)], axis=1) for p in model.planes}


def mse_of(model, S, y):
    E, _ = model.forward(S)
    return fs.fit_map(E, y)["mse"]


def head_fit(model, S, y):
    def obj(x):
        model.set_head(x)
        E, El = model.forward(S)
        loss, _, dle = fs.map_resid(E, y)
        cw = model.channel_weights()
        grads = []
        Ep = {}
        for p in model.planes:
            w = model.level_weights(p)
            Ep[p] = sum(w[l] * El[(p, l)] for l in range(LEVELS))
            grads.append(np.array([np.sum(dle * cw[p] * w[k] * (El[(p, k)] - Ep[p]))
                                   for k in range(LEVELS)]))
        if len(model.planes) == 3:
            EY = Ep[model.planes[0]]
            EC = Ep[model.planes[1]] + Ep[model.planes[2]]
            cY, cC = cw[model.planes[0]], cw[model.planes[1]]
            gwc = np.array([np.sum(dle * (EY * cY * (1 - cY) + EC * (-cC * cY))),
                            np.sum(dle * (EY * (-2 * cY * cC) + EC * (cC - 2 * cC * cC)))])
        else:
            gwc = np.zeros(2)
        g = np.concatenate(grads + [gwc])
        np.nan_to_num(g, copy=False)
        return loss, g
    best_loss, bx = fs.adam(obj, model.head_vector(), fs.ADAM_STEPS_HEAD)
    model.set_head(bx)
    return best_loss


def level_obj(model, caches, p, l, S, y, freeze_beta):
    def obj(x):
        g, P, c0, beta, sharp, L = decode6(x)
        # evaluate the candidate in place, then put back the SAME cell object:
        # fit() holds a reference to it and writes the Adam result into it
        orig = model.cells[p][l]
        cand = dict(orig)
        cand.update({"g": g, "P": P, "L": L, "sharp": sharp})
        if cand["vis"] == "curve":
            cand["knee"] = c0
            if not freeze_beta:
                cand["beta"] = beta
        s, d = s_level_grad(caches[p], l, cand, model.edge, sub=True)
        model.cells[p][l] = cand
        Sc = {q: S[q].copy() for q in S}
        Sc[p][:, l] = s
        E, _ = model.forward(Sc)
        loss, _, dle = fs.map_resid(E, y)
        cw = model.channel_weights()[p]
        w = model.level_weights(p)[l]
        sp = np.maximum(s, 0.0)
        smax = np.maximum(sp, 1e-300)
        if model.pooling == "tied":
            Elv = np.where(sp > 0, np.power(smax, 1.0 / P), 0.0)
            dEl_ds = np.where(sp > 0, Elv / (P * np.where(sp > 1e-120, sp, 1e-120)), 0.0)
            dEl_dP = np.where(sp > 0, Elv * (-np.log(smax) / (P * P)), 0.0)
            dEl_dL = np.zeros_like(s)
        else:
            Elv = np.where(sp > 0, np.power(smax, L), 0.0)
            dEl_ds = np.where(sp > 0, L * np.power(smax, L - 1.0), 0.0)
            dEl_dP = np.zeros_like(s)
            dEl_dL = np.where(sp > 0, Elv * np.log(smax), 0.0)
        coef = dle * cw * w
        gp = np.zeros(6)
        gp[0] = np.sum(coef * dEl_ds * d[:, 0])
        gp[1] = np.sum(coef * (dEl_dP + dEl_ds * d[:, 1]))
        gp[2] = np.sum(coef * dEl_ds * d[:, 2])
        gp[3] = 0.0 if freeze_beta else np.sum(coef * dEl_ds * d[:, 3])
        gp[4] = np.sum(coef * dEl_ds * d[:, 4])
        gp[5] = 0.0 if model.pooling == "tied" else np.sum(coef * dEl_dL)
        if cand["vis"] != "curve":
            gp[0] = gp[2] = gp[3] = gp[4] = 0.0     # only P (and L) move
        model.cells[p][l] = orig
        np.nan_to_num(gp, copy=False)
        return loss, gp * decode6_jac(x)
    return obj


def p10_knees(model, caches):
    out = {}
    for p in model.planes:
        for l in range(LEVELS):
            mins = []
            for recs, _, _, _ in caches[p].iter_level(l, False):
                if recs.shape[0]:
                    cs, cd, _, _ = contrast_any(recs, 1.0, model.edge)
                    mins.append(np.minimum(cs, cd))
            p10 = float(np.quantile(np.concatenate(mins), 0.10)) if mins else 0.01
            out[(p, l)] = min(max(p10, fs.C0_LO), fs.C0_HI)
    return out


def gate_scan(model, caches, p, l, S, y):
    """Best gate knee on the shared GRID_C0 axis + golden refine, or off."""
    cell = model.cells[p][l]
    cache = caches[p]

    def mse_with(vis, knee):
        cand = dict(cell)
        cand.update({"vis": vis, "knee": knee if knee is not None else cell["knee"]})
        Sc = {q: S[q].copy() for q in S}
        Sc[p][:, l] = s_level_rows(cache, l, cand, model.edge, sub=True)
        return mse_of(model, Sc, y)

    grid = {float(k): mse_with("gate", float(k)) for k in fs.GRID_C0}
    bk = min(grid, key=grid.get)
    ks = sorted(grid)
    i = ks.index(bk)
    lo = math.log(ks[i - 1]) if i > 0 else math.log(bk) - 0.5
    hi = math.log(ks[i + 1]) if i < len(ks) - 1 else math.log(bk) + 0.5
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
    f1, f2 = mse_with("gate", math.exp(x1)), mse_with("gate", math.exp(x2))
    for _ in range(16):
        if f1 > f2:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + gr * (hi - lo)
            f2 = mse_with("gate", math.exp(x2))
        else:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - gr * (hi - lo)
            f1 = mse_with("gate", math.exp(x1))
    kref = math.exp(0.5 * (lo + hi))
    best = min([(grid[bk], "gate", bk), (mse_with("gate", kref), "gate", kref),
                (mse_with("off", None), "off", None)])
    return best


def beta_search(model, caches, S, y):
    """Golden search of the shared β over [0.05, 3] (log axis)."""
    def mse_beta(b):
        for p in model.planes:
            for c in model.cells[p]:
                if c["vis"] == "curve":
                    c["beta"] = b
        Sc = build_S(model, caches, sub=True)
        return mse_of(model, Sc, y)
    lo, hi = math.log(0.05), math.log(3.0)
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
    f1, f2 = mse_beta(math.exp(x1)), mse_beta(math.exp(x2))
    for _ in range(18):
        if f1 > f2:
            lo, x1, f1 = x1, x2, f2
            x2 = lo + gr * (hi - lo)
            f2 = mse_beta(math.exp(x2))
        else:
            hi, x2, f2 = x2, x1, f1
            x1 = hi - gr * (hi - lo)
            f1 = mse_beta(math.exp(x1))
    b = math.exp(0.5 * (lo + hi))
    mse_beta(b)
    return b


def fit(model, caches, y, protocol, beta0, beta_mode="shared"):
    knees = p10_knees(model, caches)
    for p in model.planes:
        for l in range(LEVELS):
            model.cells[p][l]["knee"] = knees[(p, l)]
            model.cells[p][l]["knee_p10"] = knees[(p, l)]
            model.cells[p][l]["beta"] = beta0
    S = build_S(model, caches)
    var_y = float(np.var(y))
    mse0 = mse_of(model, S, y)
    E0, _ = model.forward(S)
    sro0 = float(scipy.stats.spearmanr(-E0, y).statistic)
    log(f"init: mse {mse0:.5f} var(y) {var_y:.5f} srocc {sro0:.4f}")
    hist = [{"sweep": 0, "mse": mse0, "srocc": sro0}]
    if protocol == "prior":
        head_fit(model, S, y)
    else:
        def snapshot():
            return (json.loads(json.dumps(model.cells)),
                    {q: v.copy() for q, v in model.wl.items()}, model.wc.copy())
        best_mse, best_state = mse0, snapshot()
        for sweep in range(1, fs.MAX_SWEEPS + 1):
            Ssub = build_S(model, caches, sub=True)
            for p in model.planes:
                for l in range(LEVELS):
                    c = model.cells[p][l]
                    if c.get("vis0") == "gate":
                        _, vis, knee = gate_scan(model, caches, p, l, Ssub, y)
                        c["vis"] = vis
                        if knee is not None:
                            c["knee"] = knee
                        Ssub[p][:, l] = s_level_rows(caches[p], l, c, model.edge, sub=True)
                    elif c["vis"] == "curve":
                        cands = []
                        for k0 in fs.GRID_C0:
                            cand = dict(c)
                            cand["knee"] = float(k0)
                            Sc = {q: Ssub[q].copy() for q in Ssub}
                            Sc[p][:, l] = s_level_rows(caches[p], l, cand, model.edge, sub=True)
                            cands.append((mse_of(model, Sc, y), float(k0)))
                        c["knee"] = min(cands)[1]
                    x0 = encode6(c["g"], c["P"], c["knee"], c["beta"], c["sharp"], c["L"])
                    per_level = beta_mode == "level" and c["vis"] == "curve"
                    bl, bx = fs.adam(level_obj(model, caches, p, l, Ssub, y, freeze_beta=not per_level),
                                     x0, fs.ADAM_STEPS_LEVEL)
                    g, P, k, b, sh, L = decode6(bx)
                    c.update({"P": P, "L": L if model.pooling == "free" else 1.0})
                    if c["vis"] == "curve":
                        c.update({"g": g, "knee": k, "sharp": sh})
                        if per_level:
                            c["beta"] = b
                    Ssub[p][:, l] = s_level_rows(caches[p], l, c, model.edge, sub=True)
                    log(f"  s{sweep} {p} l{l}: {c['vis']} knee={c['knee']:.4g} g={c['g']:.3f} "
                        f"P={c['P']:.3f} sharp={c['sharp']:.3f} L={c['L']:.3f} sub-mse {bl:.5f}")
            if beta_mode == "shared" and any(c["vis"] == "curve" for p in model.planes
                                             for c in model.cells[p]):
                b = beta_search(model, caches, Ssub, y)
                log(f"  s{sweep} shared beta = {b:.4f}")
            S = build_S(model, caches)
            head_fit(model, S, y)
            m = mse_of(model, S, y)
            hist.append({"sweep": sweep, "mse": m})
            log(f"sweep {sweep}: full mse {m:.5f} (best {best_mse:.5f})")
            improved = best_mse - m
            if m < best_mse:
                best_mse, best_state = m, snapshot()
            if improved < fs.REL_STOP * max(best_mse, 1e-12):
                break
        model.cells, model.wl, model.wc = best_state
    S = build_S(model, caches)
    E, _ = model.forward(S)
    mp = fs.fit_map(E, y)
    return {"map": mp, "history": hist, "var_y": var_y,
            "fit_srocc": float(scipy.stats.spearmanr(-E, y).statistic),
            "fit_krocc": float(scipy.stats.kendalltau(-E, y).statistic)}


def load_segments(path, planes):
    segs = json.loads(Path(path).read_text())
    ys, caches = [], {p: [] for p in planes}
    for s in segs:
        rows = list(csv.DictReader(open(s["pairs"]), delimiter="\t"))
        y = np.asarray([float(r["human_score"]) for r in rows]) * float(s.get("label_scale", 1.0))
        for p in planes:
            c = PlaneCacheF16(s["bins"][p])
            assert c.n_rows == len(y), (s["name"], p, c.n_rows, len(y))
            caches[p].append(c)
        ys.append(y)
    return np.concatenate(ys), {p: ConcatCache(v) for p, v in caches.items()}, segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit"])
    ap.add_argument("--segs", required=True)
    ap.add_argument("--planes", required=True)
    ap.add_argument("--vis", choices=["curve", "gate", "off"], required=True)
    ap.add_argument("--contrast", choices=["plain", "edge"], required=True)
    ap.add_argument("--pooling", choices=["free", "tied"], required=True)
    ap.add_argument("--constants", choices=["prior", "fit"], required=True)
    ap.add_argument("--beta", type=float, default=0.693)
    ap.add_argument("--beta-mode", choices=["shared", "level"], default="shared",
                    help="curve fits: one beta for every plane and level (golden search; default) "
                         "or a beta per plane and level fitted with the other level constants "
                         "(the talk's per-level A, B)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="")
    a = ap.parse_args()
    planes = a.planes.split(",")
    y, caches, segs = load_segments(a.segs, planes)
    cells = {p: [{"vis": a.vis, "vis0": a.vis, "knee": 0.01, "beta": a.beta, "sharp": 4.0,
                  "g": 1.0, "P": 1.0, "L": 1.0} for _ in range(LEVELS)] for p in planes}
    model = Model(planes, cells, a.pooling, a.contrast == "edge")
    log(f"{a.name}: {len(y)} rows, planes {planes}, vis {a.vis}, contrast {a.contrast}, "
        f"pooling {a.pooling}, constants {a.constants}")
    t0 = time.time()
    res = fit(model, caches, y, a.constants, a.beta, a.beta_mode)
    cw = model.channel_weights()
    art = {
        "name": a.name, "planes": planes, "vis": a.vis, "contrast": a.contrast,
        "pooling": a.pooling, "constants": a.constants, "beta0": a.beta, "beta_mode": a.beta_mode,
        "cells": {p: [{k: v for k, v in c.items() if k != "vis0"} for c in model.cells[p]]
                  for p in planes},
        "level_weights": {p: model.level_weights(p).tolist() for p in planes},
        "channel_weights": {p: cw[p] for p in planes},
        "map": res["map"], "fit": {k: v for k, v in res.items() if k != "map"},
        "fit_rows": int(len(y)), "segments": segs, "wall_s": time.time() - t0,
        "fitter": "research/2026-09-dvifm/dvifmish-eval/fit_dvifmish.py",
    }
    Path(a.out).write_text(json.dumps(art, indent=1, default=float) + "\n")
    log(f"WROTE {a.out}: fit srocc {res['fit_srocc']:.4f} mse {res['map']['mse']:.5f} "
        f"wall {art['wall_s']:.0f}s")


if __name__ == "__main__":
    main()
