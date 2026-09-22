#!/usr/bin/env python3
"""faithful lane — luma-only DVIFM in the talk's configuration + ablations.

Model (per row r, level l in 0..4, plane = ycbcr_y):
    cs_b, cd_b = plain or corner-discounted block range through phi_g
    v_b        = max(v(cs), v(cd))    (talk: merged visibility)
    e_b        = m_b^P                (max |ref-dist| to learned power)
    s_{r,l}    = mean_b(v_b * e_b)
    E_{r,l}    = s_{r,l}^L            — L free per level (talk's Lp pooling)
                 [tied mode: E = s^(1/P) — our drifted form]
    E_r        = sum_l softmax(wl)_l * E_{r,l}    (convex band mix)
    loss       = Amendment-1 refit-map MSE: yhat = A*exp(-lam*E)+B refit
                 inside every objective evaluation (identical protocol to
                 fit_standalone.py; SROCC/KROCC of -E reported alongside).

Arms (one deviation from faithful F at a time):
    faithful : band=laplacian, edge=False (plain range), vis=curve, L free,
               fit on human-label mix (cid22a + tid/kadid JPEG+JP2K nt)
    band_local : F with band=local            (needs local-band caches)
    edge_disc  : F with edge=True             (our 3x3 corner discount)
    gate       : F with vis=gate              (our two-state gate)
    tied_pool  : F with L := 1/P              (Lp folded into error power)
    fitmix     : F config fitted on cid22_dev (ssim2/100 pseudo-labels)

CLI:
  fit_faithful.py fit  --arm NAME --fitsegs seg.json --out fit.json
  fit_faithful.py eval --fit fit.json --evalsegs seg.json --out scores.json

seg.json: [{"name": str, "pairs": tsv, "bin": cache.bin,
            "rows": <json row-index list path, optional>,
            "invert_label": <bool, optional> — set when the pairs file
            carries DMOS orientation (higher=worse); emits 1-y so all
            segments share the quality convention (higher=better)}]
The pairs file's human_score column supplies y (already quality-oriented
0..1 lineage units); `rows` selects a subset of cache/pairs rows.
"""
import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, '/mnt/v/output/zensim/dvifm-screen2d-2026-09-19/tools')
sys.path.insert(0, '/home/lilith/work/zen/zensim/tools/joint_core')
import numpy as np
import scipy.stats
import fit_standalone as fs
from fit_core import PlaneCacheF16, RowView

LEVELS = 5
L_LO, L_HI = 0.02, 50.0


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------- per-level params (raw6) ----------

def encode6(g, P, c0, beta, sharp, L):
    gg = min(max(g, fs.G_LO + 1e-9), fs.G_HI - 1e-9)
    bb = min(max(beta, fs.B_LO + 1e-9), fs.B_HI - 1e-9)
    cc = min(max(c0, fs.C0_LO), fs.C0_HI)
    ll = min(max(L, L_LO), L_HI)
    return np.array([
        math.log((gg - fs.G_LO) / (fs.G_HI - gg)),
        math.log(max(P, 1e-9)),
        math.log(cc),
        math.log((bb - fs.B_LO) / (fs.B_HI - bb)),
        math.log(math.expm1(max(sharp, 1.0 + 1e-6) - 1.0)),
        math.log(ll),
    ])


def decode6(raw):
    r0 = min(max(float(raw[0]), -60.0), 60.0)
    r1 = min(max(float(raw[1]), -20.0), 20.0)
    r2 = min(max(float(raw[2]), math.log(fs.C0_LO)), math.log(fs.C0_HI))
    r3 = min(max(float(raw[3]), -60.0), 60.0)
    r4 = min(max(float(raw[4]), -30.0), 30.0)
    r5 = min(max(float(raw[5]), math.log(L_LO)), math.log(L_HI))
    return (fs.G_LO + (fs.G_HI - fs.G_LO) * fs.sig(r0), math.exp(r1),
            math.exp(r2), fs.B_LO + (fs.B_HI - fs.B_LO) * fs.sig(r3),
            1.0 + fs.softplus(r4), math.exp(r5))


def decode6_jac(raw):
    """d(phys)/d(raw) for decode6."""
    r0 = min(max(float(raw[0]), -60.0), 60.0)
    r3 = min(max(float(raw[3]), -60.0), 60.0)
    s0, s3 = fs.sig(r0), fs.sig(r3)
    return np.array([
        (fs.G_HI - fs.G_LO) * s0 * (1 - s0),
        math.exp(min(max(float(raw[1]), -20.0), 20.0)),
        math.exp(min(max(float(raw[2]), math.log(fs.C0_LO)),
                     math.log(fs.C0_HI))),
        (fs.B_HI - fs.B_LO) * s3 * (1 - s3),
        fs.sig(min(max(float(raw[4]), -30.0), 30.0)),
        math.exp(min(max(float(raw[5]), math.log(L_LO)), math.log(L_HI))),
    ])


# ---------- contrast / s ----------

def contrast_any(recs, g, edge):
    """(cs, cd, dcs/dg, dcd/dg). edge=True: min over 3x3 corners (ours);
    edge=False: plain block range = phi_g(max cmax) - phi_g(min cmin)
    (the talk's form; corners tile the 5x5 block, so argmax/argmin over
    the 4 corner extrema gives the whole-block extremum)."""
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
    mode = cell["vis"]
    if mode == "off":
        return np.ones_like(c)
    if mode == "gate":
        return (c <= cell["kappa"]).astype(np.float64)
    return fs.visibility(c, cell["c0"], cell["beta"], cell["sharp"])[0]


def s_level_rows(cache, l, cell, sub=False):
    """s_{r,l} = mean_b(max(v_s,v_d) * m^P) per row."""
    s = np.zeros(cache.n_rows)
    for recs, nz, bounds, counts in cache.iter_level(l, sub):
        if recs.shape[0] == 0:
            continue
        cs, cd, _, _ = contrast_any(recs, cell["g"], cell["edge"])
        v = np.maximum(vis_of(cs, cell), vis_of(cd, cell))
        e = np.power(recs[:, 0], cell["P"])
        den = np.maximum(counts, 1).astype(np.float64)
        s[nz] = np.add.reduceat(v * e, bounds) / den
    return s


def s_level_grad(cache, l, cell, sub=True):
    """(s[n_rows], ds/d(g,P,c0,beta,sharp) [n_rows,5]) on the block
    subsample — mirrors s_level(with_grad) + edge flag. For non-curve
    visibility (gate/off) v is not a function of the curve params, so
    only ds/dP is nonzero."""
    s = np.zeros(cache.n_rows)
    d = np.zeros((cache.n_rows, 5))
    for recs, nz, bounds, counts in cache.iter_level(l, sub):
        if recs.shape[0] == 0:
            continue
        cs, cd, dcs, dcd = contrast_any(recs, cell["g"], cell["edge"])
        curve = cell["vis"] == "curve"
        if curve:
            vs, vs_c0, vs_b, vs_s, vs_c = fs.visibility(
                cs, cell["c0"], cell["beta"], cell["sharp"])
            vd, vd_c0, vd_b, vd_s, vd_c = fs.visibility(
                cd, cell["c0"], cell["beta"], cell["sharp"])
            pick = vd > vs
            v = np.where(pick, vd, vs)
        else:
            v = np.maximum(vis_of(cs, cell), vis_of(cd, cell))
        e = np.power(recs[:, 0], cell["P"])
        den = np.maximum(counts, 1).astype(np.float64)
        s[nz] = np.add.reduceat(v * e, bounds) / den
        de = np.where(recs[:, 0] > 0,
                      e * np.log(np.maximum(recs[:, 0], 1e-300)), 0.0)
        d[nz, 1] = np.add.reduceat(v * de, bounds) / den
        if curve:
            d[nz, 0] = np.add.reduceat(
                np.where(pick, vd_c * dcd, vs_c * dcs) * e, bounds) / den
            d[nz, 2] = np.add.reduceat(np.where(pick, vd_c0, vs_c0) * e,
                                       bounds) / den
            d[nz, 3] = np.add.reduceat(np.where(pick, vd_b, vs_b) * e,
                                       bounds) / den
            d[nz, 4] = np.add.reduceat(np.where(pick, vd_s, vs_s) * e,
                                       bounds) / den
    return s, d


# ---------- forward ----------

def forward_E(cells, wl, S, tied):
    """S: (n,5) s-matrix. E = sum_l softmax(wl)_l * s_l^L_l."""
    w = np.exp(wl - wl.max())
    w = w / w.sum()
    El = np.zeros_like(S)
    for l in range(LEVELS):
        s = np.maximum(S[:, l], 0.0)
        if tied:
            P = cells[l]["P"]
            El[:, l] = np.where(s > 0, np.power(np.maximum(s, 1e-300),
                                                1.0 / P), 0.0)
        else:
            L = cells[l]["L"]
            El[:, l] = np.where(s > 0, np.power(np.maximum(s, 1e-300),
                                                L), 0.0)
    return El @ w, El, w


def build_S(cells, caches, sub=False):
    """caches: list of segment caches (same rows order as y). Each cache
    may itself be a concatenation handled upstream — here `caches` is a
    single PlaneCache-like covering all rows."""
    S = np.zeros((caches.n_rows, LEVELS))
    for l in range(LEVELS):
        S[:, l] = s_level_rows(caches, l, cells[l], sub=sub)
    return S


# ---------- fit ----------

def init_cells(cache, edge, vis, L0=1.0):
    """c0 = p10 of min(cs,cd) per level at g=1 on the fit rows; rest at
    the design-doc prior constants."""
    cells = []
    for l in range(LEVELS):
        mins = []
        for recs, nz, bounds, counts in cache.iter_level(l, False):
            if recs.shape[0]:
                cs, cd, _, _ = contrast_any(recs, 1.0, edge)
                mins.append(np.minimum(cs, cd))
        p10 = float(np.quantile(np.concatenate(mins), 0.10)) \
            if mins else 0.01
        c0 = min(max(p10, fs.C0_LO), fs.C0_HI)
        cells.append({"vis": vis, "edge": edge, "g": 1.0, "P": 1.0,
                      "c0": c0, "beta": 0.65, "sharp": 4.0, "L": L0,
                      "kappa": c0, "c0_p10_raw": p10})
    return cells


def loss_of(cells, wl, S, y, tied):
    E, _, _ = forward_E(cells, wl, S, tied)
    return fs.fit_map(E, y)


def level_obj(cache, l, cells, wl, S, y, tied):
    """(loss, grad_raw6) for level l — s recomputed per candidate."""
    def obj(x):
        g, P, c0, beta, sharp, L = decode6(x)
        cand = dict(cells[l])
        cand.update({"g": g, "P": P, "c0": c0, "beta": beta,
                     "sharp": sharp, "L": L})
        s, d = s_level_grad(cache, l, cand, sub=True)
        Sc = S.copy()
        Sc[:, l] = s
        E, El, w = forward_E(cells, wl, Sc, tied)
        loss, yhat, dle = fs.map_resid(E, y)
        smax = np.maximum(s, 1e-300)
        sp = np.maximum(s, 0.0)
        if tied:
            Elv = np.where(sp > 0, np.power(smax, 1.0 / P), 0.0)
            dEl_ds = np.where(sp > 0, Elv / (P * np.where(sp > 1e-120,
                                                          sp, 1e-120)), 0.0)
            dEl_dP = np.where(sp > 0, Elv * (-np.log(smax) / (P * P)), 0.0)
            dEl_dL = np.zeros_like(s)
        else:
            Elv = np.where(sp > 0, np.power(smax, L), 0.0)
            dEl_ds = np.where(sp > 0, L * np.power(smax, L - 1.0), 0.0)
            dEl_dP = np.zeros_like(s)
            dEl_dL = np.where(sp > 0, Elv * np.log(smax), 0.0)
        coef = dle * w[l]
        g_phys = np.zeros(6)
        g_phys[0] = np.sum(coef * dEl_ds * d[:, 0])
        g_phys[1] = np.sum(coef * (dEl_dP + dEl_ds * d[:, 1]))
        g_phys[2] = np.sum(coef * dEl_ds * d[:, 2])
        g_phys[3] = np.sum(coef * dEl_ds * d[:, 3])
        g_phys[4] = np.sum(coef * dEl_ds * d[:, 4])
        g_phys[5] = np.sum(coef * dEl_dL)
        np.nan_to_num(g_phys, copy=False)
        return loss, g_phys * decode6_jac(x)
    return obj


def head_obj(cells, wl0, S, y, tied):
    def obj(x):
        E, El, w = forward_E(cells, x, S, tied)
        loss, yhat, dle = fs.map_resid(E, y)
        Ep = El @ w
        g = np.array([np.sum(dle * w[k] * (El[:, k] - Ep))
                      for k in range(LEVELS)])
        np.nan_to_num(g, copy=False)
        return loss, g
    return obj


def grid_init_level(cache, l, cells, wl, S, y, tied, sub=True):
    """Amendment-1 grid: (c0 x beta) on the block subsample, refit map."""
    recs, bounds, counts = cache.level_arrays(l, sub=True)
    nz = cache.nz_rows[l]
    den = np.maximum(counts, 1).astype(np.float64)
    e = np.power(recs[:, 0], cells[l]["P"])
    best = None
    for c0 in fs.GRID_C0:
        for beta in fs.GRID_BETA:
            cand = dict(cells[l]); cand.update({"c0": float(c0),
                                                "beta": float(beta)})
            cs, cd, _, _ = contrast_any(recs, cand["g"], cand["edge"])
            v = np.maximum(vis_of(cs, cand), vis_of(cd, cand))
            sr = np.zeros(cache.n_rows)
            sr[nz] = np.add.reduceat(v * e, bounds) / den
            Sc = S.copy(); Sc[:, l] = sr
            E, _, _ = forward_E(cells, wl, Sc, tied)
            mse = fs.fit_map(E, y)["mse"]
            if best is None or mse < best[0]:
                best = (mse, float(c0), float(beta))
    return best


def fit_curve_arm(cache, y, cells, tied, name):
    """Full fit: sanity gate -> per-level grid+Adam -> head; <=3 sweeps."""
    wl = np.zeros(LEVELS)
    S = build_S(cells, cache, sub=False)
    var_y = float(np.var(y))
    mp0 = loss_of(cells, wl, S, y, tied)
    E0, _, _ = forward_E(cells, wl, S, tied)
    sro0, kro0 = fs._rank_stats(E0, y)
    sane = mp0["mse"] < var_y
    log(f"{name}: init refit-MSE {mp0['mse']:.4f} var(y) {var_y:.4f} "
        f"SROCC {sro0:.4f} sane={sane}")
    hist = [{"sweep": 0, "mse": mp0["mse"], "srocc": sro0}]
    best_loss = mp0["mse"]
    best_state = ([dict(c) for c in cells], wl.copy(), S.copy())
    if sane:
        for sweep in range(1, fs.MAX_SWEEPS + 1):
            for l in range(LEVELS):
                bm = grid_init_level(cache, l, cells, wl, S, y, tied)
                if bm is not None:
                    cells[l]["c0"], cells[l]["beta"] = bm[1], bm[2]
                x0 = encode6(cells[l]["g"], cells[l]["P"], cells[l]["c0"],
                             cells[l]["beta"], cells[l]["sharp"],
                             cells[l]["L"])
                bl, bx = fs.adam(level_obj(cache, l, cells, wl, S, y,
                                           tied), x0, fs.ADAM_STEPS_LEVEL)
                g, P, c0, beta, sharp, L = decode6(bx)
                cells[l].update({"g": g, "P": P, "c0": c0, "beta": beta,
                                 "sharp": sharp, "L": L})
                S[:, l] = s_level_rows(cache, l, cells[l], sub=False)
                log(f"  {name} s{sweep} l{l}: MSE {bl:.4f} "
                    f"(g={g:.3f} P={P:.3f} c0={c0:.4g} b={beta:.3f} "
                    f"s={sharp:.3f} L={L:.3f})")
            hb, hx = fs.adam(head_obj(cells, wl, S, y, tied), wl,
                             fs.ADAM_STEPS_HEAD)
            wl = hx
            hist.append({"sweep": sweep, "mse": hb})
            log(f"{name} sweep {sweep}: MSE {hb:.4f}")
            if hb < best_loss:
                best_state = ([dict(c) for c in cells], wl.copy(),
                              S.copy())
            if best_loss - hb < fs.REL_STOP * max(best_loss, 1e-12):
                best_loss = min(best_loss, hb)
                break
            best_loss = min(best_loss, hb)
        cells, wl, S = best_state
    E, _, _ = forward_E(cells, wl, S, tied)
    mp = fs.fit_map(E, y)
    sro, kro = fs._rank_stats(E, y)
    return {"cells": cells, "wl": wl.tolist(), "tied": tied,
            "map": mp, "fit": {"mse": mp["mse"], "var_y": var_y,
                               "srocc": sro, "krocc": kro,
                               "sanity_init_mse": mp0["mse"],
                               "sanity_init_srocc": sro0,
                               "sane": bool(sane)},
            "history": hist}


def fit_gate_arm(cache, y, cells, tied, name):
    """Gate ablation: per level pick (off | kappa on GRID_C0 + golden
    refine) on the subsample, then Adam-refine (P, L) + head. g/c0/beta/
    sharp stay at init (gate ignores them)."""
    wl = np.zeros(LEVELS)
    S = build_S(cells, cache, sub=False)
    var_y = float(np.var(y))
    mp0 = loss_of(cells, wl, S, y, tied)
    E0, _, _ = forward_E(cells, wl, S, tied)
    sro0, kro0 = fs._rank_stats(E0, y)
    log(f"{name}: init refit-MSE {mp0['mse']:.4f} SROCC {sro0:.4f}")
    best_loss = mp0["mse"]
    best_state = ([dict(c) for c in cells], wl.copy(), S.copy())
    for sweep in range(1, fs.MAX_SWEEPS + 1):
        for l in range(LEVELS):
            recs, bounds, counts = cache.level_arrays(l, sub=True)
            nz = cache.nz_rows[l]
            den = np.maximum(counts, 1).astype(np.float64)
            cs, cd, _, _ = contrast_any(recs, cells[l]["g"],
                                        cells[l]["edge"])
            e = np.power(recs[:, 0], cells[l]["P"])
            cmin = np.minimum(cs, cd)

            def eval_v(v):
                sr = np.zeros(cache.n_rows)
                sr[nz] = np.add.reduceat(v * e, bounds) / den
                Sc = S.copy(); Sc[:, l] = sr
                E, _, _ = forward_E(cells, wl, Sc, tied)
                return fs.fit_map(E, y)["mse"]

            mse_off = eval_v(np.ones_like(cs))
            grid = {float(k): eval_v((cmin <= k).astype(np.float64))
                    for k in fs.GRID_C0}
            bk = min(grid, key=grid.get)
            best = (grid[bk], "gate", bk)
            ks = sorted(grid)
            i = ks.index(bk)
            lo = math.log(ks[i - 1]) if i > 0 else math.log(bk) - 0.5
            hi = math.log(ks[i + 1]) if i < len(ks) - 1 else math.log(bk) + 0.5
            gr = (math.sqrt(5.0) - 1.0) / 2.0
            x1, x2 = hi - gr * (hi - lo), lo + gr * (hi - lo)
            f1 = eval_v((cmin <= math.exp(x1)).astype(np.float64))
            f2 = eval_v((cmin <= math.exp(x2)).astype(np.float64))
            for _ in range(20):
                if f1 > f2:
                    lo, x1, f1 = x1, x2, f2
                    x2 = lo + gr * (hi - lo)
                    f2 = eval_v((cmin <= math.exp(x2)).astype(np.float64))
                else:
                    hi, x2, f2 = x2, x1, f1
                    x1 = hi - gr * (hi - lo)
                    f1 = eval_v((cmin <= math.exp(x1)).astype(np.float64))
            kref = math.exp(0.5 * (lo + hi))
            mref = eval_v((cmin <= kref).astype(np.float64))
            if mref < best[0]:
                best = (mref, "gate", kref)
            if mse_off <= best[0]:
                best = (mse_off, "off", None)
            cand = dict(cells[l])
            cand["vis"] = best[1]
            cand["kappa"] = best[2]
            s_full = s_level_rows(cache, l, cand, sub=False)
            Sc = S.copy(); Sc[:, l] = s_full
            E, _, _ = forward_E(cells, wl, Sc, tied)
            fl = fs.fit_map(E, y)["mse"]
            E0c, _, _ = forward_E(cells, wl, S, tied)
            cur = fs.fit_map(E0c, y)["mse"]
            if fl < cur:
                cells[l] = cand
                S[:, l] = s_full
                log(f"  {name} s{sweep} l{l}: {best[1]}"
                    f"{'' if best[2] is None else f'@{best[2]:.4g}'} "
                    f"sub {best[0]:.3f} -> full {fl:.3f} (was {cur:.3f})")
            # continuous params still refine (P, L) with vis fixed
            if cells[l]["vis"] != "off":
                x0 = encode6(cells[l]["g"], cells[l]["P"],
                             cells[l]["c0"], cells[l]["beta"],
                             cells[l]["sharp"], cells[l]["L"])
                bl, bx = fs.adam(level_obj(cache, l, cells, wl, S, y,
                                           tied), x0, fs.ADAM_STEPS_LEVEL)
                g, P, c0, beta, sharp, L = decode6(bx)
                cells[l].update({"P": P, "L": L})
                S[:, l] = s_level_rows(cache, l, cells[l], sub=False)
        hb, hx = fs.adam(head_obj(cells, wl, S, y, tied), wl,
                         fs.ADAM_STEPS_HEAD)
        wl = hx
        log(f"{name} sweep {sweep}: MSE {hb:.4f}")
        if hb < best_loss:
            best_state = ([dict(c) for c in cells], wl.copy(), S.copy())
        if best_loss - hb < fs.REL_STOP * max(best_loss, 1e-12):
            best_loss = min(best_loss, hb)
            break
        best_loss = min(best_loss, hb)
    cells, wl, S = best_state
    E, _, _ = forward_E(cells, wl, S, tied)
    mp = fs.fit_map(E, y)
    sro, kro = fs._rank_stats(E, y)
    return {"cells": cells, "wl": wl.tolist(), "tied": tied,
            "map": mp, "fit": {"mse": mp["mse"], "var_y": var_y,
                               "srocc": sro, "krocc": kro,
                               "sanity_init_mse": mp0["mse"],
                               "sanity_init_srocc": sro0,
                               "sane": bool(mp0["mse"] < var_y)}}


# ---------- segment loading ----------

def load_segments(seg_path):
    segs = json.loads(Path(seg_path).read_text())
    out = []
    for s in segs:
        rows = list(csv.DictReader(open(s["pairs"]), delimiter="\t"))
        y = np.asarray([float(r["human_score"]) for r in rows])
        if s.get("invert_label"):
            y = 1.0 - y
        if "rows" in s:
            idx = np.asarray(json.loads(Path(s["rows"]).read_text()),
                             dtype=np.int64)
            y = y[idx]
            cache = RowView(s["bin"], idx)
        else:
            cache = PlaneCacheF16(s["bin"])
        assert cache.n_rows == len(y), (s["name"], cache.n_rows, len(y))
        out.append({"name": s["name"], "cache": cache, "y": y,
                    "ref": [r["ref_path"] for r in rows] if not s.get("rows")
                           else [rows[i]["ref_path"] for i in idx]})
    return out


def cat_caches(segs):
    """Build a virtual concatenated cache: wraps each seg cache, stacks
    rows. Implemented by materialising per-level CSR concatenation."""
    class _Cat:
        def __init__(self, caches):
            self.caches = caches
            self.n_rows = sum(c.n_rows for c in caches)
            self.nz_rows = []
            off = 0
            for l in range(LEVELS):
                rows = []
                off = 0
                for c in caches:
                    rows.extend((off + r) for r in c.nz_rows[l])
                    off += c.n_rows
                self.nz_rows.append(np.asarray(rows, dtype=np.int64))
        def iter_level(self, l, sub):
            off = 0
            for c in self.caches:
                for recs, nz, bounds, counts in c.iter_level(l, sub):
                    yield recs, nz + off, bounds, counts
                off += c.n_rows
        def level_arrays(self, l, sub=True):
            recs_l, bounds_l, counts_l, nz_l = [], [], [], []
            off_r = off_b = 0
            for c in self.caches:
                r, b, cnt = c.level_arrays(l, sub=sub)
                recs_l.append(r)
                bounds_l.append(b + off_b)
                counts_l.append(cnt)
                nz_l.append(c.nz_rows[l] + off_r)
                off_b += len(r)
                off_r += c.n_rows
            recs = np.concatenate(recs_l) if recs_l else np.zeros((0, 18))
            bounds = np.concatenate(bounds_l) if bounds_l else np.zeros(0)
            counts = np.concatenate(counts_l) if counts_l else np.zeros(0)
            self.nz_rows[l] = np.concatenate(nz_l) if nz_l else \
                np.zeros(0, dtype=np.int64)
            return recs, bounds, counts
    return _Cat([s["cache"] for s in segs])


def score_arm(art, cache):
    S = build_S(art["cells"], cache, sub=False)
    E, _, _ = forward_E(art["cells"], np.asarray(art["wl"]), S,
                        art["tied"])
    a, b, lam = art["map"]["A"], art["map"]["B"], art["map"]["lambda"]
    return E, a * np.exp(-lam * E) + b


def metrics_of(E, y):
    return {"srocc": float(scipy.stats.spearmanr(-E, y).statistic),
            "krocc": float(scipy.stats.kendalltau(-E, y).statistic),
            "n": int(len(y))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fit", "eval"])
    ap.add_argument("--arm", default="faithful")
    ap.add_argument("--fitsegs")
    ap.add_argument("--evalsegs")
    ap.add_argument("--fit-art", help="existing fit artefact (eval)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    if a.cmd == "fit":
        spec = ARM_SPECS[a.arm]
        segs = load_segments(a.fitsegs)
        cache = cat_caches(segs)
        y = np.concatenate([s["y"] for s in segs])
        log(f"fit {a.arm}: {len(y)} rows from "
            f"{[(s['name'], len(s['y'])) for s in segs]}")
        cells = init_cells(cache, spec["edge"], spec["vis"],
                           L0=1.0)
        t0 = time.time()
        if spec["vis"] == "gate":
            art = fit_gate_arm(cache, y, cells, spec["tied"], a.arm)
        else:
            art = fit_curve_arm(cache, y, cells, spec["tied"], a.arm)
        art["wall_s"] = time.time() - t0
        art["arm"] = a.arm
        art["spec"] = spec
        art["fit_rows"] = int(len(y))
        art["fit_segs"] = [(s["name"], int(len(s["y"]))) for s in segs]
        Path(a.out).write_text(json.dumps(art, indent=1,
                                        default=float) + "\n")
        log(f"WROTE {a.out} wall {art['wall_s']:.0f}s "
            f"mse {art['fit']['mse']:.3f} srocc {art['fit']['srocc']:.4f}")
    else:
        art = json.loads(Path(a.fit_art).read_text())
        out = {}
        for s in load_segments(a.evalsegs):
            E, yhat = score_arm(art, s["cache"])
            m = metrics_of(E, s["y"])
            out[s["name"]] = m
            np.savez(Path(a.out).parent / f"scores_{art['arm']}_"
                     f"{s['name']}.npz", E=E, yhat=yhat, y=s["y"])
            log(f"eval {s['name']}: SROCC {m['srocc']:.4f} "
                f"KROCC {m['krocc']:.4f} n={m['n']}")
        Path(a.out).write_text(json.dumps(out, indent=1) + "\n")


ARM_SPECS = {
    # the talk's configuration, luma-only
    "faithful":   {"edge": False, "vis": "curve", "tied": False},
    # our deviations, one at a time from faithful:
    "edge_disc":  {"edge": True,  "vis": "curve", "tied": False},
    "gate":       {"edge": False, "vis": "gate",  "tied": False},
    "tied_pool":  {"edge": False, "vis": "curve", "tied": True},
    # band/fit-mix are selected by the caches/fitsegs, same spec flags:
    "band_local": {"edge": False, "vis": "curve", "tied": False},
    "fitmix":     {"edge": False, "vis": "curve", "tied": False},
    # and the fully-drifted config for the total-gap check:
    "all_ours":   {"edge": True,  "vis": "gate",  "tied": True},
    # our actual production configuration: every drift + pseudo-label fit
    "ours_full":  {"edge": True,  "vis": "gate",  "tied": True},
}

if __name__ == "__main__":
    main()
