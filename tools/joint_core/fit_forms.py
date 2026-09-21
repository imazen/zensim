#!/usr/bin/env python3
"""X2 — DVIFM standalone constant-FORM comparison on joint-core-v1.

Three arms, identical fit rows per seed, identical head-fit and output-map
machinery (fit_standalone Amendment-1), differing ONLY in the visibility
constants' form:

  curve : fully fitted smooth curve v(c)=exp(-softplus(beta*sharp*
          (ln c - ln c0))/sharp) per (plane,level); (g,P,c0,beta,sharp)
          all fitted -> fs.fit_variant verbatim.
  gate  : two-state v(c)=[c <= kappa], or uniform 1 ("off") where a cell
          wants no masking; g=P=1 so a shipped kernel is one integer
          compare per block. Per cell: kappa scanned on the same GRID_C0
          axis + golden refine; mode = argmin(off, best-kappa) on refit
          MSE; accepted only if the full-block rescore improves.
  prior : no loss-fitted constants: g=1, P=1, beta=0.65, sharp=4,
          c0 = 10th percentile of min(cs,cd) over THIS fit subset's
          records per (plane,level). Head (level/channel weights) still
          fitted — same machinery.

Usage:
  fit_forms.py <pairs-tsv> <outdir> <surfaces_dir>
      ycbcr_y=<y.bin> ycbcr_cb=<cb.bin> ycbcr_cr=<cr.bin>
      [--init-spec spec.json] [--seeds a,b,c] [--rows 1024]
      [--score-tasks tasks.json]

Each seed draws a seeded uniform row subset of the SDR fit domain (the
core file is cell-blocked; bare strides alias — see fit_core.py), fits all
three arms on the IDENTICAL subset, then scores every arm on the eval
legs listed in --score-tasks:

  [{"name": "...", "ycbcr_y": bin, "ycbcr_cb": bin, "ycbcr_cr": bin,
    "y": "<json {y:[...], ref:[...]}>" , "rows": "<.npy/.json indices>"}]

`rows` selects a subset of the cache rows (e.g. KADID refs {1,3,5});
`y`/`ref` must be in the same order. Arm artefacts and per-(seed,arm,leg)
scores land under <outdir>.
"""
import csv
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, '/mnt/v/output/zensim/dvifm-screen2d-2026-09-19/tools')
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import scipy.stats
import fit_standalone as fs
from fit_core import PlaneCacheF16, RowView


class RowViewF32(fs.PlaneCache):
    """Row subset of an f32 cache (2d-screen caches are uncapped f32)."""

    def __init__(self, bin_path, rows):
        super().__init__(bin_path)
        rows = np.asarray(rows, np.int64)
        self.nb = self.nb[rows]
        self.row_elem = self.row_elem[rows]
        self.n_rows = len(rows)
        self.nz_rows = [np.nonzero(self.nb[:, l])[0]
                        for l in range(fs.LEVELS)]


DEFAULT_SPEC = ('/mnt/v/output/zensim/dvifm-screen-2026-09-19/specs/'
                'dvifm-local-fitted-final.json')


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------- visibility forms ----------

def vis_form(c, cell):
    mode = cell["mode"]
    if mode == "off":
        return np.ones_like(c)
    if mode == "gate":
        return (c <= cell["kappa"]).astype(np.float64)
    return fs.visibility(c, cell["c0"], cell["beta"], cell["sharp"])[0]


def s_level_form(cache, l, cell, sub=False):
    g, P = cell["g"], cell["p"]
    s = np.zeros(cache.n_rows)
    for recs, nz, bounds, counts in cache.iter_level(l, sub):
        if recs.shape[0] == 0:
            continue
        cs, cd, _, _ = fs.contrast_grad(recs, g)
        v = np.maximum(vis_form(cs, cell), vis_form(cd, cell))
        e = np.power(recs[:, 0], P)
        den = np.maximum(counts, 1).astype(np.float64)
        s[nz] = np.add.reduceat(v * e, bounds) / den
    return s


def build_S_form(cells, planes, caches, sub=False):
    S = {}
    for p in planes:
        c = caches[p]
        S[p] = np.zeros((c.n_rows, fs.LEVELS))
        for l in range(fs.LEVELS):
            S[p][:, l] = s_level_form(c, l, cells[p][l], sub=sub)
    return S


def encode_cells(params, cells):
    """Keep params.raw in sync with cells — forward_E reads phys[1]=P
    (always 1.0 for gate/prior arms; the other coordinates are inert to
    forward_E but keep the artefact self-describing)."""
    for p in params.planes:
        for l in range(fs.LEVELS):
            c = cells[p][l]
            params.raw[p][l] = fs.Params.encode(
                c["g"], c["p"], c.get("c0") or c.get("kappa") or 0.01,
                c.get("beta", 0.65), c.get("sharp", 4.0))


def head_fit(params, S, y):
    """Identical head refit to fit_variant: Adam on level/channel logits,
    gradients on FULL S, map refit inside every objective call."""
    def hobj(x):
        params.set_head(x)
        return fs.head_obj_grads(params, S, y)
    _, bx = fs.adam(hobj, params.head_vector(), fs.ADAM_STEPS_HEAD)
    params.set_head(bx)
    E, _ = fs.forward_E(params, S)
    mp = fs.fit_map(E, y)
    params.map_abl = (mp["A"], mp["B"], mp["lambda"])
    return mp["mse"], mp


# ---------- gate arm ----------

cells_global = None


def gate_cell_scan(cache, l, g, P, Ssub, params, y, p):
    """Scan kappa on the shared GRID_C0 axis (+ golden refine on log kappa)
    and the 'off' alternative for cell (p,l) on the block subsample.
    Returns (mse, mode, kappa)."""
    recs, bounds, counts = cache.level_arrays(l, sub=True)
    cs, cd, _, _ = fs.contrast_grad(recs, g)
    e = recs[:, 0] ** P
    nz = cache.nz_rows[l]
    den = np.maximum(counts, 1).astype(np.float64)
    cY, cC = params.channel_weights()
    if len(params.planes) == 1:
        cY, cC = 1.0, 0.0
    cp = cY if p == params.planes[0] else cC
    wl = params.level_weights(p)[l]
    E_const = np.zeros(cache.n_rows)
    for pp in params.planes:
        w = params.level_weights(pp)
        cpp = cY if pp == params.planes[0] else cC
        for ll in range(fs.LEVELS):
            if pp == p and ll == l:
                continue
            P2 = cells_global[pp][ll]["p"]
            E_const += cpp * w[ll] * np.power(
                np.maximum(Ssub[pp][:, ll], 1e-300), 1.0 / P2)

    def eval_v(v):
        sr = np.zeros(cache.n_rows)
        sr[nz] = np.add.reduceat(v * e, bounds) / den
        El = np.where(sr > 0, np.power(np.maximum(sr, 1e-300), 1.0 / P), 0.0)
        return fs.fit_map(E_const + cp * wl * El, y)["mse"]

    mse_off = eval_v(np.ones_like(cs))
    grid = {}
    for k in fs.GRID_C0:
        v = np.maximum(cs <= k, cd <= k).astype(np.float64)
        grid[float(k)] = eval_v(v)
    bk = min(grid, key=grid.get)
    best = (grid[bk], "gate", bk)
    ks = sorted(grid)
    i = ks.index(bk)
    lo = math.log(ks[i - 1]) if i > 0 else math.log(bk) - 0.5
    hi = math.log(ks[i + 1]) if i < len(ks) - 1 else math.log(bk) + 0.5
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    x1, x2 = hi - gr * (hi - lo), lo + gr * (hi - lo)

    def f(lk):
        k = math.exp(lk)
        v = np.maximum(cs <= k, cd <= k).astype(np.float64)
        return eval_v(v)

    f1, f2 = f(x1), f(x2)
    for _ in range(20):
        if f1 > f2:
            lo = x1; x1, f1 = x2, f2
            x2 = lo + gr * (hi - lo); f2 = f(x2)
        else:
            hi = x2; x2, f2 = x1, f1
            x1 = hi - gr * (hi - lo); f1 = f(x1)
    kref = math.exp(0.5 * (lo + hi))
    mref = f(0.5 * (lo + hi))
    if mref < best[0]:
        best = (mref, "gate", kref)
    if mse_off <= best[0]:
        best = (mse_off, "off", None)
    return best


def fit_gate(planes, caches, y, init_cells):
    global cells_global
    params = fs.Params(planes)
    cells = {p: [dict(c) for c in init_cells[p]] for p in planes}
    cells_global = cells
    encode_cells(params, cells)
    S = build_S_form(cells, planes, caches)
    var_y = float(np.var(y))
    E0, _ = fs.forward_E(params, S)
    mp0 = fs.fit_map(E0, y)
    sro0, kro0 = fs._rank_stats(E0, y)
    sane = bool(mp0["mse"] < var_y)
    log(f"gate: init refit-MSE {mp0['mse']:.4f} var(y) {var_y:.4f} "
        f"SROCC {sro0:.4f} sane={sane}")
    params.map_abl = (mp0["A"], mp0["B"], mp0["lambda"])
    best_loss = mp0["mse"]
    hist = [{"sweep": 0, "mse": best_loss}]
    for sweep in range(1, fs.MAX_SWEEPS + 1):
        Ssub = build_S_form(cells, planes, caches, sub=True)
        for p in planes:
            for l in range(fs.LEVELS):
                cell = cells[p][l]
                bm, mode, kappa = gate_cell_scan(
                    caches[p], l, cell["g"], cell["p"], Ssub, params, y, p)
                cand = dict(cell); cand["mode"] = mode; cand["kappa"] = kappa
                s_full = s_level_form(caches[p], l, cand)
                Sf = dict(S); Sf[p] = S[p].copy(); Sf[p][:, l] = s_full
                cur = fs.mse_of(params, S, y)
                fl = fs.mse_of(params, Sf, y)
                if fl < cur:
                    cells[p][l] = cand
                    S[p][:, l] = s_full
                    Ssub[p][:, l] = s_level_form(caches[p], l, cand,
                                                sub=True)
                    log(f"  gate s{sweep} {p} l{l}: {mode}"
                        f"{'' if kappa is None else f'@{kappa:.4g}'} "
                        f"sub {bm:.3f} -> full {fl:.3f} (was {cur:.3f})")
        mse, mp = head_fit(params, S, y)
        hist.append({"sweep": sweep, "mse": mse})
        log(f"gate sweep {sweep}: MSE {mse:.4f}")
        if best_loss - mse < fs.REL_STOP * max(best_loss, 1e-12):
            best_loss = mse
            break
        best_loss = mse
    encode_cells(params, cells)
    E, _ = fs.forward_E(params, S)
    mp = fs.fit_map(E, y)
    params.map_abl = (mp["A"], mp["B"], mp["lambda"])
    sro, kro = fs._rank_stats(E, y)
    return {"cells": cells, "params": params,
            "fit": {"mse": mp["mse"], "var_y": var_y, "srocc": sro,
                    "krocc": kro, "map": mp,
                    "sanity": {"init_mse": mp0["mse"], "init_srocc": sro0,
                               "init_sane": sane},
                    "history": hist}}


def prior_cells(planes, caches):
    cells = {}
    for p in planes:
        cache = caches[p]
        cells[p] = []
        for l in range(fs.LEVELS):
            mins = []
            for recs, nz, bounds, counts in cache.iter_level(l, False):
                if recs.shape[0]:
                    cs, cd, _, _ = fs.contrast_grad(recs, 1.0)
                    mins.append(np.minimum(cs, cd))
            p10 = float(np.quantile(np.concatenate(mins), 0.10)) \
                if mins else 0.01
            # p10 can be exactly 0.0 (>=10% zero-contrast blocks): clamp to
            # the registered C0_LO bound and record the raw value.
            c0 = min(max(p10, fs.C0_LO), fs.C0_HI)
            cells[p].append({"mode": "curve", "g": 1.0, "p": 1.0,
                             "c0": c0, "beta": 0.65, "sharp": 4.0,
                             "p10_raw": p10})
    return cells


def fit_prior(planes, caches, y, cells):
    params = fs.Params(planes)
    encode_cells(params, cells)
    S = build_S_form(cells, planes, caches)
    var_y = float(np.var(y))
    E0, _ = fs.forward_E(params, S)
    mp0 = fs.fit_map(E0, y)
    sro0, kro0 = fs._rank_stats(E0, y)
    log(f"prior: init refit-MSE {mp0['mse']:.4f} var(y) {var_y:.4f} "
        f"SROCC {sro0:.4f}")
    mse, mp = head_fit(params, S, y)
    E, _ = fs.forward_E(params, S)
    mp = fs.fit_map(E, y)
    params.map_abl = (mp["A"], mp["B"], mp["lambda"])
    sro, kro = fs._rank_stats(E, y)
    return {"cells": cells, "params": params,
            "fit": {"mse": mp["mse"], "var_y": var_y, "srocc": sro,
                    "krocc": kro, "map": mp,
                    "sanity": {"init_mse": mp0["mse"], "init_srocc": sro0,
                               "init_sane": bool(mp0["mse"] < var_y)},
                    "history": [{"sweep": 1, "mse": mse}]}}


def gate_init_cells(planes, caches):
    pc = prior_cells(planes, caches)
    for p in planes:
        for l in range(fs.LEVELS):
            pc[p][l].update({"mode": "gate", "kappa": pc[p][l]["c0"]})
    return pc


# ---------- arm (de)serialisation + scoring ----------

def arm_to_json(arm, planes):
    params = arm["params"]
    return {
        "planes": list(planes),
        "cells": {p: arm["cells"][p] for p in planes},
        "level_weights": {p: params.level_weights(p).tolist()
                          for p in planes},
        "channel_weights": list(params.channel_weights()),
        "map": {"A": params.map_abl[0], "B": params.map_abl[1],
                "lambda": params.map_abl[2]},
        "fit": {k: v for k, v in arm["fit"].items() if k != "history"},
        "wall_s": arm.get("wall_s"),
    }


def arm_from_json(j):
    params = fs.Params(j["planes"])
    for p in j["planes"]:
        for l in range(fs.LEVELS):
            params.raw[p][l] = fs.Params.encode(
                1.0, 1.0,
                j["cells"][p][l].get("c0") or
                j["cells"][p][l].get("kappa") or 0.01,
                j["cells"][p][l].get("beta", 0.65),
                j["cells"][p][l].get("sharp", 4.0))
        w = np.maximum(np.asarray(j["level_weights"][p]), 1e-12)
        params.wl[p] = np.log(w)
    cY, cC = j["channel_weights"]
    params.wc = np.log(np.maximum(np.asarray([cY, cC]), 1e-12))
    mp = j["map"]
    params.map_abl = (mp["A"], mp["B"], mp["lambda"])
    return {"cells": j["cells"], "params": params}


def curve_arm(fit_json):
    params = fs.params_from_fit(fit_json)
    cells = {p: [dict(mode="curve", **lv)
                 for lv in fit_json["levels"][p]] for p in fit_json["planes"]}
    return {"cells": cells, "params": params,
            "fit": {"mse": fit_json["fit_mse"], "var_y": fit_json["var_y"],
                    "srocc": fit_json["srocc_negE_fit"],
                    "krocc": fit_json["krocc_negE_fit"],
                    "map": fit_json["map"],
                    "sanity": fit_json.get("sanity"),
                    "history": fit_json.get("history")},
            "wall_s": fit_json.get("wall_s")}


def score_arm(arm, planes, caches):
    S = build_S_form(arm["cells"], planes, caches)
    E, _ = fs.forward_E(arm["params"], S)
    a, b, lam = arm["params"].map_abl
    return E, a * np.exp(-lam * E) + b


def metrics_of(E, y, map_abl):
    sro = float(scipy.stats.spearmanr(-E, y).statistic)
    kro = float(scipy.stats.kendalltau(-E, y).statistic)
    a, b, lam = map_abl
    yhat = a * np.exp(-lam * E) + b
    plcc = float(np.corrcoef(yhat, y)[0, 1])
    mse = float(np.mean((yhat - y) ** 2))
    return {"srocc": sro, "krocc": kro, "plcc": plcc, "mse": mse}


def load_task_caches(task, planes):
    f32 = task.get("fmt", "f16") == "f32"
    caches = {}
    for p in planes:
        if "rows" in task:
            idx = (np.load(task["rows"]) if task["rows"].endswith(".npy")
                   else np.asarray(json.loads(Path(task["rows"]).read_text())))
            caches[p] = (RowViewF32(task[p], idx) if f32
                         else RowView(task[p], idx))
        else:
            caches[p] = (fs.PlaneCache(task[p]) if f32
                         else PlaneCacheF16(task[p]))
    return caches


def main():
    args = sys.argv[1:]
    pairs_tsv, outdir, surfaces_dir = args[0], args[1], args[2]
    planes = {}
    seeds = [9201, 9207, 9211, 9215, 9219]
    n_rows = 1024
    tasks_path = None
    init_spec_path = DEFAULT_SPEC
    i = 3
    while i < len(args):
        a = args[i]
        if a == "--seeds":
            seeds = [int(x) for x in args[i + 1].split(",")]; i += 2
        elif a == "--rows":
            n_rows = int(args[i + 1]); i += 2
        elif a == "--score-tasks":
            tasks_path = args[i + 1]; i += 2
        elif a == "--init-spec":
            init_spec_path = args[i + 1]; i += 2
        else:
            p, _, path = a.partition("=")
            planes[p] = path; i += 1

    rows_meta = [r for r in csv.DictReader(open(pairs_tsv), delimiter="\t")]
    y_all = np.asarray([float(r["human_score"]) for r in rows_meta])
    Path(outdir).mkdir(parents=True, exist_ok=True)
    Path(surfaces_dir).mkdir(parents=True, exist_ok=True)
    tasks = json.loads(Path(tasks_path).read_text()) if tasks_path else []
    init_levels = json.loads(Path(init_spec_path).read_text())["levels"]
    pl = tuple(planes.keys())
    rpath = Path(outdir, "x2_results.json")
    if rpath.exists():                 # merge across per-seed invocations
        results = json.loads(rpath.read_text())
        results.setdefault("arms", {})
    else:
        results = {"seeds": seeds, "n_rows": n_rows,
                   "fit_pairs": pairs_tsv, "caches": planes,
                   "init_spec": init_spec_path, "arms": {}}
    results["seeds"] = sorted(set(results.get("seeds", [])) | set(seeds))
    for seed in seeds:
        rng = np.random.default_rng(seed)
        sel = sorted(rng.choice(len(rows_meta), size=n_rows,
                                replace=False).tolist())
        y = y_all[sel]
        caches = {p: RowView(b, sel) for p, b in planes.items()}
        prov = {"seed": seed, "rows": n_rows,
                "row_sample": "seeded uniform subset (cell-blocked file)",
                "pairs_tsv": pairs_tsv, "caches": planes,
                "sel": sel}
        arms = {}
        # --- curve arm: the identical fs.fit_variant pipeline ---
        out_curve = Path(outdir) / f"curve_s{seed}.json"
        art_c = fs.fit_variant(f"x2_curve_s{seed}", pl, caches, y,
                               init_levels, str(out_curve), surfaces_dir,
                               prov)
        arms["curve"] = curve_arm(art_c)
        arms["curve"]["artefact_path"] = str(out_curve)
        # --- gate arm ---
        t0 = time.time()
        gart = fit_gate(pl, caches, y, gate_init_cells(pl, caches))
        gart["wall_s"] = time.time() - t0
        arms["gate"] = gart
        Path(outdir, f"gate_s{seed}.json").write_text(json.dumps(
            {**arm_to_json(gart, pl), "provenance": prov}, indent=1) + "\n")
        # --- prior arm ---
        t0 = time.time()
        part = fit_prior(pl, caches, y, prior_cells(pl, caches))
        part["wall_s"] = time.time() - t0
        arms["prior"] = part
        Path(outdir, f"prior_s{seed}.json").write_text(json.dumps(
            {**arm_to_json(part, pl), "provenance": prov}, indent=1) + "\n")
        # --- dev scoring ---
        for task in tasks:
            name = task["name"]
            t_caches = load_task_caches(task, pl)
            yj = json.loads(Path(task["y"]).read_text())
            ty = np.asarray(yj["y"], dtype=np.float64)
            out_scores = Path(outdir) / "scores" / name
            out_scores.mkdir(parents=True, exist_ok=True)
            for aname, art in arms.items():
                E, yhat = score_arm(art, pl, t_caches)
                m = metrics_of(E, ty, art["params"].map_abl)
                np.savez(out_scores / f"{aname}_s{seed}.npz",
                         E=E, yhat=yhat, y=ty)
                rec = results["arms"].setdefault(aname, {}).setdefault(
                    str(seed), {})
                rec[name] = m
        for aname, art in arms.items():
            rec = results["arms"].setdefault(aname, {}).setdefault(
                str(seed), {})
            rec["fit"] = {k: v for k, v in art["fit"].items()
                          if k != "history"}
            rec["wall_s"] = art.get("wall_s")
            rec["sel"] = sel
            if aname == "gate":
                rec["cells"] = [{"mode": c["mode"], "kappa": c.get("kappa")}
                                for p in pl for c in art["cells"][p]]
            if aname == "prior":
                rec["cells"] = [{"c0": c["c0"], "p10_raw": c["p10_raw"]}
                                for p in pl for c in art["cells"][p]]
        rpath.write_text(json.dumps(results, indent=1) + "\n")
        log(f"seed {seed} DONE")
    log("ALL SEEDS DONE")


if __name__ == "__main__":
    main()
