#!/usr/bin/env python3
"""Screen-4: joint Adam fit of DVIFM shape constants on the block cache.

Per level l fits raw theta_l = (gamma, p, a, b, s) under the design's
parameterisation:
    g     = 0.2 + 1.8*sigmoid(gamma)     init g=1
    P     = exp(p)                        init P=1
    C0    = exp(a)                        init = TRAIN p10 of min-C~
    beta  = 1.5*sigmoid(b)                init 0.65
    sharp = 1 + softplus(s)               init 4
c_hi = inf, edge discount on, F2 centres fixed at the screen spec's
values during the fit (they are re-derived at the fitted g afterwards,
per the design's "at the baked g" rule). Beta carries a weak prior
toward 0.65 (LAMBDA_BETA). Head: linear on the 30 pooled features ->
human_score, MSE loss, trained jointly by full-batch Adam.

TRAIN fit rows only (row_index < 8000). Usage:
    fit_params.py <cache.bin> <spec.json> <train.parquet> <out.json>
"""
import sys, json, math
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dvifm_cache import F32, REC, LEVELS, BINS, load_index, load_spec

LAMBDA_BETA = 1e-3
EPOCHS = 150
LR = 3e-2
CHUNK_ROWS = 256


def sig(x):
    return 1.0 / (1.0 + np.exp(-x))


def phi(x, g):
    return np.sign(x) * np.abs(x) ** g


def dphi_dg(x, g):
    out = np.zeros_like(x)
    nz = x != 0.0
    out[nz] = phi(x[nz], g) * np.log(np.abs(x[nz]))
    return out


def vis_grads(c, c0, beta, sharp):
    """v(c)=exp(-softplus(beta*sharp*(ln c - ln c0))/sharp) for c>0,
    pinned v=1 with zero grads at c<=0. Returns v and dv/d{c0,beta,
    sharp,c}."""
    v = np.ones_like(c)
    dc0 = np.zeros_like(c)
    db = np.zeros_like(c)
    ds = np.zeros_like(c)
    dc = np.zeros_like(c)
    pos = c > 0.0
    if not np.any(pos):
        return v, dc0, db, ds, dc
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


def block_terms(recs, g, P, c0, beta, sharp, centres):
    """Per-block forward + physical-param grads for one level.

    recs: (n,18) f64. Returns dict with:
      f1b (n,), f2b (n,5)       — per-block contributions
      g_*: per-block grads wrt physical (g, P, c0, beta, sharp):
        dg_f1 (n,), dg_f2 (n,5), dP_f1, dP_f2 (n,5),
        dc0_f1, db_f1, ds_f1 (n,)
    """
    n = recs.shape[0]
    cmax_s, cmin_s = recs[:, 2:6], recs[:, 6:10]
    cmax_d, cmin_d = recs[:, 10:14], recs[:, 14:18]
    m = recs[:, 0]

    cq_s = phi(cmax_s, g) - phi(cmin_s, g)
    cq_d = phi(cmax_d, g) - phi(cmin_d, g)
    rows = np.arange(n)
    js = np.argmin(cq_s, axis=1)
    jd = np.argmin(cq_d, axis=1)
    cs = cq_s[rows, js]
    cd = cq_d[rows, jd]
    dcs = dphi_dg(cmax_s[rows, js], g) - dphi_dg(cmin_s[rows, js], g)
    dcd = dphi_dg(cmax_d[rows, jd], g) - dphi_dg(cmin_d[rows, jd], g)

    vs, vs_c0, vs_b, vs_s, vs_c = vis_grads(cs, c0, beta, sharp)
    vd, vd_c0, vd_b, vd_s, vd_c = vis_grads(cd, c0, beta, sharp)
    pick_d = vd > vs
    v = np.where(pick_d, vd, vs)
    dv_c0 = np.where(pick_d, vd_c0, vs_c0)
    dv_b = np.where(pick_d, vd_b, vs_b)
    dv_s = np.where(pick_d, vd_s, vs_s)
    dv_c = np.where(pick_d, vd_c, vs_c)
    dv_dg = dv_c * np.where(pick_d, dcd, dcs)

    e = np.power(m, P)
    de_dP = np.where(m > 0.0, e * np.log(np.maximum(m, 1e-300)), 0.0)

    cmin = np.minimum(cs, cd)
    ell = np.log(cmin + 1e-6)
    dell_dg = np.where(cd < cs, dcd, dcs) / (cmin + 1e-6)

    ctr = np.asarray(centres, dtype=np.float64)
    ell_c = np.clip(ell, ctr[0], ctr[-1])
    i = np.searchsorted(ctr, ell_c, side="left")
    j = np.clip(i - 1, 0, BINS - 2)
    t = (ell_c - ctr[j]) / (ctr[j + 1] - ctr[j])
    h = np.zeros((n, BINS))
    h[rows, j] = 1.0 - t
    h[rows, j + 1] = t
    live = ell == ell_c
    dj = ctr[j + 1] - ctr[j]
    dh_dl = np.zeros((n, BINS))
    w = live
    dh_dl[rows[w], j[w]] = -1.0 / dj[w]
    dh_dl[rows[w], j[w] + 1] = 1.0 / dj[w]

    f1b = v * e
    f2b = h * e[:, None]
    return {
        "f1b": f1b, "f2b": f2b,
        "dg_f1": dv_dg * e,
        "dg_f2": dh_dl * (dell_dg * e)[:, None],
        "dP_f1": v * de_dP,
        "dP_f2": h * de_dP[:, None],
        "dc0_f1": dv_c0 * e,
        "db_f1": dv_b * e,
        "ds_f1": dv_s * e,
    }


def main():
    bin_path, spec_path, parquet_path, out_path = sys.argv[1:5]
    spec = load_spec(spec_path)
    index = load_index(bin_path)
    train = [e for e in index if e["row_index"] < 8000]
    n_rows = len(train)
    FEAT = LEVELS * (1 + BINS)

    import pyarrow.parquet as pq
    y = pq.read_table(parquet_path, columns=["human_score"])["human_score"] \
          .to_numpy().astype(np.float64)[:n_rows]
    assert len(y) == n_rows, (len(y), n_rows)

    cache = np.memmap(bin_path, dtype=F32, mode="r")
    row_off = np.zeros((n_rows, LEVELS + 1), dtype=np.int64)  # f32 elements
    nb_row = np.zeros((n_rows, LEVELS), dtype=np.int64)       # block counts
    for i, e in enumerate(train):
        base = e["offset"] // 4  # byte offset -> f32 element index
        cnt = np.asarray(e["level_records"], dtype=np.int64)
        row_off[i, 0] = base
        row_off[i, 1:] = base + np.cumsum(cnt) * REC
        nb_row[i] = cnt

    th = np.zeros((LEVELS, 5))
    centres = []
    for l, lv in enumerate(spec["levels"]):
        th[l] = (math.log((lv["g"] - 0.2) / (1.8 - (lv["g"] - 0.2))),
                 math.log(lv["p"]), math.log(lv["c0"]),
                 math.log(lv["beta"] / (1.5 - lv["beta"])),
                 math.log(math.expm1(lv["sharp"] - 1.0)))
        centres.append(lv["f2_centers"])

    def phys(th):
        return (0.2 + 1.8 * sig(th[:, 0]), np.exp(th[:, 1]),
                np.exp(th[:, 2]), 1.5 * sig(th[:, 3]),
                1.0 + np.logaddexp(0.0, th[:, 4]))

    def epoch(th, w, b, grads):
        """Full pass over TRAIN rows. Returns (loss, X, grad...)."""
        g, P, c0, beta, sharp = phys(th)
        X = np.zeros((n_rows, FEAT))
        dX = np.zeros((n_rows, FEAT, LEVELS, 5)) if grads else None
        for l in range(LEVELS):
            f1r = np.zeros(n_rows)
            f2r = np.zeros((n_rows, BINS))
            acc = {k: np.zeros((n_rows,) + ((BINS,) if k.endswith("f2") else ()))
                   for k in ("dg_f1", "dg_f2", "dP_f1", "dP_f2",
                             "dc0_f1", "db_f1", "ds_f1")} if grads else None
            for i0 in range(0, n_rows, CHUNK_ROWS):
                i1 = min(i0 + CHUNK_ROWS, n_rows)
                parts = [cache[row_off[i, l]:row_off[i, l + 1]]
                         for i in range(i0, i1)]
                recs = np.concatenate(parts).astype(np.float64) \
                    .reshape(-1, REC)
                nb = nb_row[i0:i1, l]
                nonempty = (nb > 0).astype(np.float64)
                if recs.shape[0] == 0:
                    continue
                t = block_terms(recs, g[l], P[l], c0[l], beta[l], sharp[l],
                                centres[l])
                starts = np.cumsum(nb) - nb
                # reduceat indexes must stay in-bounds; empty segments are
                # masked by `nonempty` below so their clipped index is dead.
                starts = np.minimum(starts, recs.shape[0] - 1)
                den = np.maximum(nb, 1)
                f1r[i0:i1] = np.add.reduceat(t["f1b"], starts) / den * nonempty
                f2r[i0:i1] = (np.add.reduceat(t["f2b"], starts, axis=0)
                              / den[:, None]) * nonempty[:, None]
                if grads:
                    for k in acc:
                        red = np.add.reduceat(t[k], starts, axis=0) \
                            / den.reshape((-1,) + (1,) * (t[k].ndim - 1))
                        acc[k][i0:i1] = red * nonempty.reshape(
                            (-1,) + (1,) * (t[k].ndim - 1))
            X[:, l * 6:l * 6 + 6] = np.column_stack([f1r, f2r])
            if grads:
                dX[:, l * 6:l * 6 + 6, l, 0] = np.column_stack(
                    [acc["dg_f1"], acc["dg_f2"]])
                dX[:, l * 6:l * 6 + 6, l, 1] = np.column_stack(
                    [acc["dP_f1"], acc["dP_f2"]])
                dX[:, l * 6, l, 2] = acc["dc0_f1"]
                dX[:, l * 6, l, 3] = acc["db_f1"]
                dX[:, l * 6, l, 4] = acc["ds_f1"]
        yhat = X @ w + b
        resid = yhat - y
        beta_v = 1.5 * sig(th[:, 3])
        loss = float((resid ** 2).mean()
                     + LAMBDA_BETA * ((beta_v - 0.65) ** 2).sum())
        if not grads:
            return loss, X
        dres = 2.0 * resid / n_rows
        gw = X.T @ dres
        gb = float(dres.sum())
        # dL/dtheta_phys[l,p] = sum_rows dres_r * sum_f w_f * dX[r,f,l,p]
        gth_phys = np.einsum("r,f,rflp->lp", dres, w, dX)
        # chain raw -> physical
        dg = 1.8 * sig(th[:, 0]) * (1 - sig(th[:, 0]))
        dP = np.exp(th[:, 1])
        dc0 = np.exp(th[:, 2])
        dbeta = 1.5 * sig(th[:, 3]) * (1 - sig(th[:, 3]))
        dsharp = sig(th[:, 4])
        gth = gth_phys * np.stack([dg, dP, dc0, dbeta, dsharp], axis=1)
        gth[:, 3] += 2 * LAMBDA_BETA * (beta_v - 0.65) * dbeta
        return loss, X, gth, gw, gb

    # init head by lstsq at the spec constants
    loss0, X = epoch(th, np.zeros(FEAT), 0.0, grads=False)
    # parity gate: replayed features must match the extractor's f956..f985
    # columns on the same rows (f32-narrowing tolerance ~1e-6 relative).
    stored = pq.read_table(parquet_path,
                           columns=[f"f{956 + i}" for i in range(FEAT)])
    S = np.column_stack([stored[c].to_numpy() for c in stored.column_names])
    delta = np.abs(X[: S.shape[0]] - S)
    rel = delta / np.maximum(np.abs(S), 1e-9)
    print(f"replay-vs-extracted: max|d|={delta.max():.3e} "
          f"max-rel={rel.max():.3e}", flush=True)
    if delta.max() > 1e-4:
        raise SystemExit("replay diverges from extracted features — abort")
    w = np.linalg.lstsq(np.column_stack([X, np.ones(n_rows)]), y,
                        rcond=None)[0]
    w, b = w[:-1], w[-1]
    print(f"init loss {loss0:.4f}; lstsq head mse {((X@w+b-y)**2).mean():.4f}",
          flush=True)

    mt = np.zeros_like(th); vt = np.zeros_like(th)
    mw = np.zeros_like(w); vw = np.zeros_like(w)
    mb = vb_ = 0.0
    hist = []
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for ep in range(EPOCHS):
        loss, X, gth, gw, gb = epoch(th, w, b, grads=True)
        hist.append(loss)
        tt = ep + 1
        mt = beta1 * mt + (1 - beta1) * gth
        vt = beta2 * vt + (1 - beta2) * gth ** 2
        th -= LR * (mt / (1 - beta1 ** tt)) / (np.sqrt(vt / (1 - beta2 ** tt)) + eps)
        mw = beta1 * mw + (1 - beta1) * gw
        vw = beta2 * vw + (1 - beta2) * gw ** 2
        w -= LR * (mw / (1 - beta1 ** tt)) / (np.sqrt(vw / (1 - beta2 ** tt)) + eps)
        mb = beta1 * mb + (1 - beta1) * gb
        vb_ = beta2 * vb_ + (1 - beta2) * gb ** 2
        b -= LR * (mb / (1 - beta1 ** tt)) / (np.sqrt(vb_ / (1 - beta2 ** tt)) + eps)
        if ep % 10 == 0 or ep == EPOCHS - 1:
            g5, P5, c05, b5, s5 = phys(th)
            print(f"ep {ep:3d} loss {loss:.4f} g={np.round(g5,3)} "
                  f"P={np.round(P5,3)} beta={np.round(b5,3)}", flush=True)

    g5, P5, c05, b5, s5 = phys(th)
    out = {"schema": "dvifm-fitted-v1",
           "provenance": {"cache": str(bin_path), "spec_in": str(spec_path),
                          "rows": "row_index<8000", "epochs": EPOCHS,
                          "lr": LR, "lambda_beta": LAMBDA_BETA,
                          "head": "linear 30->1 mse", "loss_hist": hist},
           "levels": [{"g": float(g5[l]), "p": float(P5[l]),
                       "c0": float(c05[l]), "beta": float(b5[l]),
                       "sharp": float(s5[l]), "c_hi": None,
                       "f2_centers": centres[l],
                       "band": spec["levels"][l]["band"], "edge": True}
                      for l in range(LEVELS)]}
    Path(out_path).write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", out_path)
    print("final:", json.dumps({l: {k: round(v, 4) if isinstance(v, float) else v
                                    for k, v in lv.items()
                                    if k in ("g", "p", "c0", "beta", "sharp")}
                                for l, lv in enumerate(out["levels"])}))


if __name__ == "__main__":
    main()
