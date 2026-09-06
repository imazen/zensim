#!/usr/bin/env python3
"""T1-T5 bodies for the pre-registered corruption-head theory tests.

Driven by `corrhead_theories.py`; see `docs/PLAN_CORRHEAD_THEORIES_2026-09-06.md`.

Every policy is reduced to a single SCORE VECTOR `s` with the uniform firing rule
`s > T`, so a hard mask, a soft interpolation, a conditional head and a per-band
head are all compared on the same ROC machinery at matched operating points —
never at a shared threshold, which is not a comparable quantity across forms.
"""
import json, os, time
import numpy as np

import corrhead_theories as C
from corrhead_theories import (D, OUT, FP_TARGETS, QBANDS, fit_probs, log,
                               pauc, standardize, threshold_for_fp, write_tsv)


# ---------------------------------------------------------------- shared ---
def op_rows(d, s, label, note=""):
    """Detection + FP at each matched FP_honest target, and at T=0.9/0.95."""
    mk = d.masks_te()
    bh, sh, ma = mk["broad_honest"], mk["severe_honest"], mk["matched_anchor"]
    rows = []
    pa = pauc(s, d.corr_te, bh)
    pts = [("fp%g" % (t * 100), threshold_for_fp(s, bh, t)) for t in FP_TARGETS]
    pts += [("T=0.9", 0.9), ("T=0.95", 0.95)]
    for name, T in pts:
        if T is None:
            rows.append(dict(arm=label, op=name, T=None, detection=None, note="NOT REACHABLE"))
            continue
        f = s > T
        r = dict(arm=label, op=name, T=float(T),
                 detection=float(f[d.corr_te].mean()),
                 fp_honest=float(f[bh].mean()),
                 fp_severe=float(f[sh].mean()),
                 fp_anchor=float(f[ma].mean()), pauc5=pa, note=note)
        for nm, lo, hi in QBANDS:
            m = bh & (d.q >= lo) & (d.q < hi)
            r[f"fp_{nm}"] = float(f[m].mean()) if m.sum() else float("nan")
        rows.append(r)
    return rows


def boot_ci(d, s, T, mask, n=1000, seed=0, cluster=True):
    """Clustered (by source) bootstrap 95% interval of the rate of s>T on mask."""
    rng = np.random.default_rng(seed)
    hit = (s[mask] > T).astype(float)
    if not cluster:
        out = [hit[rng.integers(0, len(hit), len(hit))].mean() for _ in range(n)]
        return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))
    uniq, inv = np.unique(d.src[mask], return_inverse=True)
    sums = np.bincount(inv, weights=hit, minlength=len(uniq))
    cnts = np.bincount(inv, minlength=len(uniq))
    out = np.empty(n)
    for i in range(n):
        k = rng.integers(0, len(uniq), len(uniq))
        out[i] = sums[k].sum() / max(cnts[k].sum(), 1)
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def _base(d):
    """The incumbent d228 head's probabilities over every row (cached)."""
    if not hasattr(d, "_p"):
        d._p, _ = fit_probs(d.Xs, d.y, d.tr, d.va, "logistic")
    return d._p


# -------------------------------------------------------------------- T1 ---
def t1(d):
    log("T1: dial gating vs interpolation vs conditional head")
    p = _base(d)
    arms, rows = {}, []

    arms["a_nogate"] = p
    for G in (80.0, 90.0):
        arms[f"b_hardmask_dial<{G:g}"] = np.where(d.dial < G, p, 0.0)

    # (c) soft interpolation: P' = P * sigmoid((G - dial)/s), (G,s) by train MLE
    tr = d.tr
    best, bG, bS = -np.inf, None, None
    for G in np.linspace(40, 160, 121):
        for S in (0.25, 0.5, 1, 2, 3, 5, 8, 12, 20, 35, 60, 120, 300):
            w = 1.0 / (1.0 + np.exp(-(G - d.dial[tr]) / S))
            pp = np.clip(p[tr] * w, 1e-9, 1 - 1e-9)
            ll = float(np.sum(d.y[tr] * np.log(pp) + (1 - d.y[tr]) * np.log(1 - pp)))
            if ll > best:
                best, bG, bS = ll, G, S
    log(f"T1(c) train-MLE logistic gate: G={bG:.4g} s={bS:g}  ll={best:.1f}")
    arms[f"c_soft_G{bG:.3g}_s{bS:g}"] = p / (1.0 + np.exp(-(bG - d.dial) / bS))

    # (d) dial as an INPUT FEATURE -> quality-conditional boundary
    Xd = np.column_stack([d.Xs, d.dial])
    pd_, _ = fit_probs(Xd, d.y, d.tr, d.va, "logistic")
    arms["d_dial_as_feature"] = pd_
    # the honest q-anchor-relative variant is DROPPED: corruption and negrich
    # rows carry no q, so the value is undefined for 177,276 of 186,869 rows.

    # (e) per-band heads, band edges from TRAIN-fold dial terciles (fixed before fit)
    e1, e2 = np.percentile(d.dial[tr], [33.3333, 66.6667])
    log(f"T1(e) train dial terciles: {e1:.3f} / {e2:.3f}")
    band = np.digitize(d.dial, [e1, e2])
    pe = np.zeros(len(p))
    for b in (0, 1, 2):
        m = band == b
        if (m & tr & (d.y == 1)).sum() < 50 or (m & tr & (d.y == 0)).sum() < 50:
            log(f"  band {b}: too few train rows, falls back to the global head")
            pe[m] = p[m]
            continue
        pb, _ = fit_probs(d.Xs[m], d.y[m], d.tr[m], d.va[m], "logistic")
        pe[m] = pb
        log(f"  band {b}: n_tr={(m&tr).sum()} pos={(m&tr&(d.y==1)).sum()}")
    arms["e_per_band_heads"] = pe

    for k, s in arms.items():
        rows += op_rows(d, s, k)
    write_tsv(f"{OUT}/t1_policies.tsv", rows)

    # decision rule: detection CIs at the matched 0.5% operating point
    mk = d.masks_te()
    ci = []
    for k, s in arms.items():
        T = threshold_for_fp(s, mk["broad_honest"], 0.005)
        if T is None:
            ci.append(dict(arm=k, op="fp0.5", detection=None, note="NOT REACHABLE"))
            continue
        lo, hi = boot_ci(d, s, T, d.corr_te)
        ci.append(dict(arm=k, op="fp0.5", T=float(T),
                       detection=float((s > T)[d.corr_te].mean()),
                       ci_lo=lo, ci_hi=hi))
    write_tsv(f"{OUT}/t1_detection_ci.tsv", ci)
    np.savez_compressed(f"{OUT}/t1_scores.npz", **arms)
    json.dump({"soft_gate": {"G": bG, "s": bS}, "band_edges": [float(e1), float(e2)]},
              open(f"{OUT}/t1_params.json", "w"), indent=1)


# -------------------------------------------------------------------- T2 ---
def t2(d):
    log("T2: MLP / HGB vs logistic")
    rows, gen = [], []
    arms = {}
    for m in ("logistic", "mlp32", "mlp64_32", "hgb"):
        t = time.time()
        try:
            p, clf = fit_probs(d.Xs, d.y, d.tr, d.va, m)
        except Exception as e:
            log(f"  {m}: FAILED {e}")
            rows.append(dict(arm=m, op="-", note=f"NOT RUN: {e}"))
            continue
        log(f"  {m}: {time.time()-t:.0f}s")
        arms[m] = p
        rows += op_rows(d, p, m, note=f"{time.time()-t:.0f}s")
        # in-sample vs held-out at the arm's own matched-0.5% threshold
        T = threshold_for_fp(p, d.masks_te()["broad_honest"], 0.005)
        if T is not None:
            gen.append(dict(arm=m, T=float(T),
                            det_train=float((p > T)[d.tr & (d.y == 1)].mean()),
                            det_test=float((p > T)[d.corr_te].mean()),
                            fp_honest_train=float((p > T)[d.tr & (d.sub == "broad_honest")].mean()),
                            fp_honest_test=float((p > T)[d.te & (d.sub == "broad_honest")].mean())))

    # class-balance control: MLP has no class_weight, logistic/hgb do. Train
    # positives outnumber negatives, so run one MLP on a balanced resample to
    # show the comparison is not a class-prior artifact.
    rng = np.random.default_rng(0)
    itr = np.where(d.tr)[0]
    pos, neg = itr[d.y[itr] == 1], itr[d.y[itr] == 0]
    keep = np.concatenate([rng.choice(pos, len(neg), replace=False), neg])
    trb = np.zeros(len(d.y), bool)
    trb[keep] = True
    try:
        pb, _ = fit_probs(d.Xs, d.y, trb, d.va, "mlp32")
        arms["mlp32_balanced"] = pb
        rows += op_rows(d, pb, "mlp32_balanced", note="majority class subsampled")
    except Exception as e:
        log(f"  mlp32_balanced FAILED {e}")

    write_tsv(f"{OUT}/t2_models.tsv", rows)
    write_tsv(f"{OUT}/t2_generalization.tsv", gen)

    # per-codec at matched 0.5%
    pc = []
    bh = d.masks_te()["broad_honest"]
    for m, p in arms.items():
        T = threshold_for_fp(p, bh, 0.005)
        if T is None:
            continue
        f = p > T
        for cdc in sorted(set(d.codec[d.sub == "broad_honest"])):
            mm = d.te & (d.sub == "broad_honest") & (d.codec == cdc)
            if mm.sum():
                pc.append(dict(arm=m, codec=cdc, n=int(mm.sum()),
                               fp=float(f[mm].mean())))
    write_tsv(f"{OUT}/t2_per_codec.tsv", pc)
    np.savez_compressed(f"{OUT}/t2_scores.npz", **arms)


# -------------------------------------------------------------------- T3 ---
def t3(d):
    log("T3: what gets missed, and the FP mechanism")
    p = _base(d)
    T = 0.9
    fam_rows = []
    for f in sorted(set(d.fam[d.corr_te])):
        m = d.corr_te & (d.fam == f)
        if m.sum() < 5:
            continue
        rec = float((p[m] > T).mean())
        miss = m & (p <= T)
        fam_rows.append(dict(family=f, n=int(m.sum()), recall_T0p9=rec,
                             margin_p50=float(np.median(p[m])),
                             dial_p50=float(np.median(d.dial[m])),
                             miss_dial_p50=float(np.median(d.dial[miss])) if miss.sum() else float("nan"),
                             region_mode=_mode(d.region[m]),
                             kind_mode=_mode(d.kind[m]),
                             sev_worst_recall=_worst_sev(d, p, m, T)))
    fam_rows.sort(key=lambda r: r["recall_T0p9"])
    write_tsv(f"{OUT}/t3_per_family.tsv", fam_rows)

    # region / kind / severity breakdown of the MISSES
    br = []
    for col, name in ((d.region, "region"), (d.kind, "kind"), (d.sev, "severity")):
        for v in sorted(set(col[d.corr_te])):
            m = d.corr_te & (col == v)
            if m.sum() >= 20:
                br.append(dict(axis=name, value=v, n=int(m.sum()),
                               recall_T0p9=float((p[m] > T).mean()),
                               dial_p50=float(np.median(d.dial[m]))))
    write_tsv(f"{OUT}/t3_miss_axes.tsv", br)

    # ---- the FP side: which honest cells fire, and what is nearest to them --
    bh = d.te & (d.sub == "broad_honest")
    fired = bh & (p > T)
    fp_cells = []
    for cdc in sorted(set(d.codec[bh])):
        for nm, lo, hi in QBANDS:
            m = bh & (d.codec == cdc) & (d.q >= lo) & (d.q < hi)
            if m.sum():
                fp_cells.append(dict(codec=cdc, band=nm, n=int(m.sum()),
                                     fp=float((p[m] > T).mean()),
                                     dial_p50=float(np.median(d.dial[m]))))
    write_tsv(f"{OUT}/t3_fp_cells.tsv", fp_cells)

    # nearest-positive-family for the flagged honest cells, and nearest-honest
    # for the missed corruptions — both in the head's OWN standardized space.
    from sklearn.neighbors import NearestNeighbors
    Z = standardize(d.Xs[d.tr], d.Xs)
    pos_te = np.where(d.corr_te)[0]
    hon_te = np.where(d.te & (d.y == 0))[0]

    nn_pos = NearestNeighbors(n_neighbors=1, n_jobs=8).fit(Z[pos_te])
    fi = np.where(fired)[0]
    if len(fi):
        _, j = nn_pos.kneighbors(Z[fi])
        nf = d.fam[pos_te[j[:, 0]]]
        u, c = np.unique(nf, return_counts=True)
        o = np.argsort(-c)
        write_tsv(f"{OUT}/t3_flagged_nearest_family.tsv",
                  [dict(nearest_positive_family=u[k], n_flagged_honest=int(c[k]),
                        frac=float(c[k] / len(fi))) for k in o])
        # and restricted to the q>=95 band, the band the record blames
        hi95 = np.where(fired & (d.q >= 95))[0]
        if len(hi95):
            _, j2 = nn_pos.kneighbors(Z[hi95])
            u2, c2 = np.unique(d.fam[pos_te[j2[:, 0]]], return_counts=True)
            o2 = np.argsort(-c2)
            write_tsv(f"{OUT}/t3_flagged_q95_nearest_family.tsv",
                      [dict(nearest_positive_family=u2[k], n=int(c2[k]),
                            frac=float(c2[k] / len(hi95))) for k in o2])

    nn_hon = NearestNeighbors(n_neighbors=1, n_jobs=8).fit(Z[hon_te])
    mi = np.where(d.corr_te & (p <= T))[0]
    if len(mi):
        dist, j = nn_hon.kneighbors(Z[mi])
        sub_n = d.sub[hon_te[j[:, 0]]]
        cdc_n = d.codec[hon_te[j[:, 0]]]
        q_n = d.q[hon_te[j[:, 0]]]
        rows = []
        for f in sorted(set(d.fam[mi])):
            m = d.fam[mi] == f
            if m.sum() < 10:
                continue
            rows.append(dict(family=f, n_missed=int(m.sum()),
                             nn_dist_p50=float(np.median(dist[m, 0])),
                             nn_subclass_mode=_mode(sub_n[m]),
                             nn_codec_mode=_mode(cdc_n[m]),
                             nn_q_p50=float(np.nanmedian(q_n[m]))))
        rows.sort(key=lambda r: -r["n_missed"])
        write_tsv(f"{OUT}/t3_missed_nearest_honest.tsv", rows)


def _mode(a):
    if len(a) == 0:
        return ""
    u, c = np.unique(a, return_counts=True)
    return str(u[np.argmax(c)])


def _worst_sev(d, p, m, T):
    best = None
    for v in sorted(set(d.sev[m])):
        mm = m & (d.sev == v)
        if mm.sum() >= 5:
            r = float((p[mm] > T).mean())
            if best is None or r < best[1]:
                best = (v, r)
    return f"{best[0]}:{best[1]*100:.0f}%" if best else ""


# -------------------------------------------------------------------- T4 ---
def t4(d):
    log("T4: leave-one-family-out over all 44 families")
    p0 = _base(d)
    bh = d.masks_te()["broad_honest"]
    hi = bh & (d.q >= 85)
    T09 = 0.9
    T05 = threshold_for_fp(p0, bh, 0.005)
    base = dict(fp_all=float((p0 > T09)[bh].mean()),
                fp_hi=float((p0 > T09)[hi].mean()),
                det=float((p0 > T09)[d.corr_te].mean()))
    # evaluation-noise band (the fit is deterministic, so refit reproducibility
    # is exactly zero; the honest reference is the clustered bootstrap CI)
    lo_a, hi_a = boot_ci(d, p0, T09, bh)
    lo_h, hi_h = boot_ci(d, p0, T09, hi)
    log(f"T4 baseline fp_all={base['fp_all']*100:.2f}% CI[{lo_a*100:.2f},{hi_a*100:.2f}] "
        f"fp_hi={base['fp_hi']*100:.2f}% CI[{lo_h*100:.2f},{hi_h*100:.2f}] "
        f"det={base['det']*100:.1f}%")

    fams = sorted(set(d.fam[d.y == 1]))
    per_fam_base = {f: float((p0 > T09)[d.corr_te & (d.fam == f)].mean())
                    for f in fams}
    rows, cross = [], []
    for i, f in enumerate(fams):
        tr = d.tr & ~((d.y == 1) & (d.fam == f))
        p, _ = fit_probs(d.Xs, d.y, tr, d.va, "logistic")
        t05 = threshold_for_fp(p, bh, 0.005)
        rows.append(dict(
            removed=f, n_removed=int(((d.y == 1) & (d.fam == f) & d.tr).sum()),
            fp_all=float((p > T09)[bh].mean()),
            d_fp_all=float((p > T09)[bh].mean() - base["fp_all"]),
            fp_hi=float((p > T09)[hi].mean()),
            d_fp_hi=float((p > T09)[hi].mean() - base["fp_hi"]),
            det_all=float((p > T09)[d.corr_te].mean()),
            d_det_all=float((p > T09)[d.corr_te].mean() - base["det"]),
            recall_removed_fam=float((p > T09)[d.corr_te & (d.fam == f)].mean()),
            d_recall_removed_fam=float((p > T09)[d.corr_te & (d.fam == f)].mean()
                                       - per_fam_base[f]),
            det_at_fp0p5=float((p > t05)[d.corr_te].mean()) if t05 else None,
            noise_band=f"[{lo_a*100:.2f},{hi_a*100:.2f}]"))
        # cross-family: which OTHER families' recall moves when f is absent
        for g in fams:
            if g == f:
                continue
            m = d.corr_te & (d.fam == g)
            dr = float((p > T09)[m].mean() - per_fam_base[g])
            if abs(dr) >= 0.02:
                cross.append(dict(removed=f, other=g, d_recall=dr))
        if (i + 1) % 10 == 0:
            log(f"  LOO {i+1}/{len(fams)}")
    rows.sort(key=lambda r: r["d_fp_all"])
    write_tsv(f"{OUT}/t4_loo.tsv", rows)
    cross.sort(key=lambda r: r["d_recall"])
    write_tsv(f"{OUT}/t4_cross_family.tsv", cross)

    # greedy top-k removal
    order = [r["removed"] for r in rows]
    curve = [dict(k=0, removed="", fp_all=base["fp_all"], fp_hi=base["fp_hi"],
                  det=base["det"], det_at_fp0p5=float((p0 > T05)[d.corr_te].mean()))]
    drop = []
    for k in range(1, 9):
        drop.append(order[k - 1])
        tr = d.tr & ~((d.y == 1) & np.isin(d.fam, drop))
        p, _ = fit_probs(d.Xs, d.y, tr, d.va, "logistic")
        t05 = threshold_for_fp(p, bh, 0.005)
        curve.append(dict(k=k, removed=",".join(drop),
                          fp_all=float((p > T09)[bh].mean()),
                          fp_hi=float((p > T09)[hi].mean()),
                          det=float((p > T09)[d.corr_te].mean()),
                          det_at_fp0p5=float((p > t05)[d.corr_te].mean()) if t05 else None))
        log(f"  greedy k={k} fp={curve[-1]['fp_all']*100:.2f}% det={curve[-1]['det']*100:.1f}%")
    write_tsv(f"{OUT}/t4_greedy.tsv", curve)
    json.dump(dict(baseline=base, noise_ci_fp_all=[lo_a, hi_a],
                   noise_ci_fp_hi=[lo_h, hi_h], T09=T09, T_fp0p5=T05),
              open(f"{OUT}/t4_baseline.json", "w"), indent=1)


# -------------------------------------------------------------------- T5 ---
GROUPS = {
    "structural": ("block_", "edge_", "overlay_", "chroma_boundary", "aliasing"),
    "photometric": ("channel_", "tone_", "composite_", "noise_"),
    "geometric": ("geometric_",),
}


def _group_of(f):
    for g, pref in GROUPS.items():
        if any(f.startswith(p) or f == p for p in pref):
            return g
    return None


def t5(d):
    log("T5: family-grouped heads (mixture) vs one head")
    p0 = _base(d)
    grp = np.array([_group_of(f) or "" for f in d.fam])
    unassigned = sorted({f for f in set(d.fam[d.y == 1]) if _group_of(f) is None})
    if unassigned:
        log(f"  WARN unassigned families: {unassigned}")
    ps = []
    for g in GROUPS:
        keep = (d.y == 0) | (grp == g)
        tr = d.tr & keep
        p, _ = fit_probs(d.Xs, d.y, tr, d.va, "logistic")
        ps.append(p)
        log(f"  {g}: train pos={(tr&(d.y==1)).sum()}")
    pmax = np.max(np.vstack(ps), axis=0)
    rows = op_rows(d, p0, "single_head") + op_rows(d, pmax, "grouped_max")
    write_tsv(f"{OUT}/t5_grouped.tsv", rows)
    # per-group recall of the mixture vs the single head, at matched 0.5%
    bh = d.masks_te()["broad_honest"]
    per = []
    for lbl, s in (("single_head", p0), ("grouped_max", pmax)):
        T = threshold_for_fp(s, bh, 0.005)
        for g in GROUPS:
            m = d.corr_te & (grp == g)
            per.append(dict(arm=lbl, group=g, n=int(m.sum()),
                            recall=float((s > T)[m].mean())))
    write_tsv(f"{OUT}/t5_per_group.tsv", per)
    np.savez_compressed(f"{OUT}/t5_scores.npz", grouped_max=pmax)


# ------------------------------------------------- T6: the content confound --
def t6(d):
    """Same-source control: score every model form on the 2,016-row gate grid.

    The positives in training come from imazen-26 sources and the broad-honest
    negatives come from the ladder's own images, so a high-capacity model could
    in principle separate the two by CONTENT rather than by corruption. The gate
    grid removes that degree of freedom completely: 672 triples
    (corruption, q10, q20) from ONE reference (`gb82_dog`), which appears in no
    training fold. A model that is doing content identification cannot score
    here; a model that is detecting corruption can.

    Reports, per arm: detection vs FP on the same source's honest anchors at
    matched operating points, and the registered gate pass rates -- head alone
    (`P_corr > P_anchor`) and the DEPLOY composition
    `gate_score(perceptual, p, T) = if p > T { min(perceptual, 0) } else { perceptual }`.
    """
    import numpy as np
    z = np.load(f"{OUT}/gate_rev1.npz", allow_pickle=False)
    Xg = z["X"][:, :C.NFEAT_SLICE].astype(np.float64)
    entry = z["entry"]
    dialg = z["dial"]
    kind = np.array([e.rsplit("__", 1)[1] for e in entry])
    trip = np.array([e.rsplit("__", 1)[0] for e in entry])
    pos, q10, q20 = kind == "corruption", kind == "q10", kind == "q20"
    hon = q10 | q20
    log(f"T6 gate grid: {pos.sum()} corruption / {q10.sum()} q10 / {q20.sum()} q20, "
        f"{len(set(trip))} triples, 1 source")

    rows, gate = [], []
    for m in ("logistic", "mlp32", "mlp64_32", "hgb"):
        # `fit_head` IS the body this loop used to inline, factored out
        # 2026-09-06 so the ZCTH exporter fits the head the same way rather
        # than a fifth way. Arithmetic-neutral: same calls, same order — and
        # gated, not asserted (re-running t6 reproduces both TSVs exactly).
        _sc, clf, iso, raw = C.fit_head(d.Xs, d.y, d.tr, d.va, model=m)
        pr = 1 / (1 + np.exp(-raw(Xg)))
        p = C.rank_break(iso.predict(pr), pr)
        rows.append(dict(arm=m, pauc5=C.pauc(p, pos, hon),
                         det_at_fp0p5=_det_at(p, pos, hon, 0.005),
                         det_at_fp1=_det_at(p, pos, hon, 0.01),
                         det_at_fp5=_det_at(p, pos, hon, 0.05),
                         det_T0p9=float((p[pos] > 0.9).mean()),
                         fp_anchor_T0p9=float((p[hon] > 0.9).mean())))
        # registered gate pass rates
        pt = {t: i for i, t in enumerate(trip[pos])}
        ip, i10, i20 = np.where(pos)[0], np.where(q10)[0], np.where(q20)[0]
        o10 = np.array([pt[t] for t in trip[q10]])
        o20 = np.array([pt[t] for t in trip[q20]])
        head_only = 100.0 * (1.0 - p)
        for T in (0.9,):
            dep = np.where(p > T, np.minimum(dialg, 0.0), dialg)
            gate.append(dict(
                arm=m, T=T,
                head_pass_q10=float((head_only[ip][o10] < head_only[i10]).mean()),
                head_pass_q20=float((head_only[ip][o20] < head_only[i20]).mean()),
                deploy_pass_q10=float((dep[ip][o10] < dep[i10]).mean()),
                deploy_pass_q20=float((dep[ip][o20] < dep[i20]).mean())))
    dial_only = dict(arm="D dial alone", T="-",
                     head_pass_q10="", head_pass_q20="",
                     deploy_pass_q10=float((dialg[np.where(pos)[0]][np.array([{t: i for i, t in enumerate(trip[pos])}[t] for t in trip[q10]])] < dialg[q10]).mean()),
                     deploy_pass_q20=float((dialg[np.where(pos)[0]][np.array([{t: i for i, t in enumerate(trip[pos])}[t] for t in trip[q20]])] < dialg[q20]).mean()))
    gate.append(dial_only)
    write_tsv(f"{OUT}/t6_gate_samesource.tsv", rows)
    write_tsv(f"{OUT}/t6_gate_pass.tsv", gate)


def _det_at(p, pos, neg, tgt):
    T = threshold_for_fp(p, neg, tgt)
    return float((p[pos] > T).mean()) if T is not None else None


# ------------------------------ T7: paired bootstrap on the T4 family removals --
def t7(d):
    """Paired bootstrap of the LOO deltas.

    T4's noise band is an UNPAIRED bootstrap of one arm's own FP -- conservative,
    because two arms are evaluated on the identical test rows. The paired form
    resamples test-fold sources once and takes the DIFFERENCE on that resample,
    which is the quantity the question "does removing family F reduce FP?"
    actually asks.
    """
    import numpy as np
    p0 = _base(d)
    bh = d.masks_te()["broad_honest"]
    T09 = 0.9
    T0 = threshold_for_fp(p0, bh, 0.005)
    arms = {"__baseline__": p0}
    import csv
    top = [r["removed"] for r in
           sorted(csv.DictReader(open(f"{OUT}/t4_loo.tsv")
                                 .read().splitlines()[0:], delimiter="\t"),
                  key=lambda r: float(r["d_fp_all"]))[:5]]
    greedy = [r for r in csv.DictReader(open(f"{OUT}/t4_greedy.tsv"), delimiter="\t")]
    for f in top:
        tr = d.tr & ~((d.y == 1) & (d.fam == f))
        arms[f"-{f}"] = fit_probs(d.Xs, d.y, tr, d.va, "logistic")[0]
    k8 = greedy[-1]["removed"].split(",")
    tr = d.tr & ~((d.y == 1) & np.isin(d.fam, k8))
    arms[f"-greedy_k{len(k8)}"] = fit_probs(d.Xs, d.y, tr, d.va, "logistic")[0]

    rng = np.random.default_rng(0)
    uniq, inv = np.unique(d.src[bh], return_inverse=True)
    upos, ipos = np.unique(d.src[d.corr_te], return_inverse=True)
    out = []
    base_fp = (p0 > T09)[bh].astype(float)
    base_det = (p0 > T0)[d.corr_te].astype(float)
    draws = [rng.integers(0, len(uniq), len(uniq)) for _ in range(2000)]
    dposs = [rng.integers(0, len(upos), len(upos)) for _ in range(2000)]

    def clustered(vals, inv_, n_u, draw):
        s = np.bincount(inv_, weights=vals, minlength=n_u)
        c = np.bincount(inv_, minlength=n_u)
        return s[draw].sum() / max(c[draw].sum(), 1)

    for name, p in arms.items():
        if name == "__baseline__":
            continue
        Ta = threshold_for_fp(p, bh, 0.005)
        a_fp = (p > T09)[bh].astype(float)
        a_det = (p > Ta)[d.corr_te].astype(float)
        dfp = np.array([clustered(a_fp, inv, len(uniq), k) -
                        clustered(base_fp, inv, len(uniq), k) for k in draws])
        ddet = np.array([clustered(a_det, ipos, len(upos), k) -
                         clustered(base_det, ipos, len(upos), k) for k in dposs])
        out.append(dict(arm=name,
                        d_fp_T0p9=float(a_fp.mean() - base_fp.mean()),
                        d_fp_ci_lo=float(np.percentile(dfp, 2.5)),
                        d_fp_ci_hi=float(np.percentile(dfp, 97.5)),
                        d_det_at_fp0p5=float(a_det.mean() - base_det.mean()),
                        d_det_ci_lo=float(np.percentile(ddet, 2.5)),
                        d_det_ci_hi=float(np.percentile(ddet, 97.5))))
    write_tsv(f"{OUT}/t7_paired_bootstrap.tsv", out)


# ------------------- T8: does the dial policy still help a NONLINEAR head? --
def t8(d):
    """T1's policy question, re-asked with the T2 winner as the base head.

    T1 and T2 interact: if the dial gate exists to compensate for a LINEAR
    boundary, then a head that can already bend the boundary itself should get
    nothing from the gate. This is the test that separates "the dial carries
    information the features do not" from "the dial was a crutch".
    """
    import numpy as np
    rows = []
    for base in ("hgb", "mlp64_32"):
        p, _ = fit_probs(d.Xs, d.y, d.tr, d.va, base)
        arms = {f"{base}_nogate": p}
        for G in (80.0, 90.0):
            arms[f"{base}_hardmask_dial<{G:g}"] = np.where(d.dial < G, p, 0.0)
        Xd = np.column_stack([d.Xs, d.dial])
        arms[f"{base}_dial_as_feature"] = fit_probs(Xd, d.y, d.tr, d.va, base)[0]
        for k, s in arms.items():
            rows += op_rows(d, s, k)
        log(f"  {base} done")
    write_tsv(f"{OUT}/t8_policy_on_nonlinear.tsv", rows)


def t9(d):
    """Per-family recall of the T2 winner, against the incumbent's, at matched FP.

    T3 characterizes what the LINEAR head misses. If the nonlinear head fixes
    those families too, "what gets missed in general" is a property of the model
    form, not of the corruption catalogue -- which is a different answer to the
    user's question than the per-family table alone gives.
    """
    import numpy as np
    p0 = _base(d)
    bh = d.masks_te()["broad_honest"]
    out = []
    ph, _ = fit_probs(d.Xs, d.y, d.tr, d.va, "hgb")
    T0 = threshold_for_fp(p0, bh, 0.005)
    Th = threshold_for_fp(ph, bh, 0.005)
    for f in sorted(set(d.fam[d.corr_te])):
        m = d.corr_te & (d.fam == f)
        if m.sum() < 5:
            continue
        out.append(dict(family=f, n=int(m.sum()),
                        logistic_at_fp0p5=float((p0 > T0)[m].mean()),
                        hgb_at_fp0p5=float((ph > Th)[m].mean()),
                        logistic_T0p9=float((p0 > 0.9)[m].mean()),
                        hgb_T0p9=float((ph > 0.9)[m].mean())))
    out.sort(key=lambda r: r["hgb_at_fp0p5"])
    write_tsv(f"{OUT}/t9_per_family_best.tsv", out)
    reg = []
    for col, nm in ((d.region, "region"), (d.sev, "severity")):
        for v in sorted(set(col[d.corr_te])):
            m = d.corr_te & (col == v)
            if m.sum() >= 20:
                reg.append(dict(axis=nm, value=v, n=int(m.sum()),
                                logistic=float((p0 > T0)[m].mean()),
                                hgb=float((ph > Th)[m].mean())))
    write_tsv(f"{OUT}/t9_axes_best.tsv", reg)
