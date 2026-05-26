#!/usr/bin/env python3
"""Build + freeze input-feature routers (training-only), eval on 6 holdouts.

Router options (all fit on TRAINING corpora only):
  A. Regime classifier: label each training pair by which bake ranks it
     better *locally* (within its (corpus) group, sliding-window rank
     agreement), train logistic/GBM on features -> P(route V0_3).
  B. Near-lossless detector: a simpler hypothesis-driven router. V0_3
     wins safesyn (synthetic, near-lossless-heavy). Use a feature-space
     'high quality / low distortion' score from training corpora, route
     the most-near-lossless fraction to V0_3.
  C. Disagreement router: where the two bakes' RANK disagrees most,
     pick by a training-fit rule.

Combiner = HARD per-pair selection. Evaluate per-corpus SROCC.
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

OUT = "/mnt/v/output/zensim/moe_v39_v03_2026-05-26"
ho = np.load(f"{OUT}/ho.npy", allow_pickle=True).item()
tr = np.load(f"{OUT}/tr.npy", allow_pickle=True).item()
HOLDOUT = ["cid22", "kadid", "tid", "konjnd", "aic3", "aic4"]
V39 = {"cid22":0.8793,"kadid":0.9251,"tid":0.9317,"konjnd":0.4197,"aic3":0.8023,"aic4":0.9051}
V03 = {"cid22":0.8604,"kadid":0.9237,"tid":0.8849,"konjnd":0.2888,"aic3":0.7761,"aic4":0.9284}

def srocc(h, p):
    s = spearmanr(h, p).correlation
    return abs(s) if s == s else 0.0

def eval_combiner(route_to_v03_fn, label):
    """route_to_v03_fn(X) -> bool mask (True => use V0_3 pred)."""
    print(f"\n=== Combiner: {label} ===")
    rows = []
    ok_all = True
    for c in HOLDOUT:
        d = ho[c]
        m = route_to_v03_fn(d["X"])
        comb = np.where(m, d["p03"], d["p39"])
        s = srocc(d["human"], comb)
        frac03 = m.mean()
        ge_v03 = s >= V03[c] - 1e-4
        ok_all &= ge_v03
        rows.append((c, s, V39[c], V03[c], frac03, ge_v03))
        print(f"  {c:7s} comb={s:.4f}  V39={V39[c]:.4f} V03={V03[c]:.4f}  "
              f"f03={frac03:5.1%}  >=V03:{'Y' if ge_v03 else 'N'}")
    print(f"  --> >=V0_3 on ALL 6: {'*** YES ***' if ok_all else 'no'}")
    return ok_all, rows

# Stack training features + per-corpus group ids for local-rank labels
Xtr_list, gid_list, h_list, p39_list, p03_list = [], [], [], [], []
for gi, c in enumerate(["cid22","kadid","tid","konjnd","aic3"]):
    d = tr[c]
    Xtr_list.append(d["X"]); gid_list.append(np.full(len(d["X"]), gi))
    h_list.append(d["human"]); p39_list.append(d["p39"]); p03_list.append(d["p03"])
Xtr = np.vstack(Xtr_list)
gid = np.concatenate(gid_list)
htr = np.concatenate(h_list); p39tr = np.concatenate(p39_list); p03tr = np.concatenate(p03_list)
# sanitize
Xtr = np.nan_to_num(Xtr, nan=0.0, posinf=0.0, neginf=0.0)
print(f"Training stack: {Xtr.shape}, groups {np.bincount(gid)}")

# ---- Local-rank label: per pair, does V0_3 rank-agree with target better
# than V39 in a feature-neighborhood? Approximate with per-group windowed
# concordance on the TARGET ordering. Simpler robust proxy: within each
# group, sort by human; for each pair, local concordance of each bake's
# pred with human over a window of W neighbors in human order.
def local_concord_labels(h, p39, p03, gid, W=40):
    lab = np.zeros(len(h), dtype=int)
    for gi in np.unique(gid):
        idx = np.where(gid == gi)[0]
        order = idx[np.argsort(h[idx])]
        n = len(order)
        for k in range(n):
            lo = max(0, k - W); hi = min(n, k + W + 1)
            win = order[lo:hi]
            if len(win) < 5:
                continue
            c39 = spearmanr(h[win], p39[win]).correlation
            c03 = spearmanr(h[win], p03[win]).correlation
            c39 = 0.0 if c39 != c39 else abs(c39)
            c03 = 0.0 if c03 != c03 else abs(c03)
            lab[order[k]] = 1 if c03 > c39 + 1e-6 else 0
    return lab

print("Computing local-concordance labels (training-only)...")
ytr = local_concord_labels(htr, p39tr, p03tr, gid, W=40)
print(f"Label balance route_V0_3={ytr.mean():.3f}  (n={len(ytr)})")

scaler = StandardScaler().fit(Xtr)
Xtr_s = scaler.transform(Xtr)

# Router A: GBM classifier on features -> P(V0_3 better locally)
clfA = HistGradientBoostingClassifier(max_iter=200, max_depth=6,
                                      learning_rate=0.08, l2_regularization=1.0,
                                      random_state=17)
clfA.fit(Xtr, ytr)
# pick threshold on training to maximize mean training-corpus combined SROCC
def routerA(X, thr=0.5):
    Xc = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return clfA.predict_proba(Xc)[:, 1] >= thr

# choose threshold by training-corpus combined SROCC (NOT holdout)
best_thr, best_mean = 0.5, -1
for thr in np.linspace(0.2, 0.8, 13):
    ms = []
    for c in ["cid22","kadid","tid","konjnd","aic3"]:
        d = tr[c]
        m = routerA(d["X"], thr)
        comb = np.where(m, d["p03"], d["p39"])
        ms.append(srocc(d["human"], comb))
    if np.mean(ms) > best_mean:
        best_mean, best_thr = np.mean(ms), thr
print(f"Router A chosen thr={best_thr:.2f} (train mean SROCC={best_mean:.4f})")

eval_combiner(lambda X: np.zeros(len(X), dtype=bool), "all-V39 (baseline)")
eval_combiner(lambda X: np.ones(len(X), dtype=bool), "all-V0_3 (baseline)")
eval_combiner(lambda X: routerA(X, best_thr), f"A: GBM local-concord thr={best_thr:.2f}")

# Router A oracle-threshold scan ON HOLDOUT (diagnostic only — shows ceiling,
# NOT used for selection; reported as 'what if we could tune thr on holdout')
print("\n[diagnostic] Router A holdout threshold scan (NOT a valid selection):")
for thr in [0.3,0.4,0.5,0.6,0.7]:
    line=[]
    for c in HOLDOUT:
        d=ho[c]; m=routerA(d["X"],thr); comb=np.where(m,d["p03"],d["p39"])
        line.append(f"{c}={srocc(d['human'],comb):.3f}(f{m.mean():.2f})")
    print(f"  thr={thr}: "+" ".join(line))
