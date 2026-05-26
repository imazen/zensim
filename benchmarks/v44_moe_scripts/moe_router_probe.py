#!/usr/bin/env python3
"""Probe WHY the MoE can't beat V0_3 universally. Three diagnostics:

1. Per-pair scale mismatch: do V39 and V0_3 raw preds live on comparable
   scales? If not, ANY per-pair mix within a corpus scrambles rank.
2. AIC-4 feature separability from the 5 corpora (DESCRIPTIVE ONLY — uses
   AIC-4 membership purely to *measure* separability, NOT to fit a router
   that ships; this answers 'is the regime even separable').
3. Near-lossless detector router (hypothesis-driven, training-only):
   route the top-quality fraction to V0_3.
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

OUT = "/mnt/v/output/zensim/moe_v39_v03_2026-05-26"
ho = np.load(f"{OUT}/ho.npy", allow_pickle=True).item()
tr = np.load(f"{OUT}/tr.npy", allow_pickle=True).item()
HOLDOUT = ["cid22","kadid","tid","konjnd","aic3","aic4"]
V39 = {"cid22":0.8793,"kadid":0.9251,"tid":0.9317,"konjnd":0.4197,"aic3":0.8023,"aic4":0.9051}
V03 = {"cid22":0.8604,"kadid":0.9237,"tid":0.8849,"konjnd":0.2888,"aic3":0.7761,"aic4":0.9284}
def srocc(h,p):
    s=spearmanr(h,p).correlation; return abs(s) if s==s else 0.0

# --- 1. Raw scale of each bake per corpus ---
print("=== 1. Raw bake-pred scale per holdout corpus (p5/p50/p95) ===")
for c in HOLDOUT:
    d=ho[c]
    q39=np.percentile(d["p39"],[5,50,95]); q03=np.percentile(d["p03"],[5,50,95])
    # rank-corr between the two bakes within corpus (how much they agree on order)
    rab=spearmanr(d["p39"],d["p03"]).correlation
    print(f"  {c:7s} V39[{q39[0]:8.2f},{q39[1]:8.2f},{q39[2]:8.2f}]  "
          f"V03[{q03[0]:7.3f},{q03[1]:7.3f},{q03[2]:7.3f}]  rank-agree(V39,V03)={rab:.3f}")

# --- 2. AIC-4 separability (DESCRIPTIVE; not a shippable router) ---
print("\n=== 2. AIC-4-vs-training feature separability (descriptive only) ===")
# Train-corpus stack (the 5 we may legitimately use)
Xtr=np.vstack([np.nan_to_num(tr[c]["X"]) for c in ["cid22","kadid","tid","konjnd","aic3"]])
Xa4=np.nan_to_num(ho["aic4"]["X"])
# Can a classifier tell AIC-4 inputs apart from training inputs?
Xs=np.vstack([Xtr, Xa4]); ys=np.concatenate([np.zeros(len(Xtr)),np.ones(len(Xa4))])
sc=StandardScaler().fit(Xs); Xss=sc.transform(Xs)
lr=LogisticRegression(max_iter=2000, C=1.0)
auc=cross_val_score(lr, Xss, ys, cv=4, scoring="roc_auc").mean()
print(f"  LogReg AIC-4-vs-train separability AUC = {auc:.4f}  "
      f"(1.0=perfectly separable, 0.5=indistinguishable)")
# How do AIC-4 inputs compare to each training corpus in PCA-ish summary?
print("  (this only measures separability; the SHIPPABLE router below does"
      " NOT use AIC-4 membership)")

# --- 3. Hypothesis-driven near-lossless router (training-only fit) ---
# On safesyn, V0_3 beats V39. The synthetic corpus is near-lossless-heavy.
# Build a 'near-lossless score' from features by training a regressor on
# the TRAINING human_score (high=near-lossless for [0,1] corpora). Then
# route the most-near-lossless pairs to V0_3. Fit the routed fraction by
# training-corpus combined SROCC ONLY.
print("\n=== 3. Near-lossless router (training-only) ===")
from sklearn.ensemble import HistGradientBoostingRegressor
# stack training feats + human (normalize konjnd/safesyn scales loosely to rank)
Xq=[]; yq=[]
for c in ["cid22","kadid","tid","konjnd","aic3"]:
    d=tr[c]; Xq.append(np.nan_to_num(d["X"]))
    # rank-normalize human within group to [0,1] (high=high quality after
    # polarity fix: for kadid DMOS lower=better so use rank of -human; but
    # we only need a monotone quality proxy, so rank by the bake-agreed order)
    h=d["human"]; r=np.argsort(np.argsort(h))/max(1,len(h)-1)
    yq.append(r)
Xq=np.vstack(Xq); yq=np.concatenate(yq)
reg=HistGradientBoostingRegressor(max_iter=200, max_depth=6, random_state=17)
reg.fit(Xq,yq)
def quality_score(X): return reg.predict(np.nan_to_num(X))
# choose routed-fraction by training combined SROCC
best=(0.0,-1,None)
for frac in [0.0,0.05,0.1,0.15,0.2,0.3,0.4,0.5]:
    ms=[]
    for c in ["cid22","kadid","tid","konjnd","aic3"]:
        d=tr[c]; qs=quality_score(d["X"])
        thr=np.quantile(qs, 1-frac) if frac>0 else np.inf
        m=qs>=thr
        comb=np.where(m,d["p03"],d["p39"]); ms.append(srocc(d["human"],comb))
    if np.mean(ms)>best[1]: best=(frac,np.mean(ms),None)
frac=best[0]
print(f"  chosen routed-fraction (top-quality)->V0_3: {frac:.2f} (train mean {best[1]:.4f})")
# freeze: compute per-corpus global threshold from TRAINING quality dist
qs_tr_all=quality_score(Xtr)
gthr=np.quantile(qs_tr_all,1-frac) if frac>0 else np.inf
ok=True
for c in HOLDOUT:
    d=ho[c]; qs=quality_score(d["X"]); m=qs>=gthr
    comb=np.where(m,d["p03"],d["p39"]); s=srocc(d["human"],comb)
    ge=s>=V03[c]-1e-4; ok&=ge
    print(f"  {c:7s} comb={s:.4f} V39={V39[c]:.4f} V03={V03[c]:.4f} f03={m.mean():5.1%} >=V03:{'Y' if ge else 'N'}")
print(f"  --> >=V0_3 on ALL 6: {'*** YES ***' if ok else 'no'}")
