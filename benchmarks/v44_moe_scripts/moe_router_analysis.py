#!/usr/bin/env python3
"""MoE router over {V39, V0_3} bakes — input-feature regime router.

Methodological red line: AIC-3/AIC-4 are HOLDOUT-ONLY. The router is
trained EXCLUSIVELY on training corpora (safesyn/kadid/tid/cid22-train/
konjnd). AIC-4 labels/membership are NEVER used to design or select the
router. AIC-4 (and AIC-3) features are only loaded at frozen-eval time.

Combiner = HARD per-pair bake selection (no soft blend — raw bake
outputs are ~0.2-wide flat bands per v43 learnings; soft blend scrambles
rank). Evaluated per-corpus (SROCC is within-corpus).
"""
import glob
import numpy as np
import pyarrow.parquet as pq
from scipy.stats import spearmanr

OUT = "/mnt/v/output/zensim/moe_v39_v03_2026-05-26"
FEAT_VAL = "/mnt/v/zen/zensim-training/2026-05-15-full-features"
TRAIN = "/mnt/v/zen/zensim-training/canonical-2026-05-21/train"
SAFESYN = "/mnt/v/zen/zensim-training/canonical-2026-05-18/train/safesyn.parquet"

NFEAT = 372

def load_pred(tag, corpus):
    d = np.loadtxt(f"{OUT}/{tag}_{corpus}_perpair.tsv", skiprows=1)
    return d[:, 0], d[:, 1]  # human, pred

def load_feats(path, ncols=NFEAT):
    cols = ["f%d" % i for i in range(ncols)]
    t = pq.read_table(path, columns=cols)
    return np.column_stack([t.column(c).to_numpy(zero_copy_only=False) for c in cols])

def srocc(h, p):
    s = spearmanr(h, p).correlation
    return abs(s)  # polarity-tolerant (bakes can be distance- or score-shaped)

# ---------------------------------------------------------------------------
# 1. Holdout per-pair preds + features (6 corpora)
# ---------------------------------------------------------------------------
HOLDOUT = ["cid22", "kadid", "tid", "konjnd", "aic3", "aic4"]
HFEAT = {
    "cid22": "cid22_features_372col_2026-05-15.parquet",
    "kadid": "kadid_features_372col_2026-05-15.parquet",
    "tid": "tid_features_372col_2026-05-15.parquet",
    "konjnd": "konjnd_features_372col_2026-05-15.parquet",
    "aic3": "aic3_features_372col_2026-05-15.parquet",
    "aic4": "aic4_features_372col_2026-05-20.parquet",
}
ho = {}
for c in HOLDOUT:
    h39, p39 = load_pred("v39", c)
    h03, p03 = load_pred("v03", c)
    assert np.allclose(h39, h03), f"{c}: human mismatch between bake TSVs"
    X = load_feats(f"{FEAT_VAL}/{HFEAT[c]}")
    assert X.shape[0] == len(h39), f"{c}: feat rows {X.shape[0]} != pred {len(h39)}"
    ho[c] = dict(human=h39, p39=p39, p03=p03, X=X)
    print(f"[holdout] {c:7s} n={len(h39):5d}  V39={srocc(h39,p39):.4f}  V03={srocc(h03,p03):.4f}")

# ---------------------------------------------------------------------------
# 2. Training per-pair preds + features (5 corpora) -- ROUTER FIT ONLY
#    corpus-name -> training source (aic3 slot == safesyn)
# ---------------------------------------------------------------------------
TRAIN_CORP = ["cid22", "kadid", "tid", "konjnd", "aic3"]
TRAIN_FEAT = {
    "cid22": f"{TRAIN}/cid22_train.parquet",
    "kadid": f"{TRAIN}/kadid.parquet",
    "tid": f"{TRAIN}/tid.parquet",
    "konjnd": f"{TRAIN}/konjnd-dense.parquet",
    "aic3": SAFESYN,  # safesyn rides the aic3 slot
}
tr = {}
for c in TRAIN_CORP:
    h39, p39 = load_pred("train_v39", c)
    h03, p03 = load_pred("train_v03", c)
    X = load_feats(TRAIN_FEAT[c])
    n = min(len(h39), X.shape[0])
    tr[c] = dict(human=h39[:n], p39=p39[:n], p03=p03[:n], X=X[:n], name=c)
    print(f"[train]   {c:7s} n={X.shape[0]:6d}  V39={srocc(h39,p39):.4f}  V03={srocc(h03,p03):.4f}")

np.save(f"{OUT}/ho.npy", ho, allow_pickle=True)
np.save(f"{OUT}/tr.npy", tr, allow_pickle=True)
print("saved ho.npy, tr.npy")
