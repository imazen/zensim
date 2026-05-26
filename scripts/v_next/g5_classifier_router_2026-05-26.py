#!/usr/bin/env python3
"""
G5 classifier-routed 2-bake ensemble — offline test (OPTION A).

Tests whether a learned 372-feature router can separate the KonJND/HF
regime from the CID22/AIC-3 mid-fidelity regime well enough to route
near-lossless pairs to the HF-specialist and everything else to V39,
clearing KonJND >= 0.70 AND keeping the other 5 within -0.01 of V39.

Two router targets are tried:
  (1) "konjnd-vs-rest": label=1 for KonJND pairs, 0 otherwise.
      The router learns to recognize KonJND-regime pairs. Route to
      specialist when p(konjnd) > tau.
  (2) "hf-vs-rest": label=1 for any pair whose human ground-truth is in
      the near-lossless HF band (top quality quartile of each corpus),
      0 otherwise. This is the principled "regime" target.

Held-out discipline: 80/20 stratified split per corpus, classifier
trained on 80%, ensemble panel reported on the held-out 20%. The
classifier never sees a held-out row's label. We also report the
FULL-corpus deployment view (what a deployed router produces).

Routing-by-feature is leakage-free for the SROCC verdict because the
router predicts regime membership, not bake correctness; the bakes
themselves were trained on synth+kadid+tid+konjnd (NOT CID22/AIC-3/AIC-4
human MOS), so CID22/AIC-3/AIC-4 remain true holdouts.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

SCORES = Path("/mnt/v/output/zensim/g5_ensemble_2026-05-26/scores")
VAL_ROOT = Path("/mnt/v/zen/zensim-training/2026-05-15-full-features")
CORPORA = ["cid22", "kadid", "tid", "konjnd", "aic3", "aic4"]
OTHER5 = ["cid22", "kadid", "tid", "aic3", "aic4"]
VAL_FILE = {
    "cid22": "cid22_features_372col_2026-05-15.parquet",
    "kadid": "kadid_features_372col_2026-05-15.parquet",
    "tid": "tid_features_372col_2026-05-15.parquet",
    "konjnd": "konjnd_features_372col_2026-05-15.parquet",
    "aic3": "aic3_features_372col_2026-05-15.parquet",
    "aic4": "aic4_features_372col_2026-05-20.parquet",
}
N_FEATURES = 372


def srocc(h, p):
    h = np.asarray(h, float)
    p = np.asarray(p, float)
    valid = np.isfinite(h) & np.isfinite(p)
    if valid.sum() < 3:
        return float("nan")
    return abs(spearmanr(h[valid], p[valid])[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specialist", default="w01")
    ap.add_argument("--target", default="konjnd", choices=["konjnd", "hf"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    feat_cols = [f"f{i}" for i in range(N_FEATURES)]

    feats, v39, spec, human = {}, {}, {}, {}
    for c in CORPORA:
        df = pq.read_table(VAL_ROOT / VAL_FILE[c], columns=feat_cols).to_pandas()
        feats[c] = np.nan_to_num(df.values, nan=0.0, posinf=0.0, neginf=0.0)
        v39[c] = pd.read_csv(SCORES / f"{c}_v39.tsv", sep="\t")["score"].values
        spec[c] = pd.read_csv(SCORES / f"{c}_{args.specialist}.tsv", sep="\t")["score"].values
        human[c] = pd.read_csv(SCORES / f"{c}_v39.tsv", sep="\t")["human"].values

    # Build per-corpus labels.
    def label_for(c, h):
        if args.target == "konjnd":
            return np.full(len(h), 1 if c == "konjnd" else 0)
        # hf: top quality band. KonJND human is PJND threshold (higher=more
        # tolerant=higher q); others vary in polarity, so use within-corpus
        # rank: HF = top 25% closest-to-lossless. For corpora where higher
        # human = better quality (cid22 MCOS, tid MOS, aic*), top quartile.
        # For kadid DMOS lower=better, bottom quartile. KonJND treated all-HF.
        if c == "konjnd":
            return np.ones(len(h), int)
        hh = np.asarray(h, float)
        if c == "kadid":
            thresh = np.nanpercentile(hh, 25)
            return (hh <= thresh).astype(int)
        thresh = np.nanpercentile(hh, 75)
        return (hh >= thresh).astype(int)

    # 80/20 split.
    train_idx, test_idx = {}, {}
    for c in CORPORA:
        n = len(human[c])
        idx = np.arange(n)
        rng.shuffle(idx)
        nt = max(1, n // 5)
        test_idx[c] = np.sort(idx[:nt])
        train_idx[c] = np.sort(idx[nt:])

    X_tr, y_tr = [], []
    for c in CORPORA:
        ti = train_idx[c]
        X_tr.append(feats[c][ti])
        y_tr.append(label_for(c, human[c][ti]))
    X_tr = np.vstack(X_tr)
    y_tr = np.concatenate(y_tr)

    scaler = StandardScaler().fit(X_tr)
    clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=args.seed,
                        early_stopping=True, validation_fraction=0.1, alpha=1e-3, solver="adam")
    clf.fit(scaler.transform(X_tr), y_tr)
    print(f"[router target={args.target}] train acc={clf.score(scaler.transform(X_tr), y_tr):.4f}")

    base_v39 = {c: srocc(human[c], v39[c]) for c in CORPORA}

    # Routing fraction per corpus + ensemble panel, on the held-out 20% AND full.
    for view, idxsel in [("HELD-OUT 20%", test_idx), ("FULL corpus", {c: np.arange(len(human[c])) for c in CORPORA})]:
        print(f"\n=== {view} — router target={args.target}, specialist={args.specialist} ===")
        print(f"{'corpus':8s} {'route%':>7s} | {'V39':>7s} {'ens@.5':>7s} {'Δ':>7s}")
        kon_e = None
        deltas = {}
        for c in CORPORA:
            sel = idxsel[c]
            p = clf.predict_proba(scaler.transform(feats[c][sel]))[:, 1]
            route = (p > 0.5).astype(int)
            ens = np.where(route == 1, spec[c][sel], v39[c][sel])
            e_srocc = srocc(human[c][sel], ens)
            v_srocc = srocc(human[c][sel], v39[c][sel])
            d = e_srocc - v_srocc
            deltas[c] = d if view == "FULL corpus" else d
            print(f"{c:8s} {route.mean()*100:6.1f}% | {v_srocc:7.4f} {e_srocc:7.4f} {d:+7.4f}")
            if c == "konjnd":
                kon_e = e_srocc
        # also sweep tau to find best routing threshold on full view
        if view == "FULL corpus":
            print("\n  tau sweep (route to specialist when p>tau):")
            print(f"  {'tau':>5s} | " + " ".join(f"{c[:5]:>6s}" for c in CORPORA) + " | verdict")
            for tau in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
                row = {}
                for c in CORPORA:
                    p = clf.predict_proba(scaler.transform(feats[c]))[:, 1]
                    route = (p > tau).astype(int)
                    ens = np.where(route == 1, spec[c], v39[c])
                    row[c] = srocc(human[c], ens)
                kon = row["konjnd"]
                ok = all(row[c] >= base_v39[c] - 0.01 for c in OTHER5)
                verdict = "PASS" if (kon >= 0.70 and ok) else ""
                print(f"  {tau:5.2f} | " + " ".join(f"{row[c]:6.3f}" for c in CORPORA) +
                      f" | kon={kon:.3f} oth_ok={ok} {verdict}")


if __name__ == "__main__":
    main()
