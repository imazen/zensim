#!/usr/bin/env python3
"""
G5 refined router — push the classifier-route past its CID22/AIC-3
false-positive ceiling (experiment-rigor extension).

The konjnd-target feature router (g5_classifier_router) gets KonJND to
0.70-0.73 but breaks CID22 (-0.014 to -0.028) and AIC-3 (-0.016) via a
small but damaging false-positive rate. This script tries three
damage-limiting refinements, on the held-out 20% AND full corpus:

  R1 soft-blend on routed pairs: combined = p*spec + (1-p)*v39 where p is
     the router's calibrated p(konjnd). Caps per-pair damage on uncertain
     routes instead of a hard switch.
  R2 confidence floor: only route (hard) if p > tau (sweep high tau to
     cut false positives), reported alongside KonJND.
  R3 probability-weighted continuous mix everywhere: combined = p*spec +
     (1-p)*v39 with NO threshold. The blend itself is the regime gate.

Held-out 20% is the principled report. Verdict bar: KonJND >= 0.70 AND
CID22/KADIK/TID/AIC-3/AIC-4 each within -0.01 of V39.
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
N = 372


def srocc(h, p):
    h = np.asarray(h, float); p = np.asarray(p, float)
    v = np.isfinite(h) & np.isfinite(p)
    return abs(spearmanr(h[v], p[v])[0]) if v.sum() >= 3 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specialist", default="w01")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    fc = [f"f{i}" for i in range(N)]

    feats, v39, spec, human = {}, {}, {}, {}
    for c in CORPORA:
        df = pq.read_table(VAL_ROOT / VAL_FILE[c], columns=fc).to_pandas()
        feats[c] = np.nan_to_num(df.values, nan=0.0, posinf=0.0, neginf=0.0)
        v39[c] = pd.read_csv(SCORES / f"{c}_v39.tsv", sep="\t")["score"].values
        spec[c] = pd.read_csv(SCORES / f"{c}_{args.specialist}.tsv", sep="\t")["score"].values
        human[c] = pd.read_csv(SCORES / f"{c}_v39.tsv", sep="\t")["human"].values

    tr, te = {}, {}
    for c in CORPORA:
        idx = np.arange(len(human[c])); rng.shuffle(idx)
        nt = max(1, len(idx)//5); te[c] = np.sort(idx[:nt]); tr[c] = np.sort(idx[nt:])

    Xtr = np.vstack([feats[c][tr[c]] for c in CORPORA])
    ytr = np.concatenate([np.full(len(tr[c]), 1 if c == "konjnd" else 0) for c in CORPORA])
    sc = StandardScaler().fit(Xtr)
    clf = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=args.seed,
                        early_stopping=True, validation_fraction=0.1, alpha=1e-3).fit(sc.transform(Xtr), ytr)

    base = {c: srocc(human[c], v39[c]) for c in CORPORA}

    def report(view, idxsel):
        print(f"\n========== {view}  specialist={args.specialist} ==========")
        # Precompute p per corpus
        P = {c: clf.predict_proba(sc.transform(feats[c][idxsel[c]]))[:, 1] for c in CORPORA}

        # R3: continuous prob-weighted mix
        print("R3 continuous mix  combined = p*spec + (1-p)*v39:")
        row = {}
        for c in CORPORA:
            sel = idxsel[c]
            comb = P[c]*spec[c][sel] + (1-P[c])*v39[c][sel]
            row[c] = srocc(human[c][sel], comb)
        kon = row["konjnd"]; ok = all(row[c] >= base[c]-0.01 for c in OTHER5)
        print("   " + " ".join(f"{c}={row[c]:.3f}" for c in CORPORA) +
              f"  kon>=.70?{kon>=0.70} oth_ok?{ok} {'PASS' if kon>=0.70 and ok else ''}")

        # R1: soft-blend on routed (p>0.5) pairs, hard v39 elsewhere
        print("R1 soft-blend on routed (p>0.5), v39 elsewhere:")
        row = {}
        for c in CORPORA:
            sel = idxsel[c]
            routed = P[c] > 0.5
            comb = v39[c][sel].copy()
            comb[routed] = P[c][routed]*spec[c][sel][routed] + (1-P[c][routed])*v39[c][sel][routed]
            row[c] = srocc(human[c][sel], comb)
        kon = row["konjnd"]; ok = all(row[c] >= base[c]-0.01 for c in OTHER5)
        print("   " + " ".join(f"{c}={row[c]:.3f}" for c in CORPORA) +
              f"  kon>=.70?{kon>=0.70} oth_ok?{ok} {'PASS' if kon>=0.70 and ok else ''}")

        # R2: high-tau hard route sweep
        print("R2 high-tau hard route:")
        print(f"   {'tau':>5s} | " + " ".join(f"{c[:5]:>6s}" for c in CORPORA) + " | verdict")
        for tau in [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9]:
            row = {}
            for c in CORPORA:
                sel = idxsel[c]
                comb = np.where(P[c] > tau, spec[c][sel], v39[c][sel])
                row[c] = srocc(human[c][sel], comb)
            kon = row["konjnd"]; ok = all(row[c] >= base[c]-0.01 for c in OTHER5)
            print(f"   {tau:5.2f} | " + " ".join(f"{row[c]:6.3f}" for c in CORPORA) +
                  f" | kon={kon:.3f} oth_ok={ok} {'PASS' if kon>=0.70 and ok else ''}")

    report("HELD-OUT 20%", te)
    report("FULL corpus", {c: np.arange(len(human[c])) for c in CORPORA})


if __name__ == "__main__":
    main()
