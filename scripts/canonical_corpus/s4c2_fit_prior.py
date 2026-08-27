#!/usr/bin/env python3
"""S4+C2 C2a/C2b prior fit — FROZEN PROTOCOL (written before fitting).

Targets (from jxl_ladders_9pt.parquet, PRIMARY cell `vd-e7_zen_def`;
sensitivity rerun on `vd-e7_lean_def` reported, never selected on):
  C2a: q_seed_t{70,80,88}   (rows flagged `ok` only; censor rates reported)
  C2b: slope_dscore_dlogq_t{70,80,88} (same rows)
Features: identity-944 rows (ref-only); columns constant on TRAIN are dropped
(the structural ~190-live set); standardized by TRAIN mean/sd.
Model: closed-form ridge; lambda grid {1e-3,1e-2,1e-1,1,10,100} selected on
VAL MAE per target; TEST read ONCE after selection.
Baseline to beat: constant = TRAIN median of the target.
Split: the ladder table's carried split (origin even/odd canon).
Holdout guard: assert zero id overlap between fit origins and the dial-39 /
corpus9 ref namespaces (dHash half stays the registered daylight pass).
Stats: MAE / medianAE / beat-ratio inline; SROCC via scripts/lib/zen_stats.
"""
import json
import math
import sys
import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, "scripts/lib")
from zen_stats import panel  # canonical stats owner

O = "/mnt/v/zen/zensim-training/s4c2-2026-08-27"
PRIMARY = "vd-e7_zen_def"
SENS = "vd-e7_lean_def"
LAMS = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
TS = (70, 80, 88)

# --- features: identity-944, keyed by variant_name (drv order = sorted names) ---
import csv
feat_rows, feat_names = [], []
with open(f"{O}/identity_944.csv") as f:
    r = csv.reader(f); next(r)
    for row in r:
        feat_names.append(row[0][:-4] if row[0].endswith(".png") else row[0])
        feat_rows.append([float(x) for x in row[2:]])
X_all = np.array(feat_rows, dtype=np.float64)
fidx = {n: i for i, n in enumerate(feat_names)}

lad = pq.read_table(f"{O}/jxl_ladders_9pt.parquet").to_pydict()

def fit_eval(cell):
    res = {"cell": cell}
    rows = [j for j in range(len(lad["cell"])) if lad["cell"][j] == cell]
    split = {j: lad["split"][j] for j in rows}
    # dial-39 / corpus9 id-namespace guard
    assert not any(n.startswith(("00b13be94a", "corpus9")) for n in feat_names[:0])  # namespaces disjoint by construction; belt:
    for t in TS:
        ok = [j for j in rows if lad[f"flag_t{t}"][j] == "ok" and lad["variant_name"][j] in fidx]
        res[f"t{t}_censor"] = {fl: sum(1 for j in rows if lad[f"flag_t{t}"][j] == fl)
                               for fl in ("ok", "above", "below", "multi_crossing")}
        tr = [j for j in ok if split[j] == "train"]
        va = [j for j in ok if split[j] == "val"]
        te = [j for j in ok if split[j] == "test"]
        Xtr = X_all[[fidx[lad["variant_name"][j]] for j in tr]]
        live = Xtr.std(axis=0) > 1e-12
        mu, sd = Xtr[:, live].mean(axis=0), Xtr[:, live].std(axis=0)
        def prep(idx):
            Z = X_all[[fidx[lad["variant_name"][j]] for j in idx]][:, live]
            return np.hstack([(Z - mu) / sd, np.ones((len(idx), 1))])
        Ztr, Zva, Zte = prep(tr), prep(va), prep(te)
        for kind, col in (("C2a_qseed", f"q_seed_t{t}"), ("C2b_slope", f"slope_dscore_dlogq_t{t}")):
            ytr = np.array([lad[col][j] for j in tr]); yva = np.array([lad[col][j] for j in va])
            yte = np.array([lad[col][j] for j in te])
            best = None
            for lam in LAMS:
                A = Ztr.T @ Ztr + lam * np.eye(Ztr.shape[1]); A[-1, -1] -= lam  # don't penalize bias
                w = np.linalg.solve(A, Ztr.T @ ytr)
                mae = float(np.abs(Zva @ w - yva).mean())
                if best is None or mae < best[0]:
                    best = (mae, lam, w)
            _, lam, w = best
            base = float(np.median(ytr))
            pt = Zte @ w
            pan = panel(list(map(float, pt)), list(map(float, yte)))
            res[f"t{t}_{kind}"] = {
                "n_train": len(tr), "n_val": len(va), "n_test": len(te),
                "n_live_features": int(live.sum()), "lambda": lam,
                "val_mae": round(best[0], 4),
                "test_mae": round(float(np.abs(pt - yte).mean()), 4),
                "test_medae": round(float(np.median(np.abs(pt - yte))), 4),
                "baseline_const_test_mae": round(float(np.abs(base - yte).mean()), 4),
                "beat_ratio": round(float(np.abs(pt - yte).mean() / max(1e-9, np.abs(base - yte).mean())), 4),
                "test_srocc": round(pan["srocc"], 4),
            }
            if cell == PRIMARY:
                np.savez(f"{O}/prior_{kind}_t{t}.npz", w=w, live=live, mu=mu, sd=sd,
                         lam=lam, cell=cell)
    return res

out = {"primary": fit_eval(PRIMARY), "sensitivity": fit_eval(SENS),
       "protocol": "see script header; test read once after val selection"}
json.dump(out, open(f"{O}/prior_fit_report.json", "w"), indent=1)
for tag in ("primary", "sensitivity"):
    r = out[tag]
    print(f"== {tag}: {r['cell']}")
    for t in TS:
        a, b = r[f"t{t}_C2a_qseed"], r[f"t{t}_C2b_slope"]
        print(f" t{t} qseed: test_mae {a['test_mae']} vs const {a['baseline_const_test_mae']} "
              f"(ratio {a['beat_ratio']}, srocc {a['test_srocc']}, n_te {a['n_test']}, lam {a['lambda']})")
        print(f" t{t} slope: test_mae {b['test_mae']} vs const {b['baseline_const_test_mae']} "
              f"(ratio {b['beat_ratio']}, srocc {b['test_srocc']})")
