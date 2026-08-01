#!/usr/bin/env python3
# ===========================================================================
# AS-RUN PROVENANCE COPY — frozen 2026-07-31 (decision-surface audit gap 3:
# the seven-domain external-read runners were previously uncommitted).
# Source:        /mnt/v/output/zensim/hdr-dmean-2026-07-29/analyze.py
# sha256(source): 5dc4401343f08940ed1f89a13f2cca043186e40bda007896047df390706bd31d
# build_commit:  c4632d6257b14e7647cd6daf9e846733b4bffec8
# Protocol doc:  benchmarks/hdr_dmean_commensurability_2026-07-29.md
# Everything below the marker line is BYTE-IDENTICAL to the source file
# (verify: strip through the marker, sha256 the rest). Do NOT extend this
# file — it is an archival record of the exact as-run analysis (it may call
# scipy directly; it predates the stats-rule batch migration and is kept
# verbatim). New analyses go through scripts/external_reads/run_external_reads.py.
# Data paths inside are as-run: /mnt/v artifact paths persist; ~/tmp
# FEATS_DIR-style inputs were session scratch — the pooled/persisted tables
# in the artifact dir are the stored equivalents (see ../README.md).
# ===== byte-identical source below this line ================================
"""HDR cross-route commensurability study — analysis per PROTOCOL.md.

Inputs (produced at zensim c4632d6257b1):
  upiq_hdr_956.csv / upiq_hdr_944.csv  (380 HDR pairs, declared-HDR route)
  upiq_sdr_956.csv                     (3,779 SDR pairs, SDR route)
  upiq_hdr924_control.csv              (chunk-2 V4 control scorer rerun)
  v3_ladder/{sdr,hdr}_feats.csv        (90-pair ladder, both routes, 956)

Outputs: results.json + printed report (tee to log).
Everything is deterministic; the only fitted objects are the three
registered ridge probe heads (train = UPIQ SDR only).
"""
import csv
import json
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

BASE = "/mnt/v/output/zensim/hdr-dmean-2026-07-29"
OUT = {}

def load_feat_csv(path):
    rows = []
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        for rec in rdr:
            rows.append(rec)
    return header, rows

def col_idx(header, name):
    return header.index(name)

# ---------- load ----------
h956, r956 = load_feat_csv(f"{BASE}/upiq_hdr_956.csv")
h944, r944 = load_feat_csv(f"{BASE}/upiq_hdr_944.csv")
hsdr, rsdr = load_feat_csv(f"{BASE}/upiq_sdr_956.csv")
assert len(r956) == 380 and len(r944) == 380, (len(r956), len(r944))
assert len(rsdr) == 3779, len(rsdr)

# control scores
ctrl = {}
with open(f"{BASE}/upiq_hdr924_control.csv", newline="") as f:
    for row in csv.DictReader(f):
        ctrl[row["condition_id"]] = row["zensim_hdr924"]

# ---------- gates ----------
gates = {}
# row identity + first-944 string (== bit) equality + score equality
nfeat_off = 5  # condition_id,dataset,is_hdr,jod,score228
mismatch_944 = 0
mismatch_score = 0
mismatch_ctrl = 0
for a, b in zip(r956, r944):
    assert a[0] == b[0], (a[0], b[0])
    if a[4] != b[4]:
        mismatch_score += 1
    if a[nfeat_off:nfeat_off + 944] != b[nfeat_off:nfeat_off + 944]:
        mismatch_944 += 1
    if ctrl.get(a[0]) != a[4]:
        mismatch_ctrl += 1
gates["first944_rows_equal"] = 380 - mismatch_944
gates["score228_rows_equal_944_956"] = 380 - mismatch_score
gates["score228_rows_equal_vs_control"] = 380 - mismatch_ctrl
OUT["gates"] = gates
print("GATES:", gates)

# ---------- numpy views ----------
def to_np(header, rows):
    ds = np.array([r[1] for r in rows])
    jod = np.array([float(r[3]) for r in rows])
    score = np.array([float(r[4]) for r in rows])
    X = np.array([[float(v) for v in r[nfeat_off:]] for r in rows])
    cid = [r[0] for r in rows]
    return cid, ds, jod, score, X

cid_h, ds_h, jod_h, score_h, X_h = to_np(h956, r956)     # HDR 956
cid_s, ds_s, jod_s, score_s, X_s = to_np(hsdr, rsdr)     # SDR 956
nar = ds_h == "narwaria"
kor = ds_h == "korshunov"
assert nar.sum() == 140 and kor.sum() == 240

def sr(a, b):
    return float(spearmanr(a, b).statistic)

# ---------- Q1 baseline: readout decomposition ----------
q1_base = {
    "pooled": sr(score_h, jod_h),
    "narwaria": sr(score_h[nar], jod_h[nar]),
    "korshunov": sr(score_h[kor], jod_h[kor]),
    "sdr_pooled": sr(score_s, jod_s),
    "sdr_tid": sr(score_s[ds_s == "tid2013"], jod_s[ds_s == "tid2013"]),
    "sdr_live": sr(score_s[ds_s == "live"], jod_s[ds_s == "live"]),
}
OUT["q1_readout"] = q1_base
print("\nQ1 readout (score228 vs JOD):", json.dumps(q1_base, indent=1))

# ---------- Q1 lanes: zero-fit per-lane table ----------
LOCAL = {"DMEAN": (13, 0), "CGAIN": (14, 1), "CLOSS": (15, 2)}
lane_rows = []
for s in range(4):
    for name, (lu, lw) in LOCAL.items():
        iu = 720 + s * 51 + 17 + lu
        iw = 944 + s * 3 + lw
        for tag, idx in (("unw", iu), ("wgt", iw)):
            v = X_h[:, idx]
            row = {
                "scale": s, "lane": name, "kind": tag, "fidx": idx,
                "std_pooled": float(v.std()),
                "std_nar": float(v[nar].std()),
                "std_kor": float(v[kor].std()),
                "srocc_pooled": sr(v, jod_h),
                "srocc_nar": sr(v[nar], jod_h[nar]),
                "srocc_kor": sr(v[kor], jod_h[kor]),
            }
            row["abs_pooled"] = abs(row["srocc_pooled"])
            row["abs_nar"] = abs(row["srocc_nar"])
            row["abs_kor"] = abs(row["srocc_kor"])
            row["gap"] = (row["abs_nar"] + row["abs_kor"]) / 2 - row["abs_pooled"]
            lane_rows.append(row)
OUT["q1_lanes"] = lane_rows
print("\nQ1 lane table (|SROCC| vs JOD; u=unweighted twin, w=weighted):")
print("scale lane   kind  std_pool   |S|pool  |S|nar   |S|kor   gap")
for r in lane_rows:
    print(f"s{r['scale']}   {r['lane']}  {r['kind']}  {r['std_pooled']:.2e}  "
          f"{r['abs_pooled']:.4f}  {r['abs_nar']:.4f}  {r['abs_kor']:.4f}  {r['gap']:+.4f}")

# ---------- Q3: registered ridge transfer probes ----------
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

# merged content groups (tid<->live repeats join)
cond_meta = {}
with open("/mnt/v/datasets/upiq/upiq_subjective_scores.csv", newline="") as f:
    for row in csv.DictReader(f):
        cond_meta[row["condition_id"]] = row

parent = {}
def find(x):
    parent.setdefault(x, x)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
def union(a, b):
    parent[find(a)] = find(b)

for c in cid_s:
    m = cond_meta[c]
    key = f"{m['dataset']}:{m['content_id']}"
    find(key)
    rep = m["repeating_content_id"].strip()
    if rep and rep != "-":
        # rep like 'l-05' names the OTHER dataset's content: l-* = live, t-* = tid2013
        other_ds = "live" if rep.startswith("l-") else "tid2013"
        other_ct = ("l-i" + rep[2:]) if rep.startswith("l-") else ("t-i" + rep[2:])
        union(key, f"{other_ds}:{other_ct}")

groups = np.array([find(f"{cond_meta[c]['dataset']}:{cond_meta[c]['content_id']}")
                   for c in cid_s])
OUT["q3_n_groups"] = int(len(set(groups)))
print(f"\nQ3: {len(set(groups))} merged content groups on the SDR train side")

DMEAN_COLS = [944 + s * 3 + 0 for s in range(4)]
FEATURE_SETS = {
    "944": list(range(944)),
    "956": list(range(956)),
    "944+DMEAN": list(range(944)) + DMEAN_COLS,
}
LAMBDAS = [1e-2, 1e-1, 1.0, 10.0, 100.0]

def fit_eval(cols):
    Xtr_full = X_s[:, cols]
    keep = Xtr_full.std(axis=0) > 1e-12
    Xtr = Xtr_full[:, keep]
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
    Ztr = (Xtr - mu) / sd
    y = jod_s
    gkf = GroupKFold(n_splits=5)
    cv = {}
    for lam in LAMBDAS:
        scores = []
        for tr, te in gkf.split(Ztr, y, groups):
            m = Ridge(alpha=lam)
            m.fit(Ztr[tr], y[tr])
            scores.append(sr(m.predict(Ztr[te]), y[te]))
        cv[lam] = float(np.mean(scores))
    lam = max(cv, key=cv.get)
    model = Ridge(alpha=lam)
    model.fit(Ztr, y)
    # ONE registered HDR look
    Xev = X_h[:, cols][:, keep]
    Zev = (Xev - mu) / sd
    pred = model.predict(Zev)
    res = {
        "n_cols_kept": int(keep.sum()),
        "lambda": lam,
        "cv_srocc_by_lambda": cv,
        "sdr_cv_srocc": cv[lam],
        "hdr_pooled": sr(pred, jod_h),
        "hdr_narwaria": sr(pred[nar], jod_h[nar]),
        "hdr_korshunov": sr(pred[kor], jod_h[kor]),
    }
    return res, (model, mu, sd, keep, cols)

q3 = {}
heads = {}
for name, cols in FEATURE_SETS.items():
    q3[name], heads[name] = fit_eval(cols)
    r = q3[name]
    print(f"Q3 [{name}] kept={r['n_cols_kept']} lam={r['lambda']} "
          f"sdrCV={r['sdr_cv_srocc']:.4f} | HDR pooled={r['hdr_pooled']:.4f} "
          f"nar={r['hdr_narwaria']:.4f} kor={r['hdr_korshunov']:.4f}")
OUT["q3_heads"] = q3

# ---------- Q2: V3 ladder ----------
hl, rl_sdr = load_feat_csv(f"{BASE}/v3_ladder/sdr_feats.csv")
_, rl_hdr = load_feat_csv(f"{BASE}/v3_ladder/hdr_feats.csv")
lad_off = 5  # pair_idx,ref_idx,kind,level,score228
L_s = np.array([[float(v) for v in r[lad_off:]] for r in rl_sdr])
L_h = np.array([[float(v) for v in r[lad_off:]] for r in rl_hdr])
lad_score_s = np.array([float(r[4]) for r in rl_sdr])
lad_score_h = np.array([float(r[4]) for r in rl_hdr])
refidx = np.array([int(r[1]) for r in rl_sdr])
n_refs = refidx.max() + 1

def xroute_stats(a, b):
    pooled = sr(a, b)
    wr = [sr(a[refidx == r], b[refidx == r]) for r in range(n_refs)]
    return pooled, float(np.mean(wr)), float(np.min(wr))

q2 = {}
q2["readout_fixed"] = dict(zip(("pooled", "within_ref_mean", "within_ref_min"),
                               xroute_stats(lad_score_s, lad_score_h)))

# per-lane cross-route SROCC (verify vs the printed G1 table)
lane_x = []
for s in range(4):
    for name, (lu, lw) in LOCAL.items():
        iu = 720 + s * 51 + 17 + lu
        iw = 944 + s * 3 + lw
        lane_x.append({
            "scale": s, "lane": name,
            "unw": sr(L_s[:, iu], L_h[:, iu]),
            "wgt": sr(L_s[:, iw], L_h[:, iw]),
            "std_unw_sdr": float(L_s[:, iu].std()),
            "std_wgt_sdr": float(L_s[:, iw].std()),
        })
q2["lanes"] = lane_x

# feature-set-level: DMEAN-channel best-available per scale + live-lane mean
def set_stats(use_weighted_dmean, use_weighted_all):
    per_scale = []
    vals = []
    for s in range(4):
        row = next(r for r in lane_x if r["scale"] == s and r["lane"] == "DMEAN")
        best = row["wgt"] if use_weighted_dmean else row["unw"]
        per_scale.append(best)
    for r in lane_x:
        vals.append(r["unw"])
        if use_weighted_all:
            vals.append(r["wgt"])
        elif use_weighted_dmean and r["lane"] == "DMEAN":
            vals.append(r["wgt"])
    return per_scale, float(np.mean(vals))

for name, (wd, wa) in (("944", (False, False)), ("956", (True, True)),
                       ("944+DMEAN", (True, False))):
    ps, mv = set_stats(wd, wa)
    q2[f"set_{name}"] = {"dmean_channel_per_scale": ps, "mean_over_lanes": mv}

# probe heads on the ladder (cross-route score consistency per feature set)
for name, (model, mu, sd, keep, cols) in heads.items():
    Za = (L_s[:, cols][:, keep] - mu) / sd
    Zb = (L_h[:, cols][:, keep] - mu) / sd
    pa, pb = model.predict(Za), model.predict(Zb)
    q2[f"head_{name}"] = dict(zip(("pooled", "within_ref_mean", "within_ref_min"),
                                  xroute_stats(pa, pb)))
    print(f"Q2 head [{name}]: xroute pooled={q2[f'head_{name}']['pooled']:.4f} "
          f"within-ref mean={q2[f'head_{name}']['within_ref_mean']:.4f} "
          f"min={q2[f'head_{name}']['within_ref_min']:.4f}")
OUT["q2"] = q2

with open(f"{BASE}/results.json", "w") as f:
    json.dump(OUT, f, indent=1)
print("\nwrote results.json")
