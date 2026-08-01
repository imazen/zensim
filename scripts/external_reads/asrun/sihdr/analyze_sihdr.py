#!/usr/bin/env python3
# ===========================================================================
# AS-RUN PROVENANCE COPY — frozen 2026-07-31 (decision-surface audit gap 3:
# the seven-domain external-read runners were previously uncommitted).
# Source:        /mnt/v/output/zensim/sihdr-transfer-2026-07-29/analyze_sihdr.py
# sha256(source): 0502c6bb82d9434f43a891f3341d1cca828aa9d1a4f6e8e99fa9c1cdcbc2d074
# build_commit:  34cbd9cf03673c48d69127b7c648bc2fd7d95adc
# Protocol doc:  benchmarks/sihdr_transfer_2026-07-29.md
# Everything below the marker line is BYTE-IDENTICAL to the source file
# (verify: strip through the marker, sha256 the rest). Do NOT extend this
# file — it is an archival record of the exact as-run analysis (it may call
# scipy directly; it predates the stats-rule batch migration and is kept
# verbatim). New analyses go through scripts/external_reads/run_external_reads.py.
# Data paths inside are as-run: /mnt/v artifact paths persist; ~/tmp
# FEATS_DIR-style inputs were session scratch — the pooled/persisted tables
# in the artifact dir are the stored equivalents (see ../README.md).
# ===== byte-identical source below this line ================================
"""SI-HDR transfer study — analysis per PROTOCOL.md (pre-registered).

Inputs:
  sihdr_feats_944.csv     (this study; 2,172-pair extraction, mode 944)
  rmse_labeled.csv        (trivial comparator: display-nits RMSE, labeled rows)
  experiment_results.csv  (SI-HDR JOD labels)
  ../hdr-dmean-2026-07-29/upiq_sdr_956.csv   (UPIQ SDR train, first-944)
  ../hdr-dmean-2026-07-29/upiq_hdr_944.csv   (UPIQ HDR eval, 380)
  ../hdr-dmean-2026-07-29/results.json       (recorded baselines, gate targets)

Registered looks: L1 zero-shot, L2 within-SI-HDR scene-disjoint CV,
L3 cross-set mass addition; Q3 diagnostic family ablation.
Everything deterministic; seed 20260729 for all bootstraps.
"""
import csv
import json
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

BASE = "/mnt/v/output/zensim/sihdr-transfer-2026-07-29"
PRIOR = "/mnt/v/output/zensim/hdr-dmean-2026-07-29"
SEED = 20260729
LAMBDAS = [1e-2, 1e-1, 1.0, 10.0, 100.0]
OUT = {}

def sr(a, b):
    return float(spearmanr(a, b).statistic)

def load_feat_csv(path):
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        rows = list(rdr)
    return header, rows

# ---------- SI-HDR features ----------
h_si, r_si = load_feat_csv(f"{BASE}/sihdr_feats_944.csv")
si_off = h_si.index("f0")
assert h_si[si_off:] == [f"f{k}" for k in range(944)]
si_cid = [r[0] for r in r_si]
si_clip = np.array([r[1] for r in r_si])
si_score228 = np.array([float(r[h_si.index("score228")]) for r in r_si])
X_si_all = np.array([[float(v) for v in r[si_off:]] for r in r_si])
si_scene = np.array([c.split("-")[0] for c in si_cid])   # 'i015'
si_method = np.array([c.split("-")[2] for c in si_cid])
print(f"SI-HDR extraction: {len(r_si)} rows x {X_si_all.shape[1]} features; "
      f"{len(set(si_scene))} scenes")
OUT["coverage"] = {
    "extracted_pairs": len(r_si),
    "scenes": len(set(si_scene)),
    "crops_applied": int(sum(1 for r in r_si if r[h_si.index("crop_x")] != "0")),
    "nonfinite_pixel_rows": int(sum(1 for r in r_si
                                    if r[h_si.index("nonfinite_ref")] != "0"
                                    or r[h_si.index("nonfinite_dist")] != "0")),
}

# ---------- labels ----------
lab = {}
with open("/mnt/v/datasets/si-hdr/experiment_results/experiment_results.csv",
          newline="") as f:
    for row in csv.DictReader(f):
        if row["scene"] == "all" or row["method"] in ("input", "original"):
            continue
        lab[(row["image"], row["clip_level"], row["method"])] = float(row["jod"])
assert len(lab) == 324, len(lab)

lab_idx, lab_jod = [], []
for i, c in enumerate(si_cid):
    scene, clip, method = c.split("-")
    key = (scene, clip, method)
    if key in lab:
        lab_idx.append(i)
        lab_jod.append(lab[key])
lab_idx = np.array(lab_idx)
jod_si = np.array(lab_jod)
X_si = X_si_all[lab_idx]
sc_si = si_scene[lab_idx]
cl_si = si_clip[lab_idx]
me_si = si_method[lab_idx]
s228_si = si_score228[lab_idx]
cid_si = [si_cid[i] for i in lab_idx]
OUT["coverage"]["labeled_joined"] = int(len(lab_idx))
print(f"labeled rows joined: {len(lab_idx)} / 324 expected")
missing = [k for k in lab if k not in
           {tuple(si_cid[i].split("-")) for i in lab_idx}]
if missing:
    print("MISSING labeled pairs:", missing)
OUT["coverage"]["missing_labeled"] = ["-".join(m) for m in missing]

# trivial comparator (display-nits RMSE, labeled rows)
rmse_map = {}
try:
    with open(f"{BASE}/rmse_labeled.csv", newline="") as f:
        for row in csv.DictReader(f):
            rmse_map[row["cid"]] = float(row["rmse"])
except FileNotFoundError:
    print("WARNING: rmse_labeled.csv missing — comparator (e) skipped")
neg_rmse = np.array([-rmse_map.get(c, np.nan) for c in cid_si])

# ---------- UPIQ side (reused verbatim) ----------
hs, rs = load_feat_csv(f"{PRIOR}/upiq_sdr_956.csv")
hh, rh = load_feat_csv(f"{PRIOR}/upiq_hdr_944.csv")
nfeat_off = 5
assert len(rs) == 3779 and len(rh) == 380
cid_s = [r[0] for r in rs]
jod_s = np.array([float(r[3]) for r in rs])
X_s = np.array([[float(v) for v in r[nfeat_off:nfeat_off + 944]] for r in rs])
ds_h = np.array([r[1] for r in rh])
jod_h = np.array([float(r[3]) for r in rh])
X_h = np.array([[float(v) for v in r[nfeat_off:]] for r in rh])
assert X_h.shape[1] == 944
nar = ds_h == "narwaria"
kor = ds_h == "korshunov"
assert nar.sum() == 140 and kor.sum() == 240

recorded = json.load(open(f"{PRIOR}/results.json"))["q3_heads"]["944"]

# UPIQ content groups (prior analyze.py verbatim)
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
        other_ds = "live" if rep.startswith("l-") else "tid2013"
        other_ct = ("l-i" + rep[2:]) if rep.startswith("l-") else ("t-i" + rep[2:])
        union(key, f"{other_ds}:{other_ct}")

groups_upiq = np.array([find(f"{cond_meta[c]['dataset']}:{cond_meta[c]['content_id']}")
                        for c in cid_s])
print(f"UPIQ SDR groups: {len(set(groups_upiq))}")

# ---------- probe machinery (prior study verbatim) ----------
def fit_ridge(Xtr, ytr, groups, lambdas=LAMBDAS):
    keep = Xtr.std(axis=0) > 1e-12
    Xk = Xtr[:, keep]
    mu, sd = Xk.mean(axis=0), Xk.std(axis=0)
    Z = (Xk - mu) / sd
    gkf = GroupKFold(n_splits=5)
    cv = {}
    for lam in lambdas:
        scores = []
        for tr, te in gkf.split(Z, ytr, groups):
            m = Ridge(alpha=lam)
            m.fit(Z[tr], ytr[tr])
            scores.append(sr(m.predict(Z[te]), ytr[te]))
        cv[lam] = float(np.mean(scores))
    lam = max(cv, key=cv.get)
    model = Ridge(alpha=lam)
    model.fit(Z, ytr)
    return model, mu, sd, keep, lam, cv

def predict(head, X):
    model, mu, sd, keep, _, _ = head
    return model.predict((X[:, keep] - mu) / sd)

# ---------- breakdowns ----------
def breakdown(pred, jod, scenes, clips, methods, tag):
    res = {"pooled": sr(pred, jod)}
    blocks = defaultdict(list)
    for i, (s, c) in enumerate(zip(scenes, clips)):
        blocks[(s, c)].append(i)
    bvals = []
    for k, idxs in sorted(blocks.items()):
        if len(idxs) >= 4:
            bvals.append(sr(pred[idxs], jod[idxs]))
    res["block_srocc_mean"] = float(np.mean(bvals))
    res["block_srocc_median"] = float(np.median(bvals))
    res["block_srocc_min"] = float(np.min(bvals))
    res["n_blocks"] = len(bvals)
    per_scene = defaultdict(list)
    for i, s in enumerate(scenes):
        per_scene[s].append(i)
    svals = [sr(pred[idxs], jod[idxs]) for idxs in per_scene.values()]
    res["scene_srocc_mean"] = float(np.mean(svals))
    res["per_method"] = {}
    for m in sorted(set(methods)):
        sel = methods == m
        res["per_method"][m] = sr(pred[sel], jod[sel])
    res["per_clip"] = {}
    for c in sorted(set(clips)):
        sel = clips == c
        res["per_clip"][c] = sr(pred[sel], jod[sel])
    print(f"  [{tag}] pooled={res['pooled']:.4f} block_mean={res['block_srocc_mean']:.4f} "
          f"scene_mean={res['scene_srocc_mean']:.4f}")
    print(f"     per-method: " + " ".join(f"{m}={v:.3f}" for m, v in res["per_method"].items()))
    print(f"     per-clip: " + " ".join(f"{c}={v:.3f}" for c, v in res["per_clip"].items()))
    return res

def scene_bootstrap_ci(pred, jod, scenes, n=10000):
    rng = np.random.default_rng(SEED)
    uniq = sorted(set(scenes))
    by_scene = {s: np.where(scenes == s)[0] for s in uniq}
    vals = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by_scene[s] for s in pick])
        vals.append(sr(pred[idx], jod[idx]))
    vals = np.sort(vals)
    return [float(vals[int(0.025 * n)]), float(vals[int(0.975 * n)])]

# ================= GATE + L1 =================
print("\n=== L1: zero-shot (the exact prior 944 probe) ===")
head944 = fit_ridge(X_s, jod_s, groups_upiq)
lam_sel = head944[4]
pred_h = predict(head944, X_h)
g1 = {
    "lambda": lam_sel,
    "hdr_pooled": sr(pred_h, jod_h),
    "hdr_narwaria": sr(pred_h[nar], jod_h[nar]),
    "hdr_korshunov": sr(pred_h[kor], jod_h[kor]),
    "recorded": {k: recorded[k] for k in ("hdr_pooled", "hdr_narwaria", "hdr_korshunov")},
}
ok = (abs(g1["hdr_pooled"] - recorded["hdr_pooled"]) < 1e-6
      and abs(g1["hdr_narwaria"] - recorded["hdr_narwaria"]) < 1e-6
      and abs(g1["hdr_korshunov"] - recorded["hdr_korshunov"]) < 1e-6
      and lam_sel == recorded["lambda"])
g1["gate_reproduces_recorded"] = bool(ok)
OUT["gate_g1"] = g1
print(f"G1 gate: refit λ={lam_sel} UPIQ-HDR pooled={g1['hdr_pooled']:.6f} "
      f"nar={g1['hdr_narwaria']:.6f} kor={g1['hdr_korshunov']:.6f} -> "
      f"{'PASS' if ok else 'FAIL'}")
if not ok:
    print("GATE G1 FAILED — stopping before any SI-HDR look")
    json.dump(OUT, open(f"{BASE}/results.json", "w"), indent=1)
    sys.exit(1)

pred_si = predict(head944, X_si)
l1 = {"probe944": breakdown(pred_si, jod_si, sc_si, cl_si, me_si, "L1 probe944")}
l1["probe944"]["pooled_ci95_scene_bootstrap"] = scene_bootstrap_ci(pred_si, jod_si, sc_si)
l1["score228"] = breakdown(s228_si, jod_si, sc_si, cl_si, me_si, "L1 score228")
if not np.isnan(neg_rmse).any():
    l1["neg_rmse_nits"] = breakdown(neg_rmse, jod_si, sc_si, cl_si, me_si, "L1 -RMSE")
print(f"  L1 pooled CI95 (scene bootstrap): {l1['probe944']['pooled_ci95_scene_bootstrap']}")
OUT["l1"] = l1

# ================= L2: within-SI-HDR scene-disjoint CV =================
print("\n=== L2: within-SI-HDR nested scene-disjoint CV ===")
def nested_cv_oof(X, y, scene_groups, col_idx=None, tag=""):
    Xc = X if col_idx is None else X[:, col_idx]
    oof = np.full(len(y), np.nan)
    lam_picks = []
    outer = GroupKFold(n_splits=5)
    for tr, te in outer.split(Xc, y, scene_groups):
        keep = Xc[tr].std(axis=0) > 1e-12
        Xk = Xc[:, keep]
        mu, sd = Xk[tr].mean(axis=0), Xk[tr].std(axis=0)
        Ztr, Zte = (Xk[tr] - mu) / sd, (Xk[te] - mu) / sd
        inner = GroupKFold(n_splits=4)
        cv = {}
        for lam in LAMBDAS:
            scores = []
            for itr, ite in inner.split(Ztr, y[tr], scene_groups[tr]):
                m = Ridge(alpha=lam)
                m.fit(Ztr[itr], y[tr][itr])
                scores.append(sr(m.predict(Ztr[ite]), y[tr][ite]))
            cv[lam] = float(np.mean(scores))
        lam = max(cv, key=cv.get)
        lam_picks.append(lam)
        m = Ridge(alpha=lam)
        m.fit(Ztr, y[tr])
        oof[te] = m.predict(Zte)
    assert not np.isnan(oof).any()
    return oof, lam_picks

oof, lam_picks = nested_cv_oof(X_si, jod_si, sc_si)
l2 = {"probe": breakdown(oof, jod_si, sc_si, cl_si, me_si, "L2 OOF")}
l2["probe"]["lambda_picks"] = lam_picks
l2["probe"]["pooled_ci95_scene_bootstrap"] = scene_bootstrap_ci(oof, jod_si, sc_si)
print(f"  L2 λ picks per outer fold: {lam_picks}")
print(f"  L2 pooled CI95: {l2['probe']['pooled_ci95_scene_bootstrap']}")
OUT["l2"] = l2

# ================= L3: cross-set mass addition =================
print("\n=== L3: UPIQ-SDR + SI-HDR mass addition ===")
def zsc(v):
    return (v - v.mean()) / v.std()

y_upiq_z = zsc(jod_s)
y_si_z = zsc(jod_si)

# baseline leg (UPIQ only, z-scored target) — gate G3
head_base = fit_ridge(X_s, y_upiq_z, groups_upiq)
pred_base = predict(head_base, X_h)
base = {
    "lambda": head_base[4],
    "pooled": sr(pred_base, jod_h),
    "narwaria": sr(pred_base[nar], jod_h[nar]),
    "korshunov": sr(pred_base[kor], jod_h[kor]),
}
ok3 = (abs(base["pooled"] - recorded["hdr_pooled"]) < 1e-6
       and abs(base["narwaria"] - recorded["hdr_narwaria"]) < 1e-6
       and abs(base["korshunov"] - recorded["hdr_korshunov"]) < 1e-6)
base["gate_reproduces_recorded"] = bool(ok3)
print(f"G3 baseline (z-target): λ={base['lambda']} pooled={base['pooled']:.6f} "
      f"nar={base['narwaria']:.6f} kor={base['korshunov']:.6f} -> "
      f"{'PASS' if ok3 else 'FAIL'}")
OUT["l3_baseline"] = base
if not ok3:
    print("GATE G3 FAILED — stopping")
    json.dump(OUT, open(f"{BASE}/results.json", "w"), indent=1)
    sys.exit(1)

X_mix = np.vstack([X_s, X_si])
y_mix = np.concatenate([y_upiq_z, y_si_z])
groups_mix = np.concatenate([groups_upiq, np.array([f"sihdr:{s}" for s in sc_si])])
head_mix = fit_ridge(X_mix, y_mix, groups_mix)
pred_mix = predict(head_mix, X_h)
mix = {
    "lambda": head_mix[4],
    "n_cols_kept": int(head_mix[3].sum()),
    "cv_srocc_by_lambda": {str(k): v for k, v in head_mix[5].items()},
    "n_train": int(len(y_mix)),
    "pooled": sr(pred_mix, jod_h),
    "narwaria": sr(pred_mix[nar], jod_h[nar]),
    "korshunov": sr(pred_mix[kor], jod_h[kor]),
}
print(f"L3 mixed: λ={mix['lambda']} kept={mix['n_cols_kept']} "
      f"pooled={mix['pooled']:.4f} nar={mix['narwaria']:.4f} kor={mix['korshunov']:.4f}")

# paired bootstrap on the delta, per stratum
rng = np.random.default_rng(SEED)
def paired_boot(mask):
    idx = np.where(mask)[0]
    deltas = []
    for _ in range(10000):
        pick = rng.choice(idx, size=len(idx), replace=True)
        deltas.append(sr(pred_mix[pick], jod_h[pick]) - sr(pred_base[pick], jod_h[pick]))
    deltas = np.array(deltas)
    return {
        "delta_point": None,  # filled below
        "delta_boot_mean": float(deltas.mean()),
        "p_le_0": float((deltas <= 0).mean()),
        "p_ge_0": float((deltas >= 0).mean()),
        "ci95": [float(np.quantile(deltas, 0.025)), float(np.quantile(deltas, 0.975))],
    }

mix["delta"] = {}
for name, mask in (("narwaria", nar), ("korshunov", kor), ("pooled", np.ones(len(jod_h), bool))):
    d = paired_boot(mask)
    d["delta_point"] = float(
        (sr(pred_mix[mask], jod_h[mask])) - (sr(pred_base[mask], jod_h[mask])))
    mix["delta"][name] = d
    print(f"  Δ{name} = {d['delta_point']:+.4f}  boot mean {d['delta_boot_mean']:+.4f} "
          f"CI95 [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]  "
          f"p(Δ<=0)={d['p_le_0']:.4f} p(Δ>=0)={d['p_ge_0']:.4f}")

dn, dk = mix["delta"]["narwaria"]["delta_point"], mix["delta"]["korshunov"]["delta_point"]
if (dn >= 0.02 and dk >= -0.005) or (dk >= 0.02 and dn >= -0.005):
    verdict = "EXTENDS (registered claim rule fired positive)"
elif (dn <= -0.02 and dk <= 0.005) or (dk <= -0.02 and dn <= 0.005):
    verdict = "HARMS (mirrored negative rule fired)"
else:
    verdict = "NO MEASURABLE EXTENSION at n=324 mass"
mix["registered_verdict"] = verdict
print(f"  L3 verdict: {verdict}")
OUT["l3_mixed"] = mix

# ================= Q3: diagnostic family ablation (L2 harness) =================
print("\n=== Q3: family ablation (scene-disjoint OOF SROCC, diagnostic) ===")
FAMS = {
    "full_944": list(range(944)),
    "minus_folded720": list(range(720, 944)),
    "minus_append": list(range(0, 720)) + list(range(924, 944)),
    "minus_append2": list(range(0, 924)),
    "only_folded720": list(range(0, 720)),
    "only_append": list(range(720, 924)),
    "only_append2": list(range(924, 944)),
    "only_bandvis8": [924 + 5 * s + k for s in range(4) for k in (0, 1)],
}
q3 = {}
full_pooled = None
for name, cols in FAMS.items():
    o, lp = nested_cv_oof(X_si, jod_si, sc_si, col_idx=np.array(cols), tag=name)
    v = sr(o, jod_si)
    q3[name] = {"oof_pooled_srocc": v, "n_cols": len(cols), "lambda_picks": lp}
    if name == "full_944":
        full_pooled = v
    q3[name]["delta_vs_full"] = v - full_pooled if full_pooled is not None else 0.0
    print(f"  {name:18s} n={len(cols):3d} OOF SROCC={v:.4f} Δ={q3[name]['delta_vs_full']:+.4f}")
OUT["q3_families"] = q3

# zero-fit append2 lane table on the 324 labeled rows
lanes = []
A2 = ["BANDVIS_GAIN", "BANDVIS_LOSS", "a2_local2", "a2_local3", "a2_local4"]
for s in range(4):
    for k in range(5):
        idx = 924 + 5 * s + k
        v = X_si[:, idx]
        lanes.append({
            "fidx": idx, "scale": s, "lane": A2[k],
            "std": float(v.std()),
            "srocc": sr(v, jod_si) if v.std() > 1e-12 else None,
        })
OUT["q3_append2_lanes_zerofit"] = lanes
print("  append2 lanes zero-fit |SROCC| (std): " + ", ".join(
    f"f{r['fidx']}:{'dead' if r['srocc'] is None else format(abs(r['srocc']), '.3f')}"
    f"({r['std']:.1e})" for r in lanes))

json.dump(OUT, open(f"{BASE}/results.json", "w"), indent=1)
print(f"\nwrote {BASE}/results.json")
