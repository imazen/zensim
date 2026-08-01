#!/usr/bin/env python3
# ===========================================================================
# AS-RUN PROVENANCE COPY — frozen 2026-07-31 (decision-surface audit gap 3:
# the seven-domain external-read runners were previously uncommitted).
# Source:        /mnt/v/output/zensim/hdrvdc-conditions-2026-07-29/analyze_hdrvdc.py
# sha256(source): 0cf44170758a4335078e40c1c7e79e97d7e224d5e3088e492925e52e77b71652
# build_commit:  6b3505a57174
# Protocol doc:  benchmarks/hdrvdc_conditions_2026-07-29.md
# Everything below the marker line is BYTE-IDENTICAL to the source file
# (verify: strip through the marker, sha256 the rest). Do NOT extend this
# file — it is an archival record of the exact as-run analysis (it may call
# scipy directly; it predates the stats-rule batch migration and is kept
# verbatim). New analyses go through scripts/external_reads/run_external_reads.py.
# Data paths inside are as-run: /mnt/v artifact paths persist; ~/tmp
# FEATS_DIR-style inputs were session scratch — the pooled/persisted tables
# in the artifact dir are the stored equivalents (see ../README.md).
# ===== byte-identical source below this line ================================
"""HDR-VDC viewing-condition study — analysis per PROTOCOL.md.

Inputs:
  ~/tmp/hdrvdc-work/feats/content_NN.csv   (per-content 944 extractions,
      5 configs x 8 frames per distorted video, build 6b3505a5)
  /mnt/v/datasets/hdr-vdc/HDR_VDC_JOD_Scores.csv (labels; 464 distorted rows)
  /mnt/v/output/zensim/hdr-dmean-2026-07-29/upiq_sdr_956.csv (probe train)
  /mnt/v/output/zensim/hdr-dmean-2026-07-29/upiq_hdr_944.csv (gate eval)
  /mnt/v/output/zensim/hdr-dmean-2026-07-29/results.json     (gate values)

Outputs: results.json + printed report (tee to log) + hdrvdc_pooled_944.csv
(the per-(video,config) mean-pooled feature table, persisted as the study
asset). The only fitted objects: the registered UPIQ-SDR 944 ridge probe
(gated reconstruction) and Q3's scene-disjoint OOF heads (diagnostic).
Everything else is zero-fit. Bootstrap: 10k, seed 20260729.
"""
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

BASE = "/mnt/v/output/zensim/hdrvdc-conditions-2026-07-29"
FEATS_DIR = os.path.expanduser("~/tmp/hdrvdc-work/feats")
LABELS = "/mnt/v/datasets/hdr-vdc/HDR_VDC_JOD_Scores.csv"
DMEAN = "/mnt/v/output/zensim/hdr-dmean-2026-07-29"
SEED = 20260729
NBOOT = 10_000
NFRAMES = 8
CONFIGS = ["A", "B", "C", "D", "E"]
CONDS = [("bright", "near"), ("bright", "far"), ("dim", "near"), ("dim", "far")]
# leg -> {(lum, dist) -> config}
LEGS = {
    "i": {c: "A" for c in CONDS},
    "ii": {("bright", "near"): "B", ("bright", "far"): "B",
           ("dim", "near"): "C", ("dim", "far"): "C"},
    "iii": {("bright", "near"): "B", ("bright", "far"): "D",
            ("dim", "near"): "C", ("dim", "far"): "E"},
}
OUT = {}


def sr(a, b):
    return float(spearmanr(a, b).statistic)


# ---------------- load features (per-frame) and mean-pool ----------------
frame_feats = defaultdict(list)   # (cid, vid, cfg) -> list of (fidx, feats)
frame_s228 = defaultdict(list)
for path in sorted(glob.glob(f"{FEATS_DIR}/content_*.csv")):
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        assert header[:4] == ["key", "dim", "peak_nits", "score228"], header
        assert len(header) == 4 + 944
        for rec in rdr:
            cid, vid, fj, cfg = rec[0].split("|")
            fidx = int(fj[1:])
            frame_feats[(cid, vid, cfg)].append(
                (fidx, np.array([float(v) for v in rec[4:]])))
            frame_s228[(cid, vid, cfg)].append((fidx, float(rec[3])))

pooled_X = {}
pooled_s228 = {}
coverage = {"videos": set(), "n_frame_rows": 0, "incomplete": []}
for k, lst in frame_feats.items():
    lst.sort()
    if len(lst) != NFRAMES:
        coverage["incomplete"].append((k, len(lst)))
        continue
    pooled_X[k] = np.mean([v for _, v in lst], axis=0)
    pooled_s228[k] = float(np.mean([s for _, s in sorted(frame_s228[k])]))
    coverage["videos"].add((k[0], k[1]))
    coverage["n_frame_rows"] += len(lst)
assert not coverage["incomplete"], coverage["incomplete"][:5]
print(f"pooled: {len(pooled_X)} (video,config) vectors over "
      f"{len(coverage['videos'])} videos; {coverage['n_frame_rows']} frame rows")

# ---------------- labels ----------------
lab = list(csv.DictReader(open(LABELS)))
rows = []  # registered 464 distorted rows
for r in lab:
    if r["is_reference"] == "True":
        continue
    vid = f"{r['crf']}_{r['resolution']}"
    rows.append({
        "cid": r["content_id"], "vid": vid,
        "cond": (r["luminance_level"], r["viewing_distance"]),
        "jod": float(r["jod"]),
        "ci_width": float(r["jod_high"]) - float(r["jod_low"]),
    })
assert len(rows) == 464, len(rows)
vids_needed = {(r["cid"], r["vid"]) for r in rows}
missing = vids_needed - coverage["videos"]
assert not missing, f"missing extracted videos: {sorted(missing)[:5]}"
OUT["coverage"] = {
    "n_videos": len(coverage["videos"]),
    "n_frame_rows_pooled_over": coverage["n_frame_rows"],
    "n_label_rows": len(rows),
    "jod_ci_width_mean": float(np.mean([r["ci_width"] for r in rows])),
    "jod_ci_width_median": float(np.median([r["ci_width"] for r in rows])),
}

# ---------------- probe reconstruction (GATE before any HDR-VDC look) ----
def load_feat_csv(path):
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        return header, list(rdr)

nfeat_off = 5  # condition_id,dataset,is_hdr,jod,score228
hsdr, rsdr = load_feat_csv(f"{DMEAN}/upiq_sdr_956.csv")
h944, r944 = load_feat_csv(f"{DMEAN}/upiq_hdr_944.csv")
assert len(rsdr) == 3779 and len(r944) == 380
jod_s = np.array([float(r[3]) for r in rsdr])
X_s = np.array([[float(v) for v in r[nfeat_off:]] for r in rsdr])[:, :944]
cid_s = [r[0] for r in rsdr]
ds_h = np.array([r[1] for r in r944])
jod_h = np.array([float(r[3]) for r in r944])
X_h = np.array([[float(v) for v in r[nfeat_off:]] for r in r944])
nar, kor = ds_h == "narwaria", ds_h == "korshunov"

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
groups_s = np.array([find(f"{cond_meta[c]['dataset']}:{cond_meta[c]['content_id']}")
                     for c in cid_s])

keep = X_s.std(axis=0) > 1e-12
Xtr = X_s[:, keep]
mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
Ztr = (Xtr - mu) / sd
LAMBDAS = [1e-2, 1e-1, 1.0, 10.0, 100.0]
gkf = GroupKFold(n_splits=5)
cv = {}
for lam in LAMBDAS:
    scores = []
    for tr, te in gkf.split(Ztr, jod_s, groups_s):
        m = Ridge(alpha=lam)
        m.fit(Ztr[tr], jod_s[tr])
        scores.append(sr(m.predict(Ztr[te]), jod_s[te]))
    cv[lam] = float(np.mean(scores))
lam = max(cv, key=cv.get)
probe = Ridge(alpha=lam)
probe.fit(Ztr, jod_s)
pred_h = probe.predict((X_h[:, keep] - mu) / sd)
gate = {
    "n_cols_kept": int(keep.sum()), "lambda": lam,
    "sdr_cv_srocc": cv[lam],
    "hdr_pooled": sr(pred_h, jod_h),
    "hdr_narwaria": sr(pred_h[nar], jod_h[nar]),
    "hdr_korshunov": sr(pred_h[kor], jod_h[kor]),
}
rec = json.load(open(f"{DMEAN}/results.json"))["q3_heads"]["944"]
for k in gate:
    want = rec[k]
    got = gate[k]
    ok = (got == want) if isinstance(want, int) else abs(got - want) < 5e-7
    assert ok, f"GATE FAIL {k}: got {got} want {want}"
OUT["probe_gate"] = gate
print("PROBE GATE PASS:", json.dumps(gate, indent=1))


def probe_score(v):
    return float(probe.predict(((v[keep] - mu) / sd)[None, :])[0])


# ---------------- per-row scores for the three legs --------------------
for r in rows:
    for leg, m in LEGS.items():
        cfg = m[r["cond"]]
        r[f"probe_{leg}"] = probe_score(pooled_X[(r["cid"], r["vid"], cfg)])
        r[f"s228_{leg}"] = pooled_s228[(r["cid"], r["vid"], cfg)]

jod = np.array([r["jod"] for r in rows])
cids = np.array([r["cid"] for r in rows])
conds = [r["cond"] for r in rows]
uniq_cids = sorted(set(cids), key=int)
scores = {(sc, leg): np.array([r[f"{sc}_{leg}"] for r in rows])
          for sc in ("probe", "s228") for leg in LEGS}

# ---------------- Q1: within-condition ranking --------------------------
rng = np.random.default_rng(SEED)
boot_draws = [rng.choice(len(uniq_cids), size=len(uniq_cids), replace=True)
              for _ in range(NBOOT)]
cid_rows = {c: np.array([i for i, x in enumerate(cids) if x == c])
            for c in uniq_cids}

def cluster_pool(draw):
    return np.concatenate([cid_rows[uniq_cids[d]] for d in draw])

q1 = []
for cond in CONDS:
    mask = np.array([c == cond for c in conds])
    for sc in ("probe", "s228"):
        for leg in LEGS:
            v = scores[(sc, leg)][mask]
            jj = jod[mask]
            per_content = []
            for c in uniq_cids:
                cm = mask & (cids == c)
                per_content.append(sr(scores[(sc, leg)][cm], jod[cm]))
            # cluster bootstrap CI (resample contents within this condition)
            bs = []
            for draw in boot_draws[:2000]:   # 2k for per-cell CIs (cheap cells)
                idx = cluster_pool(draw)
                idx = idx[np.array([conds[i] == cond for i in idx])]
                bs.append(sr(scores[(sc, leg)][idx], jod[idx]))
            q1.append({
                "cond": "-".join(cond), "scorer": sc, "leg": leg,
                "srocc": sr(v, jj),
                "ci_lo": float(np.percentile(bs, 2.5)),
                "ci_hi": float(np.percentile(bs, 97.5)),
                "per_content_mean": float(np.mean(per_content)),
                "per_content_median": float(np.median(per_content)),
                "per_content_min": float(np.min(per_content)),
            })
OUT["q1"] = q1
print("\nQ1 within-condition SROCC (n=116/cond; per-content n=5..8 flagged small):")
print(f"{'cond':13s} {'scorer':6s} {'leg':4s} {'SROCC':>8s} {'[95% CI]':>18s} "
      f"{'pc_mean':>8s} {'pc_med':>8s} {'pc_min':>8s}")
for r in q1:
    print(f"{r['cond']:13s} {r['scorer']:6s} {r['leg']:4s} {r['srocc']:8.4f} "
          f"[{r['ci_lo']:7.4f},{r['ci_hi']:7.4f}] {r['per_content_mean']:8.4f} "
          f"{r['per_content_median']:8.4f} {r['per_content_min']:8.4f}")

# ---------------- Q2: cross-condition commensurability ------------------
def pooled_stats(sc, leg, row_filter=None):
    if row_filter is None:
        idx = np.arange(len(rows))
    else:
        idx = np.array([i for i in range(len(rows)) if row_filter(rows[i])])
    return sr(scores[(sc, leg)][idx], jod[idx])

def within_cond_mean(sc, leg):
    vals = []
    for cond in CONDS:
        mask = np.array([c == cond for c in conds])
        vals.append(sr(scores[(sc, leg)][mask], jod[mask]))
    return float(np.mean(vals))

q2 = {"pooled": {}, "gap": {}, "axis": {}}
for sc in ("probe", "s228"):
    for leg in LEGS:
        p = pooled_stats(sc, leg)
        q2["pooled"][f"{sc}_{leg}"] = p
        q2["gap"][f"{sc}_{leg}"] = within_cond_mean(sc, leg) - p

def boot_delta(sc, leg_a, leg_b, row_filter=None):
    """Paired cluster bootstrap of SROCC(leg_a) - SROCC(leg_b)."""
    if row_filter is None:
        keep_i = np.ones(len(rows), bool)
    else:
        keep_i = np.array([row_filter(r) for r in rows])
    deltas = np.empty(NBOOT)
    for t, draw in enumerate(boot_draws):
        idx = cluster_pool(draw)
        idx = idx[keep_i[idx]]
        deltas[t] = (sr(scores[(sc, leg_a)][idx], jod[idx])
                     - sr(scores[(sc, leg_b)][idx], jod[idx]))
    point = (pooled_stats(sc, leg_a, row_filter)
             - pooled_stats(sc, leg_b, row_filter))
    return {
        "delta": point,
        "boot_mean": float(deltas.mean()),
        "p_le_0": float((deltas <= 0).mean()),
        "p_ge_0": float((deltas >= 0).mean()),
        "ci_lo": float(np.percentile(deltas, 2.5)),
        "ci_hi": float(np.percentile(deltas, 97.5)),
    }

q2["deltas"] = {}
for sc in ("probe", "s228"):
    for a, b in (("ii", "i"), ("iii", "i"), ("iii", "ii")):
        q2["deltas"][f"{sc}_{a}-{b}"] = boot_delta(sc, a, b)

# axis decompositions
for lum in ("bright", "dim"):
    flt = lambda r, lum=lum: r["cond"][0] == lum
    q2["axis"][f"xdist_{lum}"] = {
        f"{sc}_{leg}": pooled_stats(sc, leg, flt)
        for sc in ("probe", "s228") for leg in LEGS}
    q2["axis"][f"xdist_{lum}_probe_iii-i"] = boot_delta("probe", "iii", "i", flt)
for dist in ("near", "far"):
    flt = lambda r, dist=dist: r["cond"][1] == dist
    q2["axis"][f"xlum_{dist}"] = {
        f"{sc}_{leg}": pooled_stats(sc, leg, flt)
        for sc in ("probe", "s228") for leg in LEGS}
    q2["axis"][f"xlum_{dist}_probe_ii-i"] = boot_delta("probe", "ii", "i", flt)

# headroom: per-video JOD spread across its 4 conditions
spread = []
byvid = defaultdict(list)
for r in rows:
    byvid[(r["cid"], r["vid"])].append(r["jod"])
for v in byvid.values():
    spread.append(max(v) - min(v))
q2["jod_condition_spread"] = {
    "mean": float(np.mean(spread)), "median": float(np.median(spread)),
    "q25": float(np.percentile(spread, 25)), "q75": float(np.percentile(spread, 75)),
    "max": float(np.max(spread)),
}
OUT["q2"] = q2
print("\nQ2 pooled SROCC over 464 rows (cross-luminance axis is anchor-linked"
      " — see PROTOCOL caveat):")
for k, v in q2["pooled"].items():
    print(f"  {k}: {v:.4f}   (gap {q2['gap'][k]:+.4f})")
print("Q2 paired deltas (cluster bootstrap over 16 contents, 10k, seed 20260729):")
for k, d in q2["deltas"].items():
    print(f"  {k}: Δ={d['delta']:+.4f} CI[{d['ci_lo']:+.4f},{d['ci_hi']:+.4f}] "
          f"p(Δ<=0)={d['p_le_0']:.4f} p(Δ>=0)={d['p_ge_0']:.4f}")
print("Q2 axis pools:", json.dumps({k: v for k, v in q2["axis"].items()
                                    if not k.endswith(("-i", "-ii"))}, indent=1))
print("JOD condition spread (headroom):", json.dumps(q2["jod_condition_spread"]))

# ---------------- Q3: diagnostic family attribution ---------------------
# leg (ii) features per row; target = per-condition z-scored JOD;
# scene-disjoint nested CV (outer GroupKFold(4) by content).
X_rows = np.stack([pooled_X[(r["cid"], r["vid"], LEGS["ii"][r["cond"]])]
                   for r in rows])
y_z = np.empty(len(rows))
for cond in CONDS:
    mask = np.array([c == cond for c in conds])
    y_z[mask] = (jod[mask] - jod[mask].mean()) / jod[mask].std()
groups_v = np.array([int(c) for c in cids])

FAMS = {
    "folded720": list(range(0, 720)),
    "append": list(range(720, 924)),
    "append2": list(range(924, 944)),
}
BANDVIS8 = [924 + 5 * s + d for s in range(4) for d in (0, 1)]
SETS = {"full944": list(range(944))}
for name, cols in FAMS.items():
    SETS[f"minus_{name}"] = [i for i in range(944) if i not in set(cols)]
    SETS[f"only_{name}"] = cols
SETS["only_BANDVIS8"] = BANDVIS8

def oof_srocc(cols):
    X = X_rows[:, cols]
    oof = np.empty(len(rows))
    outer = GroupKFold(n_splits=4)
    for tr, te in outer.split(X, y_z, groups_v):
        Xtr_f = X[tr]
        kp = Xtr_f.std(axis=0) > 1e-12
        Xk = Xtr_f[:, kp]
        m0, s0 = Xk.mean(axis=0), Xk.std(axis=0)
        Z = (Xk - m0) / s0
        inner = GroupKFold(n_splits=4)
        cv_l = {}
        for lam2 in LAMBDAS:
            ss = []
            for itr, ite in inner.split(Z, y_z[tr], groups_v[tr]):
                mm = Ridge(alpha=lam2)
                mm.fit(Z[itr], y_z[tr][itr])
                ss.append(sr(mm.predict(Z[ite]), y_z[tr][ite]))
            cv_l[lam2] = float(np.mean(ss))
        lam2 = max(cv_l, key=cv_l.get)
        mm = Ridge(alpha=lam2)
        mm.fit(Z, y_z[tr])
        oof[te] = mm.predict((X[te][:, kp] - m0) / s0)
    return sr(oof, y_z)

q3 = {"oof": {}}
for name, cols in SETS.items():
    q3["oof"][name] = oof_srocc(cols)
    print(f"Q3 OOF [{name}]: {q3['oof'][name]:.4f}")
full = q3["oof"]["full944"]
q3["deltas_vs_full"] = {k: float(v - full) for k, v in q3["oof"].items()}

# zero-fit append2 lane table (leg-ii features)
lanes = []
for li in range(924, 944):
    v = X_rows[:, li]
    row = {"fidx": li, "std": float(v.std()),
           "srocc_pooled": sr(v, jod) if v.std() > 0 else 0.0}
    for cond in CONDS:
        mask = np.array([c == cond for c in conds])
        row[f"srocc_{'-'.join(cond)}"] = (
            sr(v[mask], jod[mask]) if v[mask].std() > 0 else 0.0)
    lanes.append(row)
q3["append2_lanes"] = lanes
OUT["q3"] = q3
print("\nQ3 append2 zero-fit lanes (SROCC vs JOD; near-constant flagged):")
for r in lanes:
    flag = " NEAR-CONST" if r["std"] < 1e-9 else ""
    parts = []
    for c in CONDS:
        cname = "-".join(c)
        parts.append(f"{cname}={r['srocc_' + cname]:+.3f}")
    print(f"  f{r['fidx']}: std={r['std']:.2e} pooled={r['srocc_pooled']:+.4f} "
          + " ".join(parts) + flag)

with open(f"{BASE}/results.json", "w") as f:
    json.dump(OUT, f, indent=1)
print("\nwrote results.json")

# persist the pooled per-(video,config) table as the study asset
with open(f"{BASE}/hdrvdc_pooled_944.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["cid", "vid", "config", "score228_mean"]
               + [f"f{k}" for k in range(944)])
    for (cid, vid, cfg), v in sorted(pooled_X.items()):
        w.writerow([cid, vid, cfg, pooled_s228[(cid, vid, cfg)]]
                   + [repr(float(x)) for x in v])
print("wrote hdrvdc_pooled_944.csv")
