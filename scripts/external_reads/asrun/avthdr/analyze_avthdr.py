#!/usr/bin/env python3
# ===========================================================================
# AS-RUN PROVENANCE COPY — frozen 2026-07-31 (decision-surface audit gap 3:
# the seven-domain external-read runners were previously uncommitted).
# Source:        /mnt/v/output/zensim/avthdr-validation-2026-07-29/analyze_avthdr.py
# sha256(source): eec054f5e2b94682f01d14d67b5eaa0c68897f0301c6999cb802b5d10c850b07
# build_commit:  1f0f92d5075d
# Protocol doc:  benchmarks/avthdr_validation_2026-07-29.md
# Everything below the marker line is BYTE-IDENTICAL to the source file
# (verify: strip through the marker, sha256 the rest). Do NOT extend this
# file — it is an archival record of the exact as-run analysis (it may call
# scipy directly; it predates the stats-rule batch migration and is kept
# verbatim). New analyses go through scripts/external_reads/run_external_reads.py.
# Data paths inside are as-run: /mnt/v artifact paths persist; ~/tmp
# FEATS_DIR-style inputs were session scratch — the pooled/persisted tables
# in the artifact dir are the stored equivalents (see ../README.md).
# ===== byte-identical source below this line ================================
"""AVT-VQDB-UHD-1-HDR validation study — analysis per PROTOCOL.md.

Inputs:
  ~/tmp/avthdr-work/feats/content_<content>.csv  (per-content 944 extractions,
      1 config x 8 frames per encoded video, build 1f0f92d5)
  /mnt/v/datasets/avt-vqdb-uhd-1-hdr/subjective_scores/mos_ci.csv
  /mnt/v/output/zensim/hdr-dmean-2026-07-29/upiq_sdr_956.csv (probe train)
  /mnt/v/output/zensim/hdr-dmean-2026-07-29/upiq_hdr_944.csv (gate eval)
  /mnt/v/output/zensim/hdr-dmean-2026-07-29/results.json     (gate values)

Outputs: results.json + printed report (tee to log) + avthdr_pooled_944.csv.
The only fitted objects: the registered UPIQ-SDR 944 ridge probe (gated
reconstruction) and Q3's scene-disjoint OOF heads (diagnostic). Everything
else is zero-fit. Bootstrap: 10k, seed 20260729. AVT has only 5 content
clusters — cluster CIs are primary but flagged structurally tiny; row-level
bootstrap reported alongside, labeled non-cluster.
"""
import csv
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

BASE = "/mnt/v/output/zensim/avthdr-validation-2026-07-29"
FEATS_DIR = os.path.expanduser("~/tmp/avthdr-work/feats")
LABELS = "/mnt/v/datasets/avt-vqdb-uhd-1-hdr/subjective_scores/mos_ci.csv"
DMEAN = "/mnt/v/output/zensim/hdr-dmean-2026-07-29"
SEED = 20260729
NBOOT = 10_000
NFRAMES = 8
CODECS = ["av1", "hevc", "vvc"]
OUT = {}


def sr(a, b):
    return float(spearmanr(a, b).statistic)


# ---------------- load features (per-frame) and mean-pool ----------------
frame_feats = defaultdict(list)   # (content, vid) -> list of (fidx, feats)
frame_s228 = defaultdict(list)
for path in sorted(glob.glob(f"{FEATS_DIR}/content_*.csv")):
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        assert header[:4] == ["key", "dim", "peak_nits", "score228"], header
        assert len(header) == 4 + 944
        for rec in rdr:
            content, vid, fj, cfg = rec[0].split("|")
            assert cfg == "P" and rec[1] == "0" and float(rec[2]) == 1000.0
            fidx = int(fj[1:])
            frame_feats[(content, vid)].append(
                (fidx, np.array([float(v) for v in rec[4:]])))
            frame_s228[(content, vid)].append((fidx, float(rec[3])))

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
    coverage["videos"].add(k)
    coverage["n_frame_rows"] += len(lst)
assert not coverage["incomplete"], coverage["incomplete"][:5]
print(f"pooled: {len(pooled_X)} (content,vid) vectors; "
      f"{coverage['n_frame_rows']} frame rows")

# ---------------- labels ----------------
pat = re.compile(r"^(\d+)_(\d+)_(\w+)_(av1|hevc|vvc)_(.+)\.mkv$")
rows = []
orig_mos = {}
for r in csv.DictReader(open(LABELS)):
    m = pat.match(r["stimuli_file"])
    if not m:
        assert "original" in r["stimuli_file"]
        orig_mos[r["stimuli_file"]] = float(r["mos"])
        continue
    w, h, br, codec, content = m.groups()
    rows.append({
        "content": content, "vid": f"{codec}_{w}_{h}_{br}",
        "codec": codec, "res": f"{w}x{h}", "br": br,
        "mos": float(r["mos"]), "ci": float(r["ci"]),
    })
assert len(rows) == 195, len(rows)
vids_needed = {(r["content"], r["vid"]) for r in rows}
missing = vids_needed - coverage["videos"]
assert not missing, f"missing extracted videos: {sorted(missing)[:5]}"
OUT["coverage"] = {
    "n_videos": len(coverage["videos"]),
    "n_frame_rows_pooled_over": coverage["n_frame_rows"],
    "n_label_rows": len(rows),
    "mos_ci_mean": float(np.mean([r["ci"] for r in rows])),
    "mos_ci_median": float(np.median([r["ci"] for r in rows])),
    "orig_mos_excluded_descriptive": orig_mos,
}

# ---------------- probe reconstruction (GATE before any AVT look) --------
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
    scores_cv = []
    for tr, te in gkf.split(Ztr, jod_s, groups_s):
        m = Ridge(alpha=lam)
        m.fit(Ztr[tr], jod_s[tr])
        scores_cv.append(sr(m.predict(Ztr[te]), jod_s[te]))
    cv[lam] = float(np.mean(scores_cv))
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


# ---------------- per-row scores ----------------------------------------
for r in rows:
    r["probe"] = probe_score(pooled_X[(r["content"], r["vid"])])
    r["s228"] = pooled_s228[(r["content"], r["vid"])]

mos = np.array([r["mos"] for r in rows])
contents = np.array([r["content"] for r in rows])
codecs = np.array([r["codec"] for r in rows])
uniq_contents = sorted(set(contents))
scores = {sc: np.array([r[sc] for r in rows]) for sc in ("probe", "s228")}

rng = np.random.default_rng(SEED)
boot_draws = [rng.choice(len(uniq_contents), size=len(uniq_contents),
                         replace=True) for _ in range(NBOOT)]
row_draws = [rng.choice(len(rows), size=len(rows), replace=True)
             for _ in range(NBOOT)]
content_rows = {c: np.array([i for i, x in enumerate(contents) if x == c])
                for c in uniq_contents}


def cluster_pool(draw):
    return np.concatenate([content_rows[uniq_contents[d]] for d in draw])


def boot_ci(stat_fn):
    """(cluster_ci, row_ci) for a statistic over row-index arrays."""
    cl = [stat_fn(cluster_pool(d)) for d in boot_draws[:2000]]
    rw = [stat_fn(d) for d in row_draws[:2000]]
    pct = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return pct(cl), pct(rw)


# ---------------- Q1: within-set ranking --------------------------------
q1 = {}
for sc in ("probe", "s228"):
    s = scores[sc]
    pooled = sr(s, mos)
    cl_ci, rw_ci = boot_ci(lambda idx, s=s: sr(s[idx], mos[idx]))
    per_codec = {}
    for c in CODECS:
        m = codecs == c
        cidx = np.nonzero(m)[0]
        def cstat(idx, s=s, c=c):
            idx = idx[codecs[idx] == c]
            return sr(s[idx], mos[idx])
        ccl, crw = boot_ci(cstat)
        per_codec[c] = {"srocc": sr(s[m], mos[m]), "n": int(m.sum()),
                        "cluster_ci": ccl, "row_ci": crw}
    per_content = {}
    for ct in uniq_contents:
        m = contents == ct
        per_content[ct] = {"srocc": sr(s[m], mos[m]), "n": int(m.sum())}
    q1[sc] = {"pooled": pooled, "pooled_cluster_ci": cl_ci,
              "pooled_row_ci": rw_ci, "per_codec": per_codec,
              "per_content": per_content}
OUT["q1"] = q1
print("\nQ1 within-set SROCC vs MOS (n=195; 5 content clusters — flagged tiny):")
for sc in ("probe", "s228"):
    r_ = q1[sc]
    print(f"  {sc}: pooled {r_['pooled']:.4f} "
          f"clusterCI[{r_['pooled_cluster_ci'][0]:.4f},{r_['pooled_cluster_ci'][1]:.4f}] "
          f"rowCI[{r_['pooled_row_ci'][0]:.4f},{r_['pooled_row_ci'][1]:.4f}]")
    for c in CODECS:
        pc = r_["per_codec"][c]
        print(f"    {c:5s} (n={pc['n']}): {pc['srocc']:.4f} "
              f"clusterCI[{pc['cluster_ci'][0]:.4f},{pc['cluster_ci'][1]:.4f}] "
              f"rowCI[{pc['row_ci'][0]:.4f},{pc['row_ci'][1]:.4f}]")
    for ct in uniq_contents:
        pc = r_["per_content"][ct]
        print(f"    content {ct:16s} (n={pc['n']}): {pc['srocc']:.4f}")

# ---------------- Q2: codec generalization (all zero-fit) ---------------
q2 = {}
for sc in ("probe", "s228"):
    s = scores[sc]
    pooled = q1[sc]["pooled"]
    percs = {c: q1[sc]["per_codec"][c]["srocc"] for c in CODECS}
    gap = float(np.mean(list(percs.values())) - pooled)
    D = float(max(abs(v - pooled) for v in percs.values()))

    # concordance: pairs with |dMOS|>0.1, within- vs cross-codec
    dm = mos[:, None] - mos[None, :]
    dsc = s[:, None] - s[None, :]
    same = codecs[:, None] == codecs[None, :]
    iu = np.triu_indices(len(rows), 1)
    valid = np.abs(dm[iu]) > 0.1
    conc = (np.sign(dm[iu]) == np.sign(dsc[iu]))
    w_mask = same[iu] & valid
    x_mask = (~same[iu]) & valid
    conc_within = float(conc[w_mask].mean())
    conc_cross = float(conc[x_mask].mean())

    def dconc_stat(idx):
        mm = mos[idx]; ss = s[idx]; cc = codecs[idx]
        dm2 = mm[:, None] - mm[None, :]
        ds2 = ss[:, None] - ss[None, :]
        sm2 = cc[:, None] == cc[None, :]
        iu2 = np.triu_indices(len(idx), 1)
        v2 = np.abs(dm2[iu2]) > 0.1
        c2 = np.sign(dm2[iu2]) == np.sign(ds2[iu2])
        w2 = sm2[iu2] & v2
        x2 = (~sm2[iu2]) & v2
        if w2.sum() == 0 or x2.sum() == 0:
            return 0.0
        return float(c2[w2].mean() - c2[x2].mean())
    dc_cl, dc_rw = boot_ci(dconc_stat)

    # matched-MOS intercept shift per codec pair
    slope = float((np.percentile(s, 90) - np.percentile(s, 10))
                  / (np.percentile(mos, 90) - np.percentile(mos, 10)))
    pairs = {}
    for a, b in (("av1", "hevc"), ("av1", "vvc"), ("hevc", "vvc")):
        ia = np.nonzero(codecs == a)[0]
        ib = np.nonzero(codecs == b)[0]
        dm_ab = mos[ia][:, None] - mos[ib][None, :]
        close = np.abs(dm_ab) <= 0.25
        diffs = (s[ia][:, None] - s[ib][None, :])[close]
        pairs[f"{a}-{b}"] = {
            "n_pairs": int(close.sum()),
            "mean_shift": float(diffs.mean()),
            "sd_shift": float(diffs.std()),
            "mos_equiv": float(diffs.mean() / slope),
        }

    # per-codec paired delta vs pooled (cluster bootstrap)
    deltas = {}
    for c in CODECS:
        def dstat(idx, c=c):
            sub = idx[codecs[idx] == c]
            return sr(s[sub], mos[sub]) - sr(s[idx], mos[idx])
        dcl = [dstat(cluster_pool(d)) for d in boot_draws[:2000]]
        deltas[c] = {
            "delta": float(percs[c] - pooled),
            "cluster_ci": (float(np.percentile(dcl, 2.5)),
                           float(np.percentile(dcl, 97.5))),
        }
    q2[sc] = {"pooled": pooled, "per_codec": percs, "gap": gap, "D": D,
              "conc_within": conc_within, "conc_cross": conc_cross,
              "dconc": conc_within - conc_cross,
              "dconc_cluster_ci": dc_cl, "dconc_row_ci": dc_rw,
              "slope_per_mos": slope, "matched_mos_shift": pairs,
              "delta_vs_pooled": deltas,
              "n_pairs_within": int(w_mask.sum()),
              "n_pairs_cross": int(x_mask.sum())}

# registered verdict bands (probe is the named carrier)
p = q2["probe"]
band = None
max_shift = max(abs(v["mos_equiv"]) for v in p["matched_mos_shift"].values())
if p["gap"] <= 0.02 and p["dconc"] <= 0.03:
    band = "codec-agnostic"
elif p["gap"] > 0.05 or p["dconc"] > 0.06 or max_shift > 0.5:
    band = "codec-specific bias"
else:
    band = "intermediate"
q2["registered_band_probe"] = band
q2["max_abs_mos_equiv_shift_probe"] = max_shift
OUT["q2"] = q2
print("\nQ2 codec generalization (zero-fit):")
for sc in ("probe", "s228"):
    r_ = q2[sc]
    print(f"  {sc}: pooled {r_['pooled']:.4f} | per-codec "
          + " ".join(f"{c}={r_['per_codec'][c]:.4f}" for c in CODECS)
          + f" | gap {r_['gap']:+.4f} D {r_['D']:.4f}")
    print(f"    concordance within {r_['conc_within']:.4f} cross {r_['conc_cross']:.4f} "
          f"dconc {r_['dconc']:+.4f} clusterCI[{r_['dconc_cluster_ci'][0]:+.4f},"
          f"{r_['dconc_cluster_ci'][1]:+.4f}] rowCI[{r_['dconc_row_ci'][0]:+.4f},"
          f"{r_['dconc_row_ci'][1]:+.4f}] "
          f"(n_pairs w/x {r_['n_pairs_within']}/{r_['n_pairs_cross']})")
    for k, v in r_["matched_mos_shift"].items():
        print(f"    matched-MOS {k}: n={v['n_pairs']} shift {v['mean_shift']:+.4f} "
              f"(sd {v['sd_shift']:.3f}) = {v['mos_equiv']:+.3f} MOS-equiv")
    for c in CODECS:
        d = r_["delta_vs_pooled"][c]
        print(f"    d(SROCC_{c} - pooled) = {d['delta']:+.4f} "
              f"clusterCI[{d['cluster_ci'][0]:+.4f},{d['cluster_ci'][1]:+.4f}]")
print(f"  REGISTERED BAND (probe): {band} "
      f"(max |MOS-equiv shift| {max_shift:.3f})")

# ---------------- Q3: diagnostic family attribution ---------------------
X_rows = np.stack([pooled_X[(r["content"], r["vid"])] for r in rows])
y = mos.copy()
groups_v = np.array([uniq_contents.index(c) for c in contents])

FAMS = {
    "folded720": list(range(0, 720)),
    "append": list(range(720, 924)),
    "append2": list(range(924, 944)),
}
BANDVIS8 = [924 + 5 * s_ + d for s_ in range(4) for d in (0, 1)]
SETS = {"full944": list(range(944))}
for name, cols in FAMS.items():
    SETS[f"minus_{name}"] = [i for i in range(944) if i not in set(cols)]
    SETS[f"only_{name}"] = cols
SETS["only_BANDVIS8"] = BANDVIS8


def oof_srocc(cols):
    X = X_rows[:, cols]
    oof = np.empty(len(rows))
    outer = GroupKFold(n_splits=5)
    for tr, te in outer.split(X, y, groups_v):
        Xtr_f = X[tr]
        kp = Xtr_f.std(axis=0) > 1e-12
        Xk = Xtr_f[:, kp]
        m0, s0 = Xk.mean(axis=0), Xk.std(axis=0)
        Z = (Xk - m0) / s0
        inner = GroupKFold(n_splits=4)
        cv_l = {}
        for lam2 in LAMBDAS:
            ss = []
            for itr, ite in inner.split(Z, y[tr], groups_v[tr]):
                mm = Ridge(alpha=lam2)
                mm.fit(Z[itr], y[tr][itr])
                ss.append(sr(mm.predict(Z[ite]), y[tr][ite]))
            cv_l[lam2] = float(np.mean(ss))
        lam2 = max(cv_l, key=cv_l.get)
        mm = Ridge(alpha=lam2)
        mm.fit(Z, y[tr])
        oof[te] = mm.predict((X[te][:, kp] - m0) / s0)
    return sr(oof, y)


q3 = {"oof": {}}
for name, cols in SETS.items():
    q3["oof"][name] = oof_srocc(cols)
    print(f"Q3 OOF [{name}]: {q3['oof'][name]:.4f}")
full = q3["oof"]["full944"]
q3["deltas_vs_full"] = {k: float(v - full) for k, v in q3["oof"].items()}

# zero-fit append2 lane table
lanes = []
for li in range(924, 944):
    v = X_rows[:, li]
    row = {"fidx": li, "std": float(v.std()),
           "srocc_pooled": sr(v, mos) if v.std() > 0 else 0.0}
    for c in CODECS:
        m = codecs == c
        row[f"srocc_{c}"] = sr(v[m], mos[m]) if v[m].std() > 0 else 0.0
    lanes.append(row)
q3["append2_lanes"] = lanes
OUT["q3"] = q3
print("\nQ3 append2 zero-fit lanes (SROCC vs MOS; near-constant flagged):")
for r_ in lanes:
    flag = " NEAR-CONST" if r_["std"] < 1e-9 else ""
    parts = " ".join(f"{c}={r_['srocc_' + c]:+.3f}" for c in CODECS)
    print(f"  f{r_['fidx']}: std={r_['std']:.2e} "
          f"pooled={r_['srocc_pooled']:+.4f} {parts}{flag}")

with open(f"{BASE}/results.json", "w") as f:
    json.dump(OUT, f, indent=1)
print("\nwrote results.json")

with open(f"{BASE}/avthdr_pooled_944.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["content", "vid", "score228_mean"]
               + [f"f{k}" for k in range(944)])
    for (content, vid), v in sorted(pooled_X.items()):
        w.writerow([content, vid, pooled_s228[(content, vid)]]
                   + [repr(float(x)) for x in v])
print("wrote avthdr_pooled_944.csv")
