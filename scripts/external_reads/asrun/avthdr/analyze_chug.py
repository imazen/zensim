#!/usr/bin/env python3
# ===========================================================================
# AS-RUN PROVENANCE COPY — frozen 2026-07-31 (decision-surface audit gap 3:
# the seven-domain external-read runners were previously uncommitted).
# Source:        /mnt/v/output/zensim/avthdr-validation-2026-07-29/analyze_chug.py
# sha256(source): 0a6942a193cea9b2b230cdc31c9fc908b48b71c283cd7de46c36be5f6458deec
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
"""CHUG sampled FR leg — the ONE registered look (per PROTOCOL.md).

Pooled SROCC (probe + s228) vs mos_j over the sampled pairs + per-rung
SROCC (descriptive) + content-cluster bootstrap CI. Imperfect-reference
caveat attaches to any verdict sentence. Gate: the probe reconstruction
must reproduce the recorded hdr-dmean values before any mos_j contact
(same machinery as analyze_avthdr.py; deterministic).
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

BASE = "/mnt/v/output/zensim/avthdr-validation-2026-07-29"
FEATS_DIR = os.path.expanduser("~/tmp/avthdr-work/chug_feats")
SAMPLE = os.path.expanduser("~/tmp/avthdr-work/chug_sample.tsv")
CHUG_CSV = "/mnt/v/datasets/chug/chug.csv"
DMEAN = "/mnt/v/output/zensim/hdr-dmean-2026-07-29"
SEED = 20260729
NFRAMES = 8
RUNGS = ["360p_0.2M_", "720p_0.5M_", "720p_2M_", "1080p_0.5M_",
         "1080p_1M_", "1080p_3M_"]


def sr(a, b):
    return float(spearmanr(a, b).statistic)


# ---------------- probe gate (verbatim machinery) ------------------------
def load_feat_csv(path):
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        return header, list(rdr)

nfeat_off = 5
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
print("PROBE GATE PASS (chug leg):", json.dumps(gate))


def probe_score(v):
    return float(probe.predict(((v[keep] - mu) / sd)[None, :])[0])


# ---------------- load chug features + mean-pool -------------------------
frame_feats = defaultdict(list)
frame_s228 = defaultdict(list)
for path in sorted(glob.glob(f"{FEATS_DIR}/chug_batch_*.csv")):
    with open(path, newline="") as f:
        rdr = csv.reader(f)
        header = next(rdr)
        assert header[:4] == ["key", "dim", "peak_nits", "score228"]
        assert len(header) == 4 + 944
        for rec2 in rdr:
            content, rung, fj, cfg = rec2[0].split("|")
            fidx = int(fj[1:])
            frame_feats[(content, rung)].append(
                (fidx, np.array([float(v) for v in rec2[4:]])))
            frame_s228[(content, rung)].append((fidx, float(rec2[3])))
pooled_X = {}
pooled_s228 = {}
incomplete = []
for k, lst in frame_feats.items():
    lst.sort()
    if len(lst) != NFRAMES:
        incomplete.append((k, len(lst)))
        continue
    pooled_X[k] = np.mean([v for _, v in lst], axis=0)
    pooled_s228[k] = float(np.mean([s for _, s in sorted(frame_s228[k])]))
assert not incomplete, incomplete[:5]
print(f"chug pooled: {len(pooled_X)} (content,rung) vectors")

# ---------------- labels (first mos_j contact) ---------------------------
mosj = {}
for r in csv.DictReader(open(CHUG_CSV)):
    if r["ref"] == "0":
        mosj[(r["content_name"], r["bitladder"])] = float(r["mos_j"])
sample = list(csv.DictReader(open(SAMPLE), delimiter="\t"))
rows = []
missing = []
for s in sample:
    k = (s["content"], s["rung"])
    if k not in pooled_X:
        missing.append(k)
        continue
    rows.append({"content": s["content"], "rung": s["rung"],
                 "mos": mosj[k], "probe": probe_score(pooled_X[k]),
                 "s228": pooled_s228[k]})
print(f"rows with features: {len(rows)}; sampled-but-missing "
      f"(dropped at decode, see chug_drops.tsv): {len(missing)}")

mos = np.array([r["mos"] for r in rows])
contents = np.array([r["content"] for r in rows])
rungs = np.array([r["rung"] for r in rows])
uniq_contents = sorted(set(contents))
rng = np.random.default_rng(SEED)
content_rows = {c: np.array([i for i, x in enumerate(contents) if x == c])
                for c in uniq_contents}
boot_draws = [rng.choice(len(uniq_contents), size=len(uniq_contents),
                         replace=True) for _ in range(2000)]

OUT = {"gate": gate, "n_rows": len(rows),
       "n_missing_after_drops": len(missing),
       "n_contents": len(uniq_contents)}
print("\nCHUG registered look (FR vs imperfect UGC reference — caveat "
      "attaches to any verdict):")
for sc in ("probe", "s228"):
    s = np.array([r[sc] for r in rows])
    pooled = sr(s, mos)
    bs = []
    for d in boot_draws:
        idx = np.concatenate([content_rows[uniq_contents[i]] for i in d])
        bs.append(sr(s[idx], mos[idx]))
    ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    per_rung = {}
    for rg in RUNGS:
        m = rungs == rg
        if m.sum() >= 3:
            per_rung[rg] = {"srocc": sr(s[m], mos[m]), "n": int(m.sum())}
    OUT[sc] = {"pooled": pooled, "cluster_ci": ci, "per_rung": per_rung}
    print(f"  {sc}: pooled SROCC {pooled:.4f} "
          f"clusterCI[{ci[0]:.4f},{ci[1]:.4f}] (n={len(rows)})")
    for rg, v in per_rung.items():
        print(f"    {rg:12s} (n={v['n']}): {v['srocc']:.4f}")

with open(f"{BASE}/chug_results.json", "w") as f:
    json.dump(OUT, f, indent=1)
print("wrote chug_results.json")
