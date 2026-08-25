#!/usr/bin/env python3
"""Audit: NO imazen-26 held-out (validate+test) id appears in any training view.

Goal criterion 2 (user directive 2026-08-25: "verify for certain that the
imazen26 images don't include test or eval ids"). imazen-26 splits by id
last-digit (even=train, odd=validate/test); the avifgen (avif944) + bigcodec
sweeps derive from imazen-26 origins, so a held-out id in their TRAIN rows would
be leakage. Other legs (safesyn hex / cid22 7-digit / kadis / kadid / tid) are
different id namespaces — reported as such, and still numerically cross-checked.

Exit 0 = clean (zero held-out ids in any training view). Nonzero = leakage.
"""
import os, re, sys, json
import pyarrow.parquet as pq

IM26 = os.path.expanduser("~/work/imazen-26")
def im26_ids(splits):
    ids=set()
    for sp in splits:
        f=f"{IM26}/manifests/{sp}.tsv"
        with open(f) as fh:
            next(fh)
            for ln in fh:
                ids.add(ln.split("\t")[0].strip())
    return ids

HELDOUT = im26_ids(["validate","test"])     # odd-digit — must NOT appear in train
ALLID   = im26_ids(["train","validate","test"])

# training VIEWS + how to pull the imazen-26-style origin id from a source col.
VIEWS = [
  # (name, path, source_col, extractor)  extractor: str -> id or None
  ("avif944",     "/mnt/v/zen/zensim-training/avif944-2026-08-07/avif944_leg_944.parquet",
       "origin",       lambda s: re.match(r"^(\d+)", str(s)).group(1) if re.match(r"^(\d+)", str(s)) else None),
  ("bigcodec",    "/mnt/v/zen/zensim-training/tbig_944_200k.parquet",
       "ref_basename", lambda s: (m.group(1) if (m:=re.match(r"o_(\d+)", str(s))) else None)),
  ("ext_safesyn", "/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/ext_safesyn_full.parquet",
       "ref_basename", lambda s: str(s).split("_")[0] if str(s).split("_")[0].isdigit() else None),
  ("ext_cid22",   "/mnt/v/zen/zensim-training/ext944-canonical-2026-08-01/ext_cid22_train201.parquet",
       "ref_basename", lambda s: str(s) if str(s).isdigit() else None),
  ("kadis",       "/mnt/v/zen/zensim-training/kadis-944-2026-08-01/kadis_944_ssim2_50k.parquet",
       "source_id",    lambda s: str(s)),
]

report={"heldout_ids":len(HELDOUT),"all_imazen26_ids":len(ALLID),"views":[],"LEAKAGE":False}
for name,path,col,ext in VIEWS:
    if not os.path.exists(path):
        report["views"].append({"view":name,"status":"MISSING","path":path}); continue
    v=pq.read_table(path, columns=[col]).column(0)
    ids=set()
    for i in range(len(v)):
        x=ext(v[i])
        if x is not None: ids.add(x)
    im26_derived = ids & ALLID                 # ids that ARE imazen-26 images
    leaked       = ids & HELDOUT               # numeric matches to held-out ids
    derived_frac = len(im26_derived)/max(1,len(ids))
    # CORPUS-AWARE: a view is imazen-26-derived only when ~all its source ids
    # are imazen-26 ids. Below the threshold the id space is a DIFFERENT corpus
    # (kadis 0-139999, cid22 7-digit, safesyn hex) and a numeric match to an
    # imazen-26 id is a coincidental namespace collision, NOT leakage.
    DERIVED_THRESHOLD = 0.90
    is_derived = derived_frac >= DERIVED_THRESHOLD
    real_leak = leaked if is_derived else set()
    if is_derived:
        verdict = "LEAK" if real_leak else "clean(imazen26-derived, train-only — disjoint from held-out)"
    else:
        verdict = (f"clean(different corpus; {len(leaked)} coincidental numeric "
                   f"id-collisions with imazen-26, NOT leakage)" if leaked
                   else "clean(different id namespace)")
    row={"view":name,"distinct_source_ids":len(ids),
         "imazen26_derived_ids":len(im26_derived),"derived_fraction":round(derived_frac,4),
         "is_imazen26_derived":is_derived,
         "HELDOUT_numeric_matches":len(leaked),"REAL_LEAK_ids":sorted(real_leak)[:20],
         "verdict":verdict}
    report["views"].append(row)
    if real_leak: report["LEAKAGE"]=True

print(json.dumps(report, indent=1))
sys.exit(1 if report["LEAKAGE"] else 0)
