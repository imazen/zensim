#!/usr/bin/env python3
"""R6b/F17: per-shipped-bake exposure to the twelve `contrast_inc` slots.

Read set from `bake_block_profile --json` (`feature_set_slots`), transform
status from the bake's own `zentrain.feature_transforms`, measured maxima from
`r6b_audit_slots.py`'s TSV. Computes nothing itself — it intersects three
existing owners' outputs. See docs/PLAN_FEATURE_REV2_2026-09-05.md section 11.3.
"""
import json, os, subprocess, re, sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(REPO)
CI=[c*13+12 for c in range(12)]
allmax={}
for line in open("/mnt/v/output/zensim/rev2-2026-09-05/r6b/slot_audit_rev1.tsv").read().splitlines()[1:]:
    p=line.split("\t"); allmax[int(p[0][1:])]=float(p[4])
ZP=os.environ.get("ZENPREDICT_BIN", os.path.expanduser("~/work/zen/zenanalyze/target/release/zenpredict"))
BP=os.environ.get("BLOCK_PROFILE_BIN", os.path.join(REPO, "target/release/bake_block_profile"))
BAKES=[("A  v47_strict_qat_native","zensim/weights/v47_strict_qat_native_2026-05-27.bin"),
 ("B  b_sdr_linear_cid80_inclwinsor","zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin"),
 ("C  c_sdr_purity944","zensim/weights/c_sdr_purity944_2026-08-29.bin"),
 ("CHdr c_hdr_l1t1944","zensim/weights/c_hdr_l1t1944_2026-08-29.bin"),
 ("D  d_sdr_add156_id100_negrich","zensim/weights/d_sdr_add156_id100_negrich_dial_2026-09-05.bin"),
 ("BHdr bhdr_linear_shaped_cvvdpmix","zensim/weights/bhdr_linear_shaped_cvvdpmix_2026-07-12.bin")]
def parse_slots(s):
    o=set()
    for t in s.split(","):
        t=t.strip()
        if not t: continue
        if "-" in t: a,b=t.split("-"); o.update(range(int(a),int(b)+1))
        else: o.add(int(t))
    return o
BOUNDED={"winsor_p99","winsor_p999","quantile_bins","clip_then_log1p","rank_gauss","clip01"}
COMPRESS={"log1p","signed_cbrt","yeo_johnson","signed_log1p","sqrt"}
out=[]
print(f"{'bake':34s} {'width':>5s} {'|tf|':>5s} {'reads':>5s} {'F17':>4s} {'bnd':>4s} {'cmp':>4s} {'raw':>4s}  raw/compressed slots -> max")
for lab,p in BAKES:
    j=json.loads(subprocess.run([BP,"--bake",p,"--json"],capture_output=True,text=True,check=True).stdout)
    slots=parse_slots(j.get("feature_set_slots","")); hit=sorted(slots & set(CI))
    txt=subprocess.run([ZP,"inspect",p],capture_output=True,text=True).stdout
    m=re.search(r'"key": "zentrain\.feature_transforms",\s*"kind": "utf8",\s*"value_len": \d+,\s*"value_text": "(.*?)"\s*\}', txt, re.S)
    tf=(m.group(1).split("\\n") if m else [])
    tf=[t for t in tf if t]
    b=c=r=0; det=[]
    for s in hit:
        t=tf[s] if s<len(tf) else "(none)"
        if t in BOUNDED: b+=1
        elif t in COMPRESS: c+=1; det.append((s,t))
        else: r+=1; det.append((s,t))
    worst_nb=max([allmax[s] for s,t in det], default=0.0)
    print(f"{lab:34s} {j['caller_input_width']:5d} {len(tf):5d} {len(slots):5d} {len(hit):4d} {b:4d} {c:4d} {r:4d}  "
          + ", ".join(f"f{s}[{t}]={allmax[s]:.4g}" for s,t in det)[:90])
    out.append(dict(bake=lab,path=p,caller_width=j["caller_input_width"],n_transforms=len(tf),
        reads=len(slots),f17_read=len(hit),f17_slots=hit,bounded=b,compressing=c,raw=r,
        non_bounded_detail=[[s,t,allmax[s]] for s,t in det],
        worst_measured_max_non_bounded=worst_nb,
        transforms_at_f17={str(s):(tf[s] if s<len(tf) else None) for s in hit}))
json.dump(out, open("/mnt/v/output/zensim/rev2-2026-09-05/r6b/f17_exposure.json","w"), indent=1)
