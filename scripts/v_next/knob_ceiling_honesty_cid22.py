#!/usr/bin/env python3
"""Is B's low ceiling MOS-honest? (Part 3c of b_knob_validation_real_encoders_2026-07-11.md)

Reframe: does B track human MOS in the HIGH-quality regime as well as ssim2 — i.e. is
the lower dial ceiling a recalibratable scale choice, or lost discrimination? On CID22
(real encodes + human MOS), three views, pooled + jxl/webp-only:
  1. Per-(high)-MOS-band SROCC/PLCC vs MOS — ordering honesty in the top regime.
  2. Disagreement adjudication — when ssim2 ranks a high-q encode above B (or vice
     versa), does MOS side with the higher or lower ranker?
  3. Top saturation — does B pile encodes near its max (lose resolution) more than
     ssim2 in the top MOS decile?

Prereq forwards (raw pre-spline is fine — all stats here are rank/percentile based):
  ensemble_score_rows --bake zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin \
      --parquet <CANON>/val/cid22.parquet --output /tmp/cid22_B.tsv
  ensemble_score_rows --bake zensim/weights/v47_strict_qat_native_2026-05-27.bin \
      --parquet <CANON>/val/cid22.parquet --output /tmp/cid22_A.tsv
"""
import csv, re
import numpy as np

def load(path):
    return {int(r["idx"]): (float(r["human"]), float(r["score"]))
            for r in csv.DictReader(open(path), delimiter="\t")}
B = load("/tmp/cid22_B.tsv"); A = load("/tmp/cid22_A.tsv")
ss = list(csv.DictReader(open("/mnt/v/output/zensim-multicodec-probe/cid22_ssim2_scores.tsv"), delimiter="\t"))
N = 4292
MOS = np.array([B[i][0]*100 for i in range(N)])
bB = np.array([B[i][1] for i in range(N)]); bA = np.array([A[i][1] for i in range(N)])
s2 = np.array([float(ss[i]["ssim2"]) for i in range(N)])
codec = np.array([re.search(r"/compressed/[^/]+/([^/]+)/", ss[i]["dist_path"]).group(1) for i in range(N)])

def srocc(a, b):
    if len(a) < 3: return np.nan
    def rk(x):
        o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[o]=np.arange(len(x)); return r
    ra, rb = rk(a), rk(b); n=len(a); return 1 - 6*np.sum((ra-rb)**2)/(n*(n*n-1))
def plcc(a, b): return np.corrcoef(a, b)[0,1] if len(a) > 2 else np.nan

def run(mask, label):
    mos, b, s, a = MOS[mask], bB[mask], s2[mask], bA[mask]
    print(f"\n########## {label}  (n={mask.sum()}) ##########")
    print("--- View 1: SROCC / PLCC vs MOS in high-MOS bands ---")
    print(f"{'band':18} {'n':>5} {'SROCC_B':>9} {'SROCC_ss2':>10} {'SROCC_A':>9}   {'PLCC_B':>8} {'PLCC_ss2':>9}")
    for name,lo,hi in [("MOS 60-70",60,70),("MOS 70-80",70,80),("MOS 80-90",80,90),("MOS 85-92",85,92),("MOS>=75 (high-q)",75,101)]:
        m=(mos>=lo)&(mos<hi)
        if m.sum()<10: print(f"{name:18} {m.sum():5d}  (n<10 skip)"); continue
        print(f"{name:18} {m.sum():5d} {srocc(b[m],mos[m]):9.4f} {srocc(s[m],mos[m]):10.4f} {srocc(a[m],mos[m]):9.4f}   {plcc(b[m],mos[m]):8.4f} {plcc(s[m],mos[m]):9.4f}")
    def pct(x):
        o=np.argsort(x,kind="mergesort"); r=np.empty(len(x)); r[o]=np.arange(len(x)); return r/(len(x)-1)
    pB,pS,pM = pct(b),pct(s),pct(mos)
    high = np.maximum(pB,pS) >= 0.75
    print("--- View 2: top-region disagreement — who does MOS side with? (mean |metric_pct - MOS_pct|) ---")
    for nm,d in [("ssim2>B (B under-ranks vs ssim2)", high&(pS-pB>=0.10)),
                 ("B>ssim2 (ssim2 under-ranks vs B)", high&(pB-pS>=0.10))]:
        if d.sum()<10: print(f"  {nm}: n={d.sum()} (skip)"); continue
        db=np.mean(np.abs(pM[d]-pB[d])); ds=np.mean(np.abs(pM[d]-pS[d]))
        print(f"  {nm}: n={d.sum():4d}  |B-MOS|={db:.3f}  |ssim2-MOS|={ds:.3f}  -> MOS agrees with "
              f"{'B' if db<ds else 'ssim2'}  (MOS_pct={np.mean(pM[d]):.3f}, B={np.mean(pB[d]):.3f}, ss2={np.mean(pS[d]):.3f})")
    topdec = mos >= np.percentile(mos,90)
    def sat(v): thr=v.min()+0.95*(v.max()-v.min()); return np.mean(v[topdec]>=thr)
    print(f"--- View 3: top-MOS-decile saturation (frac within 5% of max; higher=more ceiling pile-up) ---")
    print(f"  B={sat(b):.3f}  ssim2={sat(s):.3f}  A={sat(a):.3f}   | within-top-decile SROCC vs MOS: "
          f"B={srocc(b[topdec],mos[topdec]):.3f} ssim2={srocc(s[topdec],mos[topdec]):.3f}")

run(np.ones(N,bool), "ALL CODECS")
run(np.isin(codec, ["libjxl","cld_webp"]), "jxl + webp ONLY (where the ceiling gap showed)")
