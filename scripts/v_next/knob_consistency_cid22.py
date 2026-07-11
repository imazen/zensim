#!/usr/bin/env python3
"""Validate a bake as a KNOB against human MOS on CID22 (real encoders + human judgment).
See benchmarks/b_knob_validation_real_encoders_2026-07-11.md.

Knob quality = when you dial metric M to a target, how consistent is the true
perceptual quality (MOS / CVVDP) across content?
  - eta^2(REF | M-decile): fraction of REF variance pinned down by M's level (higher=better)
  - residual SD(REF | M-decile): spread of true quality at a fixed M target (lower=better)
Compared across candidate knobs {B, A, ssim2, cvvdp} x references {MOS, CVVDP}.
Rank-INDEPENDENT of SROCC: a metric can rank well yet be a content-dependent poor knob.

Prereq forwards (raw pre-spline output is fine — quantile binning is monotone-invariant):
  ensemble_score_rows --bake zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin \
      --parquet <CANON>/val/cid22.parquet --output /tmp/cid22_B.tsv
  ensemble_score_rows --bake zensim/weights/v47_strict_qat_native_2026-05-27.bin \
      --parquet <CANON>/val/cid22.parquet --output /tmp/cid22_A.tsv
"""
import csv
import re
import numpy as np

B_TSV = "/tmp/cid22_B.tsv"   # idx, human(MCOS/100), raw_B
A_TSV = "/tmp/cid22_A.tsv"   # idx, human, A
SSIM2 = "/mnt/v/output/zensim-multicodec-probe/cid22_ssim2_scores.tsv"  # ref,dist,mcos,ssim2
CVVDP = "/mnt/v/zen/zensim-eval/cid22_cvvdp_scores_2026-05-17.tsv"      # ref,dist,cvvdp_imazen_v0_0_1
N = 4292


def load_scores(path):
    return {int(r["idx"]): (float(r["human"]), float(r["score"]))
            for r in csv.DictReader(open(path), delimiter="\t")}


B = load_scores(B_TSV)
A = load_scores(A_TSV)
ss = list(csv.DictReader(open(SSIM2), delimiter="\t"))
cv = list(csv.DictReader(open(CVVDP), delimiter="\t"))
assert len(ss) == len(cv) == len(B) == N

mis = sum(abs(B[i][0] * 100 - float(ss[i]["mcos"])) > 1e-2
          or ss[i]["ref_path"] != cv[i]["ref_path"] or ss[i]["dist_path"] != cv[i]["dist_path"]
          for i in range(N))
print(f"alignment mismatches (parquet<->ssim2 mcos, ssim2<->cvvdp ref/dist): {mis}  (0 = aligned)")

MOS = np.array([B[i][0] * 100 for i in range(N)])
bB = np.array([B[i][1] for i in range(N)])
bA = np.array([A[i][1] for i in range(N)])
s2 = np.array([float(ss[i]["ssim2"]) for i in range(N)])
cvv = np.array([float(cv[i]["cvvdp_imazen_v0_0_1"]) for i in range(N)])
codec = np.array([re.search(r"/compressed/[^/]+/([^/]+)/", ss[i]["dist_path"]).group(1) for i in range(N)])
u, c = np.unique(codec, return_counts=True)
print("codecs:", dict(zip(u.tolist(), c.tolist())))


def srocc(a, b):
    def rk(x):
        o = np.argsort(x, kind="mergesort"); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r
    ra, rb = rk(a), rk(b); n = len(a)
    return 1 - 6 * np.sum((ra - rb) ** 2) / (n * (n * n - 1))


def eta2_resid(metric, ref, nbins=10):
    order = np.argsort(metric, kind="mergesort")
    bins = np.array_split(order, nbins)
    grand = ref.mean(); n = len(ref); between = 0.0; within_sd = []; binmeans = []
    for b in bins:
        m = ref[b].mean(); binmeans.append(m)
        between += len(b) * (m - grand) ** 2
        within_sd.append(np.std(ref[b]))
    eta2 = (between / n) / np.var(ref)
    mono = all(binmeans[i] <= binmeans[i + 1] for i in range(len(binmeans) - 1)) or \
        all(binmeans[i] >= binmeans[i + 1] for i in range(len(binmeans) - 1))
    return eta2, float(np.mean(within_sd)), mono, binmeans


for REFNAME, REF in [("MOS", MOS), ("CVVDP", cvv)]:
    print(f"\n=== knob consistency vs {REFNAME} (pooled across all codecs) ===")
    print(f"{'knob M':8} {'SROCC':>8} {'eta^2':>8} {'resid SD':>10} {'bin-mono':>8}")
    for name, M in [("B", bB), ("A", bA), ("ssim2", s2), ("cvvdp", cvv)]:
        if name.lower() == REFNAME.lower():
            continue
        e, rsd, mono, _ = eta2_resid(M, REF)
        print(f"{name:8} {srocc(M, REF):8.4f} {e:8.4f} {rsd:10.3f} {str(mono):>8}")

print("\n=== B decile table vs MOS ('dial B to bin -> what MOS do you get') ===")
order = np.argsort(bB, kind="mergesort"); bins = np.array_split(order, 10)
print(f"{'B decile':10} {'n':>5} {'B raw range':>18} {'mean MOS':>9} {'SD MOS':>8}")
for k, b in enumerate(bins):
    print(f"{k:<10} {len(b):5d} [{bB[b].min():7.3f},{bB[b].max():7.3f}] {MOS[b].mean():9.2f} {np.std(MOS[b]):8.2f}")
