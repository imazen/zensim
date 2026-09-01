#!/usr/bin/env python3
"""Paired per-reference bootstrap: is a candidate's within-image CID22 SROCC
distinguishable from ssim2's, given only 49 references?

Every SROCC is `panel --batch` (zenstats). This script does the PAIRING and
the resample, which is what a paired bootstrap is; it computes no correlation
itself. The pairing is exact: `bake_verdict --per-pair-output --per-pair-refs`
and the stored `cid22_ssim2.tsv` are the SAME 4,292 pairs in the SAME order
(verified index-wise, max |model_human − MCOS/100| = 0.0), so a reference's
pair set is identical for every arm.

Resample unit = the REFERENCE, because that is the unit the eval population
actually samples (49 CID22 sources) and the unit a per-ref mean averages over.
"""
import csv, os, sys, random, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "lib"))
from zen_stats import panel_batch  # noqa: E402

O = os.environ.get("O", "/mnt/v/output/zensim/ssim2-bar-2026-08-31")
RM = "/mnt/v/output/zensim/reports/refmetrics"
B_ITERS = int(os.environ.get("BOOT", "10000"))
SEED = int(os.environ.get("SEED", "20260901"))

def model_rows(path):
    rows = [l.rstrip("\n").split("\t") for l in open(path) if l.strip()]
    h = rows[0]; i = {n: k for k, n in enumerate(h)}
    return [(float(r[i["pred"]]), float(r[i["human"]]), r[i["ref"]]) for r in rows[1:]]

# corpus -> (peer tsv, human col, metric col, human scale divisor)
CORPORA = {
    "cid22": ("cid22_ssim2.tsv", "MCOS", "ssim2", 100.0),
    "csiq": ("csiq_ssim2_gpu.tsv", "human_score", "ssim2_gpu", 1.0),
    "aic3": ("aic3_ssim2_heldout.tsv", "jnd", "ssim2_gpu", 1.0),
    # LIVE is deliberately absent: its peer table and the verdict per-pair dump
    # hold the same 779 pairs in DIFFERENT row order (max index-wise target
    # difference 1.12, vs 0.0 for the three above), so the pairing that makes
    # this test valid does not hold and a join key does not exist in either
    # file. Reported unpaired in the note instead of silently mis-paired.
}
CORPUS = os.environ.get("CORPUS", "cid22")

def peer_rows():
    f, hc, mc, div = CORPORA[CORPUS]
    r = list(csv.DictReader(open(f"{RM}/{f}"), delimiter="\t"))
    return [(float(x[mc]), float(x[hc]) / div, os.path.basename(x["ref_path"])) for x in r]

arms = {"ssim2": peer_rows()}
for name in ("B", "ADD156", "W10L9P", "W10L9PH"):
    p = f"{O}/pp_{name}_{CORPUS}.tsv"
    if os.path.exists(p):
        arms[name] = model_rows(p)

n = len(arms["ssim2"])
for k, v in arms.items():
    assert len(v) == n, f"{k}: {len(v)} != {n}"
# targets must be identical index-wise, else the pairing is a fiction
base = [t for _, t, _ in arms["ssim2"]]
for k, v in arms.items():
    d = max(abs(t - b) for (_, t, _), b in zip(v, base))
    assert d < 1e-9, f"{k}: targets differ index-wise by {d}"

# Reference grouping comes from the PEER table (names); the model dumps carry
# integer ids, and the two agree because the row order is identical.
refs = [r for _, _, r in arms["ssim2"]]
groups = {}
for i, r in enumerate(refs):
    groups.setdefault(r, []).append(i)
keys = sorted(groups)
print(f"# corpus={CORPUS}: {len(keys)} references, {n} pairs, {B_ITERS} bootstrap resamples, seed {SEED}")

# Per-reference SROCC vectors, one panel --batch call per arm.
perref = {}
for name, rows in arms.items():
    jobs = [(f"{name}__{j}",
             [rows[i][0] for i in groups[k]],
             [rows[i][1] for i in groups[k]]) for j, k in enumerate(keys)]
    res = panel_batch(jobs, stats="srocc")
    perref[name] = [r["srocc"] for r in res]
    print(f"{name}\tper_ref_mean={statistics.fmean(perref[name]):.4f}\t"
          f"median={statistics.median(perref[name]):.4f}\tn={len(keys)}")

# ---- pooled, reference-clustered paired bootstrap ----------------------
# The headline rank axis. Resampling REFERENCES (not pairs) is the campaign's
# own convention for a CID22 CI (the F8 floor was derived reference-clustered),
# and it is the honest unit: the 4,292 pairs are 49 clusters, not 4,292
# independent draws.
rng0 = random.Random(SEED)
pooled_idx = [[rng0.randrange(len(keys)) for _ in keys] for _ in range(B_ITERS)]
pooled_jobs = {}
for name, rows in arms.items():
    jobs = [(f"{name}_p{j}",
             [rows[i][0] for k in s_ for i in groups[keys[k]]],
             [rows[i][1] for k in s_ for i in groups[keys[k]]])
            for j, s_ in enumerate(pooled_idx)]
    # point estimate first
    pt = panel_batch([(f"{name}_pt", [r[0] for r in rows], [r[1] for r in rows])],
                     stats="srocc")[0]["srocc"]
    pooled_jobs[name] = (pt, jobs)
print(f"\n# pooled {CORPUS} SROCC (reference-clustered bootstrap)")
print("arm\tpooled_srocc")
pooled_boot = {}
for name, (pt, jobs) in pooled_jobs.items():
    res = panel_batch(jobs, stats="srocc")
    pooled_boot[name] = [r["srocc"] for r in res]
    print(f"{name}\t{pt:.4f}")
print("\ncandidate\tpooled\tssim2\tdelta\tCI95_lo\tCI95_hi\tP(cand>ssim2)")
for name in [k for k in pooled_boot if k != "ssim2"]:
    d = [a - b for a, b in zip(pooled_boot[name], pooled_boot["ssim2"])]
    ds = sorted(d)
    lo = ds[int(0.025 * B_ITERS)]; hi = ds[int(0.975 * B_ITERS) - 1]
    pwin = sum(1 for x in d if x > 0) / B_ITERS
    print(f"{name}\t{pooled_jobs[name][0]:.4f}\t{pooled_jobs['ssim2'][0]:.4f}\t"
          f"{statistics.fmean(d):+.4f}\t{lo:+.4f}\t{hi:+.4f}\t{pwin:.3f}")

print("\n# within-image (per-reference) paired bootstrap")
rng = random.Random(SEED)
idx_sets = [[rng.randrange(len(keys)) for _ in keys] for _ in range(B_ITERS)]
print("\ncandidate\tmean\tssim2\tdelta\tCI95_lo\tCI95_hi\tP(cand>ssim2)")
for name in [k for k in perref if k != "ssim2"]:
    d = [a - b for a, b in zip(perref[name], perref["ssim2"])]
    boots = [statistics.fmean([d[i] for i in s]) for s in idx_sets]
    boots.sort()
    lo = boots[int(0.025 * B_ITERS)]
    hi = boots[int(0.975 * B_ITERS) - 1]
    pwin = sum(1 for b in boots if b > 0) / B_ITERS
    print(f"{name}\t{statistics.fmean(perref[name]):.4f}\t"
          f"{statistics.fmean(perref['ssim2']):.4f}\t{statistics.fmean(d):+.4f}\t"
          f"{lo:+.4f}\t{hi:+.4f}\t{pwin:.3f}")
