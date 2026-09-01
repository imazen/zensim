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

BAND MODE (`BAND_LO` / `BAND_HI`, added by the hfnl944 lane 2026-09-01).
Restricts every arm to the rows whose SHARED human target falls in
`[BAND_LO, BAND_HI)`, so the pairing the whole test rests on is untouched — the
band is cut on the target, which is identical index-wise across arms by the
assertion above. This is what makes a NEAR-LOSSLESS clause answerable on human
labels: the exam's near-lossless corpora (`hf_nearlossless`, `hfnlproxy`) are
ssim2 SELF-TARGETS, so the opponent scores 1.0 there by construction and no
model can ever beat it; the high-MOS band of CID22 is the same zone measured
against people. In band mode two extra guards run, because a restricted range
is where the appendix-V band defects live:

  * references with fewer than `PER_REF_MIN_ROWS` (3) in-band pairs, or with no
    spread on either vector, are DROPPED — the same filter
    `zenstats::per_group_srocc` applies, so a per-ref mean here means what
    `bake_verdict`'s does;
  * `srocc_signed` is printed for every arm's point estimate, and the minimum
    over all bootstrap draws is printed, so a reader can confirm the |SROCC|
    the fast bootstrap path returns is equal to the signed value on every draw
    rather than assuming it.

Default (no band) output is unchanged, byte for byte — the extra lines are
band-mode only, so `paired_boot_10k.txt` still reproduces.
"""
import csv, os, sys, random, statistics
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "lib"))
from zen_stats import panel_batch  # noqa: E402

O = os.environ.get("O", "/mnt/v/output/zensim/ssim2-bar-2026-08-31")
RM = "/mnt/v/output/zensim/reports/refmetrics"
B_ITERS = int(os.environ.get("BOOT", "10000"))
SEED = int(os.environ.get("SEED", "20260901"))
BAND_LO = float(os.environ["BAND_LO"]) if os.environ.get("BAND_LO") else None
BAND_HI = float(os.environ["BAND_HI"]) if os.environ.get("BAND_HI") else None
BANDED = BAND_LO is not None or BAND_HI is not None
# `zenstats::per_group_srocc`'s own floor (bake_verdict PER_REF_MIN_ROWS).
PER_REF_MIN_ROWS = 3

def model_rows(path):
    rows = [l.rstrip("\n").split("\t") for l in open(path) if l.strip()]
    h = rows[0]; i = {n: k for k, n in enumerate(h)}
    return [(float(r[i["pred"]]), float(r[i["human"]]), r[i["ref"]]) for r in rows[1:]]

# corpus -> (peer tsv, human col, metric col, human scale divisor)
CORPORA = {
    "cid22": ("cid22_ssim2.tsv", "MCOS", "ssim2", 100.0),
    "csiq": ("csiq_ssim2_gpu.tsv", "human_score", "ssim2_gpu", 1.0),
    "aic3": ("aic3_ssim2_heldout.tsv", "jnd", "ssim2_gpu", 1.0),
    # LIVE and AIC-4 added 2026-09-01 (hybrid lane) — and the reason the
    # original exclusion was right is worth keeping, because it is ROOT-scoped,
    # not corpus-scoped. MEASURED, same peer table, three different dumps of
    # the SAME 779 LIVE pairs:
    #
    #   372 root  (`2026-08-30-full-features-372`, shipped B)  max |Δ| = 1.12
    #   944 root  (`ext944-canonical-2026-08-01`, W10L9PH)     max |Δ| = 0.0
    #   pools-944 (`r1b-pools944-2026-08-30`,     W10L9PH)     max |Δ| = 0.0
    #
    # So the exam's note ("its peer table and the verdict per-pair dump hold
    # the same 779 pairs in DIFFERENT row order") is TRUE of the 372 root and
    # FALSE of every 944-class root — the row order of `ext_live.parquet` at
    # 944 width already matches `live_ssim2_gpu.tsv` exactly. AIC-4 matches on
    # the 944 roots too (max |Δ| = 0.0 over 300 rows). Nothing here weakens the
    # guard: the index-wise assertion below still runs on every arm, so a
    # 372-root dump fed to CORPUS=live aborts loudly instead of pairing a
    # fiction. KonJND stays out: its peer table is the DILUTED 1,008-ref ruler
    # and the JPEG-504 cut is a `dist_path` filter the peer row applies but
    # this script does not, so pairing it needs a join, not an index.
    "live": ("live_ssim2_gpu.tsv", "human_score", "ssim2_gpu", 1.0),
    "aic4": ("aic4_ssim2_gpu.tsv", "human_score", "ssim2_gpu", 1.0),
}
CORPUS = os.environ.get("CORPUS", "cid22")

def peer_rows():
    f, hc, mc, div = CORPORA[CORPUS]
    r = list(csv.DictReader(open(f"{RM}/{f}"), delimiter="\t"))
    return [(float(x[mc]), float(x[hc]) / div, os.path.basename(x["ref_path"])) for x in r]

arms = {"ssim2": peer_rows()}
# ARMS is env-overridable so a lane can add a candidate without editing the
# default list — which stays exactly as the exam ran it, so the committed
# `paired_boot_10k.txt` reproduces from a flagless invocation.
for name in os.environ.get("ARMS", "B ADD156 W10L9P W10L9PH").split():
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

# ---- band restriction (optional) --------------------------------------
# Cut on the SHARED target, so every arm keeps exactly the same rows and the
# pairing survives. Applied before grouping so the per-ref filter below sees
# in-band counts.
band_note = ""
if BANDED:
    keep = [i for i in range(n)
            if (BAND_LO is None or base[i] >= BAND_LO)
            and (BAND_HI is None or base[i] < BAND_HI)]
    assert keep, "band selection is empty"
    arms = {k: [v[i] for i in keep] for k, v in arms.items()}
    base = [base[i] for i in keep]
    n = len(keep)
    band_note = (f" band=[{BAND_LO}, {BAND_HI}) span={max(base) - min(base):.6f}")

# Reference grouping comes from the PEER table (names); the model dumps carry
# integer ids, and the two agree because the row order is identical.
refs = [r for _, _, r in arms["ssim2"]]
groups = {}
for i, r in enumerate(refs):
    groups.setdefault(r, []).append(i)
keys = sorted(groups)
if BANDED:
    # Same filter zenstats::per_group_srocc applies, so a per-ref mean on a
    # band means what bake_verdict's per_ref_mean means on the full corpus.
    def usable(k):
        idx = groups[k]
        if len(idx) < PER_REF_MIN_ROWS:
            return False
        for arm in arms.values():
            xs = [arm[i][0] for i in idx]
            if all(x == xs[0] for x in xs):
                return False
        ys = [base[i] for i in idx]
        return any(y != ys[0] for y in ys)
    dropped = [k for k in keys if not usable(k)]
    keys = [k for k in keys if k not in set(dropped)]
    assert keys, "no reference survives the per-ref floor in this band"
    band_note += (f" refs_kept={len(keys)} refs_dropped={len(dropped)}"
                  f" (<{PER_REF_MIN_ROWS} in-band pairs or no spread)")
print(f"# corpus={CORPUS}:{band_note} {len(keys)} references, {n} pairs, "
      f"{B_ITERS} bootstrap resamples, seed {SEED}")
if BANDED:
    # State the SIGN once per arm. The bootstrap below uses panel's |SROCC|
    # fast path; on a restricted range that is only equal to the signed value
    # while nothing crosses zero, which the printed bootstrap minimum checks.
    sgn = panel_batch([(f"sgn_{k}", [r[0] for r in v], [r[1] for r in v])
                       for k, v in arms.items()], stats="full")
    print("# signed point estimate (pooled, in-band), panel --batch stats=full:")
    for k, r in zip(arms, sgn):
        print(f"#   {k:10s} srocc_signed={r['srocc_signed']:+.6f}  |srocc|={r['srocc']:.6f}")

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
if BANDED:
    print("# min |SROCC| over all bootstrap draws (a value > 0 means the fast "
          "|.| path equals srocc_signed on every draw):")
    for name in pooled_boot:
        print(f"#   {name:10s} min={min(pooled_boot[name]):.6f}")
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
