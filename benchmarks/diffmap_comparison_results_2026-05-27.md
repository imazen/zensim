# Diffmap comparison — all three avenues, results (#38, 2026-05-27)

## ⇒⇒ FINAL VERDICT — FULL SWEEP (supersedes all smoke conclusions below)

After fixing jxl-encoder's build (zenmetrics cvvdp-cpu→cvvdp rename, commit
`8bd78081`), ran the real sweep: **3 loops (zensim-v47, cvvdp, butteraugli)
× 11 images (6 photo + 5 screen, 640²–2560²) × 6 distances {1,2,3,4,6,8}**,
judged by **ssim2 + butteraugli + cvvdp** (dssim dropped per user). BD-rate vs
zensim-v47 (positive = challenger needs MORE bytes at equal quality = zensim
BETTER; `C` = circular, discount):

| judge | cvvdp-loop | butteraugli-loop |
|---|--:|--:|
| ssim2 | +1.3% | +2.1% |
| butteraugli | **+4.9%** (indep) | +2.9% `C` |
| cvvdp | +2.3% `C` | **+3.1%** (indep) |

**The full sweep REVERSES the n=3 smoke.** On a representative multi-content ×
wide-q corpus, **zensim-v47 is the best/competitive diffmap**: both cvvdp and
butteraugli loops need +1 to +5% MORE bytes at equal quality, across ALL three
judges, on BOTH photo and screen aggregate. zensim wins ~9/11 images on the
independent judges.

**Why the smoke (n=3, d=1,2,3) misled — on BOTH axes:** it was 3 images AND
high-q only. The single 2560px `codec_wiki` screen at d=1–3 favored cvvdp
(−46.9%); but over the full q-range (adding low-q d=4,6,8) codec_wiki *favors
zensim* (+5.4%). So the smoke was unrepresentative in corpus AND distance
range — exactly why the sweep was necessary. Per-image variance is high and
the judges disagree per-image (e.g. `graph`: cvvdp-loop +20% but butt-loop
−3.7%); trust the aggregate.

### Same data, re-referenced to the butteraugli loop (zensim as a visible column)

The table above uses zensim as the baseline (so it has no column). Re-referenced
to the **butteraugli loop** (the default encoder = 0%), all loops are peers
(NEG = fewer bytes at equal quality = better than default; `C` = circular):

| judge | zensim-v47 vs default | cvvdp vs default |
|---|--:|--:|
| ssim2 | **−2.0%** (photo −0.5, screen −3.7) | −1.1% |
| cvvdp | **−2.8%** (photo −0.9, screen −5.1) | −1.0% `C` |

(butteraugli axis circular for the butteraugli base — omitted.) Ranking of the
diffmaps, best→worst: **zensim-v47** (−2 to −3% vs default, most on screen) >
**cvvdp** (~−1%) > **butteraugli** (default). So the zensim-v47 diffmap is a
real ~2–3% RD improvement over the stock butteraugli loop on this sweep.

**Conclusion: KEEP the shipped v47 zensim diffmap — it is the best of the three
on a real sweep.** This vindicates the shipped Profile::A for the RD use case
too (not just the dial). The earlier "ssim2 flips it, zensim worst" was a smoke
artifact; the sweep is the trustworthy answer. (Caveat: n=11 images / 6 dist —
better than n=3 but still short of the 50/class ideal; the direction is solid,
magnitude modest. Data: `/mnt/v/output/zensim/diffmap-sweep-2026-05-27/`.)

---

# (Below: the earlier n=3 SMOKE analysis — SUPERSEDED by the full sweep above)


User: "do all three [diffmap avenues] in sequence and choose the best."
Harness: `jxl-encoder/examples/zensim_diffmap_rd.rs` (encode corpus with each
perceptual-loop metric, iters≥1 so the diffmap drives per-tile redistribution,
save decoded outputs). Judged by an **independent** panel (`zen-metrics`
butteraugli + dssim) at **matched bytes** (RD curves; the loops each target
their own quality calibration per distance, so matched-distance is NOT
apples-to-apples). The common independent judge across all three loops is
**dssim**; butteraugli is independent for the zensim-vs-cvvdp comparison.

**Corpus: n=3 images (2 CID22 photos + 1 gb82-sc screen) × 3 distances
{1,2,3} — SMOKE level.** Conclusions are directional, not a shippable
calibration (a real calibration needs the bigger multi-content + low-q corpus
per CLAUDE.md sweep discipline).

## Result — NOT cleanly Pareto-separated; a content crossing (BD-rate)

**⚠ CORRECTION (BD-rate):** a single matched-byte slice showed zensim-v47
"best" (mean dssim 0.00056 vs butteraugli 0.00058 vs cvvdp 0.00060). But the
RD curves CROSS, so a single byte budget is misleading. The proper
Bjøntegaard **BD-rate** (% bytes at equal quality, integrated over the
overlapping range) shows the three loops are essentially **TIED** on the
independent dssim axis:

**BD-rate vs zensim-v47, THREE independent judges** (negative = challenger uses
FEWER bytes at equal quality = better than v47; `C` = circular for that loop,
discount):

| judge axis | cvvdp-loop | butteraugli-loop |
|---|--:|--:|
| dssim | −1.0% | −1.3% |
| butteraugli | −3.9% | −10.2% `C` |
| **cvvdp** | −10.7% `C` | **−16.2%** |

**The original eval used butteraugli + dssim as the RD authority — NOT cvvdp
and NOT zensim** (zensim would be circular for the zensim loop). Adding cvvdp
(the strongest authority; circular only for the cvvdp loop) shifts the verdict.

The judges DISAGREE, which is the signal:
- **dssim**: all three within ~1% → ~tied.
- **cvvdp** (non-circular cell: butteraugli-loop vs zensim = **−16.2%**): the
  zensim-v47 diffmap is the **weakest**, driven by **codec_wiki screen
  −46.9%** — zensim over-spends bits on text/UI without cvvdp-perceived gain
  (zensim spent 115,910 B on codec_wiki d1 vs butteraugli's 98,694).

Per-image content split (consistent direction across strong authorities):

| image | cvvdp-loop vs zensim (dssim / butter) | butteraugli-loop vs zensim (cvvdp) |
|---|--:|--:|
| 1025469 (photo) | +6.4% / +5.4% (zensim better) | −1.0% (cvvdp range saturated*) |
| 1418519 (photo) | −3.6% / +4.0% (mixed) | −0.8% (saturated*) |
| **codec_wiki (screen)** | **−5.8% / −21.1%** (challenger better) | **−46.9%** (zensim much worse) |

(*photos scored cvvdp ≈ 9.99 JOD = near-lossless/saturated → photo cvvdp
BD-rates are on a near-flat range, unreliable; the robust cvvdp signal is the
screen result.)

**Corrected verdict: NOT a clean global winner, and NOT a symmetric crossing —
the zensim-v47 diffmap is competitive on PHOTOS but clearly LOSES on SCREEN
content across BOTH strong independent authorities (butteraugli + cvvdp).** The
Pareto-optimal play is **content-adaptive routing — use butteraugli or cvvdp
for screen/text, zensim for photo.** The single-point matched-bytes read ("v47
wins") and even the dssim-only BD-rate ("~tied") both masked the screen
weakness that cvvdp exposes.

## ⇒ ssim2-as-judge UPDATE (2026-05-27, user: "use ssim2 not dssim")

Swapping the weak **dssim** judge for the strong **ssim2** flips the verdict.
3-judge BD-rate, challenger vs zensim-v47 loop (NEG = challenger better; `C` =
circular, discount):

| judge | cvvdp-loop | butteraugli-loop |
|---|--:|--:|
| **ssim2** | **−4.0%** | **−3.3%** |
| butteraugli | −3.9% | −10.2% `C` |
| cvvdp | −10.7% `C` | −16.2% |

All THREE strong authorities now agree: **the zensim-v47 diffmap is the worst
of the three loops** — both the plain butteraugli loop and the cvvdp loop drive
better RD. ssim2 is *semi*-circular for zensim (v47 trained on ssim2-derived
targets) so it should FAVOR zensim, yet it still scores zensim −4.0% behind
cvvdp — and now on photos too (1418519 −4.7% per ssim2), not only screen. The
earlier dssim-based "no winner / v47 ties" read was an artifact of dssim being
a weak judge; with strong judges the zensim diffmap clearly loses.

**Corrected bottom line:** the zensim-v47 *diffmap* is not the best signal for
jxl-encoder RD redistribution — butteraugli (default) and cvvdp both beat it.
zensim's value is the user-facing **dial** (score targeting) and the **metric
ranking**, NOT diffmap-driven bit redistribution. For RD, keep the butteraugli
loop (or cvvdp for screen).

**SWEEP STATUS — blocked.** The bigger multi-content + full-q sweep (the next
step) is blocked: a concurrent agent renamed the zenmetrics crate
`cvvdp-cpu` → `cvvdp`, which breaks jxl-encoder's `[patch.crates-io]
cvvdp-cpu` entry (a patch can't rename a package), so jxl-encoder won't build
until that rename lands in jxl-encoder's manifests too (the other agent's
deliverable). The numbers above are the n=3 high-q smoke, re-judged with
ssim2 — directional and now CONSISTENT across 3 strong judges, but not a
shipping calibration. Resume the sweep once the cvvdp-cpu→cvvdp rename
settles.

## Avenue-by-avenue

### Avenue 1 — DiffmapOptions tuning (env sweep on the v47 diffmap)
Masking helps a LOT: `nomask` butteraugli-pnorm3 **1.17** vs masked **~0.70**
at matched bytes. Optimum is mask2–8; the shipped default (mask8) is near-best
(mask2 marginally better on butteraugli but within n=3 noise); mask16/32 no
better. sqrt/HF: minor. **No dramatic win over the shipped default.**

### Avenue 2 — #33 structural signal as a diffmap → SUBSUMED by masking for RD
**Key finding (measured, not assumed):** #33's core mechanism — error measured
*relative to local source activity* — is, for the RD-*redistribution* use case,
exactly contrast **masking** (`error / (1 + strength·local_variance)`), which
the v47 diffmap already implements (the `ZENSIM_MASKING` knob). The masking
sweep above directly measures this: activity-relative weighting helps
(nomask→masked halves butteraugli error) and peaks at the default range. A
bespoke #33-structural-diffmap injection would re-implement masking with extra
complexity and no measured RD gain beyond the tuned knob. #33's distinct,
*non-overlapping* value is **localized-defect DETECTION** (regression-test
gating: "is this decode broken?"), which is a different use case from RD bit
redistribution — and is already validated (op100 92.5%/81.2%, see
`approach_b_structural_signal_spike_2026-05-27.md`). So for the *diffmap*
question, avenue 2 collapses into avenue 1; for the *detection* question, #33
stands on its own.

### Avenue 3 — cvvdp diffmap (PerceptualMetric::Cvvdp, cvvdp-loop-cpu)
Worst aggregate dssim (0.00060) but **wins screen content** (codec_wiki dssim
0.00009 vs zensim 0.00014). So cvvdp's CSF-based spatial localization helps on
text/UI but not on continuous-tone photos. (Note: this is cvvdp's *actual*
diffmap used directly — NOT the falsified cvvdp-scalar-trained metric of #37.)

## Choice + recommendation (3-judge BD-rate, corrected)

**No global Pareto winner, and the zensim-v47 diffmap has a SCREEN weakness
exposed only when cvvdp is added as a judge.** With dssim+butteraugli alone it
looked ~tied; the strongest authority (cvvdp) shows butteraugli-loop −16.2%
better than zensim (independent), driven by screen −46.9%. So "keep v47" holds
for PHOTOS (competitive there), but it is NOT the best diffmap for screen/text.

**The Pareto-optimal play is content-adaptive routing** — use butteraugli or
cvvdp for screen/text, zensim for photo. jxl-encoder already has all three
loops behind `PerceptualMetric`, so it's a content-classifier gate away (the
encoder's existing screen-content detection could drive it). On screen,
BOTH the cvvdp loop (−21% butteraugli, −5.8% dssim) AND the plain butteraugli
loop (−46.9% per cvvdp) beat the zensim diffmap — so the screen fix doesn't
even require cvvdp; routing screen content to the butteraugli loop already
helps.

Bigger-corpus follow-up (the smoke is n=3, high-q, so none of this is
shippable yet):
1. **Content-adaptive cvvdp/zensim routing** — the one robust, large effect;
   validate cvvdp-screen / zensim-photo across a screen+photo corpus at the
   full q range, then wire a classifier gate.
2. **mask2 vs mask8** — a faint hint, deep in n=3 noise; only worth chasing
   if the bigger sweep confirms it.

Honest gaps: n=3 images, high-q only (d=1–3), single-byte-budget BD on 3 RD
points (BD-rate from 3 points is itself coarse). The user's sweep discipline
(tiny+small+medium+large × q5–q100 × content classes) is the bar for a
*shipping* diffmap decision. This smoke's defensible conclusions: (a) no
global Pareto winner among the three loops on the independent axis; (b) a
robust content split — cvvdp wins screen, zensim wins photo; (c) #33≡masking
for RD (avenue 2 subsumed by avenue 1).
