# Diffmap comparison — all three avenues, results (#38, 2026-05-27)

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

## Result — the shipped v47 zensim diffmap (default) wins the smoke

**Three loops at matched bytes (independent dssim, lower=better):**

| loop (diffmap) | mean dssim | per-image |
|---|--:|---|
| **zensim-v47 (#32, default mask8)** | **0.00056** | wins both photos |
| butteraugli (reference) | 0.00058 | mid |
| cvvdp (#37 / avenue 3) | 0.00060 | **wins screen** (codec_wiki) |

zensim-v47 drives the best aggregate RD; cvvdp edges it on screen content,
zensim wins photos. (Diffs are small + n=3 — directional only.)

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

## Choice + recommendation

**Winner (smoke): keep the shipped v47 zensim diffmap with default mask8** —
best aggregate RD, and it already embodies the #33 activity-relative mechanism
in tuned form. No change to ship.

Two threads worth a bigger-corpus follow-up (NOT shippable on n=3):
1. **mask2 hint** — marginally better butteraugli than mask8 on the smoke;
   confirm on the multi-content + low-q corpus before changing the default.
2. **cvvdp-for-screen** — cvvdp's screen-content RD edge suggests a
   content-adaptive diffmap choice (zensim for photo, cvvdp for screen) could
   beat either alone. Worth a real RD sweep across content classes.

Honest gaps: n=3, high-q only (d=1–3), dssim-primary. The user's sweep
discipline (tiny+small+medium+large × q5–q100 × content classes) is the bar
for a *shipping* diffmap decision; this smoke ruled the direction (v47-default
holds; cvvdp not a clear win; #33≡masking for RD).
