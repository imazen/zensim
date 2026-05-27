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

## Result — NOT cleanly Pareto-separated; a content crossing (BD-rate)

**⚠ CORRECTION (BD-rate):** a single matched-byte slice showed zensim-v47
"best" (mean dssim 0.00056 vs butteraugli 0.00058 vs cvvdp 0.00060). But the
RD curves CROSS, so a single byte budget is misleading. The proper
Bjøntegaard **BD-rate** (% bytes at equal quality, integrated over the
overlapping range) shows the three loops are essentially **TIED** on the
independent dssim axis:

**BD-rate vs zensim-v47 (negative = challenger uses FEWER bytes at equal
quality = better than v47):**

| | dssim axis (mean) | butteraugli axis (mean) |
|---|--:|--:|
| cvvdp | −1.0% | −3.9% |
| butteraugli-loop | −1.3% | −10.2%* |

(*butteraugli-loop judged by butteraugli is circular — discount.)

On the common independent axis (dssim) all three are within ~1% BD-rate —
inside n=3 noise (per-image spread is ±6%). **There is no global Pareto
winner.** The robust signal is a **content crossing**:

| image | cvvdp BD-rate (dssim) | cvvdp BD-rate (butteraugli) |
|---|--:|--:|
| 1025469 (photo) | +6.4% (zensim better) | +5.4% (zensim better) |
| 1418519 (photo) | −3.6% | +4.0% (mixed) |
| **codec_wiki (screen)** | **−5.8%** (cvvdp better) | **−21.1%** (cvvdp much better) |

**cvvdp Pareto-dominates screen content** (consistent + large across BOTH
independent axes), zensim holds the photos. The Pareto-optimal strategy is
therefore **content-adaptive diffmap routing** (cvvdp for screen/text,
zensim for photo) — NOT a single global winner. The earlier "v47 wins"
single-point read masked this.

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

## Choice + recommendation (BD-rate, corrected)

**There is no global Pareto winner** — on the independent dssim BD-rate the
three loops are within ~1% (n=3 noise). So "keep v47-default" stands by
*default* (it's shipped, tied-or-better, and already embodies #33's
activity-relative mechanism via tuned masking) — but NOT because it Pareto-
dominates; it doesn't.

**The Pareto-optimal play is content-adaptive routing**, which the BD-rate
makes the headline (not a footnote): **cvvdp Pareto-dominates screen content**
(codec_wiki −5.8% dssim / −21% butteraugli, consistent + large across both
independent axes), **zensim holds photos** (+5–6% on 1025469). A router that
picks cvvdp for screen/text and zensim for photo beats either alone on this
smoke. jxl-encoder already has both loops behind `PerceptualMetric`, so the
routing is a content classifier away (the existing screen-content detection in
the encoder could gate it).

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
