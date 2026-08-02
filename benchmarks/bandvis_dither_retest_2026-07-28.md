# BANDVIS dst-side dither retest — real content, the +5% fix decision (2026-07-28)

> **2026-08-02 UPDATE:** the plane was subsequently BUILT (opt-in
> `append2_dst_activity`, default OFF) and adjudicated for SOTA-944 P1.5 —
> both pre-registered masking arms FAILED their suppression gates and this
> doc's DEFER verdict is CONFIRMED by direct measurement (mechanism:
> flatness masks inside `bounded_excess` are ratio-cancelled; at the
> resonant scale banding contours ARE local activity). See
> `bandvis_dst_activity_2026-08-02.md`.

Decides append2 REMAINDERS #3: is the Y-only dst-activity plane (~+5% CPU,
the V3(b) dst-masking fix) warranted? Evidence base: the
**grain-pathology-2026-07-28** starter tranche
(`/mnt/v/output/grain-pathology-2026-07-28/`, 1,272 pairs, foldapp2-944,
DATA_PROVENANCE §grain-pathology-2026-07-28) — the first REAL-content
posterize+dither corpus — overlaid on the LIVE-YT-Banding external run
(`bandvis_lyb_validation_2026-07-28.md`). Analysis:
`benchmarks/bandvis_dither_retest_2026-07-28.py` (pure analysis; generates
nothing). Pinned context: V3(b) measured dst-dither cross-fire 1.55×
(ordered) / 1.72× (noise) on the 256² synthetic ramp; LYB measured LOSS as
the workhorse (−0.441 folds) with GAIN weak (−0.15).

Corpus slice used: posterize_dither = luma posterize {6,5,4,3}-bit ×
{undithered, Bayer o8x8, Floyd-Steinberg} over 26 imazen-26 origins ×
{native, s1024} (546 pairs; 6-bit s1024-only); plus the other three
families for polarity context. Dither is luma-only; no true blue-noise
mask (FS is the blue-noise-ish surrogate).

## (1) Masking suppression on real content: there is none — dither FIRES, reproducing the fixture

GAIN_dithered / GAIN_undithered at matched (source, size, bits), groups
with GAIN_undithered < 1e-4 (=C_BV) excluded (≤4 per cell):

| dither | s0 | s1 | s2 | s3 |
|---|--:|--:|--:|--:|
| bayer med [q25,q75] | 1.62 [1.03, 10.50] | 1.56 [1.15, 6.80] | 1.65 [1.17, 4.16] | 1.11 [0.84, 1.98] |
| bayer frac>1 | 0.76 | 0.84 | 0.88 | 0.60 |
| fs med | 1.55 [1.07, 11.05] | 1.42 [1.10, 7.01] | 1.42 [1.05, 4.70] | 1.26 [0.79, 4.32] |
| fs frac>1 | 0.80 | 0.82 | 0.80 | 0.60 |

The fixture numbers (1.55/1.72) are REPRODUCED on photographic/document/
synthetic content at s0–s2 (medians 1.42–1.65, upper quartiles 4–11×).
s3 is mildest (1.11–1.26, frac>1 = 0.60): pooled downscaling averages the
dither pattern out before the resonant banding scale. Caveat: the
undithered denominator carries the corpus's ~±1-code YCbCr-roundtrip
floor, which if anything UNDERSTATES the multiplier.

## (2) Cross-fire magnitude: squarely inside the real-banding GAIN band

Absolute GAIN medians, pooled sizes (n=26 for b6, else 52):

| bits | mode | s0 | s1 | s2 | s3 |
|---|---|--:|--:|--:|--:|
| b6 | none | 0.0172 | 0.0135 | 0.0102 | 0.0068 |
| b6 | bayer | 0.0339 | 0.0198 | 0.0131 | 0.0051 |
| b5 | bayer | 0.0437 | 0.0279 | 0.0205 | 0.0084 |
| b4 | bayer | 0.0421 | 0.0327 | 0.0244 | 0.0118 |
| b3 | bayer | 0.0482 | 0.0353 | 0.0299 | 0.0191 |
| b3 | fs | 0.0502 | 0.0319 | 0.0274 | 0.0214 |

Scales with bits mildly (b6→b3 ≈ +45%), but even at **6-bit** — banding
near/below threshold — dither GAIN (0.033–0.034 at s0) already sits at
the LYB real-banding p25–p50. Against the LIVE-YT-Banding distorted-video
distribution (frame-sampled, same feature/units; an amplitude reference,
not a matched protocol):

| scale | LYB GAIN med [q25,q75] | LYB p90 | dithered pairs ≥ LYB-p50 | ≥ p90 |
|---|--:|--:|--:|--:|
| s0 | 0.0588 [0.0394, 0.0765] | 0.1050 | **37.1%** | 26.1% |
| s1 | 0.0378 [0.0256, 0.0592] | 0.0758 | **37.6%** | 29.1% |
| s2 | 0.0334 [0.0259, 0.0459] | 0.0555 | 36.3% | 25.8% |
| s3 | 0.0354 [0.0280, 0.0433] | 0.0533 | 19.2% | 14.8% |

**A third of dither-dst pairs reach median-or-worse real-banding GAIN
amplitude** (a quarter reach p90). The cross-fire is NOT small in absolute
terms: an untrained consumer of GAIN alone would see dither as banding at
damaging magnitude. (Whether a *trained* head is damaged is §4.)

**Workhorse check — LOSS on the posterize modes** (medians):

| mode | s0 | s1 | s2 | s3 |
|---|--:|--:|--:|--:|
| none | 0.1674 | 0.1055 | 0.0650 | 0.0488 |
| bayer | 0.2007 | 0.0677 | 0.0245 | 0.0130 |
| fs | 0.1788 | 0.0608 | 0.0210 | 0.0093 |

LOSS legitimately fires on undithered posterize (plateau-flattening
destroys in-band ref micro-structure — the same removal-side signal that
carried LYB). Dst-dither suppresses LOSS at s1–s3 by ~35–80%: added dst
curvature "fills in" b_dst. Note this is DIRECTIONALLY defensible —
dithered quantization genuinely looks less banded (that is dither's
purpose) — so partial LOSS suppression is not pure error; but the
magnitude is worth knowing for calibration.

## (3) Polarity across the other families (GAIN/LOSS medians, s0..s3)

| family/variant | GAIN s0..s3 | LOSS s0..s3 | read |
|---|---|---|---|
| regrain/den (nlmeans) | .061 .060 .038 .029 | **.104** .070 .042 .021 | LOSS = detail-loss semantics ✓; GAIN 0.06 on pure denoise = smoothing-plateau cross-fire (same class as V3(b), no dither needed) |
| regrain/rg05 | .119 .143 .087 .049 | .036 .016 .016 .014 | grain at ½σ SUPPRESSES the denoise-LOSS 3× and fires GAIN |
| regrain/rg10 | .109 .117 .086 .052 | .070 .025 .016 .013 | matched regrain restores LOSS only partially |
| regrain/rg20 | .083 .087 .077 .052 | **.142** .049 .021 .015 | 2×σ OVERSHOOTS the visibility band: b_dst rolls off past δ_hi, LOSS *re-rises above den* and GAIN falls — the band-pass cap, coherent with design |
| jxl/d1_plain vs d1_noise | .065 vs .067 (s0) | .043 vs .041 | content-noise estimation mostly not engaged (flag in corpus log) — weak evidence column |
| jxl/d1_photon3200 − d1_plain | +.022 (s0) | ≈0 | photon noise = GAIN, LOSS untouched |
| jxl/d3_photon800 | .156 .085 .044 .022 | .045 .033 .023 .009 | distance + noise compound; largest GAIN in the corpus |
| av1 q128 pn24−pn0 (paired) | **+.018 +.010 +.007 +.004** | **−.007 −.014 −.008 −.004** | grain synthesis adds GAIN and MASKS the banding-LOSS signal by ~10–25% of its q128 magnitude |
| av1 q80 pn24−pn0 | +.009 +.006 +.006 +.005 | ≈0 −.001 −.001 −.001 | milder at higher quality |

Polarity surprises worth pinning: (i) pure DENOISE fires GAIN at
banding-band amplitude (0.06) — smoothing creates soft plateaus; the
V3(b) class is broader than dither. (ii) rg20's LOSS > den's LOSS — the
band-pass contrast cap makes over-regrained content read as *more*
structure-loss than the denoise it papers over. (iii) AV1 film-grain
synthesis partially suppresses LOSS — semantically defensible (grain
visually masks banding; that is why the tool exists) but a calibration
drift the head should see grain lanes alongside.

## (4) The trainer-side gate already exists: TEXTURE_DISSIM separates dither from banding at AUC 0.977

Classes: dithered-dst (n=364) vs undithered-dst (n=182), posterize family.
Rank AUC per single feature (0.5 = inseparable; values <0.5 = separable
with dithered LOW):

| lane | AUC | direction |
|---|--:|---|
| **texture_dissim_s3 (app Y, local 9)** | **0.023** | dithered LOW |
| texture_dissim_s2 | 0.033 | dithered LOW |
| contrast_gain_s3 | 0.078 | dithered LOW |
| contrast_loss_s0 | 0.082 | dithered LOW |
| bandvis_gain_s1 (the confusion itself) | 0.782 | dithered HIGH |
| mscn_diff_s0 | 0.695 | dithered HIGH |

Physics: dither RESTORES ref-like local variance inside plateaus →
`1 − bounded_sim(var₁, var₂)` collapses toward 0; undithered banding both
flattens texture (variance deficit) and leaves step-flank variance spikes
→ texture_dissim stays high. At the pooled level the 924 vector already
carries an almost-perfect dither-vs-banding discriminator, and it is
**independent of GAIN** (corr with bandvis_gain_s1 within dithered rows:
+0.04). Operating points (Youden):

- `texture_dissim_s3 <= 1e-4`: catches **95.3%** of dithered at **6.0%**
  false-flag on undithered.
- `contrast_gain_s3 <= 0.0088`: 93.4% @ 18.1%.

A head trained on the 944 vector with these lanes unmasked can learn
"GAIN high AND texture_dissim_s3 high → banding; GAIN high AND
texture_dissim_s3 ≈ 0 → dither/regrain" with zero new CPU.

## Decision matrix → **DEFER the +5% plane; RESHAPE = trainer-side gating (zero cost)**

| option | evidence for | evidence against |
|---|---|---|
| **Fix NOW (+5% dst-activity plane)** | cross-fire real on real content: 1.4–1.65× at s0–s2, 26–37% of dither pairs at real-banding GAIN amplitude (§1–2) | GAIN — the polarity the fix rescues — is already the WEAK polarity on real banding MOS (LYB −0.15 vs LOSS −0.44); +5% CPU buys specificity for a feature whose predictive value is unproven; and §4 shows the specificity is purchasable for free |
| **DEFER (do nothing)** | LOSS (the workhorse) is only partially suppressed and directionally defensibly so (§2); head sees MSCN/HF/texture lanes anyway | leaves GAIN un-gated for any consumer that reads it as a scalar banding score outside a trained head |
| **RESHAPE (trainer-side 2-feature gate)** | texture_dissim_s3 separates dither-vs-banding at AUC 0.977, independent of GAIN, already computed, zero cost (§4) | pooled-level only — if BANDVIS is ever consumed as a per-tile MAP (CAMBI-style localization), pooled gating does not localize and the dst-activity plane (or A8 soft-tile) regains relevance |

**Verdict: DEFER the +5% dst-activity plane; adopt the RESHAPE.** The
cross-fire is real and material (the fixture pin generalizes to real
content almost unchanged), but (a) it afflicts the polarity that LYB
already showed weak, (b) the workhorse LOSS survives dither at defensible
magnitude, and (c) the 944 vector already contains an AUC-0.977,
GAIN-independent, zero-cost discriminator. Spending +5% CPU to rescue
GAIN's specificity before the LOO-on-944 bake (REMAINDERS #2) can even
show GAIN earns weight would be paying for specificity nobody has yet
proven they need. Concretely:

1. Keep append2 as landed; do NOT build the dst-activity plane now.
2. In the 944 bake/LOO round: keep `texture_dissim_*` and
   `contrast_gain/loss_*` UNMASKED alongside BANDVIS, and evaluate GAIN's
   LOO jointly with texture_dissim_s3 (the gate pair), not alone.
3. Re-open the fix ONLY if (i) LOO shows the head failing to exploit the
   gate, or (ii) a per-tile BANDVIS map becomes a requirement — in which
   case prefer the A8 soft-tile pooling route (REMAINDERS #5) over a raw
   activity plane.
4. Pin separately: pure-denoise GAIN cross-fire (0.06 with no dither) and
   AV1-grain LOSS masking (−10–25%) — neither is addressed by a
   dst-DITHER gate; both are visible to the head via contrast/texture
   lanes; both belong in the LOO read-out.

## Limitations

- Corpus dither is luma-only Bayer/FS at 4 bit depths over 26 origins;
  no true blue-noise mask, no chroma dither, starter scale (546
  posterize pairs). Fleet-scale expansion may sharpen the quartiles but
  the effect directions are consistent across all four scales and both
  dither types.
- LYB overlay is cross-protocol (frame-sampled AV1 video vs still
  pairs): used as an amplitude reference for "real banding GAIN levels",
  not as a matched experiment.
- The AUC-0.977 separation is measured on THIS corpus's dither
  (regular Bayer + FS error diffusion); photon-noise/film-grain textures
  land in the same texture_dissim-low class by mechanism (variance
  restoration), but that specific AUC was not re-measured per family.
- Pooled-image features throughout; no per-tile locality claim.

## Reproduce

```
python3 benchmarks/bandvis_dither_retest_2026-07-28.py \
  --dataset /mnt/v/output/grain-pathology-2026-07-28 \
  --lyb-per-video ~/tmp/lyb-out/lyb_per_video.csv
# per-group ratio dump: ~/tmp/bandvis-retest/suppression_ratios.csv
```
