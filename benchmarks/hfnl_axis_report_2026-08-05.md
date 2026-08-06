# The HF-NL-proxy axis, under the microscope (2026-08-05)

**Registered:** `benchmarks/sota944_campaign_2026-08-03.md` APPENDIX O (plan frozen +
pushed before any number; results in O.R). **Data:** 30-model per-reference SROCC
matrix + provenance at `/mnt/v/output/zensim/reports/hfnl-axis-2026-08-05/`; compact
machine-readable summary `benchmarks/hfnl_axis_2026-08-05.json` (feeds the gauntlet
HF-NL panel). Every SROCC herein comes from zenstats via `panel --batch`
(`srocc_signed`) or `bake_verdict` itself; means/CIs are arithmetic over those values.

**Prompted by** (user): "everyone is bombing HF-NL pretty badly compared to other
graphs … is it just a graph axis problem or do all models suck at it" + "k128 isn't
that bad". Short answers: **the axis was partly broken (80 board cells sign-flipped —
now repaired), the repaired axis is highly reliable, the top models do NOT bomb it —
they sit at the level independent perceptual metrics reach — and K128 is indeed
"not that bad": significantly above its class median, far below C.**

---

## 0. What this axis is

`rank.hfnlproxy.per_ref_mean`: over the 944-root near-lossless slice
(`ext_hfnlproxy.parquet` — the ssim2 ≥ 91 band of the held-out bigcodec TEST views;
11,356 pairs, 757 scoreable refs, ~11 pairs/ref median, codec mix **73.6% avif** /
14.7% jpeg / 6.1% webp / 5.6% jxl), group the pairs by reference image, compute the
**signed** Spearman correlation between the model's score and the ssim2-derived
target **within each reference**, and average over references. It answers: *"inside
one image's near-lossless ladder, does the model order the variants the way ssim2
does?"* — the question a target-hitting dial must get right in the q75-100 zone.

Pooled SROCC on this corpus is dominated by cross-image scale (the target spans 9
ssim2 points across images vs ~3.7 within one) and is NOT the headline — B reads
0.503 pooled beside 0.825 per-ref on the same rows.

## 1. The sign-flip repair (found by this study's reproduction gate)

Recomputing per-ref means from fresh per-pair dumps reproduced the board to ≤1e-15
for 21 of 24 gate-eligible models — and produced an **exact negation** for 8. Audit
of all 91 board cells with negative pooled signed SROCC (86 single re-verdicts + the
5 negative-pooled wave-5/6 ensembles from their frozen member lists): pooled
`srocc_signed` reproduced bit-identically on every one, and **80 cells were exact
per-ref sign flips**.

Mechanism: before the orientation pin (`730a386e`, 2026-08-04 16:49),
`per_group_srocc` ran `Orientation::Auto`, which points the per-ref statistic at the
**pooled** sign — on this corpus a noise-level quantity (|pooled| < 0.26 on every
affected cell). Every pre-pin verdict with pooled < 0 stored
`per_ref_mean = −(true value)`. Post-pin cells (W10 lane, all featsub) and all
pooled-positive cells are untouched.

**Repaired 2026-08-05**: all 80 board fullevals corrected in place via the new
sha-gated `promote_fulleval.py --repair-rank-orientation hfnlproxy` (refuses unless
every orientation-independent field is float-identical and the flip exact; superseded
values kept in `rank_graft_sources`). Post-repair scan: 233/233 board cells match the
pinned convention. Registry: `eval_annotations.json`
`hfnl-preauto-orientation-flip-REPAIRED`. Consequential doc corrections (campaign
O.R0): the arm-B "+0.19310280" HF-NL bar value is truly **−0.193** (arm B is
hfnl-inverted); "EM4 fails the HF-NL row" is **reversed**; the wave-5 "volatility
−0.115…+0.211" range shrinks to +0.041…+0.211; W8A's +0.20/+0.36/+0.42 are
−0.20/−0.36/−0.42.

Corrected board (233 cells): span **−0.486 … +0.848**, median **+0.168**.

## 2. Is the axis reliable? Yes — quantified

30-model matrix (755 common scoreable refs), registered procedures:

| statistic | value | meaning |
|---|---|---|
| Split-half model-ranking SROCC (20 shuffles, seed 4242) | **0.9919 ± 0.0028** | rank models on odd refs, again on even refs: same ordering |
| … Spearman–Brown full-length | **0.996** | the axis ordering is highly reliable |
| Split-half PLCC | 0.9983 → SB 0.9991 | |
| Per-model \|half₁−half₂\| | median 0.026, p90 0.034 | a single model's mean moves ~±0.03 between ref halves |
| Marginal 95% CI half-width (B=10,000, seed 777, shared resamples) | 0.023–0.047 per model | |
| **Axis LSD** — median paired 95% Δ half-width (435 pairs) | **0.039** (p10 0.022, p90 0.047) | **Δ < 0.04 is noise; Δ ≥ 0.05 is essentially always real** |

The registered CALL 1 threshold (r_SB ≥ 0.9 ⇒ reliable) is passed with margin. The
"HF-NL looks terrible" impression was three separable things: (i) 80 genuinely
wrong (flipped) numbers, now fixed; (ii) a weak *population* (the mid-944-MLP mass —
real, §4); (iii) presentation: the gauntlet's hfnlproxy *pooled scatter* cell is
range-restricted and looks like a shotgun blast even for a per-ref-excellent model —
use the new HF-NL panel instead.

## 3. The ceiling: what a strong metric can even get here — CORRECTED 2026-08-05 (full coverage)

> **COVERAGE CORRECTION (2026-08-05, user-prompted).** The first version of this
> section claimed "avif has no butteraugli/cvvdp anywhere" and computed the
> reference ceiling on a 118-ref / 2,410-pair non-avif subset. The data-location
> audit confirmed the avif 4-metric fleet fill (`fill-avif-b0..b7[-cpu]`,
> `s3://codec-corpus/jobs/`) was queued 2026-07-02 and **descoped by user
> directive the same day** at ~0.2% done (PLAN_BEAT_A amendment) — so no
> sidecar covered avif. The slice's own cells were then backfilled **locally**
> (no fleet spend, no re-encode): 8,360 avif members ranged-GET sha-verified
> from the mandfix4 box tars + 23 residual non-avif gaps, scored with the fill4
> metric implementations (0 failures). **Slice 4-metric coverage: 26.2% →
> 100.0% (11,356/11,356).** Agreement gates: bit-identical vs the 17
> Jul-2-fleet blob cells that did run (butteraugli 6e-8); the 4 old cvvdp
> mode-B NaN cells re-scored non-null. Sidecar (+R2+Tower):
> `fill4-6codec-2026-07-01/hfnl_avifgap_4metric_sidecar_2026-08-05.parquet`
> (sha `64ce4278…`). Campaign: APPENDIX O.R7; registry:
> `hfnl-ceiling-subset-superseded-fullcorpus`.

The target is ssim2-derived, so ssim2-vs-itself is +1.0 by construction. With
full coverage the reference rows are computed on the **same footing as every
model's axis number** — per-ref signed SROCC, the axis min-3 rule, all 757
scoreable refs — so reference rows and model `per_ref_mean`s compare directly.
The pre-fill registered subset values (118 non-avif refs; within-subset reads
only) are preserved in the second column and in the JSON as `subset_mean`.

| row (per-ref mean) | FULL corpus (757 refs) | old subset (118 refs, 0% avif) |
|---|---|---|
| ssim2-self (trivial ceiling) | +1.000 | +1.000 |
| **dssim** (negated) | **+0.833** | +0.786 |
| IW-SSIM | +0.763 | +0.655 |
| butteraugli (negated) | +0.733 | +0.420 |
| ColorVideoVDP | +0.660 | +0.549 |
| best learned: ADD156 / Ebothg / b_sdr / GL λ=1 | **+0.831 / +0.829 / +0.825 / +0.85** | (subset: +0.70–0.73 band) |
| winner_dial / C(W10L9) / v47 | +0.644 / +0.733 / +0.725 | +0.650 / +0.620 / +0.707 |
| mid-944 MLP single (class median, §4) | **+0.093** [IQR −0.04, +0.27] | +0.11 … +0.49 (subset flattered them) |

Codec-stratified reference rows (min-3 rule per stratum; read within a column —
per-stratum ladders differ in length and range restriction):

| codec (share of pairs) | n refs | dssim(neg) | iwssim | butter(neg) | cvvdp |
|---|--:|--:|--:|--:|--:|
| zenavif (73.6%) | 757 | **+0.828** | **+0.762** | **+0.740** | **+0.657** |
| zenjpeg (14.7%) | 173 | +0.797 | +0.440 | +0.446 | +0.514 |
| zenjxl (5.6%) | 51 | +0.699 | +0.667 | +0.510 | +0.514 |
| zenwebp (6.1%) | 71 | +0.429 | +0.396 | +0.232 | +0.297 |

Two corrected readings:

1. **The independent-metric band is 0.66–0.83, not 0.42–0.79** — the old subset
   dramatically understated it (butteraugli +0.42 → +0.73) because the non-avif
   subset is the *hard* part of the corpus. **The era-additive top models
   (ADD156 +0.831, Ebothg +0.829, b_sdr +0.825) and the sparsity-trained GL
   cells (+0.81–0.85) sit AT the dssim row (+0.833) — within the axis LSD
   (~0.039) of the best independent reference.** The top learned models are not
   merely "inside the band"; they match its ceiling.
2. **The avif majority is the EASIEST stratum for every reference metric** (all
   four stratified rows peak at avif; webp is hardest for everyone). So the
   mid-944-MLP deficit that concentrates in the avif cells (§1's occlusion
   finding, confirmed here) is unambiguously **model behavior** — independent
   metrics order avif near-lossless ladders *better* than the ladders the
   subset measured. "The axis is intrinsically hard" survives only in the weak
   form: even dssim tops out at ~0.83 per-ref against an ssim2-derived target.

## 4. The family pattern (corrected board, 233 cells)

| class | n | median | IQR | max |
|---|--:|--:|---|--:|
| era linear/additive (Ebothg, B, winner_dial) | 3 | **+0.825** | [+0.734, +0.827] | +0.829 |
| era MLP (v47) | 1 | +0.725 | — | +0.725 |
| 944 BVLS/blend heads | 63 | +0.415 | [+0.083, +0.507] | +0.611 |
| 944 featsub (input-restricted MLP) | 23 | +0.216 | [+0.151, +0.816] | +0.848 |
| era bridge (EM4 @944 root) | 1 | +0.132 | — | +0.132 |
| ensembles | 11 | +0.119 | [+0.108, +0.154] | +0.211 |
| 944-MLP single | 125 | **+0.093** | [−0.041, +0.272] | +0.800 |
| distilled (ens students) | 6 | −0.012 | [−0.103, +0.047] | +0.217 |

Three NEW era measurements this study (never on the board before): ADD156 **+0.831**,
v02_bvls +0.787, bhdr +0.634.

**The sparse story is about training pressure, not input count.** The featsub class
is bimodal: post-hoc top-K contribution masks (K64…K944) sit at −0.17…+0.22 — like
ordinary 944-MLPs — while sparsity-*trained* cells (group-lasso GL*, pilot-λ) sit at
0.71–0.85 with a clean λ gradient (PILOT0 +0.21 → λ=0.01 +0.44 → λ=0.1 +0.81 →
λ=1 +0.85). Together with the era-additive top of the table: **what preserves
near-lossless ordering is sparsity/simplicity pressure during training** — dense
944-MLP fits smear the tiny within-ladder signal, and the exceptions inside the
944-MLP class (nt223 +0.784, KFG75 +0.800, W10L9 +0.733) are exactly the cells whose
recipes re-weighted toward the real-codec near-threshold mass.

## 5. K128, precisely ("k128 isn't that bad" — confirmed)

- K128_s2501 **+0.174** [95% CI +0.134, +0.209]; s2503 +0.145. Corrected-board ranks
  114 and 124 of 233; 27 cells sit inside s2501's CI band (mid-board is crowded).
- **vs the 944-MLP median cell** (C_em944_s71, +0.093): Δ **+0.081 [+0.058, +0.105]**
  — significantly ABOVE the class median (s2503 too: +0.052 [+0.026, +0.078]).
- **vs C** (W10L9_s4003_packed, +0.733): Δ **−0.561 [−0.597, −0.526]** — ~12× the
  axis LSD. The appendix-J "K128 concedes hfnl −0.57 vs C" read stands exactly.
- Both are true at once: K128 is a *slightly-better-than-typical* 944-MLP on this
  axis; the typical 944-MLP is the weak population. K64, by contrast, is genuinely
  inverted (−0.17) — the pinned value, not a flip.

## 6. Range restriction: real, modest, not the story

Per-ref target span: median **3.68 ssim2 pts** (p10 1.74, p90 5.08). Correlation of
per-ref span with per-ref SROCC across the 30 models: median **+0.12**
(range −0.10…+0.31) — weak. The registered-secondary wide-band view (span ≥ median,
378 refs) lifts every model ~+0.05–0.07 without reordering (B 0.825→0.873, W10L9
0.733→0.801, K128 0.172→0.261). So narrow ladders depress the absolute level
uniformly; they do not create the family gap. With ~11 pairs/ref, per-ref SROCC is
coarsely quantized — that granularity is what the LSD absorbs.

## 7. How to read this axis (the panel legend, in words)

1. **Per-ref, not pooled.** The scoreboard number is the per-reference mean; the
   pooled scatter is range-restricted and misleading here by construction.
2. **Signed, quality-oriented.** Positive = orders a ladder the way ssim2 does;
   negative = inverted. Since the 2026-08-05 repair, all 233 board cells follow
   this convention (pin `730a386e`).
3. **The LSD.** Treat Δ < 0.04 as noise, Δ ≥ 0.05 as real (median/p90 paired
   bootstrap; per-model CIs ±0.02–0.05).
4. **The ceiling is ~0.83, not 1.0.** Independent metrics reach 0.66–0.83 on the
   full corpus (2026-08-05 coverage fill; dssim tops the band) and the
   era-additive/GL cells sit AT that ceiling (0.83-0.85). Read 0.7+ as
   "excellent", 0.4-0.7 as "useful", |x| < 0.15 as "no reliable ladder
   signal", ≤ −0.15 as "inverted — dial hazard in the HF zone".
5. **It is an ssim2-agreement axis**, not human truth: the human check for the HF
   zone is JPEG-AI-SDR25.

## 8. Limitations

Registered confounds (O.2.10), post-correction status: ssim2-derived target
(still applies); ~11 pairs/ref quantization (still applies); era models ride the
exact-row-identity 372 slice (`derive_hfnlproxy_372.py`; still applies). The
"reference-ceiling subset excludes avif entirely" confound is **RESOLVED** by
the 2026-08-05 coverage fill — reference rows are now full-corpus (757 refs,
100.0% pair coverage on all four metrics); the subset values remain in the JSON
as `subset_mean` for within-subset reads. New caveat: the codec-stratified rows
have per-stratum range restriction (shorter within-stratum ladders for
jpeg/jxl/webp) — read within a column. Ensemble cells have corrected scalar
means but no per-ref distribution here (the instrument loads one ZNPR per
member; distributions computable via `--ensemble` re-runs if ever needed). The
`--per-pair-refs` dump + this study's manifests + `hfnl_metrics_full.parquet`
(the merged 4-metric slice table in the report dir) make the whole analysis
re-runnable in ~5 minutes.
