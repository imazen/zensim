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

## 3. The ceiling: what a strong metric can even get here

The target is ssim2-derived, so ssim2-vs-itself is +1.0 by construction (computed,
118/118 groups: +1.0000). Independent references exist only for the non-avif cells
(fill4 sidecar; avif has no butteraugli/cvvdp anywhere). Registered subset = refs
with ≥6 covered cells: **118 refs / 2,410 pairs, 0% avif** — every row below on
identical pairs; within-subset reads only, never the axis headline.

| row (subset, per-ref mean) | value |
|---|---|
| ssim2-self (trivial ceiling) | +1.000 |
| **dssim** (negated) | **+0.786** |
| IW-SSIM | +0.655 |
| ColorVideoVDP | +0.549 |
| butteraugli (negated) | +0.420 |
| best learned: FS_GL0p3 / FS_PILOT1 / v47 / ADD156 | **+0.734 / +0.729 / +0.707 / +0.703** |
| winner_dial / C(W10L9) / B | +0.650 / +0.620 / +0.607 |
| mid-944 MLP cells | +0.11 … +0.49 |

Side observation: mid-944 cells do noticeably better on this non-avif subset than
on the full corpus (mid-944 median cell +0.093 full → +0.270 subset) while the
era/sparse class barely moves — **the 944-MLP deficit concentrates in the avif
cells**, which are 73.6% of the axis. A codec-stratified per-ref view is the
natural follow-up instrument.

Reading: **even strong independent perceptual metrics agree with ssim2's
near-lossless ordering at only 0.42–0.79 per-ref.** The axis is intrinsically hard.
The top learned models are *inside* the independent-metric band — above cvvdp,
IW-SSIM and butteraugli, ~0.05 under dssim (the target's nearest kin, expected to be
the top non-self row). "All models suck at it" is false; the mid-944-MLP deficit
below that band is real model behavior.

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
4. **The ceiling is ~0.79-0.85, not 1.0.** Independent metrics reach 0.42–0.79 on
   matched pairs; era-additive/GL cells reach 0.83-0.85 on the full corpus. Read
   0.7+ as "excellent", 0.4-0.7 as "useful", |x| < 0.15 as "no reliable ladder
   signal", ≤ −0.15 as "inverted — dial hazard in the HF zone".
5. **It is an ssim2-agreement axis**, not human truth: the human check for the HF
   zone is JPEG-AI-SDR25.

## 8. Limitations

Registered confounds (O.2.10) all apply: ssim2-derived target; reference-ceiling
subset excludes avif (73.6% of the axis's pairs) entirely; ~11 pairs/ref
quantization; era models ride the exact-row-identity 372 slice
(`derive_hfnlproxy_372.py`). Ensemble cells have corrected scalar means but no
per-ref distribution here (the instrument loads one ZNPR per member; distributions
computable via `--ensemble` re-runs if ever needed). The `--per-pair-refs` dump +
this study's manifests make the whole analysis re-runnable in ~5 minutes.
