# Codec-Target Metric Goals

The zensim codec-target metric is a **user-facing quality dial**.
Users type a target score; the codec hits it. Every goal below
derives from that use case. Rank correlation with human MOS
(SROCC) is a means, not an end — a metric can have perfect SROCC
and still be a broken dial (clamped range, non-monotone, codec-
dependent).

**Reference:** Mohammadi, Jenadeleh, Sneyers, Saupe & Ascenso,
"Evaluation of Objective Image Quality Metrics for High-Fidelity
Image Compression" (IEEE Access 2026, DOI 10.1109/ACCESS.2026.3669417).
This paper's findings directly shape goals G5, G6, G8, G9, G10,
the HF/MF split, and the validation pipeline.

---

## G1 — Full-dial dynamic range

The score distribution on a representative multi-codec corpus
must span the usable dial:

| Measure | Threshold |
|---|---|
| Score p5 (across the corpus) | ≤ 25 |
| Score p95 | ≥ 85 |
| No flat zone wider than 5 score units | Verified by: no 5-unit bin with > 20% of all rows |

**Why:** v10 was clamped at ~55 for everything below butter=3.
A codec binary-searching for "score=30" got stuck. v11 fixed this
with tanh_scale=30; future bakes must not regress it.

**Trainer lever:** `--dynamic-range-floor-weight`,
`--dynamic-range-sigma-threshold`, `--tanh-output-head-scale`.

## G2 — JND semantic anchor

The visually-lossless threshold (KonJND PJND mean) must land at
a declared integer:

| Measure | Threshold |
|---|---|
| Mean score at KonJND PJND pairs | 60 ± 5 |
| Std across KonJND refs at PJND | ≤ 10 |
| Z-RMSE at KonJND PJND pairs (σ-normalized) | ≤ 0.80 |

**Why:** "score 60 = just-noticeable difference" is the user-
facing semantic contract. Codecs targeting "visually lossless"
binary-search to score ≥ 60. The Z-RMSE gate (from Mohammadi
2025 § VIII) penalizes anchor errors proportional to how
CONFIDENT humans are about that stimulus's quality — an anchor
miss on a low-σ (high-consensus) PJND pair is far worse than
the same absolute miss on a high-σ pair.

**Trainer lever:** anchor loss at score 60
(`--anchor-parquet`, `--anchor-loss-weight`), konjnd aggregation
head (`--konjnd-aggregation-weight`).

## G3 — Strict monotonicity

Higher codec quality must produce higher scores. Measured on a
50-image × 19-q JPEG sweep (q5 → q100):

| Measure | Threshold |
|---|---|
| Strict monotonicity rate | ≥ 93% |
| Tied-pair rate | ≤ 5% |

**Why:** a codec binary-search that encounters a reversal
(score(q=80) < score(q=75)) oscillates or picks the wrong q.

**Trainer lever:** `--monotonicity-reg`,
`--monotonicity-margin`.

## G4 — Cross-codec equivalence

The same perceptual quality across codecs must produce the same
score. Measured on matched-quality (butter ≤ 2.5) pairs across
≥ 3 codec families (JPEG, WebP, JXL) on 10+ images:

| Measure | Threshold |
|---|---|
| p50 \|Δscore\| across codecs at matched quality | ≤ 1.5 |
| \|Δscore\| / dial span | ≤ 2.5% |

**Why:** "score 70 from JPEG" should mean the same as "score 70
from JXL." A picker that routes between codecs based on score
needs this. Mohammadi Figure 9 shows that even CVVDP has
per-source-per-codec scatter — no metric achieves perfect cross-
codec equivalence. Our threshold accounts for this.

**Trainer lever:** `--cross-codec-eq-weight`,
`--cross-codec-rank-preserve-weight`.

## G5 — High-fidelity rank fidelity (HF: ≤ 1 JND)

Rank quality in the near-lossless range where compression
artifacts are at or below the visibility threshold. This is the
range Mohammadi 2025 calls **HF** (High Fidelity, 115 AIC-3
scores with perceived impairment ≤ 1 JND). Learning-based
metrics systematically underperform conventional metrics here
because their training data rarely includes near-lossless pairs
(Mohammadi § X-A-1). This is our KonJND failure zone.

| Measure | Floor | Aspiration |
|---|---|---|
| AIC-3 HF-subset SROCC | ≥ 0.70 | ≥ 0.85 (CVVDP=0.851) |
| AIC-3 HF-subset PWRC | ≥ 4.0 | ≥ 5.5 (CVVDP=5.92) |
| AIC-3 HF-subset Z-RMSE | ≤ 15.0 | ≤ 10.0 (CVVDP=9.45) |
| KonJND-1k val SROCC | ≥ 0.70 | ≥ 0.85 |
| KonJND-1k PWRC | ≥ 0.65 | ≥ 0.75 |

**Why:** Mohammadi Table 2-3 shows CVVDP at 0.851 HF SROCC
and 0.826 MF SROCC — the HF range is HARDER for every metric.
SSIMULACRA2 scores 0.806 HF / 0.657 MF SROCC. IW-SSIM 0.867
HF / 0.825 MF. Our v11 ship's KonJND SROCC 0.285 means we're
failing catastrophically in the HF range. The 0.70 floor is the
"codec can target visually lossless" bar; 0.85 is "competitive
with CVVDP in HF."

**Trainer lever:** konjnd aggregation head, HF-weighted training
pairs (pairs with ground-truth ≤ 1 JND get higher loss weight),
input feature transforms (YJ finding: transforms unlock IW-pool
features that carry HF signal).

## G6 — Medium-fidelity band coverage (MF: > 1 JND)

No 10-band bin (B0..B9, width-10 on the 0-100 scale) where the
metric loses to ssim2 by more than 0.10 SROCC on any held-out
corpus. This covers the **MF** (Medium Fidelity, > 1 JND) range
— the "visible distortion" regime where most codecs operate at
web-delivery quality settings.

| Measure | Threshold |
|---|---|
| max(ssim2_srocc − zensim_srocc) across (corpus, band) in MF range | ≤ 0.10 |
| max gap in HF range (B7-B9) | ≤ 0.15 (relaxed — HF is structurally harder) |

**Why:** Mohammadi Table 2 shows that even the best metrics
(CVVDP, IW-SSIM) have lower HF scores than MF. Holding HF to
the same 0.10 threshold as MF would block every bake. The
relaxed 0.15 HF threshold is achievable by IW-SSIM-class
metrics.

**Trainer lever:** per-band weighting (not yet implemented —
the trainer applies uniform loss across the score range). Future:
stratified sampling with HF pairs upweighted 2-3×.

## G7 — Compression-corpus rank (advisory)

CID22 aggregate SROCC is the gold-standard generalization check
for the codec-compression use case. **Advisory, not blocking**
(per CLAUDE.md 2026-05-14 ship policy):

| Measure | Threshold |
|---|---|
| CID22 aggregate SROCC | ≥ 0.85 (advisory) |
| CID22 PWRC | ≥ 4.5 (advisory) |
| CID22 Z-RMSE | ≤ 30.0 (advisory) |

**Why:** a bake that drops CID22 by 0.005 while gaining +0.05
on HF IS the winning trade. CID22 is informative, not
determinative. Note: CID22's ground truth uses pairwise
comparison (Thurstone model) not absolute MOS — its σ-per-stimulus
is derived from bootstrapping, not from direct observer variance.
Z-RMSE on CID22 is therefore an approximation.

**Trainer lever:** training-target choice (mix_cv40_iw60 vs pure
ssim2 vs other), input feature transforms, CID22-train subset
weighting.

## G8 — Z-RMSE (σ-normalized prediction accuracy)

Z-RMSE (Mohammadi 2025 § VIII, Equation 6) is:

```
Z-RMSE = √( (1/n) Σᵢ ((S_trans,i − S_subj,i) / σᵢ)² )
```

where σᵢ is the per-stimulus standard deviation of subjective
scores (from bootstrap on AIC-3, from direct observer variance
on KonJND/TID/KADID). Z-RMSE is **the single best stat for "does
this metric track the consensus when there IS one"** — it
penalizes errors on high-consensus stimuli (low σ) more than on
ambiguous ones (high σ).

Crucially, Z-RMSE is proportional to the negative log-likelihood
of the predictions under a Gaussian model of observer noise
(Equation 10-11): minimizing Z-RMSE is equivalent to maximizing
the probability that the metric IS the generative model of human
judgment. This is a stronger claim than "metric ranks correctly"
(SROCC).

| Measure | Floor | Aspiration |
|---|---|---|
| AIC-3 All Z-RMSE | ≤ 30.0 | ≤ 10.0 (CVVDP=9.45) |
| AIC-3 HF Z-RMSE | ≤ 20.0 | ≤ 10.0 (CVVDP HF best) |
| AIC-3 MF Z-RMSE | ≤ 8.0 | ≤ 4.0 (CVVDP MF ~4.60) |
| KonJND Z-RMSE | ≤ 0.80 | ≤ 0.50 |

**Why (matters for codec-target):** a codec binary-searching to
hit "score=70" needs the metric to be RIGHT on stimuli where
humans agree about quality. Z-RMSE measures exactly that.
SROCC + PLCC only measure RELATIVE ordering / linearity —
the metric could be perfectly ranked but consistently wrong
by +5 points on high-consensus stimuli.

**Trainer lever:** σ-weighted MSE loss: `loss = (1/n) Σ
((predicted − target) / σ_target)²` where σ_target is either
the observer σ from the corpus (KonJND, AIC-3) or a
target-derived proxy (bootstrap σ on synthetic pairs). This
directly optimizes Z-RMSE at training time. NOT yet implemented.

## G9 — DS-AUC (same-vs-different classification)

DS-AUC (Mohammadi 2025 § VII): Area Under the ROC curve for
classifying stimulus pairs as "same" vs "different" perceptual
quality. Ground-truth labels from a 2AFC binomial hypothesis
test on subjective data (p < 0.05 → "different").

| Measure | Floor | Aspiration |
|---|---|---|
| AIC-3 DS-AUC | ≥ 0.70 | ≥ 0.85 (CVVDP=0.846) |

**Why (matters for codec-target):** a codec targeting "visually
lossless" needs the metric to answer a BINARY question: "is
this encode perceptibly different from the reference?" DS-AUC
measures exactly this. SSIMULACRA2's DS-AUC is 0.571 — barely
better than a coin flip. This means ssim2-based training
targets cannot teach a metric when to STOP compressing.

**Current gap:** we don't compute DS-AUC anywhere in the
pipeline. AIC-3 provides the 2AFC response data needed to
derive ground-truth labels. Implementation: add a
`ds_auc(metric_scores, ground_truth_labels)` function to
`bake_verdict`.

**Trainer lever:** none directly (DS-AUC is a binary
classification measure; the metric produces a continuous score).
Indirectly: Z-RMSE optimization in the HF range should improve
DS-AUC because correctly predicting per-stimulus quality in
the near-threshold range improves the threshold-crossing
accuracy that DS-AUC measures.

## G10 — Per-source-per-codec stability

Mohammadi Figure 9 shows that metric performance varies wildly
by source image AND by codec. A metric that scores 0.95 SROCC
aggregate but has one source image where JPEG scores land 2 JND
away from the identity line is broken for that source.

| Measure | Threshold |
|---|---|
| Max per-source RMSE (across all codecs + quality levels) | ≤ 2× median per-source RMSE |
| Per-codec bias (mean residual per codec) | ≤ 0.3 JND |

**Why:** the cross-codec equivalence goal (G4) measures the
AVERAGE cross-codec spread. G10 measures the WORST-CASE
per-source-per-codec failure. A picker that routes images to
codecs based on score needs both.

**Current gap:** `bake_verdict` computes per-corpus aggregates,
not per-source-per-codec scatter. Need: for each (source,
codec, quality) triple in the AIC-3 dataset, compute the
residual against the logistic-transformed subjective score.
Report the per-source RMSE distribution and flag outlier
sources.

**Trainer lever:** per-source loss weighting (upweight sources
where the residual is large). Not yet implemented.

## G11 — Display-dependent quality (display profiles)

CVVDP's structural advantage over every other metric in
Mohammadi 2025 is its **display model**: it takes display
parameters (pixels per degree, peak luminance, ambient light,
contrast ratio) as input and modulates sensitivity functions
accordingly.

**CORRECTED 2026-05-26 (measured, was backwards):** an earlier
version of this section claimed compression artifacts are "MORE
visible on a phone." The CVVDP physics says the opposite for the
pixel-density axis, and we MEASURED it: scoring the same KADID
pairs at `modern_oled_phone_indoor` (≈110 PPD) vs `standard_4k`
(≈75 PPD) gives **higher** JOD (artifacts LESS visible) at the
higher PPD. Reason: a fixed pixel-scale artifact subtends a
SMALLER visual angle at higher PPD → higher spatial frequency →
deeper CSF rolloff → less visible. So **higher PPD → pixel-level
artifacts LESS visible, not more.** The "more visible on a phone"
intuition comes from the *other* axes — phones are often held
CLOSER (which LOWERS effective PPD and raises visibility) and run
brighter (higher contrast sensitivity). The net per-display
direction is not assumable from one parameter; it must be
**measured by running CVVDP at that display's actual
(geometry + photometry)** — which is exactly what the
`zensim-b-phone` bake does.

For a codec-target metric, this still means: **"score 70" should
mean different byte budgets for different displays** — but the
sign and magnitude of the shift are an empirical CVVDP-at-display
output, not a fixed "mobile is stricter" rule.

### Display profile parameters

The minimum viable display model needs three numbers:

| Parameter | Definition | Example: iPhone 14 Pro | Example: 1080p desktop |
|---|---|---|---|
| **PPD** (pixels per degree) | PPI × (π/180) × viewing_distance_cm / 2.54 | ~67 | ~53 |
| **peak_nits** | Peak display luminance | 2000 | 350 |
| **ambient_lux** | Typical ambient light | 500 (indoor) | 200 (office) |

PPD is the load-bearing parameter. At higher PPD, the human
visual system resolves finer spatial frequencies → more
compression artifacts become visible → the JND threshold shifts
toward lower distortion → "score 60" means a tighter encode.

### Implementation strategy (three tiers)

**Tier 1 (immediate, no retraining):** post-network PPD-dependent
score shift. The v0.3 MLP produces a display-agnostic score.
A per-display-profile affine `score_display = α(ppd) + β(ppd) ·
score_agnostic` shifts the output based on PPD. The α/β
coefficients are fit from a CVVDP-anchored sweep: for N images
× M quality levels, compute both zensim and CVVDP-at-PPD,
regress the affine per PPD bracket.

This is architecturally identical to the existing
`zentrain.per_codec_calibration` metadata — the bake carries a
`zentrain.per_display_calibration` payload mapping PPD brackets
to (α, β) pairs, and the runtime applies the shift when the
caller provides a display hint.

```rust
let z = Zensim::new(ZensimProfile::codec_target())
    .with_display(DisplayProfile::iphone_14_pro());
let score = z.compute(&source, &distorted)?;
// score accounts for Retina PPD — tighter than desktop
```

**Tier 2 (medium-term, retraining):** PPD as an input feature.
Add PPD (or log2(PPD)) as the 373rd input feature to the MLP.
Train on a corpus that includes the same (ref, dist) pair
scored at multiple PPD values (from CVVDP). The network learns
to modulate its sensitivity at different viewing conditions.
This is a richer model than the Tier 1 affine because the
PPD interaction with specific feature types (e.g., high-freq
artifact detection features are PPD-sensitive; color shift
features are not) can be learned.

**Tier 3 (long-term, architectural):** multi-scale feature
extraction conditioned on PPD. The blur kernel widths in
zensim's 4-scale Gaussian pyramid are currently fixed at 5px
radius × 1 pass. On a 67 PPD display, a 5px blur corresponds
to ~0.075° of visual angle; on a 53 PPD display, the same blur
is ~0.094°. Making the blur radii PPD-dependent (so they always
correspond to the same angular extent) would let the feature
extraction itself be display-aware. This requires changes to
`zensim/src/metric.rs::compute_with_config_inner` and would
invalidate existing bakes (new feature schema).

### Named display profiles

Ship a set of named profiles as `const` in the `zensim` crate:

```rust
pub struct DisplayProfile {
    pub ppd: f32,
    pub peak_nits: f32,
    pub ambient_lux: f32,
}

impl DisplayProfile {
    pub const DESKTOP_1080P: Self = Self { ppd: 53.0, peak_nits: 350.0, ambient_lux: 200.0 };
    pub const DESKTOP_4K_27: Self = Self { ppd: 93.0, peak_nits: 600.0, ambient_lux: 200.0 };
    pub const MACBOOK_RETINA: Self = Self { ppd: 99.0, peak_nits: 500.0, ambient_lux: 300.0 };
    pub const IPHONE_14_PRO: Self = Self { ppd: 67.0, peak_nits: 2000.0, ambient_lux: 500.0 };
    pub const IPHONE_16_PRO: Self = Self { ppd: 69.0, peak_nits: 2000.0, ambient_lux: 500.0 };
    pub const IPAD_PRO_M4: Self = Self { ppd: 80.0, peak_nits: 1600.0, ambient_lux: 400.0 };
    pub const TV_4K_55_3M: Self = Self { ppd: 56.0, peak_nits: 1000.0, ambient_lux: 50.0 };
    pub const PRINT_300DPI_30CM: Self = Self { ppd: 115.0, peak_nits: 100.0, ambient_lux: 500.0 };
    pub const WEB_GENERIC: Self = Self { ppd: 60.0, peak_nits: 350.0, ambient_lux: 200.0 };
}
```

Codecs default to `WEB_GENERIC` (the current display-agnostic
behavior). When a caller provides a specific profile, the score
shifts by the MEASURED CVVDP-at-display delta (see the corrected
physics note above — do NOT assume "higher PPD = stricter"; at
fixed distance higher PPD makes pixel-scale artifacts LESS visible
via CSF rolloff). The sign/magnitude per profile comes from
running CVVDP at that display's geometry + photometry, not a
fixed rule.

### Measurement

| Measure | Threshold |
|---|---|
| Cross-PPD consistency: same (ref, dist) at PPD 53 vs 110 → score delta | non-zero + sign matches measured CVVDP-at-display (NOT assumed "more visible on mobile" — see corrected physics note) |
| Cross-PPD rank preservation: rank order of quality levels at PPD X matches rank order at PPD Y | SROCC ≥ 0.99 (ranks should not invert) |
| CVVDP-anchor fit R² per PPD bracket | ≥ 0.90 for Tier 1 affine |

**Current gap:** zensim has zero display awareness. Every score
assumes the same implicit viewing condition. CVVDP's Mohammadi
paper numbers (0.960 SROCC, 9.45 Z-RMSE) are for `ccfl lcd,
64.27 ppd` — a specific display model. Our eval numbers are
display-agnostic. Comparing directly is slightly unfair to
zensim (display-agnostic metric vs display-tuned metric), but
it's also the gap we need to close.

**Trainer lever:** Tier 1 needs no trainer changes — just a
CVVDP-anchored calibration sweep. Tier 2 needs PPD as a 373rd
feature + multi-PPD training corpus (each pair scored at 3-5
PPD values via CVVDP). Tier 3 is architectural.

---

## Priority order

When goals conflict, resolve in this order:

1. **G1 (dynamic range)** — a clamped dial is unusable
2. **G3 (monotonicity)** — a non-monotone dial is unreliable
3. **G2 (JND anchor)** — the semantic contract
4. **G8 (Z-RMSE)** — the probabilistic accuracy anchor
5. **G4 (cross-codec)** — the picker contract
6. **G5 (HF rank)** — the visually-lossless range
7. **G11 (display profiles)** — the display-dependent quality
8. **G9 (DS-AUC)** — same-vs-different classification
9. **G10 (per-source stability)** — worst-case guard
10. **G6 (MF band coverage)** — the "no dead zones" guard
11. **G7 (CID22 rank)** — advisory generalization check

G11 is ranked above G9/G10 because display-dependent scoring is
the structural feature that differentiates a codec-target metric
from a generic IQA metric. CVVDP does this; we should too. The
Tier 1 implementation (post-network affine, no retraining) is
achievable within the current architecture.

G8 (Z-RMSE) is ranked 4th — above G4 (cross-codec) — because
Z-RMSE is the paper's key finding: it's the only stat that
accounts for observer uncertainty, and it's the correct
optimization objective under the Gaussian observer model
(Mohammadi Equation 7-11). A metric with good SROCC but bad
Z-RMSE is a metric that ranks stimuli correctly but gives wrong
absolute scores — useless for a codec binary-searching to a
target.

---

## Validation policy — `--val-policy goals`

The current trainer uses `--val-policy min` (worst-corpus SROCC)
to select the "best epoch" checkpoint. This is wrong:

- It optimizes for a SINGLE stat (SROCC) on a SINGLE corpus
  (whichever is worst, usually konjnd_dense which oscillates
  with cyclic LR) — violating the "SROCC-only verdicts BANNED"
  principle.
- It ignores G1, G2, G3, G4, G8, G9, G10 entirely — these are
  never measured during training, only post-hoc.
- The YJ-AT retrain showed the pathology: best val_min was
  epoch 10 (transient post-init) while every corpus improved
  through epoch 299.

**Replace with `--val-policy goals`:** at each validation
checkpoint, compute a weighted score against G1–G11 (G11 is
display-dependent and only activates when the trainer carries
multi-PPD anchors — zero-cost when absent):

```
goal_score = (
    w1  * dial_range_ok(sweep)              # G1: p5 ≤ 25 ∧ p95 ≥ 85
  + w2  * jnd_anchor_ok(konjnd_preds)       # G2: |mean - 60| ≤ 5 ∧ Z-RMSE ≤ 0.80
  + w3  * mono_rate(sweep)                  # G3: strict_mono ≥ 0.93
  + w4  * cross_codec_ok(eq_pairs)          # G4: p50|Δ| ≤ 1.5
  + w5  * hf_rank(aic3_hf_preds)            # G5: HF SROCC ≥ 0.70
  + w6  * mf_band_coverage(val_corpora)     # G6: max band gap ≤ 0.10
  + w7  * cid22_rank(cid22_preds)           # G7: advisory SROCC ≥ 0.85
  + w8  * zrmse_quality(aic3_preds, sigmas) # G8: Z-RMSE ≤ 30 (All)
  + w9  * ds_auc(aic3_preds, gt_labels)     # G9: DS-AUC ≥ 0.70
  + w10 * source_stability(per_src_rmse)    # G10: max/median ≤ 2
)
```

Each term is 0.0–1.0 (soft gate: linear ramp from threshold to
target). Weights follow the priority order.

**Implementation cost per val checkpoint (~3s total):**

| Check | Forward passes | Wall time |
|---|---:|---|
| G1/G3: q-sweep 50 img × 19 q | 950 | ~0.5s |
| G2: KonJND anchor forward | ~200 | ~0.1s |
| G4: cross-codec-eq parquet | ~500 | ~0.3s |
| G5: AIC-3 HF subset forward | ~115 | ~0.1s |
| G6: per-band SROCC on val corpora | 0 (reuses val preds) | ~0.1s |
| G7: CID22 SROCC | 0 (reuses val preds) | ~0.1s |
| G8: Z-RMSE on AIC-3 | 0 (reuses G5 preds + stored σ) | ~0.1s |
| G9: DS-AUC on AIC-3 | 0 (reuses preds + stored 2AFC labels) | ~0.1s |
| G10: per-source RMSE from AIC-3 | 0 (reuses preds) | ~0.1s |

Incremental cost vs current val: +~1700 forward passes per
epoch (~1.5s on 7950X). Total val cost ~3s/epoch. Acceptable
for 300 epochs (15 min overhead across full training).

**Starter weights** (tunable per experiment):

| Goal | Weight | Rationale |
|---|---:|---|
| G1 dynamic range | 3.0 | Broken dial is unusable |
| G3 monotonicity | 2.5 | Non-monotone dial is unreliable |
| G8 Z-RMSE | 2.5 | Probabilistic accuracy (Mohammadi key finding) |
| G2 JND anchor | 2.0 | Semantic contract |
| G4 cross-codec | 1.5 | Picker contract |
| G5 HF rank | 1.5 | Visually-lossless range (our biggest gap) |
| G9 DS-AUC | 1.0 | Same-vs-different gate |
| G10 per-source stability | 0.5 | Worst-case guard |
| G11 display profiles | 0.5 | Display-dependent (Tier 1: CVVDP-anchored affine; 0 cost when absent) |
| G6 MF band coverage | 0.5 | Guard rail |
| G7 CID22 rank | 0.5 | Advisory |

---

## Statistical significance — MRR + Wilcoxon

Per Mohammadi § IX: when comparing two bakes, SROCC deltas
alone are insufficient. Two stats:

1. **Meng-Rosenthal-Rubin (MRR) test** for paired SROCC
   comparison: accounts for the shared subjective scores
   between the two metrics being compared. Reports Z-statistic
   + p-value. Mohammadi Table 4 shows that SROCC 0.907 vs 0.917
   can be statistically indistinguishable (gray cell).

2. **Pairwise Wilcoxon Signed-Rank test** on residuals: non-
   parametric, tests whether the median paired residual
   difference is zero. Reports direction + p-value + effect
   size r = Z/√N.

**Policy:** any ship-or-no-ship verdict must cite BOTH tests.
"CID22 SROCC dropped 0.045" is not a verdict — "MRR p=0.003,
Wilcoxon p=0.008 r=0.12" is. If MRR p > 0.05, the delta is
not statistically significant regardless of its magnitude.

**Implementation:** add `mrr_test(srocc_a, srocc_b, n)` and
`wilcoxon_signed_rank(residuals_a, residuals_b)` to
`bake_verdict`. Emit p-values + effect sizes in every comparison
table. This is ~50 LOC of pure Rust stats (Equation 12-16 from
the paper).

---

## 4-parameter logistic transform

Per Mohammadi § VI (Equation 1): before computing PLCC, RMSE,
OR, Z-RMSE, and DS-AUC, raw metric scores are mapped through
a 4-parameter logistic:

```
S_trans = B2 + (B1 - B2) / (1 + exp(-(s_obj - B3) / B4))
```

fitted globally across ALL stimuli for each metric via nonlinear
least-squares (minimize RMSE between S_trans and S_subj). This
is more principled than our current per-corpus affine calibration
because:

- It handles the sigmoid saturation at both ends of the scale
- It's fit once, globally — not per-corpus (avoids overfitting
  to individual corpus structure)
- SROCC and KT are unaffected (rank-invariant under monotone
  transforms) but PLCC, RMSE, OR, Z-RMSE, and DS-AUC all
  depend on the transform

**Policy:** `bake_verdict` should fit the 4-parameter logistic
ONCE per bake across all validation stimuli (CID22 + AIC-3 +
KonJND pooled), then compute PLCC / RMSE / OR / Z-RMSE / DS-AUC
on the transformed scores. Current approach (per-corpus affine)
is kept as a secondary report for back-compat.

---

## SA-ST curves (PWRC visualization)

Per Mohammadi § VII and Figure 4: the SA-ST (Sorting Accuracy
vs Sensory Threshold) curve plots how well a metric ranks
stimulus pairs as a function of the JND threshold at which pairs
are considered "perceptually different." A metric with high SA
at high ST is good at distinguishing stimuli that are HARD to
tell apart (the HF range). Figure 4 shows CVVDP dominating at
all ST thresholds; most learning-based metrics collapse at high
ST.

**What to implement:**
1. For each pair (i, j) where quality(i) ≠ quality(j):
   - ST(i,j) = |MOS(i) - MOS(j)| / σ_pooled(i,j)
   - SA(metric, θ) = fraction of pairs with ST ≥ θ where the
     metric correctly ranks them
2. Plot SA vs θ for θ ∈ [0, 4] (0 = all pairs, 4 = only pairs
   where quality difference is > 4σ)
3. Report AUC of the SA-ST curve (= PWRC) + SA at specific
   thresholds: θ=0 (All), θ=0 to 1 (HF range), θ=1+ (MF range)

**Where:** add to `bake_verdict` output; render in the zenpredict-
viz compare panel so two bakes' SA-ST curves overlay.

---

## Measurement

All goals are measured by an enhanced `bake_verdict` that
computes:

1. Full Mohammadi panel (SROCC, PLCC, KROCC, OR, PWRC, Z-RMSE)
   at aggregate + 10-band + HF/MF split
2. DS-AUC on AIC-3
3. Per-source-per-codec scatter + outlier flagging
4. MRR + Wilcoxon when comparing two bakes
5. SA-ST curve + AUC
6. 4-parameter logistic fit (global)
7. G1-G10 pass/fail table with soft scores

A single `bake_verdict --full-mohammadi --compare <baseline.bin>`
invocation should emit all of the above.

---

## Current v11 ship scorecard

| Goal | v11 ship | YJ-AT retrain | Pass? (v11) |
|---|---|---|---|
| G1 p5 ≤ 25 | p5 = 28 | TBD | ✗ marginal |
| G1 p95 ≥ 85 | p95 ≈ 93 | TBD | ✓ |
| G2 JND = 60 ± 5 | mean 60 | TBD | ✓ |
| G3 mono ≥ 93% | 92.78% | TBD | ✗ marginal |
| G4 p50 \|Δ\| ≤ 1.5 | 1.37 | TBD | ✓ |
| G5 HF SROCC ≥ 0.70 | KonJND 0.285 | KonJND 0.666 | ✗ |
| G6 MF max gap ≤ 0.10 | TBD | TBD | ? |
| G7 CID22 ≥ 0.85 | 0.860 | 0.816 | ✓ |
| G8 Z-RMSE ≤ 30 | TBD | TBD | ? |
| G9 DS-AUC ≥ 0.70 | TBD | TBD | ? |
| G10 max/med ≤ 2 | TBD | TBD | ? |

v11 passes 3/11 cleanly, marginal on 2, fails G5 decisively,
5 unmeasured (G6/G8/G9/G10/G11 — pipeline doesn't compute them).

**First-order priority for the pipeline:**
1. Implement G8 (Z-RMSE) + G9 (DS-AUC) in `bake_verdict`
   (< 100 LOC each)
2. Score v11 + YJ-AT on the full G1-G11 table
3. Tier 1 display calibration (G11): run a CVVDP-anchored sweep
   at 3 PPD values on the safesyn corpus, fit per-PPD (α, β),
   bake as `zentrain.per_display_calibration` metadata
4. THEN decide the next retrain direction

---

## Appendix: paper reference numbers (SOTA targets)

From Mohammadi 2025 Table 2-3 (AIC-3 full-resolution,
300 stimuli, sorted by SROCC):

| Metric | SROCC All | SROCC HF | SROCC MF | Z-RMSE All | Z-RMSE HF | DS-AUC |
|---|---:|---:|---:|---:|---:|---:|
| **CVVDP** | **0.960** | **0.852** | **0.893** | **9.45** | **14.50** | **0.846** |
| IW-SSIM | 0.944 | 0.867 | 0.825 | 10.48 | — | 0.836 |
| HDR-VDP-3 | 0.929 | 0.873 | 0.768 | 10.62 | — | 0.836 |
| MS-SSIM | 0.927 | 0.855 | 0.763 | 10.26 | — | 0.836 |
| SSIMULACRA1 | 0.907 | 0.854 | 0.691 | 10.19 | — | 0.831 |
| SSIMULACRA2 | 0.905 | 0.831 | 0.687 | 10.06 | — | 0.806 |
| VIF | 0.905 | 0.839 | 0.684 | 10.29 | — | 0.856 |
| BUTTERAUGLI | 0.893 | 0.857 | 0.640 | 9.92 | — | 0.806 |

Our v11 ship's CID22 SROCC (0.860) positions us between
BUTTERAUGLI and SSIMULACRA2 in the "All" column. On the AIC-3
corpus (which we hold out), we score 0.776 — below BUTTERAUGLI's
0.893 and well below CVVDP's 0.960. The gap is largest in the
HF range, matching the paper's finding that learning-based
metrics fail near-lossless.
