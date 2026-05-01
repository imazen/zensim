# Low-quality (SSIM2 25-60) performance — diagnosis & improvement plan

## Diagnosis

### Per-band SROCC vs SSIM2 ground truth (synthetic, 218k pairs)

| band | V0_4-smooth | V0_5 | V0_4-smooth-konjnd | V0_6 dct_hf | training share |
|---|--:|--:|--:|--:|--:|
| ≤ 0 | 0.93 | 0.98 | 0.87 | 0.96 | 6.0% |
| 0-25 | 0.83 | 0.92 | 0.70 | 0.91 | 7.4% |
| **25-40** | **0.84** | **0.87** | **0.75** | **0.90** | **7.8%** |
| **40-60** | **0.94** | **0.95** | **0.89** | **0.96** | **17.4%** |
| 60-75 | 0.95 | 0.95 | 0.90 | 0.97 | 20.9% |
| 75-90 | 0.98 | 0.98 | 0.91 | 0.98 | 28.0% |
| ≥ 90 | 0.86 | 0.87 | 0.73 | 0.86 | 12.4% |

### Per-band-calibrated MAE in SSIM2 units (synthetic)

| band | V0_4-smooth | V0_5 | V0_6 dct_hf |
|---|--:|--:|--:|
| ≤ 0 | 7.54 | **5.13** | 6.26 |
| 0-25 | 3.95 | **2.41** | 2.58 |
| **25-40** | 2.27 | 1.97 | **1.65** |
| **40-60** | 1.70 | 1.77 | **1.39** |
| 60-75 | 1.13 | 1.16 | **0.86** |
| 75-90 | 0.77 | 0.73 | **0.69** |
| ≥ 90 | 0.95 | 0.94 | **0.92** |

**V0_6 dct_hf wins the 25-90 band** on both rank and calibration.
V0_5 wins ≤ 25 (because of its wider dynamic range that maps low-q
with more resolution — the 700-magnitude outliers helped SROCC there
but hurt smoothness).

### Where the loss is

1. **Training data is biased toward high quality.** The synthetic
   training set has 28% of pairs in [75, 90] SSIM2 but only 7.4% in
   [0, 25] and 7.8% in [25, 40]. For a web-compression metric where
   most production decisions live in [25, 60], that's the wrong
   distribution.

2. **The synthetic q-grid is sparse below q=70.**
   `training_safe_synthetic.csv` has q levels {5, 10, 15, 20, 25, 30,
   40, 50, 60, 70, 75, 80, 87, 90, 95, 100} — 7 levels in [5, 60]
   (gaps: 30→40, 40→50, 50→60) vs 9 in [70, 100] (gaps: 70→75 = 5 q
   apart, 87→90, 90→95, 95→100 = 5 q apart). High-q is denser per
   q-step, and high-q maps to a narrow SSIM2 range, which compounds
   the imbalance.

3. **Human-MOS within-band SROCC is ~0 at 40-60 for every metric
   including SSIMULACRA 2 itself** (V0_5 = 0.063, ref SSIM2 = 0.012).
   This isn't a metric defect — it's the "narrow-band-MOS-noise" floor:
   when you slice human MOS to a thin SSIM2 band, intra-band MOS
   variance is mostly inter-rater noise, no metric can rank cleanly.
   Don't use within-band human SROCC as a fix target.

## Improvement plan

Ranked by expected ROI / cost ratio for the 25-60 SSIM2 band.

### 1. Densify the synthetic q-grid in [25, 60] *(highest ROI, ~half a day of compute)*

The grid currently jumps q=30 → q=40 → q=50 → q=60. Add intermediate
levels q ∈ {32, 35, 38, 42, 45, 48, 52, 55, 58, 65}, run the existing
generator, append to the dataset. Expected effect:
- Doubles the per-pair density in the operationally-critical band.
- Triples the training-pair share in [25, 60] SSIM2 from ~25% to ~50%.
- Cost: ~6h on the existing GPU pipeline (3,579 sources × 6 codecs
  × 10 new q levels ≈ 215k new pairs, generator runs at ~10 pairs/s
  with cache hits).

This is required by CLAUDE.md's "Sweep Discipline" rule for
source-informing benchmarks: web-focused codec work demands dense
low-q sampling, and the current grid violates that.

### 2. Loss reweighting / sampler bias *(cheapest, half-day of code)*

Modify `mlp_train.rs` triplet sampler to upsample pairs whose SSIM2
target is in [0, 60]. Concretely: in `train_mlp`, when sampling a
batch of pairs, draw 50% from band-[0, 60] and 50% from band-[60+]
instead of uniform-by-pair. This gives the MLP roughly equal gradient
contribution from low-q and high-q regions. Expected effect: +0.02-0.03
SROCC in the 25-60 band, possibly −0.005 in 75-90 (acceptable
tradeoff for a web-target metric).

### 3. Add low-q-targeted zenanalyze features *(no new training data needed)*

V0_6 dct_hf already wins 25-90 with three features
(dct_compressibility_y/uv + high_freq_energy_ratio). Per the V0_6
sweep, adding more features didn't help — likely because the features
were redundant with each other, not because more is bad. Try a
heavy-distortion-targeted combo:

- `AqMapStd` (adaptive-quant map variance — high under heavy quant)
- `NoiseFloorY`, `NoiseFloorUV` (block-noise floor — directly
  measures heavy quant artifacts)
- `PatchFraction` (fraction of 8×8 blocks that look smoothed-out)
- `LaplacianVariancePeak` (peak edge response — drops under heavy blur)

Combined with `dct_compressibility_y/uv`, these specifically describe
*how* the image was destroyed at low q, which is what the MLP needs
to predict ranking there.

### 4. Multi-target training: SSIM2 + DSSIM *(medium effort, validated payoff)*

Per CID22 paper Table 7, SSIMULACRA 2 is "very good" in 25-90 but
*DSSIM is "very good" in 0-50* and SSIMULACRA 2 dips slightly below
20. Joint regression on (gpu_ssim2, dssim) — using two output heads,
a sum-of-pairwise-losses, or alternating-objective training — should
inherit the better-of-each-metric per band.

`training_safe_synthetic.csv` already has a `dssim` column populated
on most pairs. Implementation cost: extend `mlp_train.rs` to accept
a list of targets and emit one logit per target, train against the
sum of per-target RankNet losses, then at inference time output a
single distance via a learned linear combination. The rank
correlation with both targets simultaneously is achievable (synthetic
SROCC against either target stays > 0.99 individually).

### 5. Inject a small fraction of KADID training data *(careful — distribution-shift risk)*

KADID-10k contains 10,125 pairs heavily weighted toward 0-50 SSIM2
(it's a heavy-distortion dataset). Including 30% of KADID
source-disjointly in the training pool — alongside synthetic — would
inject direct human-MOS supervision in the low-q range that the
synthetic-only training lacks.

**Risk:** the CID22 paper notes that <5% of KADID's distortions are
compression-relevant (they include blur, noise, color shifts that no
codec produces). The MLP might learn to recognize KADID-style
distortions that don't generalize back to codec output.

**Mitigation:** use only KADID's compression-distortion subset
(`dist_type ∈ {jpeg2k, jpeg}` per KADID's distortion-type column,
which the loader could filter on). That's roughly 405 pairs (4% of
KADID), giving a small but high-signal injection. Same for TID2013
JPEG/JPEG2000 subsets.

### 6. Curriculum training *(moderate effort, uncertain payoff)*

Start the MLP on low-q pairs only (SSIM2 ≤ 60) for the first 30-50
epochs, then expand to the full dataset. Forces the model to develop
representations that handle heavy distortion before getting drowned
out by the easy 75-90 majority. Adam's adaptive learning rates
should let the early-locked-in low-q features survive the second
phase.

## Recommended sequence

| step | effort | expected gain in 25-60 SROCC | tooling needed |
|---|---|---|---|
| 1. Densify q-grid 25-60 | ~6h GPU | +0.02-0.04 | existing generator |
| 2. Sampler bias toward low-q | ~2h code + retrain | +0.02 | mlp_train.rs change |
| 3. Heavy-distortion zenanalyze features | ~4h sweep | +0.005-0.015 | gen_zenanalyze + retrain |
| 4. Multi-target (SSIM2 + DSSIM) | ~1d code + retrain | +0.01 | mlp_train.rs extension |
| 5. KADID/TID compression subset injection | ~half day | +0.005 (or regression risk) | filter loader + retrain |
| 6. Curriculum training | ~1d | uncertain | mlp_train.rs phases |

**Combined upper-bound expectation (steps 1+2+3): +0.05-0.08 SROCC in
the 25-60 band**, bringing V0_6-equivalent from ~0.90 to ~0.95-0.97 —
matching its 60-90 performance. Steps 4-6 are speculative; do steps
1-3 first, re-measure, then decide.

## What to do *right now* if a metric is shipping today

V0_6 dct_hf already has the best per-band performance and the best
calibration MAE in the 25-90 band. Ship it as the user-facing target
metric, with a note that the 25-40 band has roughly 0.90 rank fidelity
to ground-truth SSIM2 and 1.65 SSIM2-units of expected calibration
error per-band (vs 0.69 SSIM2-units in the 75-90 band). For
web-compression decisions in [40, 90], its precision is operationally
fine.
