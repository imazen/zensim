# Coarse features beyond 228: legacy evidence and next experiment

September 13 follow-up to the sampling-filter screen. The question is whether
costlier features beyond the 228 basic/peak set contributed at ¼ and ⅛
resolution, allowing their fine-scale computation to be omitted.

**Yes.** There is both current structural evidence and historical measured
score influence. These are different claims: nonzero first-layer weights do
not establish importance, and one-feature contribution does not establish
that a feature is indispensable after refitting correlated features.

## Current baked read sets

Rebuilt `bake_block_profile` from the current source. Its old executable on
this machine predated the dense-ID fix and incorrectly classified packed
positions as feature IDs; its first output was discarded. The rebuilt
`--lines` option reports the existing Rust owner's caller-line L2 norms,
joined to each bake's declared feature IDs. No wire-format parser was added.
The following are exact nonzero counts outside f227 in the named September 6/7
bakes, grouped by the existing feature definition map:

| Model | Full | ½ | ¼ | ⅛ |
|---|---:|---:|---:|---:|
| A, legacy 372 | 30 | 32 | 32 | 34 |
| B, legacy 372 | 8 | 6 | 5 | 4 |
| BHdr, legacy 372 | 15 | 14 | 5 | 11 |
| C, legacy 944 | 118 | 131 | 131 | 131 |
| CHdr, legacy 944 | 124 | 139 | 139 | 139 |
| D, basic 156 | 0 | 0 | 0 | 0 |

The [JSON audit](data/coarse_legacy_features_2026-09-13.json) preserves model hashes and per-ID norms. Not every ID beyond 228 is expensive: that range also contains cheap raw
moments and statistics sharing existing blur work. Norm magnitudes are
not comparable contribution percentages across features with different
transforms/scalers or across models. This table establishes that coarse
features were actually read, including by the current dense-ID twins.

## Historical influence, with chronology preserved

The August 4 `bake_contrib` runs predate this sampling experiment and the
September dense-ID repacking. The method sets one standardized input to its
training mean, then measures the absolute change in the complete model's
score. The canonical tool's baseline was parity-gated to the scoring owner.
Read the existing [run metadata](bake_contrib_2026-08-04.meta) for exact
bakes, source roots and limitations. We have **not** rerun their protected
corpora, reused old feature values for the new models, or treated an August
candidate as today's shipped C. The figures below use the recorded
`mean_abs_imazen26` column, not correlations pooled across unrelated corpora.

For legacy B, representative coarse extras were:

| ID | Legacy feature | Resolution | Mean absolute score change |
|---|---|---:|---:|
| f267 | X masked artifact fourth-moment pool | ¼ | 3.468 |
| f279 | B masked artifact fourth-moment pool | ¼ | 1.381 |
| f351 | B IW artifact fourth-moment pool | ¼ | 1.027 |
| f297 | B masked artifact fourth-moment pool | ⅛ | 1.640 |
| f369 | B IW artifact fourth-moment pool | ⅛ | 1.091 |

Source: [B-primary contribution table](bake_contrib_b_sdr_372primary_2026-08-04.tsv).
These are substantial responses to individual features in score units,
though this intervention is not a realistic codec change or removal-refit.
B also had stronger fine-scale extras; the evidence does not justify deleting
those from its existing bake and expecting unchanged quality.

Across three legacy 944 candidates (C_co3a_s1301, C_em944_s31 and
H_co3abpg_s2507), prominent coarse extras included ⅛-scale blockiness
f658/f687/f716: individual mean absolute changes about 0.24–0.47 points on
that historical imazen slice. Quarter-scale blockiness f571/f600/f629 and
GMS_DEV2 f849 also contributed. The sums of individual absolute changes for
coarse extras were about 23–27% of the corresponding sums over all features.
That is a descriptive aggregate of overlapping interventions, **not** the
fraction of score explained, marginal quality, or a joint-ablation result.
Sources: [C_co3a](bake_contrib_C_co3a_s1301_2026-08-04.tsv),
[C_em944](bake_contrib_C_em944_s31_2026-08-04.tsv),
[H_co3abpg](bake_contrib_H_co3abpg_s2507_2026-08-04.tsv).

## Compute opportunity and concrete next screen

A per-pixel expensive family evaluated only at ¼ and ⅛ sees
`1/16 + 1/64 = 5/64` of full-resolution area. Relative to that family at all
four legacy levels, the ideal pixel work is `(5/64)/(85/64) = 1/17`, or 5.9%.
This excludes shared conversion, pyramid construction, fixed costs, padding
and retained-map overhead; it is not a measured total-model speedup.

At the time of this audit the plan had global family switches. Selecting only coarse feature
IDs currently does **not** guarantee that the expensive fine-scale kernels
or their intermediate planes disappear. Add private per-scale compute masks
at the kernel/retention dispatch, keep canonical feature IDs, and verify both
consumed-feature parity and actual skipped work. In particular, coarse extras
must not accidentally reactivate full-resolution X/B via a global family flag.

The next bounded screen should compare the current full-Y/coarse-XYB 190
control with the same basic/peak inputs plus¼/⅛ masked/IW features (72 added
IDs, 262 total), then coarse-only 944 gradient/blockiness/append groups.
Start with the unchanged box pyramid, fresh Rev3 extraction and paired fits;
measure scalar cost and public-API spatial interventions before broadening
training. This directly tests compute saved, while retaining full-resolution
luma for small errors. The sampling screen's salt/pepper and R/B-swap failures
remain required checks, not a reason to declare this proposal qualified.

The [subsequent implementation and measured screen](coarse_pool_dispatch_2026-09-13.md)
supersedes that dispatch limitation for v1 masked/IW. It also confirms an
important distinction from the July research: these legacy pools have no
supported spatial integrands. Keep them as scalar cost controls; the next
spatial candidates need the existing v2 families instead.

## Channel-swap features to screen next

Later user review lowers this proposal's priority: all 20 sampling models
already reject the whole-image R/B swap with negative scalar scores. Treat
that as a scalar rejection check, not the main spatial acceptance criterion.
Aliasing, salt-and-pepper and ordinary JXL artifact sensitivity take priority.
The proposal below remains unimplemented research context.

Detection and useful repair guidance are separate problems. The
[September 8 D228 corruption screen](../docs/CANONICAL_CORRUPTION_REFIT_2026-09-08.md#honest-cost-follow-up--after-the-first-d228-screen-before-new-fits)
already caught all RGB swaps it tested, but penalized some honest near-lossless
outputs. Low B weighting is therefore not an established sole cause. In this
new sampling packet, the box/Y190 model gives the R/B swap a very low scalar
score while its 8-pixel repair correlation is negative. More scalar penalty
alone cannot establish useful spatial steering.

A proposed cheap additional signal is a **channel correspondence margin**.
For each RGB permutation P, accumulate `E(P) = mean(sum_i (A_i-B_P(i))²)`;
compare identity error with the minimum over the five nonidentity
permutations. `max(0, E(identity)-min_P E(P))` is zero for exact identity
and for an exactly achromatic reference, but can expose swapped channel
correspondence that channel-energy summaries hide. Bound/normalize it using
an explicitly specified chroma-energy floor. The nine cross-channel second
moments let these errors share one accumulation pass; actual conversion and
retention costs still need measurement. This is a proposal, not a new
implemented or trained feature.

Screen global and local pooled versions against existing D228 inputs, with
unchanged honest-output protection. Cover all five permutations, local and
whole-image swaps, grayscale/inert controls, ordinary chroma subsampling,
desaturation, color shifts and near-lossless codec negatives. Coarse sampling
may miss small swapped regions, so do not assume it is sufficient. Before a
fit, define the color domain, scale/feature IDs and local contribution rule
in the Rust feature/attribution owners. Train through the existing owner and
evaluate complete final bakes through BakeScorer, including actual block
repairs and measured extraction cost. It must improve separation and spatial
guidance without adding another end-user control.
