# Product training packet — September 14, 2026

Preregistered before source sampling, extraction or fitting. The nine MT913
candidates remain frozen. This follows the original product goal's requirement
for representative train-side development, not a refit to their eval outputs.

Recover the existing keyed W-LIN7 training bitstreams before generating another
codec corpus. Read only identity, source, codec, knob and path columns from its
208,169-row key sidecar: no historical feature or target column. Its URI manifest
declares train and pins the keys. Apply the September 8 source-family map, then
exclude historical reserved content before any image read. Reconstruct the old
ceiling screen's five reserved codec families from its source-only pair keys
and original hash-order algorithm: origins 1220/1634/7004/7050/7058/8134.
Also exclude 8462/9066 from the old corruption screens. Exclude every mapped
relative of these sources. Preserve the earlier AVIF eval8 reservation by
excluding suffix-8 origins and their families from this new training packet.
Canonical validation/test families remain excluded. No reservation is renamed.

Bounded screen selection uses original renditions up to 1024 on their longest
side. For every remaining origin and codec, retain at most ten equally spaced
numeric knob settings including endpoints, and at most two distinct reference
renditions per setting chosen by encoded-filename SHA order. Selection uses no
quality label. Include identity at each selected reference rendition. Retain
all source classes present; record missing codec/source cells and size limits.
This is a bounded training screen, not a claim of exhaustive feature capability
or large-image qualification. Full-resolution cost and spatial tests remain
separate requirements.

Assign whole source families by SHA-256 modulo ten: 0/1 development, 2
calibration, 3–9 fitting. All partitions remain within TRAIN; development and
calibration are for training decisions, never independent qualification. Pin
original file bytes and all assignments before native extraction. Extract
mathematically versioned Rev3 full944 and same-buffer fast-ssim2 through the
existing Rust extractor/audit. Provisional zero targets in the extraction input
are explicitly forbidden as fitting labels. Only verified fresh peer scores
become codec targets. Deduplicate by decoded reference/distorted pixel identity
within a source, retaining membership provenance and checking target equality.

Human supervision may reuse the already admitted native KADID/TID TRAIN packet
after hash/schema checks; split training KADID sources for internal development,
keep TID training-only, and keep codec absolute supervision separate from human
within-reference ranking. Model architecture/mixture/scale comparisons require
their own bounded training-only registration after admission is measured.
Use the existing trainer, public BakeScorer and Rust panel owners. No new scorer,
metric kernel, optimizer or controller. Historical best-of-all constraints,
transform/identity caveats and seed-order findings must inform that registration.

Artifacts: `~/work/zensim-validation-2026-09-14/product-train/`. Missing inputs,
ambiguous roles or nonfinite output block admission; no silent dropped rows.

## Registered first fit comparison

Admission measured 10,499 codec/identity pairs, 148 sources and 141 families
across 19 source classes. TRAIN partitions contain 7,947 fitting, 1,629
development and 923 calibration pairs. There are no exact decoded duplicates
or identical reference pixels crossing these partitions. Native human TRAIN
reuse supplies 7,000 fitting rows and 1,000 internal-development KADID rows;
the latter's re-extracted 944,000 features match exactly and carry pixel identity.

Run exactly 24 fits: y60/H32, y60/H128, local120/H128 and full944/H128, each
plain and existing `--nonneg-distance`, each at seeds 7101/7103/7107 with order
seeds +10000. These use the already declared native scale read sets; no new
feature formulas or filters. H128 comparisons control capacity, and H32 measures
the minimal head alternative. The nonnegative-distance control comes from the
September 6 best-of-all program, including its documented convexity restriction.
No transforms that move zero, post-hoc clipping, synthetic zero-feature identity
assumptions, or new activation implementation. Full944 has reference-dependent
identity features; only actual decoded identity gets the public API identity rule.

Use equal sampling weight for two groups: human within-reference **rank only**,
and codec within-reference **rank plus absolute score**. Human MOS affine units
are not a calibrated codec dial. Codec absolute targets are fresh signed
SSIMULACRA2 scores, including actual identities and negative tails. Keep MSE
weight 1, 32 epochs, 8,192 pairs/epoch, stratified pair sampling and f32 output.
Training-only checkpoint monitoring; no auto-eval or calibration/development
group in fitting. Record per-seed coverage, completed time and final bytes.
Retain refusal/failure artifacts, and do not substitute the best seed.

Densify through the existing Rust owner, proving original/packed served scores
equal on both internal-development populations. Assess every seed and complete
uniform three-member ensemble through public BakeScorer and canonical Rust
panels/scatter. Report signed rank, within-reference and per-codec behavior,
raw tails and score range; do not use MAE as the selection objective. The first
comparison leaves calibration rows unused. It establishes development evidence,
not qualification, and does not open EVAL. Any spatial check must use admitted
TRAIN sources; complete native targeting/RD and runtime gates remain outstanding.

## Sampling correction, registered before the replacement fits

Stopped the first campaign after eight completed fits and retained both active
fit logs. Its declared 1:1 mixture did not hold: the legacy global stratified
schedule visits 194 human versus 2,002 codec eligible strata, ignoring relative
group weights. It produced only 8.8806% human pairs and reached 86.9004% of rows;
1,958 singleton reference/band cells are ineligible. These fits cannot represent
the registered balanced experiment. This is a mismatch with existing sampler
semantics, not evidence against any feature set. The trainer now warns about
the actual group shares; existing sampling sequences stay unchanged.

Use the existing uniform within-reference sampler for the replacement campaign.
It respects group draw weights and reaches singleton-band rows when a reference
has other rows. Correct its same-row collision bias analytically from TRAIN
reference counts: for group g, `a_g = mean_ref(1 - 1/n_ref)` is the probability
of a usable pair; draw weight `1/a_g` makes expected usable group mass equal.
Weights are human 1.0082131044119642 and codec 1.2121805295885542. This is source
count accounting, not label tuning. All other registered fit parameters remain
unchanged. Replacement artifacts use `fits-uniform/` and model prefix `PT914U`.

The existing Rust `subset_sim` replays all three sample seeds before fitting.
Uncorrected uniform weights give human shares 54.426–54.703%; corrected weights
give 49.881–50.198%, with 100% row coverage for every seed. Match each actual
training sample-sequence digest to its replay before accepting the fit.
No development score, EVAL row, or calibration label chose this correction.

## Measured TRAIN development results

All 24 corrected fits and all eight complete uniform ensembles finished. The
three actual sampling digests match the preregistered Rust replay. Every training
row is reached; usable human pair share is 49.881–50.198%. Individual fits took
24.84–50.32 seconds with two concurrent workers; these are training wall times,
not inference benchmarks. Calibration remains unused.

The score-export owner formerly rounded targets and scores to six decimal
places. Its output now preserves round-trip precision. Re-scoring the same
final model bytes verifies **63,096 raw/packed row predictions bit-exactly**.
No model was refit for this change. The original rounded dumps remain intact.
All current panels use the exact replay.

The Rust panel/scatter owners measured all 32 served alternatives. Codec
development contains 1,380 distorted pairs plus 249 identities from 23 sources
and 11 classes. Only 57 distorted pairs have proxy target >=91. Human
development has 1,000 KADID pairs from eight TRAIN sources. These are internal
development populations, not EVAL.

| Three-member ensemble | Human SROCC | Distorted codec SROCC | Codec seed range | >=91 SROCC | Shape out4 | JXL spatial cases |
|---|---:|---:|---:|---:|---:|---|
| y60_h32_plain | 0.8389 | 0.8780 | 0.8767–0.8780 | 0.4600 | 5.51% | 5 pass |
| y60_h32_nonneg | 0.8488 | 0.8743 | 0.8738–0.8746 | 0.4800 | 5.65% | 5 pass |
| y60_h128_plain | 0.8467 | 0.8804 | 0.8762–0.8801 | 0.5164 | 6.16% | 4 pass, 1 fail |
| y60_h128_nonneg | 0.8403 | 0.8732 | 0.8721–0.8743 | 0.3867 | 6.01% | 2 fail, 3 pass |
| local120_h128_plain | 0.8654 | 0.8982 | 0.8922–0.9009 | 0.6386 | 9.13% | 4 pass, 1 fail |
| local120_h128_nonneg | 0.8292 | 0.8917 | 0.8881–0.8927 | 0.4557 | 6.67% | 2 fail, 3 pass |
| full944_h128_plain | 0.9193 | 0.9605 | 0.9542–0.9589 | 0.5174 | 5.65% | 5 unsupported |
| full944_h128_nonneg | 0.9196 | 0.9578 | 0.9564–0.9569 | 0.2369 | 7.68% | 5 unsupported |
| Matched B | unmeasured | 0.8488 | — | 0.4072 | 9.42% | unmeasured here |
| Matched D | unmeasured | 0.8494 | — | 0.3885 | 9.20% | unmeasured here |

`out4` is the canonical shape-normalized envelope statistic on distorted pairs;
it is not the raw score-error outlier rate. Full Mohammadi, signed rank, raw
density/slope/tails, normalized geometry, per-source/class/codec and per-reference
panels are retained in `assessment-panels/RESULT.json`; the committed
[summary](product_train_2026-09-14.results.json) includes every seed, ensemble,
matched baseline and evidence hashes. Do not substitute this table for the
full composite. Small/constant panels are explicitly non-informative; undefined
statistics are null. Of 1,033 codec/rendition ladders only 32 have four or more
observations, limiting conclusions about per-image targeting.

All eight ensembles remain at or below 100 on this development population,
without post-hoc clipping. This is measured population behavior, not a universal
model guarantee: three individual plain seeds still emit scores above 100
(seven y60/H128 seed-7103 rows, four seed-7107 rows, and three full944/H128
seed-7107 rows). Full944 plain improves distorted codec rank over matched B/D
and has the strongest scalar result in this bounded comparison. Nonnegative
distance does not consistently improve the panels; at full944 it substantially
reduces near-lossless rank. A 57-pair high-quality slice is too small to establish
near-lossless capability. Proxy agreement cannot prove perceptual superiority.

## Spatial evidence and remaining work

Frozen metadata-only selection picked five JXL q50 TRAIN development cases:
photo, report, plot, screenshot and AI product. Their longest sides are 96–256
pixels, block size 32. Existing Rust decode-list output matched every fresh
native extraction pixel hash. All 40 complete-ensemble spatial calls match the
cached scalar within 7.11e-15, with finite block records. Both y60/H32 heads
pass M2 >=.99 and M3f >=.70 on every case. Several H128 cases fail M2. Full944
reports unsupported refinement IDs on all ten calls; its partial-map values
are not passing spatial evidence. These finite pixel interventions are not
native encoder allocation or RD results.

The first spatial invocation supplied unnormalized explicit weights; all calls
refused before scoring. The corrected invocation uses normalized thirds and
identical frozen cases/models. A first panel serialization attempt refused
undefined small-group statistics; the completed harvest reuses exact saved
scores and records those statistics as null. Failure artifacts are retained.

The next train-side questions are consequential: which full944 contributions
explain the scalar advantage, which can be retained at inexpensive scales with
complete spatial support, and how do they behave on representative corruption
and high-quality controls? Keep the small H32 spatial controls and full944
scalar reference while testing that tradeoff. This run does not test fractional
scales, new kernels or an exhaustive feature ceiling. New candidates require
frozen independent EVAL, native bounded 1/2/3-shot targeting, native spatial RD,
corruption composition, and controlled p95/runtime/memory before shipping.

Validation: 19 Rust sampler tests, CI-exact Clippy, API documentation snapshot,
formatting, and script hygiene pass. The scored real-model replay and matched
B/D native audits passed. **No model is qualified; the full goal remains active.**
