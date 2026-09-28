# Frozen ensembles in native JXL steering — September 14, 2026

Seven of eleven frozen compositions complete the existing native JXL experiment.
Four correctly refuse unsupported spatial terms. The encoder now accepts complete,
hash-bound ensembles through the existing Rust `BakeScorer::ensemble` surface.
A real neutral-control quantization bug is fixed. **No model is promoted:** these
small-image diagnostics do not resolve the previous human-rank/dial failures,
corruption specificity, HDR, or production performance requirements.

The served report is `/zensim/reports/jxl-ensemble-native-2026-09-14/index.html`;
the full and fair summer gauntlets add the latest native-JXL discussion and
per-model measured/unsupported annotations while preserving all prior panels,
scalar targeting results and product qualification states.

## Scientific contract and chronology

The frozen MT913 candidates and matched B/D retain their complete original bytes,
weights and formula revisions. The September 14 public TEST permission supersedes
the older blanket ban, but this experiment does not read public TEST or secret
holdouts. Seed curves use the nine TRAIN families admitted by the preceding
scalar-targeting correction. The eight original validation families are unchanged.
Later reservations exclude 6068/9066/8462 from the old twelve-family seed packet.
No feature selection, checkpoint, weights, gain or policy is fitted to these results.

A TRAIN smoke on source 2010 precedes cohort admission. Each supported composition
then uses 21 distance knots, three arms (scalar zero updates, neutral two updates
with H3 gain 0, active two updates with H3 gain 10), effort 8/Zenjxl, bin 8, no automatic
resampling and the existing delivered U8/dither decode. These are opaque sRGB8,
roughly 256-pixel-long-edge sources; this is not native HBD/ICC/HDR qualification.

All per-image/arm bounds are measured before target search. Search never receives
those ladders; witnesses only admit assessment requests. The same target must
have a measured witness within one score unit in all three arms. Values outside
the sampled envelope, or inside without a witness, are coverage gaps rather than
assumed failures or proofs of codec impossibility. Five fixed targets and five
neutral-ladder quantiles are considered per image. The tolerance of one point is
the existing numerical protocol, not a universal perceptual error bar.

## Corrected native execution

Previously the loop recomputed integer quantizers/global scale from an unchanged
floating-point field at its first comparison and final emission. Gain0 therefore
changed 13/21 source 2010 bitstreams. Skipping zero-gain clamping did not fix it;
that failed hypothesis/binary is retained and the speculative code removed.
Preserving the seed's original discrete state whenever the float field is unchanged
fixes all 21. Across seven supported models and smoke/TRAIN/EVAL ladders, **2,646/2,646
neutral pairs match bytes, decoded pixels and score**. Active steering changes
2,642 of those paired bitstreams. Every paired neutral/scalar selected target
output also matches exactly.

The candidate mount accepts `ensemble:<manifest>` with name, ordered member
paths/SHA-256 hashes and convex weights. Existing singleton `bake:` arithmetic
is preserved. The existing native targeting example binds calibration to the
whole manifest hash, every member hash, its executable and actual formula1/3;
it no longer forces formula1. There is no new public Rust API or alternate scorer.
The lockfile admits the already-pinned zensim/zenresize dependency closure; the
existing cubecl pin stays unchanged.

## Coverage and target errors

The seven supported models execute **4,914 targeting cases**, 8,041 search encodes,
4,914 terminal decode/score verifications, 3,969 TRAIN ladder encodes, 3,528 EVAL
bound encodes and 441 successful smoke encodes. There are 36 negative-target cases.
Native reconstruction/comparison/map evaluations total 31,977 including ladders.
The failed initial controls and unsupported attempts are additional retained work.

Three-shot TRAIN-curve absolute errors below are in original score units. Scalar
and neutral have identical score/error distributions. Different models admit
different witnessed targets; this is not a controlled ranking between models.
All 1/2/3-shot, midpoint, coverage, class and paired-family results are included
in the canonical analyzer outputs.

| Composition | Joint targets | Scalar median / p95 | Active median / p95 |
|---|---:|---:|---:|
| MT913_full944_h128_ens5 | unsupported | 180 unavailable spatial terms | not measured |
| MT913_full944_h256_ens5 | unsupported | 180 unavailable spatial terms | not measured |
| MT913_local120_h128_ens5 | 46 | 1.042 / 2.939 | 0.948 / 4.001 |
| MT913_selected619_h128_ens5 | unsupported | 30 unavailable spatial terms | not measured |
| MT913_y40_h32_ens5 | 39 | 0.056 / 1.458 | 0.233 / 1.117 |
| MT913_y40_h128_ens5 | 38 | 0.099 / 1.086 | 0.192 / 1.347 |
| MT913_y60_h32_ens5 | 35 | 0.088 / 0.539 | 0.167 / 1.436 |
| MT913_y60_h128_ens5 | 35 | 0.123 / 2.028 | 0.102 / 0.906 |
| MT913_linear60 | 41 | 0.219 / 1.007 | 0.389 / 1.009 |
| MT914_matched_B | unsupported | 40 unavailable spatial terms | not measured |
| MT914_matched_D | 39 | 0.248 / 0.753 | 0.281 / 0.871 |

## Independent perceptual RD diagnostics

The pinned existing zenmetrics CPU owner evaluates all 8,442 reference/output pairs
with both SSIMULACRA2 and Butteraugli (16,884 judge scores). The existing RD analyzer
compares active output bytes to interpolated scalar bytes at equal independent
judge score, without extrapolation. Positive means smaller; negative means larger.
The table gives **mean percentage bytes saved on the active distance ladder**,
SSIM2 / Butteraugli, by class. Each class has only two source families; its numerous
ladder points are correlated and must not be treated as independent images.
Full distributions, medians and separately deduplicated targeted-output comparisons
remain in `analysis_summary.json`. Direct matched-quality confirmation is still needed.

| Composition | Document | Graphic | Photo | Screen |
|---|---:|---:|---:|---:|
| MT913_local120_h128_ens5 | -1.52% / -0.01% | -0.46% / -0.97% | +0.64% / +0.16% | -0.03% / -2.24% |
| MT913_y40_h32_ens5 | +1.23% / -0.48% | +1.09% / -0.00% | +0.58% / -0.32% | +0.52% / -0.69% |
| MT913_y40_h128_ens5 | -1.68% / -1.69% | +1.04% / +0.39% | +0.51% / +0.09% | +0.12% / -0.23% |
| MT913_y60_h32_ens5 | +1.48% / -0.57% | +0.60% / +0.10% | +0.81% / +0.14% | +0.34% / -0.26% |
| MT913_y60_h128_ens5 | -2.33% / -0.75% | -0.44% / +0.13% | +0.93% / +0.42% | +0.08% / -0.76% |
| MT913_linear60 | +0.61% / -1.19% | -0.96% / -0.37% | -0.59% / -0.10% | +0.60% / -0.53% |
| MT914_matched_D | +2.45% / +0.57% | +0.43% / +0.77% | +0.58% / +0.29% | +1.73% / +0.22% |

The judges disagree for several new models: for y60/H32, document SSIM2 estimates
1.48% saved while Butteraugli estimates 0.57% extra bytes. Local120 loses 2.24% on
screens by Butteraugli; y60/H128 loses 2.33% on documents by SSIM2. Matched D has
positive class means under both judges in this small diagnostic, but its prior
dial failures remain and this does not qualify shipping. Tight target error alone
cannot establish useful spatial allocation. These results do not authorize tuning
frozen models against the EVAL images.

## Validation and limitations

Twenty-one independent sample replays (one selected output per supported model
and arm) have exact Rust-decoded pixel hashes and complete f32 scores through the
canonical extractor. All 21 also decode with libjxl v0.12.0. The latter checks
compatibility, not cross-decoder pixel equality. All output identities are bound
by the existing analyzer. Six actual CLI negative controls refuse nonprimary hash,
invalid/changed weights, formula/config mismatch and TRAIN/EVAL family overlap
before creating outputs; source-file hashes may already have been read.

The analyzer now records uninstrumented JXL `consumed_non_neutral_maps` as null,
not a misleading zero. Actual map evaluations are measured and byte intervention
is checked separately. Initial assemblies are retained. Timing records include
competing builds and are descriptive; no production speedup or latency claim is
made. All model bytes, manifests, calibrations, commands, output bitstreams/PNGs,
judges, checks and source snapshots are bound in the replay bundle. Source images
retain their original hash-bound paths; relocation requires explicit path mapping.

Remaining product work includes TRAIN-driven correction of ranking/dial failure,
reliable spatial allocation and direct RD confirmation, severity-adjudicated
corruption training/specificity, native color/HBD/HDR assessment and quiet-machine
latency/memory qualification. AVIF/JPEG/WebP native steering remains separate.

The existing codec compatibility suite separately opens two published CID22
validation-reference images and two screenshot fixtures, with no human labels
or Zensim model selection; that exposure is recorded in ENGINEERING_EXPOSURE.json.
Its first run failed because its legacy decoder path requires a missing shared
library. The rerun explicitly selects the verified libjxl v0.12 binary.

Local validation passes: JXL workspace and native-example Clippy, scoped format,
all six loop tests, the default suite with the explicit v0.12 decoder, and both
registered RD-regression tests. Zensim CI-exact Clippy, four target-analyzer tests,
616-script lint, and full/fair board render/data gates also pass. The original
failed decoder-path run is retained. No CI wait or package release.
