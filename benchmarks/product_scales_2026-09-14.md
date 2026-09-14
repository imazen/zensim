# Product TRAIN scale frontier — September 14, 2026

Later September 14 timing-admission correction: the model comparison owner used
`GateConfig::disabled()`. Its saved `unreliable=false` flag cannot establish
quiet conditions; the existing process records and descriptive medians remain
historical evidence. The [native latency study](native_latency_2026-09-14.md)
enables strict gating and retains actual individual-call rounds. No older
performance qualification is inferred or retroactively repaired.

Registered after the [mixed-TRAIN baseline](product_train_2026-09-14.md), before
new fits. That experiment supports a scalar advantage for full944 and a spatial
advantage for small local features. This comparison tests whether richer coarse
features recover scalar accuracy with less extraction work and complete spatial
support. It does not use the frozen MT913 EVAL failures to select a candidate.

Reuse the exact admitted product-TRAIN tables, family splits, signed fresh
SSIMULACRA2 codec targets and human rank supervision. Keep calibration unused;
never read EVAL/TEST. Reuse verified native Rev3 features without changing any
formula, scale geometry or filter. Earlier prime/filter studies are historical
evidence with different data/protocols; their internal test segments remain
inaccessible. They do not establish the present product frontier.

Exactly six profiles, three paired seeds 7101/7103/7107, plain H128 unless
specified. Use the preceding corrected uniform mixture, sample seeds +10000,
32 epochs, 8192 draws/epoch, MSE weight 1 and no auto-eval. Match actual sampler
digests to the existing Rust replay. Do not select a best seed. Every complete
three-member ensemble is uniformly served through Rust BakeScorer.

| Profile | Basic features | Rich v2/append/append2 features | Width | Head |
|---|---|---|---:|---:|
| basic228 | All basic/peaks | None | 228 | 128 |
| full800 | All basic/peaks | All four scales | 800 | 128 |
| local549 | Local120 | Scales 1,2,3 | 549 | 128 |
| local406 | Local120 | Scales 2,3 | 406 | 128 |
| y346 | Y60 | Scales 2,3 | 346 | 128 |
| y346 | Y60 | Scales 2,3 | 346 | 32 |

Local120 and Y60 are the exact prior declared ID sets. Each rich scale has
87 v2, 51 append and 5 append2 slots. Scales 0/1/2/3 mean original, half,
quarter and eighth dimensions in the native pyramid. The full800 control
removes only legacy weighted pools f228..371; it tests whether those unsupported
terms explain full944's lift. Coarse variants remove actual v2 passes at finer
resolutions. A count reduction alone is not a speedup. Keep reference-only and
SDR structural-zero identities explicit, and use the existing Rust densifier.

Assess all seeds and complete ensembles through the current round-trip Rust
score exporter and canonical panel/scatter owners: human and codec development,
distorted-only, near-lossless, per-codec/source/class and reference groups. Retain
seed ranges, full geometry and raw tails; MAE remains auxiliary. Compare the
preceding frozen TRAIN controls on these same rows. Degenerate small groups
cannot establish rank; record undefined statistics as null.

Before any performance claim, verify native consumed features and complete
pixel/cached scores on the same five metadata-selected TRAIN JXL pairs from the
preceding campaign. Run every ensemble's finite block interventions with the
same M2 >=.99 / M3f >=.70 bars and reject nonfinite records. Unsupported terms
remain unsupported, even when a partial map produces a correlation. Diagnose
serving defects on TRAIN before adapting models. No arbitrary zeroing of invalid
terms or relaxed gates. Complete spatial support is required before prepared
map benchmarking. Measure actual serving cost with the existing owner on quiet
1MP/4MP ST workloads; retain timings separately from inference correctness.

Stop after 18 fits, six ensembles, the specified scalar/spatial assessments
and measured runtime or explicit reasons it is not admissible. No new kernel,
optimizer or controller. This is a bounded TRAIN development comparison,
not an exhaustive feature ceiling, human superiority claim or qualification.
Native targeting/RD, corruption composition and frozen full EVAL remain required.

Artifacts: `~/work/zensim-validation-2026-09-14/product-scales/`.

## Serving defect reproduced on TRAIN and corrected

The public API regression uses a one-input v2 IW-MSE model at each native scale,
with 128x128, odd 97x131 and padded 17x9 images. Before the correction, its
scale-1 attribution is NaN. This is a serving bug: the retention pass visited
omitted cells with sample count zero, deriving `0 * (1 / 0)` coefficients even
though those cells have no active sensitivity. An initial MSE-only diagnostic
correctly reported an unsupported free-feature extraction variant; the
weighted-MSE case exercises actual supported v2 extraction.

The existing retention owner now skips zero-sensitivity scales/channels before
coefficient derivation. It does not replace invalid active values with zero or
change any feature formula. The regression checks sequential/threaded execution,
scratch reuse, scalar/feature parity, finite nonzero gains, and the weighted-pool
sum identity on unpadded geometry. The broader attribution suite passes 36 tests
with two existing ignored diagnostics. Real TRAIN before/after maps are retained.

Both y346 ensembles previously had nonfinite blocks on all five JXL cases. All
ten calls become finite after the fix, but each still fails four of five spatial
cases. The other four ensembles' spatial values are unchanged. This fixes the
NaN defect; it does not establish that these richer features guide useful repairs.
All six complete ensembles pass native consumed-feature equality and pixel/cached
score audits on the same five original JXL pairs. No fit or calibration changed.

## Measured scalar/spatial frontier

All 18 fits and six uniform three-member ensembles finished. Actual sampling
digests match the prior replay, and 47,322 raw/packed development predictions
are bit-exact. Assessment uses the same 1,000 human TRAIN development pairs and
1,380 distorted codec pairs plus 249 identities. Full Rust panels/scatter and
every seed are retained; EVAL, TEST and the 923 calibration rows remain untouched.

| Ensemble | Human SROCC | Distorted codec SROCC | Codec seed range | High-quality SROCC (57 pairs) | Spatial pass / 5 | Scalar median 1024² / 2048² ms |
|---|---:|---:|---:|---:|---:|---:|
| basic228_h128 | 0.9048 | 0.9532 | 0.9476–0.9505 | 0.7011 | 2 | 17.01 / 79.87 |
| full800_h128 | 0.9177 | 0.9583 | 0.9494–0.9583 | 0.4597 | 1 | 46.32 / 209.72 |
| local549_h128 | 0.8957 | 0.9574 | 0.9542–0.9552 | 0.4085 | 1 | 24.53 / 113.70 |
| local406_h128 | 0.9037 | 0.9519 | 0.9475–0.9509 | 0.5106 | 1 | 19.19 / 88.29 |
| y346_h128 | 0.8904 | 0.9469 | 0.9404–0.9464 | 0.2311 | 1 | 12.83 / 56.42 |
| y346_h32 | 0.8548 | 0.9443 | 0.9393–0.9436 | 0.3117 | 1 | 12.77 / 56.39 |

On the same rows, the prior full944 plain ensemble had human SROCC .9193 and
distorted codec SROCC .9605; its current interleaved scalar medians are
53.30 / 240.65 ms. Basic228 recovers much of this bounded scalar performance
at 17.01 / 79.87 ms, while removing the rich v2 passes. The cheaper Y346
alternatives lose more scalar rank and fail four spatial cases each. Full800
and local549 approach full944 scalar rank but their repair maps fail four cases.
These point estimates and seed ranges do not establish population superiority.

Removing legacy weighted pools alone retains most of full944's measured rank
while lowering the scalar median from 53.30 to 46.32 ms at 1024². That supports
keeping full800 as a control for those pools; it does not make its poor spatial
response acceptable or establish a universal feature-removal result.

Full800 has one above-100 distorted score; y346/H128 has three and y346/H32
one. Every such row remains in the panels. The full Mohammadi OR and the
shape-normalized out4 envelope are separate statistics: neither is substituted
for the complete product gate. Only 57 high-quality distorted rows limit
near-lossless conclusions. Complete outliers, density and per-codec/source/class
results are in the [hashed summary](product_scales_2026-09-14.results.json) and
`assessment-panels/RESULT.json`; there is no complete selection composite here.

## Runtime, memory and disposition

The existing benchmark interleaves final complete ensembles on a deterministic
textured RGB8 pair, CPU 8 pinned, one worker, 30 accepted one-call rounds at
each geometry. Its owner marks neither run unreliable. The table gives medians,
not p95; the owner does not retain/export raw rounds. Machine and competing
process records are retained. Named D is current-process timing context, not
a repaired historical qualification. No inference speedup is inferred from
feature counts or fit times.

Basic228 prepared score+map medians are 79.55 / 339.27 ms, versus its scalar
17.01 / 79.87 ms. Its cost ratio and failed repair cases prevent a production
spatial claim. The prior small H32 control is 8.31 / 37.82 ms scalar and
23.78 / 107.46 ms prepared. Rich-v2 prepared sessions remain explicitly refused
by the current basic/peak-only contract; legacy attribution availability does
not silently expand that API. The new scalar calls and before/after spatial
measurements all execute official Rust surfaces.

Fourteen isolated RSS measurements cover all six scalar profiles at both sizes
and basic228 prepared at both sizes, 30 repeated calls per process. Every total
process peak lies below 128 bytes/pixel +64 MiB; these include program/input
buffers and conservatively bound worker allocations for these inputs. They are
not incremental estimates or native codec-worker measurements. Full receipts
are in `rss/RESULT.json`.

**None of these six ensembles passes all five TRAIN spatial cases.** Retain
basic228 as the next inexpensive scalar reference and y60/H32 as the spatial
control. The next consequential question is which added basic/HF/peak pools
buy basic228’s scalar improvement, and whether removing hard-max work or
changing head capacity preserves that improvement with accurate cheaper maps.
This is preferable to assuming more rich coarse features solve steering.
Do not spend EVAL on candidates that still fail their registered TRAIN gates.
Frozen qualification, broader corruption/aliasing/channel-swap coverage, native
bounded targeting/RD, and controlled p95 remain required. The full goal is active.
