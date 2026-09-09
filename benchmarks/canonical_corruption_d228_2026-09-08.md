# D's existing features can detect RGB swaps; this companion is not shippable

September 8, 2026, later than the all-372 serving refusal. The user requested
a corruption head within **D's current extraction regime**. This replaces the
previous proposed extraction-plan expansion. [Registration and stopping rules](../docs/CANONICAL_CORRUPTION_REFIT_2026-09-08.md).

**Result: all tested non-inert RGB swaps detected, but honest-output protection
fails.** No named profile or default model changes. No model qualifies.

## What was trained and served

The existing canonical trainer fits HGB on f0..227: all 156 basic plus 72 peak
features. D already materializes these, including X/Y/B, through `Plan::for_bake`.
The head adds no image pass or masked/IW extraction. Its 100-tree inference has
a separate cost. The full 372-column source contract remains intact; only
estimator inputs are gathered by explicit IDs. ZCTH also declares those IDs.

Use the existing completed content-admission receipt and the same source roles:
8 fit origins / 5,504 unique pairs; 4 separate calibration origins / 2,709 pairs;
8 validation origins / 5,679 pairs. Full pixel scoring includes all 15,060 raw
attempts and deduplicates before reporting. Source weights, HGB settings,
calibration and P>0.9 deadband remain registered. Seeds 4101/4103/4107 have the
same numerical model and results; their provenance bytes differ. These are
not three independent generalization experiments.

Every seed exports and runs through `BakeScorer` in the pinned native extractor.
For each, Rust/sklearn parity on 13,892 f32-derived feature rows is 0 raw ulp,
0 probability error and 0 fire disagreements. Full pixel versus cached and
stored-f32 **composed scores** agree exactly on all 15,060 attempts, and all
678 identity attempts remain exactly 100. Consumed-feature comparison passes.

One honest training JXL row (index 8580, origin 2010) has a confidence difference
of 0.063934 between the full-precision pixel features and their stored-f32
rounding; neither fires, so scores agree. Thus probability parity is exact for
the exported model on the *same input precision*, not between every pixel and
rounded feature row. This distinction matters for the cost-4 failure below.

## Validation result, fixed bars

| Measure | Result | Required |
|---|---:|---:|
| Corruption detection | 5,330/5,353 = 99.57% | >=95% |
| Real-bug detection | 312/317 = 98.42% | >=90% |
| Honest native codec outputs lowered | 20/304: 15 AVIF, 5 JXL | 0 |
| All honest outputs lowered | 22/326 = 6.75% | <=1% |
| Corruptions strictly below their honest q20 anchor | 4,877/5,353 = 91.11% | >=99% |

The q20 failure is largely an honest-anchor false positive: origin 3311's q20
score falls from 44.45 to 0, producing 458 ordering failures on that origin.
A gate which catches more corruptions but also floors the honest anchor cannot
satisfy this product requirement.

Channel-operation reporting now joins the original, hash-bound generator
metadata by row ID; it does not guess operation names from filenames. Counts
below are unique source/pixel pairs within each operation, across full/local
regions and opacity levels. Operations can share pixels; do not sum them as
independent observations. Inert attempts remain honest identities.

| Swap | D alone below q20 | Historical head detected | New D228 head detected |
|---|---:|---:|---:|
| R/G | 8/120 | 116/120 | 120/120 |
| R/B | 10/120 | 116/120 | 120/120 |
| G/B | 16/117 | 114/117 | 117/117 |

All these new swap cases also pass strict below-q20 ordering. The historical
head comparison is a replay on the same packet, not new held-out evidence for
its older training recipe.

## Is B weighting the cause?

D is strongly luminance-led. The existing Rust contribution owner, rebuilt to
report actual declared IDs, measures X/Y/B shares of **2.64% / 92.17% / 5.19%**
on the 5,504 fit rows. These are sums of mean absolute raw-output changes when
each standardized input is zeroed; they are neither coefficient percentages
nor a causal whole-channel ablation. D reads 9 X, 12 Y and 7 B basic IDs.

The positive XYB transform uses opponent color signals: B is derived from a
mixed cone response minus Y, not the original RGB blue value. A channel swap
can alter X and Y as well. Swaps retain much spatial structure, and on gray
pixels can be inert. A perceptual score and a bug detector also have different
objectives. Low learned chroma influence is a plausible contributor to D's
weak swap ordering, but this experiment does not establish B weighting as the
sole cause. Crucially, the detector reads the unweighted X/Y/B feature values,
and the successful swap detection demonstrates useful information is present.

## Bounded honest-cost follow-up: none advances

The first recipe already falsely lowers two honest JXL outputs on the four
training calibration origins. A threshold above the largest honest raw response
would retain only 83.97% of their corruptions. Before further fits, register
honest **fit** weight multipliers 4, 16 and 64; leave the scaler, calibration,
tree shape, threshold and acceptance bars unchanged. Select only an arm whose
three seeds pass every bar on calibration. These arms score **only the twelve
training origins** through Rust; their parity matrices exclude validation too.

| Honest fit multiplier | Calibration outcome | Status |
|---|---|---|
| 4 | One pixel/stored-f32 fire disagreement in seed 4101 | Serving parity refused; no quality report, later seeds not run |
| 16 | Detection 2,524/2,546; 2/163 honest lowered; ordering 2,532/2,546 | All three seeds fail honest protection |
| 64 | Detection 2,488/2,546; 1/163 honest lowered; ordering 2,499/2,546 | All three seeds fail zero native lowering and ordering |

No arm is selected. Validation is not scored for any of these three cost arms.
The registered cost sweep stops here. Next work is broader admitted honest
codec/chroma/tone examples and source coverage, plus an explicit training versus
serving precision contract for the sharp tree boundary. Keep D's feature regime;
another threshold sweep or a wider image extractor is not supported by this
experiment. The eight-source validation screen is small and repeatedly exposed;
any eventual finalist still requires untouched qualification sources.

## Reproduction and artifacts

Packet: `/mnt/v/output/zensim/canonical-corruption-d228-2026-09-08/`.
Windows copy: `~/work/zensim-validation-2026-09-08/canonical-corruption-d228/`.

`FIT_MANIFEST.json`, `REGISTRATION.md`, `fit-final/seed-*/` bind the main fit;
`FIT_MANIFEST-cost*.json`, `REGISTRATION-cost-followup.md`, `cost*/` bind the
training-only follow-up. Each completed seed retains its head, parity inputs,
Rust logs, raw pixel audit, scorecard and failed `SCREEN.json`. The first
Fortran-order NPZ refusal remains in `fit/`; the corrected export writes a
C-contiguous parity matrix, using the parity owner's declared-ID scatter.
No failed output is relabeled complete.

Seed-4101 candidate: **201,446 bytes**, SHA-256
`85add97278bd7676ce8c06dc5383117f94866ad22bc83c128f1737f2c6c303d3`.
`source-main-fit/` preserves the exact successful trainer (SHA-256
`4ad357cc086c422d1e71d816ad6026b103a90f0ad3525cce08de6234f41367db`),
reporter and table validator; the later cost-mode code has its own source
archive. Use this snapshot for the original artifact's provenance bytes.
Load with `CorruptionHead::from_bytes` and attach via
`BakeScorer::new(&base)?.with_corruption_head(&head, None)?`; use the pinned
D base in the manifest and the `corruption-head` / `feature-regime-v2` build.
This is a reproducible research candidate, not a recommended production model.

The existing speed instrument now includes `bake_surface` and
`bake_surface_corruption`, measuring the real complete API and its declared
extraction plan. The older hand-assembled Off-pool arms are diagnostics, not
valid evidence of D+228 serving cost. Commands, binaries and raw measurements
are retained in the packet.

Timing is **inconclusive**: at one Rayon thread, CPU affinity 4, 256² and
1024² synthetic pairs, the instrument repeatedly reaches its wall budget with
only one completed round and reports gate waits/noise. Launcher exclusions do
not resolve it in this session. No marginal-overhead, speedup or latency pass
is claimed from these readings. The plan/serving evidence establishes unchanged
image extraction; it does not establish the complete candidate's speed budget.

Controls refuse an out-of-regime ID set, duplicate IDs, an unregistered fit
cost and overlapping source roles. Existing historical report fields reproduce
exactly; six cost-16/64 reports and parity inputs prove validation exclusion.
Local compilation, Clippy and script checks are recorded in `logs/`.
