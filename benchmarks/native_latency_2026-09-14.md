# Native TRAIN feature-profile latency screen

September 14, 2026. This study measures the existing complete Rust models,
including prepared score/map work. It does not change a model or qualify one.

The earlier root model benchmark explicitly used `GateConfig::disabled()` and
an older zenbench dependency. Its `unreliable=false` field was not evidence of
quiet-machine admission. Historical medians remain descriptive; the prior
model quality and spatial failures remain unchanged.

The root benchmark now shares the reviewed zenbench revision with the standalone
speed benchmark, enables the owner's strict resource gate, fixes actual call
counts to one, and saves paired rounds. A new optional SHA-bound manifest feeds
native RGB8 PNG pairs through the existing zen_io decode owner. It checks every
case's declared TRAIN-development role before any pixel read, original-byte
and decoded-pixel hashes, dimensions, and conflicts with synthetic geometry.
Canonical source-family admission is still verified by the experiment driver;
a self-declared role string alone is not split authority.

## Registered population and method

Six frozen ensembles inherit the spatial-coverage/family-control registrations.
The frozen full944/H128 plain ensemble adds a seventh scalar comparison; its
prepared steering remains explicitly unsupported. All member hashes and uniform
one-third weights are checked. No fitting, calibration, EVAL or TEST access.

For each of five content classes, choose the largest previously admitted and
decoded TRAIN pair, breaking ties by row ID. Selection uses geometry, not scores.

| Row | Content | Native size | JXL q |
|---|---|---|---:|
| 405 | Photo | 672x896 | 90 |
| 2428 | Report | 791x1024 | 5 |
| 4589 | Plot | 1024x1024 | 5 |
| 5325 | Screenshot | 216x384 | 90 |
| 6569 | AI product | 768x768 | 5 |

The existing deterministic texture at 1024² and 2048² supplements geometry
coverage. It remains synthetic; there is no native 4MP case in this packet.
One rendition per class and JXL-only bytes limit generalization.

Four runs cover native/synthetic and scalar/prepared modes on a Ryzen 9 9950X3D,
CPU 8, one Rayon worker, Rev3, a release build without target-cpu=native.
Each group requires 30 completed one-call rounds and no owner unreliability flag.
Raw durations exclude warmup and gate waits. Decode and manifest verification
happen outside measured calls. Prepared callbacks construct a fresh worker per round, prepare the reference
outside the timed call, and use bin8. The measured call computes the complete
score/map plus one half-image rectangle query. It can include first-comparison
scratch allocation; this is not established long-lived-worker steady-state cost.
They do not time a dense finite-response query sweep or prove map accuracy.

Scalar groups interleave seven models with fast-ssim2 and D under current Rev3
arithmetic. D is explicitly named `D_current_revision`: it is timing context,
not qualification against the frozen production Rev1 baseline. Prepared groups
interleave six supported models. Modes run separately: their p95 ratios match
inputs but are not within-round paired mode comparisons. The original protocol's
"paired map/scalar" wording overstates that design; a retained method note
corrects it. No paired mode speedup/significance claim is made.

Percentiles use NumPy's linear sample quantile on actual one-call raw durations.
Per-case distributions, hashes, execution metadata and original failures are
retained. Strict-gate acceptance is reported separately from the broader release
contract: raw samples and an owner flag do not establish every contention,
memory, input, quality, corruption, targeting or native-RD requirement.


## Gate defect found during execution

The first two native suites each recorded 15 noisy checks while exporting
`unreliable=false`. Inspection showed that the timing engine never propagated
`ResourceGate::is_unreliable()` into its saved result. Moreover, that threshold
is per group and permits some noisy checks: 15 checks across five groups does
not itself prove any one group exceeded its threshold. Thus a false flag alone
cannot certify every round clean, even after propagation is fixed.

The initial driver's `accepted` fields implement the registered but insufficient
flag check; they remain immutable execution records and are superseded by the
final assessment's explicit refusal of quiet qualification. All actual timings
remain descriptive. No run is repeated merely to obtain a favorable or quieter
result, and noisy rounds are not silently removed to manufacture 30 clean ones.
The owner correction records each pre-round check as clean, flagged, or disabled
and propagates the strict verdict. Its forced-failure regression is separate
from this model timing campaign.


## Measured runtime tradeoffs

All four runs completed: 420 rounds and **3,150 individual-call observations**.
They recorded 15, 15, 5 and 1 noisy checks respectively. No run qualifies as a
quiet performance gate. The following are descriptive observed percentiles;
they are not release passes or paired mode speedup estimates.

| Model | Synthetic scalar p95, 1MP / 4MP ms | Synthetic prepared p95, 1MP / 4MP ms | Native map/scalar p95 ratio range |
|---|---:|---:|---:|
| basic228_h128 | 18.23 / 81.01 | 86.10 / 338.29 | 3.53–4.69 |
| basic228_h32 | 17.72 / 80.22 | 85.02 / 336.00 | 3.16–4.59 |
| y60_h32 | 8.68 / 38.64 | 26.96 / 107.66 | 1.84–2.91 |
| local120_h128 | 16.68 / 77.03 | 55.92 / 219.66 | 2.07–3.26 |
| basic156_h128 | 18.12 / 80.19 | 60.37 / 233.93 | 2.20–3.24 |
| basic192l8_h128 | 17.74 / 80.25 | 62.61 / 246.85 | 2.61–3.42 |
| full944_h128 | 54.66 / 243.22 | unsupported | unsupported |


All ensembles have three members. The p95 ratios on native inputs match each
model and image across the separate scalar and prepared runs. Per-image raw
samples and all p50/p95/min/max values remain in the structured assessment.

The observations reinforce the measured cost problem: basic228 prepared maps
cost roughly 3.2–4.7 times scalar on these native cases. Removing peaks reduces
map work, but local120/basic156/basic192l8 still exceed three times scalar on
at least one native case. y60 has the lowest cost, but its known human-ranking
and spatial limitations are unchanged. On the synthetic 1MP case even its
observed map/scalar p95 ratio is approximately 3.11. Full944 exceeds the absolute
scalar budgets in both synthetic geometries and has no supported prepared path.
Contention prevents treating these observations as controlled gate verdicts.

Do not select a shipping model from latency alone. Read the existing
[TRAIN scalar assessment](product_train_2026-09-14.md),
[scale profiles](product_scales_2026-09-14.md),
[peak profiles](product_peaks_2026-09-14.md), and
[spatial failures](spatial_diagnosis_2026-09-14.md) alongside these costs.
Reducing actual retained-map work remains consequential; repeated coefficient
ablations with unchanged extraction do not answer this result.

## Verification and follow-through

Eight negative admission controls refused before measurement: protected
manifest role, a later protected case with earlier nonexistent pixel paths,
wrong manifest hash, wrong PNG hash, wrong decoded-pixel hash, wrong dimensions,
conflicting geometry, and missing manifest hash. All native pairs decoded through
the existing owner and matched registered RGB8 hashes at original dimensions.
An initial build failed on SHA digest formatting; the existing bytewise hex
form fixed it before any timing. Its failure log is preserved.

The gate regression uses impossible available-RAM requirements and a zero noisy
check allowance. It fails on the old engine, then passes after the correction.
The owner also records enabled-clean versus disabled/unknown round checks, keeps
old JSON readable, and preserves unreliability when aggregating multiple runs.
These controls do not require deliberately loading or heating the machine.

No production feature arithmetic, model weights, calibration, public scoring
API or native codec policy changed. Memory/RSS qualification was not rerun;
prepared reference setup is excluded from the per-call timing and requires its
own accounting. HDR, other codecs, representative 4MP images, corruption and
native rate-distortion results remain outstanding. Future timing campaigns must
preregister sufficient rounds and count only explicitly admitted observations;
this packet's noisy runs remain intact and are not recycled into a clean pass.


The corrected owner is pushed at `53941021fd20158bb52af6654df003ca6ead9caa`.
Both local benchmark roots pin it. A separate final native-input smoke runs all
seven scalar ensembles for three rounds on the admitted photo pair, verifying
21 observations with explicit pre-round status (three clean, zero flagged).
That smoke verifies integration, not qualification and not a replacement for
any original timing run. The original 3,150 observations used owner16fb8fdd;
both measured binaries and source/lock hashes are retained separately.

[Structured assessment](native_latency_2026-09-14.results.json) and the
[served packet](http://localhost:3300/zensim/reports/native-latency-2026-09-14/native_latency_2026-09-14.md)
include registrations, raw measurements, admission controls, the forced gate
failure, correction checks and execution receipts. Source and complete local
artifacts remain in `~/work/zensim-validation-2026-09-14/native-latency/`.
