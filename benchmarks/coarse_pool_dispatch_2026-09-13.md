# Coarse masked/IW dispatch: real work removed, spatial limitation retained

September 13, after the [legacy feature audit](coarse_legacy_features_2026-09-13.md)
and user review of the spatial gallery. This implements the first bounded
scale-selection comparison. It does not qualify a new model or finish the
fundamental feature overhaul.

## Registered comparison and implementation

The four box-pyramid layouts, unchanged Rev3 arithmetic and paired seeds
4004/4005/4006 were fixed before fitting or changing dispatch:

| Layout | Inputs | Actual work requested |
|---|---:|---|
| y190 | 190 | Full-resolution Y basic/peaks; coarse XYB basic/peaks |
| coarse262 | 262 | y190 plus masked/IW at ¼ and ⅛ |
| y310 | 310 | Basic/peaks and masked/IW at every scale, omitting full-resolution X/B |
| full372 | 372 | All v1 channels, scales and pool families |

`Plan` now derives a private masked/IW scale mask from declared feature IDs.
Normalization, layout widening and plan unions preserve it. Selecting either
masked or IW enables their shared activity/weighted-pool chain at that scale;
there is no fictitious saving from dropping one of the two output groups.
The streaming walk selects Full versus Peaks before executing each scale,
and emits structural zeros for omitted pool slots. Full-resolution X/B may
also be omitted for these channel-local v1 pools. Sampling contracts retain
their previous basic/peak-only admission; this does not expand that contract.

Existing public extraction still computes its requested complete family set.
Serving and research requests select work through existing feature IDs, with
no new public toggle, feature formula, SSIM kernel or model format. Family-only
feature-set shorthand is not emitted for a scale-restricted request; explicit
IDs and per-slot provenance remain authoritative. Wider v2 families retain
their existing conservative dispatch.

An instrumented test records widths at the actual weighted-pool kernel entry:
for 256×192 input, full runs at widths 256/128/64/32, coarse262 only at 64/32,
and y190 never enters it. This removes the fine activity/sigma/pool work,
not merely the corresponding coefficients. Conversion and pyramid generation
remain; no whole-pipeline 1/17-cost or allocation-free claim is made.

## Runtime measurement

Ryzen 9 9950X3D, 1024×1024 synthetic textured/color-error pair, Rev3, actual
H32 bakes through `BakeScorer::compute`. The same **original fitted bake bytes**
are used before and after. Each executable interleaves four layouts with
zenbench, 15 samples per layout, pinned to core 0 or cores 0–7. Builds,
training and tests finish before timing. The existing benchmark retains its
exclusive lock but disables the known self-detecting process-name gate;
process inspection showed no other sustained heavy workload.

Final repeated sweep, median milliseconds:

| Layout | Before ST | After ST | Before MT8 | After MT8 |
|---|---:|---:|---:|---:|
| y190 | 15.330 | 13.270 | 3.740 | 4.111 |
| coarse262 | 34.512 | 13.643 | 6.425 | 4.189 |
| y310 | 34.500 | 16.535 | 6.458 | 4.857 |
| full372 | 34.484 | 32.825 | 6.464 | 6.578 |

The useful within-build finding is that **72 additional coarse pool inputs
cost about 2.8% ST and 1.9–2.8% MT8 over y190** across the two final sweeps.
Previously coarse262 paid essentially full372 extraction cost. Fine-scale Y
weighted pools cost more, as expected from their extra full-resolution work.

The unchanged-work controls also move across executables/sweeps: y190 shifts
about 13% ST, and MT8 varies noticeably. The mechanism behind that baseline
shift is unresolved. Do not attribute the entire raw cross-build reduction
to the scale mask, or advertise it as a general production speedup. Original
and repeated medians, MADs and counts are retained in the
[compact data](data/coarse_pool_dispatch_2026-09-13.json).

Cached spatial-call timings were also recorded at the same geometry. They
include scoring and the available map work, but the weighted-pool candidates
have incomplete map coverage. Those timings cannot establish usable steering
performance or a native codec rate/distortion improvement.

## Fast fitting, real pixel checks and negative results

The existing [feature-screen recipe](coarse_pool_screen_2026-09-13.json) uses
the same admitted 264-pair T2 JXL packet, SSIM2 proxy labels, reference-level
roles and Rust trainer as the previous screen. Initial fitting/auditing took
53.53 seconds. After implementation, a fresh run including seven fixed
corruption/JXL cases per model completed in **61.09 seconds**: 12 fits,
3,168 final pixel/cache comparisons, 84 spatial checks and 4,416 actual
32-pixel block replacements, all within its 300-second deadline.

Scores match the initial run exactly on all 3,168 paired comparisons, and
the final pixel audit reports zero consumed-feature discrepancy. Bake file
hashes differ because fresh runs bind different extraction-cache/input paths
and provenance; byte-reproducible artifacts are not claimed. Timing uses the
same preserved first-run bakes on both executables, independent of this refit.

The spatial owner reports **63 UNSUPPORTED, 15 PASS and 6 FAIL** cells.
All weighted-pool candidates have unsupported terms. This confirms the
distinction already recorded in the July research: legacy v1 masked/IW
pooling lacks spatial integrands; the v2 bounded masked/IW families are
different features with existing attribution support. A partial map is never
accepted as a passing complete candidate.

Across the three seeds, scalar scores on the fixed photo corruptions are:

| Layout | Salt-and-pepper | 2× nearest-neighbor aliasing | Whole-image R/B swap |
|---|---:|---:|---:|
| y190 | 84.8–100.6 | −193.2 to −113.9 | −221.0 to −215.1 |
| coarse262 | 92.1–103.8 | −153.2 to −116.0 | −339.6 to −292.6 |
| y310 | 91.4–108.7 | −162.8 to −104.9 | −305.4 to −272.0 |
| full372 | 93.0–105.6 | −101.8 to −35.3 | −260.3 to −242.6 |

Thus coarse pools did **not** fix salt-and-pepper in this fit. These models
were trained on ordinary JXL metric labels, not corruption judgments; this
does not prove the features lack useful corruption information. Neither
proxy agreement nor spatial correlation establishes acceptable visual quality
on the ordinary JXL examples. The user's observed artifact sensitivity and
aliasing concerns remain product requirements. Whole-image R/B rejection is
already satisfied here; its spatial correlation has lower product priority.

## Validation and next feature work

The library/integration run passes 584 tests (14 ignored). Additional Rev3
tests cover exact consumed-feature parity under serial/parallel extraction,
odd and reflected-small geometry, impulses and periodic grids; public
`BakeScorer` scalar/cache/spatial score parity; declared-linear HDR parity;
and actual skipped pool execution. CI-exact workspace clippy, script lint,
formatting and the no-default-feature build pass. Spatial-input negative
controls reject changed manifests, unadmitted origins and invalid block sizes
before extraction or training. No public API changed and no protected
holdout or release operation ran.

Next extend scale planning to the **existing v2 spatializable feature
families**, resolving cross-scale gradient dependencies explicitly. Screen
coarse bounded errors/gradient/blockiness while retaining fine Y impulse and
aliasing information. Preserve canonical feature parity and test actual
retained-plane/kernel savings. Keep these v1 variants as scalar cost controls;
do not spend another broad fit sweep trying to promote unsupported maps.
The proxy packet is an iteration instrument, not the objective for choosing
fundamental perceptual features or calibrating the user's target score.

Full artifacts are in `~/work/zensim-validation-2026-09-13/coarse-pools/`:
registration, preserved before/after instruments, recipes, fits and final
screen, block JSON, audit rows, timing scripts/results and check logs. The
compact data records binary/bake/report hashes; source changes are this commit.
