# Bounded finite response: accurate, still too expensive at the codec grid

September 14, 2026. **TRAIN development only; no model qualifies.** The
32-column prototype passes the fixed accuracy matrix. Reusing scratch and
updating/restoring only affected pyramid rows preserves every prediction.
However, a finest-grid cost probe takes 3.14 seconds for basic228/H32 on one
1024² image. Reject this per-query kernel-replay approach for dense steering.

[Structured evidence](tile_response_2026-09-14.results.json) accompanies this
record. The [served packet](http://localhost:3300/zensim/reports/tile-response-2026-09-14/tile_response_2026-09-14.md)
contains both registrations, source patches, measured source, outputs, exact
score witnesses and controls. Complete local artifacts, including binaries, are
in `~/work/zensim-validation-2026-09-14/tile-response/`. Experimental source is
archived and removed from the implementation. The only production-file change
corrects a stale comment claiming that max/L8 features are discarded/deprecated.

## Registered method and evidence separation

The [exact band experiment](band_response_2026-09-14.md) established that finite
feature recomputation plus complete model inference reproduces actual repair
gains. Its full-width bands were too costly. This experiment fixes tile width
at 32 columns inside the canonical 32-row bands; it does not search tile sizes.

Horizontal blur reads the tile plus its real input halo. Only its outputs and
raw center pixels are cropped before calling the existing vertical blur/feature
kernel, preserving the original band's vertical extent and initialization.
Unchanged cells reuse cached raw sums; affected cells recompute all basic/peak
statistics, including new maxima. Existing finalizers use global scale pixel
counts. No new SSIM formula, model, feature era or public API is introduced.

Cropping the horizontal running sum changes f32 rounding, so this is explicitly
an **approximate response estimator**. For canonical features `F`, tile features
`T`, base pair `P`, and repaired pair `P_R`, the prediction is:

`F_hat(P_R) = F(P) + T(P_R) - T(P)`

The implementation evaluates `F(P) + (T(P_R) - T(P))`, preserving its operation
order. Public `BakeScorer::score_features_with_identity` evaluates the complete
frozen ensemble on that vector. This anchors the base exactly and accounts for
finite pool/head behavior, but does not claim exact repaired feature equality.
Original RGB equality controls the identity shortcut.

The registered response gates are signed rank ≥ .99, p95 absolute gain error
≤ .01 points, maximum ≤ .05, and no wrong sign when actual gain magnitude exceeds
.05. The p95 budget is 1% of the product's one-point three-shot p95 target-error
bar. These estimator gates do not change scalar precision, the exact band's
`1e-5` contract, M2 diagnostics, or product qualification. Because the observed
maximum is below .01, it also bounds p95: no separate p95 estimate is claimed.

Reuse the same 23 admitted TRAIN development pairs, six frozen ensembles and
1,626 rectangles as the preceding spatial screen. Original/decoded bytes,
pixel identities, source/family admission, model bytes and prior public-pixel
repair reports are hash verified. Predictions are produced before comparing
with saved repaired scores. No fit, calibration, EVAL, TEST or terminal data.
The original seven unavailable nomination cells remain unavailable.

## Accuracy and exact-preserving implementation work

All 138 model/image checks pass. There are 9,756 scored repairs per phase.

| Frozen ensemble | Minimum signed rank | Maximum absolute gain error |
|---|---:|---:|
| basic228/H128 | .999754 | .007198 |
| basic228/H32 | .999827 | .008803 |
| y60/H32 | .999952 | .001167 |
| local120/H128 | .999555 | .001528 |
| basic156/H128 | .999555 | .001471 |
| basic192l8/H128 | .999931 | .002926 |

Every base score matches its full public pixel score exactly. No material
wrong-sign prediction occurs. These are gains under ideal reference replacement,
not native encoder effects or new human-quality evidence. The original models'
M2 diagnostics and production-map failures remain unchanged.

After the fixed accuracy matrix passed, a separately recorded implementation
step reused the twelve tile scratch vectors and replaced per-query RGB copying
and full pyramid reconstruction with a reusable working pyramid. Reference XYB
values replace exactly the reflected logical coordinates in the requested
rectangle. Existing 2x downsampling recomputes affected full-width row spans;
the baseline rows are restored after each query. This is row-local work, not
an optimal rectangular downscale implementation or an interior/boundary shortcut.

All 9,756 resulting scores, 138 base scores, query geometries and recomputed-cell
counts are exactly equal to phase one. The measured binary passes 5,472 synthetic
public-feature comparisons across six geometries, including both dimensions below
64, odd dimensions, borders and band seams. Working pyramids equal full RGB-repair
pyramids at every level, and restoration equals the original pyramid. Empty and
identity cases pass. Wrong-role and wrong-revision controls refuse before pixels.

## Cost and the actual JXL caller

The initial fixed matrix finishes in 6.89 seconds; the reuse/local-pyramid matrix
finishes in 2.07 seconds. These process totals include all six heads and diagnostic
work. They are feasibility observations, not a paired performance qualification.
Complete final basic228 dense sweeps cost 3.93–17.30 scalar calls across the
registered geometries; every case remains above the three-call advancement bar.
Reference setup, RGB mismatch counting, working storage, query restoration and
diagnostic bookkeeping are retained. The shared path extracts full228 even for
narrower models. No optimized narrow-model or memory/p95 qualification is claimed.

The current JXL loop at commit `87f6aec7` defaults to bin8 and queries 8-aligned
transform rectangles. Large images in the accuracy matrix use block64, so a
separately registered cost screen evaluates all 16,384 8×8 rectangles on the
largest-area admitted pair (tie rule: lowest row ID), row 4589 at 1024², JXL q5.
This is a finest-grid workload, not a claim about this image's native transform
partition. It uses the exact same measured binary and frozen models.

For basic228/H32, the complete dense8 work takes **3.144 seconds**, versus a
single scalar control of **53.035 ms**, or **59.28×**. Components shared across
the six model evaluations are 2.858 seconds tile replay, 67.94 ms pyramid updates,
and 28.05 ms restoration; setup is 90.42 ms. H32 inference alone takes 88.86 ms;
H128 inference takes 195.78 ms. The reported per-model total subtracts other
heads' measured inference time but retains shared diagnostic overhead.

Dense8 response accuracy is **unmeasured**: the saved large-image pixel witnesses
cover block64. The cost screen performs no native encode or RD measurement.
Its timing is not p95 and cannot qualify or represent every codec partition.
It does establish that the tested per-query replay cannot advance as a cheap
dense-map implementation, even after eliminating most pyramid and allocation work.

## Next action

Archive this fixed implementation and preserve it as a bounded-response reference.
Avoid another tile-size or head-capacity sweep to disguise the work-per-query
problem. Investigate sharing moment updates across queries and computing only
changed interior/boundary contributions through existing owners. Account for
new extrema, global pooling and head cost explicitly. Check the caller's query
geometry before choosing the representation; a coarse grid cannot silently
replace arbitrary finite rectangle responses or an 8-aligned codec map.

The full scalar, corruption, attainable-target, native-RD, HDR, memory, p95 and
frozen qualification requirements remain open. No default or advertised capability
changes, and no candidate is promoted.
