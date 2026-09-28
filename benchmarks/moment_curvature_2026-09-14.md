# Finite moment curvature diagnosis — September 14, 2026

Registered before diagnostic execution. Reuse the twenty immutable TRAIN
block-32 reports from [basic/peak decomposition](product_peaks_2026-09-14.md):
four frozen ensembles, five source-family-admitted JXL cases, 588 actual pixel
repairs. No fitting, extraction, new pixel intervention, calibration, EVAL or
TEST access. Existing saved base features, sensitivities, per-feature weighted
changes and family decompositions are sufficient. Preserve their hashes.

Hypothesis: finite L2/L4/L8 root curvature explains the remaining basic228/H32
map failure, and correcting it is worth implementation in the retained owner.
Test all twenty reports, not just the known failed screenshot.

For a root-pooled feature F=(M)^(1/p), the observed contribution is
A=g(F'-F). Its first-order change in moment space is
B=g(F'^p-F^p)/(p F^(p-1)). Compute this stably as
A[1+(F'/F)+...+(F'/F)^(p-1)]/p. Existing per-feature weighted changes recover
F' when g is nonzero. The curvature residual is C=A-B. For each family, map
prediction D has total feature-linearization error A-D=C+(B-D). The latter
includes changed neighborhoods, coarse footprints, spatial approximations and
numerical error; do not name it exclusively blur error.

Use the existing Rust `diffmap_block_coherence --refinement-analysis` owner.
Add private diagnostic helpers and optional output under `moment_curvature`;
no library public API, feature arithmetic or served model change. Only basic
L2/L4 and peak L8 IDs receive this decomposition. Mean, HF and max contributions
remain unchanged. Zero sensitivity contributes zero; singular zero-base roots
with nonzero changes, negative reconstructed roots or nonfinite results make
the affected diagnostic incomplete rather than silently supplying zero.

Report D+C versus actual score changes, separately for SSIM roots, edge roots,
L8 and their union. Also report aggregate squared residuals before/after and
the fraction of feature-linearization squared error removed. This is an oracle:
C depends on the measured repaired image and cannot be a runtime input. It
estimates the value of removing this particular error, not the accuracy of any
unimplemented frozen-signal approximation. M2 remains unchanged and separately
limits nonlinear-head approximation.

Advance finite-root implementation only if the all-root oracle passes M3f>=.70
on all five basic228/H32 cases and lowers its aggregate squared residual versus
observed feature linearization. Otherwise preserve the negative result and
investigate the remaining spatial error before adding buffers or compute.
Do not change any product gate or promote any candidate through this diagnostic.

Validate the helper against independent raw-power algebra, full removal,
increases, tiny changes, signed sensitivities and singular/invalid inputs.
Verify all prior report fields are identical after the owner extension and the
decomposition reconstructs recorded family differences. Build/test locally,
retain full source and artifact provenance, and push reviewed evidence.

Artifacts: `~/work/zensim-validation-2026-09-14/moment-curvature/`.

## Measured result

The registered mechanism screen passes. All twenty reports have finite, complete
oracle decompositions; every prior diagnostic field is exactly unchanged. No new
image was decoded or scored. This is evidence to implement and measure a
base-image approximation, not evidence that such an implementation already works.

| Ensemble | M3f pass before / 5 | All-root oracle pass / 5 | Reduction in squared map-versus-feature error |
|---|---:|---:|---:|
| basic156_h128 | 5 | 5 | 33.7% |
| basic192l8_h128 | 5 | 5 | 55.5% |
| basic192max_h128 | 4 | 5 | 91.0% |
| basic228_h32 | 4 | 5 | 92.9% |

M2 is unchanged. The H128 heads that failed M2 still fail that separate gate;
this table measures M3f alone, not complete spatial qualification. Neither
scalar predictions nor above-identity failures have changed.

| basic228/H32 TRAIN row | Existing M3f | L8-only curvature oracle | All-root curvature oracle |
|---|---:|---:|---:|
| 396 | 0.8247 | 0.9374 | 0.9406 |
| 2436 | 0.8754 | 0.9341 | 0.9702 |
| 4597 | 0.9167 | 0.9667 | 0.9667 |
| 5316 | 0.6739 | 0.7339 | 0.9148 |
| 6577 | 0.9629 | 0.9817 | 0.9812 |

For basic228/H32, the aggregate squared error against observed feature
linearization falls from 111.4110 to 7.9474 (92.9%). The failed screenshot
case rises from .6739 to .9148 with all-root correction; L8 alone gives .7339.
The all-root oracle improves every one of its five map correlations. These
corrections use each repaired image’s observed features. They are unavailable
to the encoder before that edit, and must never be used as runtime input.

## Verification and next implementation

Two mathematical tests cover raw-power algebra, full removal, increases, tiny
changes, signed weights and singular/invalid cases. Three command-line controls
prove inconsistent feature/family records are refused, singular roots mark the
diagnostic incomplete without zero fill, and legacy records lacking root
metadata retain their previous analysis. An initial full run and a second run
with the stronger input consistency guard give identical scientific outputs.
All twenty old report dictionaries remain exactly equal after removing the new
optional field. CI-exact Clippy, focused tests, formatting and script hygiene
checks pass; no public library API or inference implementation changed.

The next implementation must derive finite moment removals from the base image
alone, through the retained attribution owner and the complete Rust scoring
surface. Preserve existing density as a control, keep scalar/model semantics
fixed, and measure both reconstruction accuracy and prepared/query cost. Actual
edits change blur neighborhoods and coarse footprints, so a frozen-signal
finite-pool formula can still fail even though this oracle succeeds. Do not
install a costly per-feature map representation before that mechanism check.

The broader product blockers remain: human-ranking loss in the small head,
above-100 distorted scores, corruption handling, representative spatial/native
RD and bounded targeting, HDR and controlled p95. No new candidate is qualified
or evaluated on EVAL; TEST and calibration remain untouched. Full results and
input/tool hashes are in the [diagnostic summary](moment_curvature_2026-09-14.results.json).
