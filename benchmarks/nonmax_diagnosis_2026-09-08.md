# SSIM attribution diagnosis after finite max queries

Registered after `c1e41f77`, with no model, feature, scalar or map arithmetic
change. The prior saved-data oracle localized all 15 spatial failures to
non-max terms. This continuation decomposes the existing coherence owner
into SSIM, edge, MSE, HF, L8 and max contributions. It saves per-feature
linear changes so future subdivisions need no repeated pixel interventions.

Use all 15 failed training cells and eight matched controls: all eight FIT
origins at distance 1.6168174743652344, block size 32. No CAL, validation or
terminal reads, fitting or new encoding. The comparison uses the unchanged
four-member F3/D blend and existing canonical decoded inputs. Every emitted
score and original report field must remain exact. Family reconstruction
has a registered absolute 1e-6 plus relative 1e-5 tolerance; report actual
residuals. Compare each one-family oracle substitution with the actual
complete Rust-surface score change, without treating an oracle as deployable.

## Family result

The 23 cells perform 6,812 actual pixel interventions, 6,835 candidate pixel
comparisons, 23 complete candidate maps and 92 additional family density
maps. Every old report field is exact. Observed/predicted family sums agree
with their totals to maximum 5.56e-16 / 7.89e-15 absolute error.

| Replaced family | Original spatial failures resolved, of 15 |
|---|---:|
| SSIM | 15 |
| Edge | 2 |
| MSE | 0 |
| HF | 0 |
| L8 | 0 |
| Max | 0 |

SSIM-only oracle correlations on the failed cells range 0.81889–0.98310.
The complete spatial release screen stays **FAIL**. Oracle substitutions
are diagnostics and are never runtime inputs. The artifact retains all
per-cell results, including control regressions under some substitutions.

The initial run stopped after one control because diagnostic eligibility
mistook the 372-slot identity extent for the model's active feature set.
Eligibility now requires exactly zero sensitivities above f227; the current
candidate consumes f0..227. The failed first attempt remains in the artifact.
Six malformed/existing-output refusal controls pass.

## Precision investigation

The largest SSIM contribution errors include negative score changes after
replacing a block with reference pixels. Register a direct f64 SSIM window
reference on the exact retained f32 XYB pyramid, using the existing ignored
`dump_ssim_moment_explosion` instrument's new precision mode. Select the
largest SSIM-error block in row73-b8, row220-b8, row223-b8, plus photographic
control row34-b32, before precision results. This isolates moment arithmetic
from pixel conversion and pyramid rounding.

The independent reference uses centered moments and
`d = mean(error)^2 + (1-mean(error)^2)*var(error)/(var(ref)+var(dist)+C2)`.
Require agreement with independently evaluated raw-moment f64 algebra within
1e-9. Require exact zero change outside the changed pyramid samples' box
support. Reconstruct original and intervention SSIM feature contributions
against the Rust-surface report. JSON parsing moved one initial feature by
one ULP; its reconstruction check now uses absolute 1e-12, matching the
weighted-delta check. This is not feature-era admission or relaxed product
qualification. Preserve that initial failure separately.

Artifact: `/mnt/v/output/zensim/nonmax-diagnosis-2026-09-08/`.

## Precision result: repair the extractor before tuning allocation

The reference confirms nonlocal numerical effects in all four selected
cases. The production walk reproduces the saved SSIM features and weighted
intervention changes within the JSON reconstruction tolerance. Base-plane
f64 algebra agrees within 5.50e-12 (both phases pass the registered 1e-9
check); its signals remain exactly unchanged
outside the mathematical box support of altered pyramid samples.

| Case | Production SSIM linear gain | f64-reference SSIM linear gain | Changed production signals outside support |
|---|---:|---:|---:|
| Photo row34-b32 | -0.0556935 | -0.0594308 | 41,579 |
| Document row73-b8 | +0.0978643 | +0.0007933 | 6,972 |
| Screen row220-b8 | -0.0938128 | +0.0001203 | 4,540 |
| Screen row223-b8 | +0.1080043 | +0.0577065 | 1,893 |

Counts use exact nonzero comparisons, including small rounding effects; they
are not counts of perceptually material changes. Maximum out-of-support
signal changes reach 0.000904–0.001060. Maximum base-signal discrepancy versus
the direct f64 reference is 0.001012–0.003247. The table uses the original
model sensitivities as a diagnostic ruler; these are **not scores served by
a new model**, and no corrected feature vector is passed off as old-era data.
The photograph's negative SSIM contribution also shows that not every negative
component is numerical: mixed-sign learned sensitivities remain relevant.

This changes the immediate action. First repair SSIM moment precision and
locality in the canonical extraction owner, with an explicit feature-era
contract. The current raw second moments and covariance subtract nearly
equal large values, and running f32 sums carry perturbation drift beyond
local support. Changing only the final dissimilarity subtraction cannot
recover precision already lost upstream. Evaluate a stable error-moment
form against this reference, preserve all legacy-era routes, and measure
complete extraction cost. Re-extract affected training/evaluation features
before candidate fitting or qualification; do not mix eras or relabel old
bakes. Then repeat spatial checks with the corrected served candidate.

The July C2b record's broad claim that residual error is an unavoidable
finite-removal floor does not explain these current SSIM failures. Its
negative allocation experiments remain valid evidence for their tested
variants; they do not justify fitting a spatial map to numerical drift.
Corruption, native codec RD/targeting, HDR and full release qualification
remain incomplete. No default model or product arithmetic changes here.

Verification: 432 library tests pass (six ignored), plus the explicitly run
precision instrument, 23 family-analysis replays and six refusal controls.
CI-exact Clippy, scoped Rust formatting and the 605-script lint pass. No
public API, production code path or model bytes change. The only core-source
addition is inside the existing test module; full-product qualification is
not inferred from these checks.
