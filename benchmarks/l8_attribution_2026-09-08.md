# L8 attribution for the cheaper complete candidate

Registered after `88a7abab`, before implementation or new image results.
The fixed F_nonneg32 three-seed + D composition currently omits 72 peak
terms. Its 36 L8 terms use the same removal-consistent moment linearization
as the existing L2/L4 maps. The other 36 terms are hard maxima: two tied
maxima in separate blocks make an exact additive removal map impossible
(each separate removal changes nothing; removing their union changes max).
Keep max IDs explicitly unsupported. Do not call a max subgradient an exact
finite-removal integrand, and do not enable native steering on partial maps.

Extend only the candidate route in `zensim/src/attribution.rs`. Reuse retained
f32 SSIM/error signals and canonical eighth-power arithmetic, with f64
coefficient `-s/(8*N*f^7)` to avoid f32 coefficient overflow near zero.
Retain SSIM window spreading, residual pixel allocation and sum-preserving
scale/bin folds. No scalar extraction or packed model changes. Legacy
caller-supplied-gradient APIs retain their documented basic/v2 coverage.

The public behavior delta is L8 density and corresponding coverage reporting
from `BakeScorer::compute_with_ref_and_attribution`; no signatures or public
symbols change. The concrete caller is the existing complete ensemble audit,
followed by native JXL only after complete spatial qualification.

Preregistered checks: all 36 terms reconstruct canonical `-s*f/8` within
2e-5 relative tolerance (floor 1e-12), every runtime SIMD tier, exact scalar
and consumed-feature parity, finite-moment derivative and concavity bounds,
zero/near-zero/concentrated signals, odd/tiny geometry, bin/session reuse,
legacy APIs, and existing HDR/serving tests. Replay the same 264 training
pairs for blend/F mean/D endpoint; D maps must remain exact, F/blend must
report precisely 36 active max IDs. These checks do not establish finite
pixel-edit accuracy, human ranking, spatial RD or model qualification.

Full preregistration, immutable inputs and subsequent evidence:
`/mnt/v/output/zensim/l8-attribution-2026-09-08/`.

## Result

All 36 L8 terms now contribute through the Rust candidate surface. Each
reconstructs `-s*f/8` within the registered tolerance on ten runtime SIMD
configurations, with exact scalar/feature parity. Zero/near-zero signal,
finite-moment derivative/curvature, padded/tiny geometry, aligned bin queries
and cross-model session reuse pass. Legacy basic/full/stale behavior remains
unchanged. These are moment-removal checks; actual finite pixel edits also
alter neighboring signals and still require intervention qualification.

The immutable 264-pair training replay runs blend, equal F mean and D endpoint:
792 complete pixel/cache/spatial audits. All CSV feature tables and all scalar,
pixel identity, model identity and work-count fields match exactly. All 264 D
maps and twelve identities per arm remain identical. Every nonidentity F/blend
map changes, removing exactly its 36 L8 IDs from the old unsupported list.
The remaining max IDs number 36 on 241 rows and 35 on eleven rows per F/blend
arm; f168 already had zero local sensitivity on those eleven old records.
The union is 36. All 252 nonidentity maps still have incomplete coverage.

Retained test corrections, with no relaxed numerical or release bars:

- The initial near-zero fixture's sensitivity 16 produced a coefficient below
  f32::MAX. Sensitivity 256 actually exercises overflow in an unsafe f32
  coefficient implementation; the f64 path remains finite.
- Moving process-wide SIMD dispatch inside parallel library tests caused five
  unrelated bit-parity failures. The permutation probes now run through the
  public candidate surface in the existing isolated SIMD test executable.
- The first replay verifier incorrectly assumed all 72 old peak sensitivities
  were nonzero on every row. Immutable old records disprove that assumption.
  The corrected verifier requires the exact old per-row list minus L8 IDs.

Verification: 461 core/golden/fold/SIMD/allocation/input tests plus 16 candidate
serving tests pass (six ignored), CI-exact Clippy and 605-script lint pass,
API snapshots pass unchanged, and basic-only / candidate-without-threads
feature builds pass. No latency, RSS, HDR spatial or model qualification is
claimed; no new encodes, training, human labels or holdouts were evaluated.

## Next: exact max removal needs a separate rectangle query

An independent mathematical reference establishes a compact route: after
zeroing a signal rectangle, its remaining maximum is the maximum of four
full-image strips outside that rectangle. Row and column prefix/suffix maxima
answer this in O(1) per term with O(width+height) retained storage. Ties and
second maxima are handled exactly. All 7,365 exhaustive rectangles across
25 zero/tied/distinct/concentrated planes match explicit brute-force removal.
This is a reference experiment, not shipped Rust support or an additive map.

Implement it in the existing attribution owner and bind a separate finite
rectangle estimate to the complete candidate. The recovered JXL caller
already queries variable-size 8-aligned tile rectangles, so a fixed-bin-only
max approximation would not meet its contract. Preregister the exact public
delta and validate source-to-coarse coordinates, reflected/padded geometry,
empty/full rectangles and SSIM neighborhoods before clearing max coverage.
Then run actual finite pixel/codec interventions and independent spatial RD.
Keep root-curvature and model/gate discontinuity limitations explicit.
