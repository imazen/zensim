# Public API diff — zensim 0.2.7 (published) → HEAD (0.3.0, `feat/streaming-372-phase1`)

Generated 2026-05-24 with `cargo public-api -p zensim -sss diff 0.2.7`
(blanket / auto-trait / auto-derived impls suppressed).

- **Published**: `zensim 0.2.7` (crates.io, 109 KB, 8.9 K downloads).
- **Local HEAD**: `zensim 0.3.0`, branch `feat/streaming-372-phase1`.
- **Raw diff log**: `/tmp/zensim_pubapi_diff.log` (331 lines, full with
  blanket impls), `/tmp/zensim_pubapi_diff_clean.log` (signal-only).
- **Method**: `cargo public-api 0.50.2`, both crates built with the
  same nightly toolchain. `cargo-public-api` filters `#[doc(hidden)]`
  items by default — see §5 for the doc-hidden surface that is not
  visible to this diff but exists in `src/profile.rs`.

A consumer migrating from 0.2.7 to the upcoming 0.3.0 sees **one
removal**, **two struct-mutating attribute changes**, and **a large
additive surface** (new `codec_calibration` module, 14 new profile
constructors, 1 new enum variant, 5 new `ZensimError` variants, 8 new
`Zensim` methods, 2 new free functions, 2 new statics). The one
removal — `ZensimResult::nan()` — is semver-major; the rest of the
breakage is the `#[non_exhaustive]` attribute being added to
`ProfileParams` and `ZensimError` (also semver-major, but trivial to
fix by adding `_ => ...` arms in downstream match expressions or
switching field-literal constructors to the documented setters /
`Default::default()` builder pattern). `latest()` keeps its signature
but now returns `PreviewV0_3` (was `PreviewV0_2`) — score scale stays
approximately stable per the profile shipping policy, but consumers
that pinned to exact byte output need to be told.

## 1. Removed (SEMVER-MAJOR breaks)

| Item | Notes |
|---|---|
| `pub fn zensim::ZensimResult::nan() -> Self` | Removed. Downstream code constructing a sentinel "NaN result" must use the documented field-literal pattern or build a result via `compute(...)`. |

That's it for hard removals.

## 2. Changed signatures / attributes (SEMVER-MAJOR effects, sigs stable)

| Item | Before (0.2.7) | After (HEAD) |
|---|---|---|
| `zensim::profile::ProfileParams` | `pub struct ProfileParams` | `#[non_exhaustive] pub struct ProfileParams` |
| `zensim::ZensimError` | `pub enum ZensimError` (4 variants) | `#[non_exhaustive] pub enum ZensimError` (9 variants) |

Both attribute changes are semver-major in Rust's rules:

- `#[non_exhaustive] struct`: downstream cannot construct
  `ProfileParams { compute_iw_features: ..., extended_features: ..., ... }`
  via field-literal syntax from outside the defining crate. They must
  use `Default::default()` (or whatever builder the crate exposes)
  and then set fields. **No signature change**, but the construction
  syntax breaks.
- `#[non_exhaustive] enum`: downstream `match ZensimError::X { ... }`
  expressions without a wildcard arm fail to compile.

No function or method signature shifted (generics, args, returns,
bounds all identical between the published API surface and HEAD's).
`pub fn latest() -> Self` has the same signature but its return value
changed (`PreviewV0_2` → `PreviewV0_3`). Score scale is approximately
stable per the variant doc's shipping policy, but exact byte output
on golden tests will differ. Surface this explicitly in the changelog.

## 3. Behavior changes worth flagging (no signature change)

| Item | Before | After |
|---|---|---|
| `ZensimProfile::latest()` | `PreviewV0_2` (linear-weights, 0.9960 SROCC on concordant safe-synth) | `PreviewV0_3` (Tuner v5 MLP bake `v_tuner_v11_2026-05-24.bin`, 372→128→128, CID22 0.860 / KonJND 0.285 / AIC-3 0.776) |
| `ZensimProfile::codec_target()` | did not exist | returns `PreviewV0_3` — same as `latest()` today, but the constructor exists specifically so codec crates can opt into bake rotations without per-codec edits. |
| `ZensimProfile` Display / `name()` of new variants | n/a | new `name()` strings: `"zensim-preview-v0.3"` (and similar for the doc-hidden V0_4 / V0_5 family — see §5). |

## 4. Added (SEMVER-MINOR additions)

Public, doc-visible. Each item is an additive change consumers can
adopt immediately on bumping to 0.3.0.

### 4.1 New module: `zensim::codec_calibration`

Two new types (also re-exported at the crate root as
`zensim::CalibrationAffine` and `zensim::CodecCalibration`):

```text
pub struct CalibrationAffine { pub alpha: f32, pub beta: f32 }
    pub const IDENTITY: Self
    pub fn apply(&self, raw: f32) -> f32
    pub fn invert(&self, calibrated: f32) -> Option<f32>

pub struct CodecCalibration {
    pub jpeg: CalibrationAffine,
    pub webp: CalibrationAffine,
    pub zenpng: CalibrationAffine,
    pub avif: CalibrationAffine,
    pub zenjxl: CalibrationAffine,
}
    pub const IDENTITY: Self
    pub const PREVIEW_V0_5_TUNER: Self
    pub fn lookup(&self, codec_name: &str) -> Option<CalibrationAffine>
```

Used by `Zensim::compute_with_codec_hint(...)` (see §4.3) and by the
PreviewV0_5Tuner ship to apply per-codec post-network affine
calibration. Consumers that want to plug their own calibration in can
construct a `CodecCalibration { ... }` literal — the struct fields
are public and the type is **not** `#[non_exhaustive]`.

### 4.2 `ZensimProfile` — new variant + 11 new constructors

| Added item | Returns | Use case |
|---|---|---|
| `PreviewV0_3` (enum variant) | — | Canonical shipping profile in 0.3.x. 372-input MLP, full 0..100 dial. |
| `pub const fn balanced() -> Self` | `PreviewV0_5Balanced` (doc-hidden) | Balanced ship — Pareto-better across all 5 eval corpora. |
| `pub const fn balanced_v2() -> Self` | `PreviewV0_5BalancedV2` (doc-hidden) | Same as `balanced()` + PCHIP spline calibration (JND=60, JOD=30). |
| `pub const fn balanced_v3() -> Self` | `PreviewV0_5BalancedV3` (doc-hidden) | V10 reallocated score-space (JND=80, JOD=50). |
| `pub const fn compression() -> Self` | `PreviewV0_5Compression` (doc-hidden) | Codec-selection / quality-dial ship. Wins CID22 + AIC-3. |
| `pub const fn compression_v2() -> Self` | `PreviewV0_5CompressionV2` (doc-hidden) | Compression + V10 PCHIP calibration. |
| `pub const fn compression_v3() -> Self` | `PreviewV0_5CompressionV3` (doc-hidden) | Compression + V10 reallocated score-space. |
| `pub const fn ensemble() -> Self` | `PreviewV0_5Ensemble` (doc-hidden) | Routes between Balanced + Compression via classifier; tracks `max(...)` per corpus. |
| `pub const fn tuner() -> Self` | `PreviewV0_5Tuner` (doc-hidden) | Codec auto-targeting (monotonic dial); NOT for cross-corpus ranking. |
| `pub const fn tuner_v3() -> Self` | `PreviewV0_5TunerV3` (doc-hidden) | Tuner + V10 spline (JND=60, JOD=30, dial full [0,100]). |
| `pub const fn tuner_v4() -> Self` | `PreviewV0_5TunerV4` (doc-hidden) | Tuner + V10 reallocated score-space. |
| `pub const fn cross_codec() -> Self` | `PreviewV0_5CrossCodec` (doc-hidden) | **Deprecated 2026-05-20** (`#[deprecated(since = "0.5.0", note = "dial-broken — use compression_v2() or balanced_v2()")]`). Cross-codec equivalence loss; PCHIP falsified. |
| `pub const fn codec_target() -> Self` | `PreviewV0_3` | **Canonical stable alias for "the bake all zen codecs target."** Use this in codec crates so bake rotations flow through automatically. |

All constructors return one of the `#[doc(hidden)]` enum variants
listed in §5 except `codec_target()` (which returns the documented
`PreviewV0_3`).

### 4.3 `Zensim` — 8 new methods

```text
pub fn compute_extended_features(&self, source, distorted) -> Result<ZensimResult, ZensimError>
pub fn compute_with_codec_hint(&self, source, distorted, codec_hint: Option<&str>) -> Result<ZensimResult, ZensimError>
pub fn compute_streaming_strips(&self, source, distorted, strip_inner, strip_margin) -> Result<ZensimResult, ZensimError>
pub fn compute_streaming_strips_default(&self, source, distorted) -> Result<ZensimResult, ZensimError>
pub fn compute_with_ref_streaming_strips(&self, &PrecomputedReference, distorted, strip_inner, strip_margin) -> Result<ZensimResult, ZensimError>
pub fn compute_with_ref_streaming_strips_default(&self, &PrecomputedReference, distorted) -> Result<ZensimResult, ZensimError>
pub fn max_pixels(&self) -> Option<usize>
pub fn with_max_pixels(self, max_pixels: usize) -> Self
```

(All `source` / `distorted` are `&impl zensim::source::ImageSource`.)

The streaming variants are the headline new feature of this branch
(`feat/streaming-372-phase1`). `compute_with_codec_hint` is the entry
point for codec-specific calibration via §4.1's `CodecCalibration`.
`with_max_pixels` adds a resource cap that pairs with the new
`ZensimError::ImageTooLarge` variant.

### 4.4 `PrecomputedReference` — 2 new accessors

```text
pub fn height(&self) -> usize
pub fn width(&self) -> usize
```

Type already existed in 0.2.7; just adds dimension getters.

### 4.5 `ZensimError` — 5 new variants

```text
ZensimError::FeatureWeightsLengthMismatch
ZensimError::ImageTooLarge
ZensimError::ModelForwardFailed { reason: &'static str }
ZensimError::ModelLoadFailed { reason: &'static str }
ZensimError::UnsupportedPixelFormat
```

Existing variants (`DimensionMismatch`, `ImageTooSmall`,
`InvalidDataLength`, `InvalidStride`) are unchanged. Enum is now
`#[non_exhaustive]` (see §2).

### 4.6 New free functions

```text
pub fn score_features_with_profile(profile: ZensimProfile, features: &[f64], width: u32, height: u32) -> Result<f64, ZensimError>
pub fn score_features_with_profile_and_codec(profile: ZensimProfile, features: &[f64], width: u32, height: u32, codec_hint: Option<&str>) -> Result<f64, ZensimError>
```

Skips the feature-extraction pass — caller supplies pre-computed
features. Useful when the same image pair is scored under multiple
profiles, or for the picker-training pipelines that already cache
features in parquet sidecars.

### 4.7 New profile statics

```text
pub static zensim::profile::LINEAR_WEIGHTS_PREVIEW_V0_1: [f64; 228]
pub static zensim::profile::LINEAR_WEIGHTS_PREVIEW_V0_2: [f64; 228]
```

Renamed-for-clarity exposure of the linear weight tables for V0_1
and V0_2. The existing `WEIGHTS_PREVIEW_V0_1` / `WEIGHTS_PREVIEW_V0_2`
statics from 0.2.7 are also still present (unchanged).

### 4.8 New `ProfileParams` fields

`ProfileParams` (now `#[non_exhaustive]`) gains 5 public fields:

```text
pub compute_iw_features: bool
pub extended_features: bool
pub extrapolate_score: bool
pub skip_score_mapping: bool
pub soft_clamp_score: bool
```

Combined with the `#[non_exhaustive]` attribute, downstream code that
previously constructed `ProfileParams { ... }` directly must switch
to `ProfileParams { ..Default::default() }` or the builder pattern.

## 5. `#[doc(hidden)]` surface not in the diff

`cargo public-api` filters items marked `#[doc(hidden)]` from its
output by default. The following enum variants live on
`ZensimProfile` in HEAD but are NOT visible above — they exist as
public-but-unstable surface, intentionally hidden from rustdoc:

```text
PreviewV0_4
PreviewV0_5
PreviewV0_5Balanced
PreviewV0_5Compression
PreviewV0_5Ensemble
PreviewV0_5Tuner
PreviewV0_5CrossCodec (deprecated)
PreviewV0_5TunerV2
PreviewV0_5TunerV3
PreviewV0_5BalancedV2
PreviewV0_5CompressionV2
PreviewV0_5BalancedV3
PreviewV0_5CompressionV3
PreviewV0_5TunerV4
PreviewV0_5TunerV4Calibrated
PreviewV0_5BalancedV3Calibrated
PreviewV0_5CompressionV3Calibrated
```

These are reachable via the documented constructors (`balanced()`,
`compression()`, `tuner_v4()`, etc.). Per the project's API audit
discipline (`cleanup(api): hide/demote/delete unstable items per
audit`, commit `d92c6fa`), they ARE part of the technical public API
but are not advertised as a stable migration target. Treat additions
/ rotations within this surface as patch-level changes, not minor
bumps — that is the explicit shipping policy in `CLAUDE.md`'s
"Don't bump the crate version when rotating a trail's bake" rule.

## 6. Semver verdict

Bump is **0.2.7 → 0.3.0** (MINOR-for-0.x, which is semver-MAJOR in
the sense of breaking-change-allowed). Justification:

- `ZensimResult::nan()` removed → semver-major.
- `#[non_exhaustive]` added to `ProfileParams` and `ZensimError` →
  semver-major.
- `latest()` return value changed (signature stable) → not strictly a
  semver break (per `ZensimProfile` doc's "scores stay approximately
  stable" contract), but a perceptible behavior shift that consumers
  with golden-file tests need to be told about in the changelog.

Everything else is purely additive. A 0.3.0 release that lands on
crates.io should:

1. Run `cargo semver-checks --manifest-path zensim/Cargo.toml` and
   confirm the major-bump justifications match the three items above.
2. Land a CHANGELOG entry in the `[0.3.0]` section under `Removed`
   (the `nan()` constructor), `Changed` (`ProfileParams`/`ZensimError`
   non-exhaustive, `latest()` now returns `PreviewV0_3`), `Added`
   (everything in §4 above).
3. Pair with a methodology doc at
   `benchmarks/v_tuner_v11_methodology_2026-05-24.md` per the
   "methodology doc paired with each crate release" rule.

## 7. Reproduction

```sh
# Pure published-vs-HEAD diff (full blanket impls)
cargo public-api -p zensim diff 0.2.7 2>&1 | tee /tmp/zensim_pubapi_diff.log

# Signal-only (drops blanket / auto-trait / auto-derived impls)
cargo public-api -p zensim -sss diff 0.2.7 2>&1 \
    | tee /tmp/zensim_pubapi_diff_clean.log
```

Run from `/home/lilith/work/zen/zensim/zensim/` (workspace member
directory, not workspace root).
