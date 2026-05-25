# zensim public API review — 2026-05-24

Senior code review of the public surface of the `zensim` crate at
`/home/lilith/work/zen/zensim/zensim/`, version 0.3.0.

Target reader: a senior Rust engineer evaluating whether to depend on
this crate from a production codec / image-quality pipeline.

Scope: `lib.rs`, `profile.rs`, `metric.rs`, `codec_calibration.rs`,
`source.rs`, plus public items in `diffmap.rs`, `mapping.rs`,
`iw_pool.rs`, `streaming.rs`, `zenpixels_compat.rs`, `error.rs`.

---

## Top findings (10)

1. **`ZensimProfile` has 20 variants, 18 of them `#[doc(hidden)]`. The
   only two visible are `PreviewV0_1` and `PreviewV0_2`** — every
   variant from `PreviewV0_3` onward, including the recently-shipped
   `PreviewV0_3` (Tuner v11), is hidden in rustdoc. A consumer reading
   the docs sees an enum whose "latest" is 0.2 — but `latest()`
   actually returns `PreviewV0_3`. This is the single biggest documentation
   defect on the crate.

2. **The user explicitly wants `latest()` removed.** Doing so without
   un-hiding `PreviewV0_3` leaves users with **no constructible
   non-legacy profile**. The cleanup PR MUST un-hide at least
   `PreviewV0_3` (the canonical codec-target ship) before the
   `latest()`-removal lands; otherwise the public surface effectively
   reverts to 0.2-era profiles only.

3. **`codec_calibration::CodecCalibration` / `CalibrationAffine` are
   DEAD public API.** Grep across all sibling crates + the zensim
   workspace itself (excluding markdown docs) finds zero callers. The
   actual per-codec affine runtime lives in
   `metric::PerCodecCalibration` (internal) and is gated by bake
   metadata (`zentrain.per_codec_calibration`) + the `codec_hint:
   Option<&str>` parameter on `Zensim::compute_with_codec_hint`. The
   `codec_calibration` *module* documents a path that is no longer
   how the code works. Delete or hide the module.

4. **Three different "compute" entry points overlap confusingly:
   `compute`, `compute_with_codec_hint`, and
   `score_features_with_profile{,_and_codec}`.** Each pair exposes the
   same option (codec hint) twice. The pattern should be either (a) a
   single entry with optional hint, or (b) a builder
   (`ComputeOptions`) — not three near-identical free functions plus
   two methods. Currently the public surface mints API mass for one
   optional, rarely-passed parameter.

5. **`Zensim::compute_streaming_strips` / `_default` and
   `compute_with_ref_streaming_strips` / `_default` quadruple the
   method surface for one knob (`strip_inner` / `strip_margin`).** The
   `_default` variants are sugar that wraps the geometry-aware
   variants; `_inner=256/margin=128` is hardcoded inside the crate.
   Move strip geometry into `Zensim` state (
   `Zensim::with_strip_geometry(usize, usize)`) and collapse to two
   methods. Saves four entry points.

6. **`ZensimResult::nan()` is `pub` but `#[doc(hidden)]` AND only
   used by `zensim-validate`.** Internal tooling helper that does NOT
   belong on the public surface. Either move it to a `pub(crate)`
   helper accessed via a `#[cfg(any(feature = "training", test))]`
   submodule, or delete it from the trait surface (zensim-validate can
   construct one itself).

7. **`PrecomputedReference` exposes `width()` / `height()` but no way
   to interrogate the pyramid scales it was built with.** Consumers
   doing multi-profile mixing or strip-geometry decisions have no
   stable way to read `num_scales` back. Either expose
   `num_scales(&self)` or constrain `PrecomputedReference` to a single
   public scale count.

8. **`ZensimError` does not implement `From<UnsupportedFormat>`.** The
   `zenpixels` adapter's `try_from_slice`/`try_from_buffer` return a
   separate error type, so a consumer must `match` twice or write a
   manual conversion. Adding the `From` impl (gated by the same
   `zenpixels` feature) costs one line and saves every caller.

9. **`approx_ssim2 / approx_dssim / approx_butteraugli` are
   profile-blind.** They use hardcoded power-law coefficients
   calibrated for V0_1/V0_2 on 344k synthetic pairs — but `raw_distance`
   for `PreviewV0_3`+ is a totally different scale (post-MLP, often
   already in `[0, 100]` score units when `skip_score_mapping` is true).
   Calling `approx_ssim2()` on a `Zensim::new(ZensimProfile::latest())`
   result silently returns garbage. **Footgun.** Either bind these to
   profiles that actually produce the calibrated distance, gate by
   `params.skip_score_mapping`, or delete.

10. **`Zensim::new` does NOT set `max_pixels`.** A service feeding
    untrusted dimensions accepts arbitrary image sizes — at 4K, zensim
    allocates ~470 MB. The `with_max_pixels` builder is documented but
    its absence is silent. Make `Zensim::new` require an explicit cap
    (or default to a sane cap like 256 MP) and ship
    `Zensim::new_uncapped(profile)` for callers that genuinely want no
    limit. Resource-exhaustion is the easiest production foot-shoot
    this crate enables today.

---

## One-PR cleanup proposal (5 bullets)

If I were doing a single high-leverage cleanup PR, this is what's in it:

1. **Delete `codec_calibration` module.** `CodecCalibration`,
   `CalibrationAffine`, `CodecCalibration::PREVIEW_V0_5_TUNER`, the
   `IDENTITY` consts, `lookup`, the four tests — none of it is wired
   into the runtime. The real per-codec calibration is metadata-driven
   inside `metric.rs`. Removing 230 LOC + 4 re-exports + one module
   docstring.

2. **Delete `latest()` and `codec_target()` is the new entry
   point.** Re-export `ZensimProfile::PreviewV0_3` as a non-hidden
   variant (it's the actively-shipped Tuner v11 bake and the canonical
   codec target). Document `PreviewV0_3` as **the** general-purpose
   profile; deprecate `PreviewV0_1`/`PreviewV0_2` with a `#[deprecated]`
   note pointing at `PreviewV0_3`.

3. **Hide or delete the entire `PreviewV0_4`/`PreviewV0_5*` variant
   block from the public enum.** All 17 are `#[doc(hidden)]` and most
   carry "DO NOT USE for general ranking" caveats in their docs. Move
   the bake bytes + `ProfileParams` definitions to `pub(crate)`, keep
   `PreviewV0_3` + `PreviewV0_1` + `PreviewV0_2` as the only visible
   variants. If any specific V0_5 variant is genuinely needed by an
   external consumer (e.g. zensim-gpu wants `PreviewV0_5Tuner` for
   strict-monotonicity dial work), promote it explicitly with a
   visible doc explaining when to use it.

4. **Fix the `approx_*` footgun.** Either (a) move those methods into
   a `Linear` newtype that's only constructible from V0_1/V0_2 results,
   or (b) return `Option<f64>` and document that the methods only
   apply when `params.skip_score_mapping == false`. Right now silently
   wrong for the shipped profile.

5. **Add `Zensim::new(profile, max_pixels: usize)` as the primary
   constructor; rename current `Zensim::new` to `Zensim::new_uncapped`
   with a doc warning.** Forces resource-safety into the type
   signature. Codec callers (`zenwebp::EncodeConfig::target_zensim`)
   already know their dimensions, so the explicit cap is free for them.

---

## Findings (ranked, ~80 items)

### P0 — ship-blocking

P0-01. **`ZensimProfile::latest()` returns a `#[doc(hidden)]`
variant.** A caller reading the rustdoc cannot find documentation
for what `latest()` resolves to. `pub fn latest() -> Self {
Self::PreviewV0_3 }` but `PreviewV0_3` is hidden. Either un-hide
`PreviewV0_3` (preferred) or document the resolution target on
`latest()`. Per user directive: remove `latest()` entirely.
(`profile.rs:749`)

P0-02. **`approx_ssim2`/`approx_dssim`/`approx_butteraugli` silently
break on the shipped `PreviewV0_3` profile.** The power-law fits are
against V0_2-era raw distance distribution; V0_3's raw output is
already in [0, 100] score units. Calling `result.approx_ssim2()` on
a `latest()`-built Zensim returns garbage with no warning. Either
gate the methods or remove them. (`metric.rs:757-791`)

P0-03. **`Zensim::new(profile)` accepts arbitrary image sizes by
default.** A 16K × 16K image allocates ~7 GB of scratch in the
streaming pipeline. Default `max_pixels = None`. Library safety
posture should default to "deny large" not "deny nothing." Make
`max_pixels` mandatory on construction or default to ~256 MP.
(`metric.rs:996-1007`)

P0-04. **`codec_calibration` module is unused public API.** No
runtime callers; no sibling crate callers; no test callers. The
actual per-codec affine is metadata-driven. Document confusion.
Delete the module + re-exports (`lib.rs:235`).

P0-05. **`PreviewV0_5CrossCodec` is marked `#[deprecated]` AND
`#[doc(hidden)]`.** Two layers of "do not use" for an item that's
also `pub`. Deprecation suffices; `doc(hidden)` is redundant noise.
Drop the `doc(hidden)`. (`profile.rs:351-360`)

P0-06. **The `ProfileParams` struct is `pub` but only constructible
via `pub fn custom(...)` (gated by `feature = "training"`).** Outside
the `training` feature, the only thing a consumer can do with a
`&ProfileParams` is observe `weights` (the only `pub` field). Other
fields (`mlp_bytes`, `mlp_bytes_b3`, `ensemble_classifier_bytes`)
are `pub(crate)`. Make the struct `pub(crate)` and provide a thin
public introspection wrapper (`ProfileInfo`) that exposes only what
a downstream tool needs (name, n_features, n_scales, has_mlp). Saves
public-surface mass on a 14-field struct. (`profile.rs:1023-1183`)

P0-07. **The `ProfileParams::custom` function takes 6 positional
`f64` / `usize` / `u8` parameters with no newtypes.** Three of them
(`score_mapping_a`, `score_mapping_b`, `blur_radius`) are easily
confused. Add `Default` impl + `with_*` setters, or replace with a
builder. (`profile.rs:1192-1221`)

P0-08. **The `Zensim` struct's `compute_with_codec_hint` only
applies the hint when the loaded bake carries
`zentrain.per_codec_calibration` metadata. Most production V0_5*
bakes do not.** Silent no-op behavior is documented but easy to miss.
Either return `Result<ZensimResult, Error>` distinguishing "hint
ignored" from "hint applied," or fold the hint into builder state
where the wiring is explicit. (`metric.rs:1090-1109`)

P0-09. **`compute_streaming_strips_default` hardcodes `(256, 128)`
inside the crate.** External callers cannot read the constants back
to size their own buffers consistently. Promote to `pub const
DEFAULT_STRIP_INNER: usize = 256; pub const DEFAULT_STRIP_MARGIN:
usize = 128;` and use them in both the body and the doc.
(`metric.rs:1300-1306`, `metric.rs:1378-1384`)

P0-10. **`Zensim::new` doesn't set `parallel = false` even when
the `threads` feature is disabled.** Compilation succeeds, runtime
falls through to a single-threaded path, but the API leaks the
illusion of multi-threaded behavior. Gate `parallel` by the
`threads` feature, or document explicitly that `parallel` is a
no-op without the feature. (`metric.rs:1001-1007`)

### P1 — high-priority cleanup

P1-11. **`ZensimProfile::ensemble()` / `tuner()` / `cross_codec()`
/ `compression_v2()` / `tuner_v3()` etc. are all aliases that
return `pub` variants.** That's 13 `pub const fn` aliases each one
line — but they are NOT covered by the `non_exhaustive` discipline
applied to the enum itself: deleting `PreviewV0_5Compression`
breaks `compression()` for all callers. Either drop the aliases or
deprecate the underlying variants too. Aliases as primary surface
is fine; aliases AS WELL AS direct-variant access doubles the
test/maintenance burden. (`profile.rs:756-873`)

P1-12. **`ZensimProfile::name()` is `pub` and returns
`&'static str` like `"zensim-preview-v0.3"`.** Format is informal
and undocumented (no escape contract, no version-pinning promise,
no enum→string round-trip guarantee). Either add a `FromStr` impl
+ document the wire format, or `pub(crate)` and let
`fmt::Display` carry the public surface. (`profile.rs:939-969`)

P1-13. **`ZensimError` is `Copy`** (line 9). That's a footgun:
adding ANY variant that carries owned data (e.g.
`ModelLoadFailed { reason: String }` instead of `&'static str`)
will silently break `Copy` and downstream code. Document this
constraint or replace `Copy` with `Clone` to leave room. The
`reason: &'static str` strings in the two MLP errors are
informative for built-in failures but cannot carry contextual
data — a caller cannot tell *which* bake or *which* layer
failed. Replace with `&'static str` + a numeric error code, or
relax `Copy` and use `String`. (`error.rs:9-71`)

P1-14. **`ZensimError` doesn't implement `From<std::io::Error>`
or any other common ecosystem error.** Consumers integrating
zensim into a pipeline that reads images from disk must write
manual conversions. Cheap to add. (`error.rs`)

P1-15. **`UnsupportedFormat` is a separate error type only under
the `zenpixels` feature.** Should be folded into `ZensimError`
via a variant (`UnsupportedZenpixelsFormat(&'static str)`) so
callers don't deal with two error types. (`error.rs:74-78`)

P1-16. **`PreviewV0_5` and `PreviewV0_5Balanced` are documented
as "semantically equivalent"** with shared bake bytes via
`profile.rs:982`. That's a documented alias inside the enum
itself, which `#[non_exhaustive]` then has to silently preserve.
Either delete `PreviewV0_5` (call all callers update to
`PreviewV0_5Balanced` directly) or formally deprecate it.

P1-17. **`Zensim::compute_extended_features` returns 300 features
but documents "the score is identical to `compute()`."** That's
only true when the underlying profile's MLP doesn't read the
extra 72 features. Document precisely which profiles preserve
score parity. (`metric.rs:1133-1145`)

P1-18. **`PrecomputedReference` is `pub` but has only `width()`
/ `height()` as public methods.** Cannot be constructed publicly
(only via `Zensim::precompute_reference`). The struct is opaque.
Make it `#[derive(Debug)]` + add a `pub fn num_scales(&self)` so
callers can size their own scratch buffers consistently.
(`streaming.rs:2062-2091`)

P1-19. **`ZensimScratch` is `pub` and has only `new()` +
`Default`** but its `dst_planes` field is `pub(crate)` — so it's
opaque-but-public. Either make it opaque (delete `Default`
derive, keep `new`), or expose enough to make it useful for
external tooling. (`streaming.rs:2034-2046`)

P1-20. **`DiffmapResult` is `pub` AND `#[non_exhaustive]` but has
fields `result/diffmap/width/height` all private.** The
`#[non_exhaustive]` is redundant (no public fields), but its
presence implies a future field add. Document the stability
promise or drop the attribute. (`diffmap.rs:550-556`)

P1-21. **`DiffmapOptions` mixes pub fields with `#[non_exhaustive]`
struct semantics.** Construction via `Default` + field overrides
is the documented pattern, but `impl From<DiffmapWeighting>`
gives a back-door that bypasses other fields. Inconsistent.

P1-22. **`mapping::*_to_ssim2` etc. functions are calibrated on
V0_2 raw distances** but operate on the public `score: f64`
(0–100). For `PreviewV0_3`+ where scores are MLP-derived, the
mapping accuracy claim ("MAE 4.7 SSIM2 points") no longer holds.
Document this restriction or feature-gate by profile.
(`mapping.rs:1-17`)

P1-23. **`distance_to_score` is `pub(crate)` but `dissimilarity_to_score`
+ `score_to_dissimilarity` are `pub`.** Three near-identical
score-space transforms, one private. Pick one paradigm and
document the relationship in the module docstring.
(`metric.rs:315, 806, 814`)

P1-24. **`score_features_with_profile` takes `width: u32, height:
u32` but every other entry point takes `usize`.** Mixed unit
types in the same crate. Pick one. (`metric.rs:402-409`)

P1-25. **`score_features_with_profile_and_codec` is exposed in
`lib.rs` as `pub use metric::{...}` but is undocumented at the
re-export site.** The `pub use` line should carry a `///`
explaining when to use the `_and_codec` variant. (`lib.rs:261`)

P1-26. **`ZensimResult::nan()` is `pub fn nan()` but
`#[doc(hidden)]`.** `pub` + `doc(hidden)` is the worst of both
worlds — visible to grep / tooling, invisible to docs. Move to
a `#[cfg(any(feature = "training", test))]` block or to
`pub(crate)` accessed via a `ZensimResultBuilder` for tooling.
(`metric.rs:684-700`)

P1-27. **`ZensimResult::profile()` returns whatever profile was
*last set* via `with_profile`, NOT the profile that produced the
features.** A `nan()` result has `PreviewV0_1`, then `compute()`
overwrites with the current profile. Calling
`result.approx_ssim2()` on a NaN result returns 100 (because
`raw_distance.is_nan()` ≤ 0.0 check), looking like a perfect
match. Footgun. (`metric.rs:691-700` + `metric.rs:757-791`)

P1-28. **`RgbSlice::new` panics on length mismatch** while
`try_new` returns `Result`. The "panicking constructor + try_*
fallback" pattern is common, but for a library that's used in
codec hot loops this means a single corrupted decode can panic
the encoder. Document recommendation: use `try_new` in codec
paths. Or just delete `new` and force `try_new`. (`source.rs:167-169`)

P1-29. **`RgbaSlice::new` defaults to `AlphaMode::Straight`** but
`StridedBytes::new` defaults to `AlphaMode::Unknown`. Two
different default-alpha semantics for two near-identical
constructors. Pick one. (`source.rs:219-232, 338-339`)

P1-30. **`AlphaMode::Unknown` is silently treated as
`Straight`** (`is_straight()`). That's a footgun — a caller
constructing `StridedBytes` without thinking about alpha gets
straight compositing even when their data may be premultiplied.
Either remove `Unknown` (force explicit choice) or document
loudly. (`source.rs:90-104`)

P1-31. **`AlphaMode::Opaque` is documented as "equivalent to
RGBX/BGRX"** but the enum is `#[derive(Default)]` with `Unknown`
as default. The `Default` impl is fine; the doc just doesn't
reflect that `Opaque` is the safe-for-RGB choice. Minor.
(`source.rs:84-104`)

P1-32. **`StridedBytes::with_color_primaries` takes
`primaries: ColorPrimaries` by value** (3 enum variants, all
`Copy`) and returns `self` — fine. But the *only* way to set
primaries is through this builder; there's no `try_*` variant
that returns `Result`. Inconsistent with the `try_new` pattern.
(`source.rs:427-430`)

P1-33. **`ColorPrimaries::Bt2020`** is documented with a 2%-mid-tone
caveat in the variant doc, recommending callers "linearize
externally and use `LinearF32Rgba`." If that's the recommendation,
remove the variant — it's a misuse magnet. Or add a `Bt2020Linear`
variant that's correct. (`source.rs:25-32`)

P1-34. **`PixelFormat::has_alpha`** does `!matches!(self,
Srgb8Rgb)` instead of an explicit enum match. Adding a new
non-alpha variant (e.g. `Srgb16Rgb`) silently flips its result.
Use exhaustive match. (`source.rs:75-78`)

P1-35. **`PixelFormat::bytes_per_pixel` is `pub`** but
`PixelFormat::has_alpha` is also `pub` — neither has a
`Self::Srgb16Rgb` or `Self::Srgb8Bgr` for completeness. The
format set covers what the crate processes; document the
restriction. (`source.rs:62-78`)

P1-36. **`ImageSource: Sync` bound** but no `Send` bound. Rayon's
`par_iter` over `&dyn ImageSource` requires both. The crate uses
`Sync` for the shared-reference parallel access, but a future
caller wanting to send an owned `Box<dyn ImageSource>` to a
worker thread will fail. Add `Send + Sync` if appropriate.
(`source.rs:111`)

P1-37. **`SubsetView` is `pub(crate)`** but if zensim-gpu wanted to
do its own strip aggregation against a parent ImageSource (which
the GPU pipeline arguably already does internally), there's no
public way to wrap a sub-range. Promote to `pub` or document the
non-extension contract. (`source.rs:478-506`)

P1-38. **`FeatureView::new` returns `Option<Self>`** but every
accessor returns `Option<f64>` or `f64`. Mixed Option semantics:
`new` returning `None` means "length didn't match," while
`ssim_max(...)` returning `None` means "peaks not present" — both
are public, both `Option<f64>`. Document the distinction or use
distinct error types. (`metric.rs:3042-3066, 3152-3173`)

P1-39. **`FeatureView` has no Iterator impl** — every accessor
is hand-rolled. A `pub fn iter_at(scale, ch) -> impl Iterator<Item
= (&'static str, f64)>` would save consumers from writing 19
boilerplate calls per channel. (`metric.rs:3038-3232`)

P1-40. **`FeatureView`'s `ssim_max`/`art_max`/etc. return
`Option<f64>` but `ssim_mean`/`art_mean`/etc. return `f64`
directly.** The Option-vs-not split is by feature tier (basic vs
peaks). Document that, or unify (return Option from all and let
callers `.unwrap()` for the always-present block).
(`metric.rs:3090-3173`)

P1-41. **`CH_X / CH_Y / CH_B` are `#[cfg(feature = "training")]`
gated as 0/1/2 `pub const`s.** Three of the most useful constants
in the crate, hidden behind a training feature, named without an
enum's type safety. Promote to a `pub enum Channel { X, Y, B }`
in the unconditional surface; use the integer value via
`as usize`. (`metric.rs:3028-3036`)

P1-42. **`FEATURES_PER_CHANNEL_BASIC` (= 13) is unconditional
`pub`** but `FEATURES_PER_CHANNEL_WITH_PEAKS / EXTENDED / IW` are
`#[cfg(feature = "training")]`-gated via the `pub use` line
(`lib.rs:272-278`). The basic count is also re-exported under
`training`. Inconsistent gating. Either unconditional all four
(they're trivial integer constants) or gate all four. The same
holds for `FEATURES_PER_SCALE`. (`metric.rs:2961-3005`)

P1-43. **`WEIGHTS` is `pub const WEIGHTS: &[f64; 228]` gated by
`#[cfg(any(feature = "training", test))]`** — but the alias points
at `WEIGHTS_PREVIEW_V0_2`. The two static arrays
(`WEIGHTS_PREVIEW_V0_1` / `WEIGHTS_PREVIEW_V0_2`) are
unconditional `pub`. So a consumer who knows the canonical
`WEIGHTS_PREVIEW_V0_2` exists can bypass the training gate. Either
gate the underlying arrays too, or make `WEIGHTS` unconditional.
(`metric.rs:3319` + `profile.rs:2244-2482`)

P1-44. **`LINEAR_WEIGHTS_PREVIEW_V0_1` / `LINEAR_WEIGHTS_PREVIEW_V0_2`
are `pub use ... as ...` aliases** of the unprefixed names. Two
spellings of the same array. The "keep forever for source
compatibility" promise locks the surface forever. Add a single
`#[deprecated]` on the unprefixed names so 0.4.x can drop them.
(`profile.rs:2715-2731`)

P1-45. **`PROFILE_PREVIEW_V0_1` etc. statics are `static` not
`const` and are referenced through `profile.params()`** — a
function returning `&'static ProfileParams`. The function pointer
indirection through `mlp_bytes: Option<fn() -> &'static [u8]>`
defers the `include_bytes!` call but the bake bytes are still
embedded at compile time. The runtime indirection is purely
cosmetic. Replace `fn() -> &'static [u8]` with
`Option<&'static [u8]>` for clarity. (`profile.rs:1058-1073`)

P1-46. **`compute_iw_features` and `extended_features` flags on
`ZensimConfig` interact in non-obvious ways**: `compute_iw_features
= true` is documented as "Implies `extended_features = true` for
the standard IW layout" — but the code does not auto-set
`extended_features`. Caller must set both. Either auto-set or
return an error from `compute_*` when only `iw` is set. Currently
"don't ship `iw` without `extended` set" is a silent contract.
(`profile.rs:1098-1108`)

P1-47. **The `ZensimConfig::compute_all_features` field is
documented as "for weight training" but is `pub`**. Weight
training is gated by `#[cfg(feature = "training")]`. Inconsistent.
Either gate the field or document why training-only fields are
public. (`metric.rs:213-218`)

P1-48. **`BlurKernel::Box { passes: u8 }` AND `ZensimConfig::blur_passes:
u8` AND `ZensimConfig::blur_kernel: BlurKernel` all coexist.** The
field doc says "Overrides `blur_passes` when set" — but
`blur_kernel` is `#[allow(dead_code)] // planned: not yet wired
into blur dispatch`. A dead-but-public config field is technical
debt. Delete it until wired up. (`metric.rs:121-204`)

P1-49. **`DownscaleFilter` has 4 variants but only `Box2x2` is
implemented** — `Mitchell`, `Lanczos`, `MitchellBlur(f32)` are
all `#[cfg(feature = "zenresize")] #[allow(dead_code)]`. The
`zenresize` feature is itself commented out in Cargo.toml ("not
yet published"). So the entire variant set is unreachable.
Delete or hide. (`metric.rs:147-167`)

P1-50. **`ZensimConfig::iw_strength: f32` mirrors
`extended_masking_strength: f32`** with identical default (4.0)
and similar semantics. Pure config debt for two near-identical
knobs. Consider folding into a single `MaskingConfig {strength,
polarity: Mask|IwWeight}` struct. (`metric.rs:230-260`)

P1-51. **`compute_zensim_with_config` accepts `&[[u8; 3]]`
directly**, bypassing the `ImageSource` trait. It exists for the
training pipeline. `#[doc(hidden)]` + `#[cfg(any(feature =
"training", test))]` — but it's `pub` and re-exported. Why not
`pub(crate)`? (`metric.rs:3244-3299`)

P1-52. **`precompute_reference_with_scales` is `pub` +
`#[doc(hidden)]`** with a docstring that says "Research /
feature-extraction API; not stable." Two-mode visibility — `pub`
for `zensim-validate`, `doc(hidden)` for everyone else. Either
move to a `pub(crate)` API with `pub use` in `zensim-validate`'s
build script, or accept the public-but-unstable contract and drop
the `doc(hidden)`. (`metric.rs:484-508`)

P1-53. **`compute_zensim_with_ref_and_config` is `pub` +
`#[doc(hidden)]` + `#[cfg(feature = "training")]`** — gated three
ways. Same critique as P1-52. (`metric.rs:518-547`)

P1-54. **`PrecomputedReference::new` is `pub(crate)`** but the
docstring describes it as the construction path. Promote to `pub`
or remove the docstring's call-site language. (`streaming.rs:2096`)

P1-55. **`PrecomputedReference::from_linear_planar` is
`pub(crate)`** — the public entry point is
`Zensim::precompute_reference_linear_planar`. Confusing layering:
the `Zensim` method holds the public API but the actual
constructor is module-private. Fine if you commit to going
through `Zensim`, but document that route. (`streaming.rs:2201-2214`)

P1-56. **`Zensim::compute_with_ref_into` signature requires
`scratch: &mut ZensimScratch`** but the scratch struct is opaque;
callers cannot inspect what they're paying for in memory. Document
"holds dst planes" or expose `scratch.capacity_bytes() -> usize`.
(`metric.rs:1398-1427`)

P1-57. **`compute_with_ref_and_diffmap` takes `options: impl Into<DiffmapOptions>`**
but `compute_with_ref` does not take any options. Asymmetric.
Either both take options or neither. (`diffmap.rs:608-685`)

P1-58. **The `ZensimResult::profile()` accessor returns the
profile that scored the features**, but `ZensimResult::nan()`
fixes the profile to `PreviewV0_1`. A NaN result's profile is
a lie. Either drop the accessor or carry an `Option<ZensimProfile>`.
(`metric.rs:691-732`)

P1-59. **`ZensimResult::mean_offset()` returns `[f64; 3]`** —
positive raw indices, no axis labels. Should return a struct
`ChannelMeans { x: f64, y: f64, b: f64 }` for clarity, or
document the index order on the method signature line.
(`metric.rs:732-738`)

P1-60. **`approx_ssim2 / approx_dssim / approx_butteraugli`
clamp behavior is inconsistent**: ssim2 clamps `max(-100.0)`
floor, dssim clamps low end at 0.0, butteraugli doesn't clamp at
all. Either document the clamping policy or make them uniform.
(`metric.rs:757-791`)

### P2 — nits / consistency

P2-61. **`#[non_exhaustive]` discipline is inconsistent.**
`ZensimProfile` is non-exhaustive (good, lots of variants).
`ZensimConfig` is non-exhaustive. `ProfileParams` is non-exhaustive.
But `Zensim` (the primary API struct) is NOT non-exhaustive — adding
a new `Zensim` field is a breaking change. Apply consistently.
(`metric.rs:986`)

P2-62. **`ZensimError` is `#[non_exhaustive]` (good).** But
`UnsupportedFormat` (gated by `zenpixels`) is not. Apply both.
(`error.rs:9, 78`)

P2-63. **The `#[doc(hidden)] pub` pattern appears 6+ times** on
`PreviewV0_*` variants, `ZensimResult::nan`,
`compute_zensim_with_config`, `compute_zensim_with_ref_and_config`,
`precompute_reference_with_scales`. Each one is a debt note —
either commit to the public surface or move to `pub(crate)`.

P2-64. **`Zensim` impls are split into 3 blocks**: unconditional,
`#[cfg(feature = "classification")]`, `#[cfg(feature = "training")]`.
The third block adds `compute_with_params` + `compute_all_features`.
The structure is fine but the split is not documented at the
top of `metric.rs`. Add a module-level overview.

P2-65. **`Zensim::with_parallel` returns `Self` (consuming builder)
but `with_max_pixels` also returns `Self`.** Standard pattern, but
the methods are not annotated `#[must_use]`. Ignored return values
would silently leak the configuration. (`metric.rs:1011-1037`)

P2-66. **`Zensim::profile()` / `parallel()` / `max_pixels()` are
named without a `get_` prefix (good, follows Rust convention) but
`width()` / `height()` on `PrecomputedReference` and `DiffmapResult`
follow the same — and `score()` / `raw_distance()` / `features()`
on `ZensimResult` also follow.** This is fine; just confirming
internal consistency. Note: `into_features()` is the consuming
variant, good.

P2-67. **`DiffmapResult::into_parts(self)` returns
`(ZensimResult, Vec<f32>, usize, usize)` — a 4-tuple.** Position
1 and 2 are obvious, positions 3 and 4 are width/height. Should
return a struct `DiffmapParts { result, diffmap, width, height }`
for self-documenting destructure. (`diffmap.rs:577-579`)

P2-68. **`ZensimResult` is `#[derive(Debug, Clone)]`** but the
underlying features Vec can be large (228+ f64s). A `Clone` of a
ZensimResult allocates 1.8 KB. Document the clone cost or
provide a lightweight `ZensimResultRef<'_>` view. Minor.

P2-69. **`DiffmapResult` is NOT `Clone`.** Wrapping a large
diffmap (Width × Height f32s) in a non-Clone-able struct is
defensible; just document. (`diffmap.rs:550`)

P2-70. **Doctest in `lib.rs:9-18` uses `Zensim::new(ZensimProfile::latest())`
but the example then uses 8×8 images with all-zero pixels.**
The result is a valid `ZensimResult` with score=100 from the
identical-image short-circuit, but the doctest passes silently
because the only assertion is `println!`. Add a `.unwrap()` or
`assert!(result.score() > 0.0)`. Currently the doctest exercises
the short-circuit path, not the actual compute path. (`lib.rs:9-18`)

P2-71. **Same doctest applies to the encoder closed-loop example
(`lib.rs:46-78`)** — the per-window compute is exercised but with
identical zero pixels, so it always hits the short-circuit. Fix
both. (`lib.rs:46-78`)

P2-72. **`pub use diffmap::{DiffmapOptions, DiffmapResult,
DiffmapWeighting};` and `pub use streaming::{PrecomputedReference,
ZensimScratch};` are at the bottom of `lib.rs`** in separate `pub
use` blocks. Group by "primary API" / "diffmap" / "streaming" with
section headers like the existing ones at line 229.

P2-73. **`pub use error::ZensimError;` is at line 230** but the
`ZensimError` docs reference `Zensim::with_max_pixels`. Forward
references between modules are fine, but the doc reads
"`[`Zensim::with_max_pixels`]`" with the full crate-relative path.
Should resolve cleanly; verify in `cargo doc`. (`error.rs:31`)

P2-74. **The `mlp` module is `pub(crate)` (good)** but its name
appears in zero rustdocs. The runtime crate is `zenpredict`. If
zensim ever exposes a way to load custom bakes (suggested in the
lib.rs comment at line 207), it should call the runtime by its
crate name, not `mlp`. (`lib.rs:212`)

P2-75. **`cvvdp_features.rs` / `xyb_lms_features.rs`** are
`pub mod` under `#[cfg(feature = "training")]` (`lib.rs:225-227`).
They're called "EX-4 extended feature modules" — research output
that hasn't been promoted. Document the lifecycle: when does
research-only code get demoted to `pub(crate)` or deleted?

P2-76. **`NUM_SCALES = 4` is `pub(crate)`** at the bottom of
`lib.rs`. It's referenced by `ZensimConfig::num_scales`'s default.
Promote to `pub const DEFAULT_NUM_SCALES: usize = 4` so users
can reference it. (`lib.rs:305`)

P2-77. **`Zensim::compute_all_features` (training-gated) returns
the same `ZensimResult` type as `compute()`** but the result's
feature vector length is different (300/372 instead of 228).
Type-system can't distinguish. Acceptable for a research API;
document the contract. (`metric.rs:1473-1487`)

P2-78. **`Zensim` is `#[derive(Clone, Debug)]`** — Clone makes
sense (the struct is 24 bytes), Debug shows fields. But the
private `parallel`/`max_pixels` fields will print to Debug
without explanation. Implement `Debug` manually if you want
to control the output.

P2-79. **The crate `repository` field in Cargo.toml points to
`https://github.com/imazen/zensim`** but the workspace path is
`/home/lilith/work/zen/zensim/`. Confirm the repo URL is right
before publishing.

P2-80. **`Cargo.toml` defaults to `["avx512", "imgref", "threads"]`** —
that's reasonable for desktop targets but unfriendly for embedded
or `no_std`-curious consumers. Document the feature matrix at the
top of `lib.rs`. The README probably has this; if so, link from
`lib.rs`. (`lib.rs:1-194`)

P2-81. **`zenpredict` dep is a workspace dep** with no version
floor in the public Cargo.toml. The CLAUDE.md context says
"crates.io 0.1.0 is v2-only; v3 unpublished on main since commit
6b552a5." So zensim ships against a path/git ref of zenpredict
that doesn't exist on crates.io. **This makes the crate
unpublishable as-is.** Either publish zenpredict 0.2 first or
vendor the runtime. (`Cargo.toml:51-53`)

P2-82. **`Cargo.toml` excludes `tests/`** from the published
package. That's standard, but it means `cargo test` from a
`cargo install zensim --tests` user finds nothing. Document or
move tests to `examples/`. Minor.

P2-83. **`zenpixels-compat` is `mod zenpixels_compat;`** but
the module name has an underscore while the feature is named
`zenpixels`. Consistent within the crate but the file naming
convention is awkward (mixed dash/underscore). Cosmetic.

---

## Severity legend

- **P0**: ship-blocking — semantic correctness issue, footgun, or
  consumer-facing visible inconsistency.
- **P1**: high-priority cleanup — public surface bloat, error type
  quality, missing documentation, dead public API, version-stability
  smells.
- **P2**: nits / consistency — naming, attribute discipline, doc
  formatting.

---

## Cross-cutting recommendations

1. **Apply `#[non_exhaustive]` uniformly** to every public enum +
   struct that may grow fields. `Zensim`, `UnsupportedFormat` are
   the holdouts.

2. **Pick one error story.** Either `ZensimError` is the One Error
   (fold `UnsupportedFormat` in), or document a clear taxonomy. The
   current split is accidental, not designed.

3. **Pick one "score helper" convention.** `score_to_dissimilarity`,
   `dissimilarity_to_score`, `distance_to_score` (private),
   `score_features_with_profile`, `score_features_with_profile_and_codec`,
   plus the `approx_*` methods on `ZensimResult`. Three different
   surface patterns for "convert these numbers." Consolidate.

4. **Promote `PreviewV0_3` as the visible canonical profile** —
   currently the actual ship is `#[doc(hidden)]`. A senior Rust
   engineer reading the docs sees V0_1/V0_2 and concludes the crate
   is stuck in 2026-Q1. The user's directive (delete `latest()`)
   only makes sense if a non-hidden modern profile exists for
   callers to name.

5. **Hide everything labelled "research" / "training" / "experimental"
   behind the `training` feature OR `pub(crate)`.** The current mix
   of `pub` + `#[doc(hidden)]` + `#[cfg(feature = "training")]`
   creates uncertainty about every item's real status. The grep
   pattern `^pub fn` in `metric.rs` finds 27 items, of which 8 are
   `#[doc(hidden)]`. That's confusing surface area.

6. **Document the codec-sweep contract.** The user noted "don't
   remove the per-profile machinery that benchmark sweeps depend
   on." That machinery is `ZensimConfig::custom`,
   `Zensim::compute_with_params`, `WEIGHTS`,
   `FEATURES_PER_CHANNEL_*`, `ZensimResult::features()`. These are
   the sweep API. Promote them OR document the gate clearly in a
   module docstring at `metric.rs:1` so reviewers understand
   why they exist on the public surface.

7. **Defer `codec_calibration` cleanup behind a `#[deprecated]`
   stub** rather than immediate deletion — in case external callers
   exist that grep doesn't find (private repos, vendored copies).
   One release of `#[deprecated]`, then delete.

---

## What NOT to delete (codec + sweep accounting)

Per the user's note: "don't remove things codecs actually need;
don't remove the per-profile machinery that benchmark sweeps
depend on." Specifically verified in this review:

- **`Zensim::compute_with_ref_and_diffmap`** — used by
  `zenwebp::EncodeConfig::target_zensim` (the codec auto-targeting
  reference implementation). Keep.
- **`Zensim::compute_with_ref` / `precompute_reference` /
  `compute_with_ref_into` / `ZensimScratch`** — used by zenwebp
  for encoder iteration loops. Keep.
- **`RgbSlice` / `RgbaSlice` / `DiffmapWeighting::Trained` /
  `DiffmapOptions`** — codec-side entry points. Keep.
- **`score_features_with_profile_and_codec`** — used by
  `zensim-gpu::opaque` for CPU-bit-equivalent rescoring of GPU
  features. Keep.
- **`ZensimProfile::PreviewV0_2` and `WEIGHTS_PREVIEW_V0_2`** —
  used by zensim-gpu (`ZensimProfile::latest()` = V0_2 at the
  zensim-gpu reference version). Keep; might want to bump
  zensim-gpu to track V0_3.
- **`ZensimConfig` + `Zensim::compute_with_params` +
  `compute_zensim_with_config`** — used by `zensim-regress`'s
  training path. Keep behind `feature = "training"`.
- **`FEATURES_PER_CHANNEL_BASIC` + `FeatureView`** — used by
  `zensim-validate` for parquet emission. Keep.

What IS safe to delete (no grep hits, no doc references in active
docs):

- `codec_calibration::*` (no runtime callers, replaced by metadata
  mechanism)
- `ZensimResult::nan()` (only used by zensim-validate; build a
  local helper there)
- `LINEAR_WEIGHTS_PREVIEW_V0_1 / LINEAR_WEIGHTS_PREVIEW_V0_2`
  aliases (kept "forever" per docstring; replace with deprecation
  and delete in 0.4)
- `BlurKernel::Box` variant's `blur_kernel` field on `ZensimConfig`
  (dead-code'd in the dispatch path)
- `DownscaleFilter::{Mitchell, Lanczos, MitchellBlur}` variants
  (gated by an unpublishable feature)

---

## Summary

The zensim public surface is **functional but bloated**. The crate
has accumulated 18 hidden profile variants, two separate
error-conversion mechanisms (none of which `From`-converts cleanly),
five "compute" entry points where two would suffice, and 230 LOC of
unused `codec_calibration` machinery whose role has been silently
taken over by bake metadata.

The single highest-leverage cleanup is **promoting `PreviewV0_3` to
visible status AND deleting `latest()`**. That single change unblocks
a sensible deprecation story for V0_1/V0_2, makes the public enum
match the actual shipping bake, and gives the user-typed
`ZensimProfile::PreviewV0_3` parity with the doc-hidden internal
selection logic.

After that, the next most useful PR is folding the codec_hint into
either a builder or a single `compute(source, distorted, opts)` entry
— the current `compute` + `compute_with_codec_hint` split adds API
mass for one rarely-passed parameter and is mirrored unnecessarily
in `score_features_with_profile{,_and_codec}`.

For a consumer evaluating "should I depend on this crate?": the
**stability story is murky**. `#[non_exhaustive]` is applied
inconsistently. 18 of 20 profile variants are hidden but `pub`. The
"approx_ssim2" footgun on V0_3+ is real and undocumented. The
`zenpredict` dep is at a path/git ref that's not on crates.io,
making downstream publishability of a crate that depends on zensim
fragile. **Recommend deferring adoption until 0.4 ships with the
cleanups in this review applied.**
