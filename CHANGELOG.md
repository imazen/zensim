# Changelog

## [Unreleased]

### QUEUED BREAKING CHANGES
<!-- Breaking changes that ship together in the next major (or minor for 0.x).
     Persist across patch releases. Only clear when the breaking release ships. -->
- `ZensimError` is now `#[non_exhaustive]` — pattern matching outside this
  crate must include a wildcard arm. New `ImageTooLarge` and
  `FeatureWeightsLengthMismatch` variants ride on this attribute.
- `ProfileParams` is now `#[non_exhaustive]` — external code can no longer
  construct it via struct literal. Pick one of the canonical
  `ZensimProfile::Preview*` variants instead. This unlocks future
  internal field additions (e.g. V0_4 MLP dispatch's `mlp_bytes`)
  without further breaking bumps.

These two together require a `0.2.x → 0.3.0` minor bump on next release.

### Added (zensim, unreleased)
- `ZensimProfile::PreviewV0_4` — MLP-scored profile shipping a real
  trained 228 → 64 LeakyReLU → 1 network (`zensim/weights/v0_4_2026-05-07.bin`,
  61 KB ZNPR v2). Trained 2026-05-07 on the 2.37M-row unified-corpus
  Parquet against `score_ssim2` with `MSE + 0.5·RankNet` loss; best
  epoch 44, val_srocc=0.9547, test_srocc=0.9814 on a source-disjoint
  held-out split. Final layer is flipped to "distance" semantics
  (0=identical, 100=worst) so the runtime `100 - 1·d^1` mapping with
  `(score_mapping_a, score_mapping_b) = (1.0, 1.0)` produces ssim2-scale
  output (0..100, 100=identical). The MLP runtime is internal-only —
  consumers that want to bake/load custom MLP weights should depend on
  `zenpredict` directly.

### Changed (zensim, unreleased)
- MSRV bumped to **1.93** (transitive minimum from `zenpredict` 0.1.0
  via the new V0_4 path).
- `Zensim::with_max_pixels(usize)` / `Zensim::max_pixels()` — opt-in cap on
  `width × height` per image, enforced before allocation. Default `None`
  (no cap). Use when feeding untrusted dimensions to avoid runaway allocation.
- `try_score_from_features` — `Result`-returning replacement for the
  panicking `score_from_features` (now deprecated, kept as a wrapper).
- `PrecomputedReference::width()` / `height()` — public accessors so callers
  can verify dimensions before passing distorted images to `compute_with_ref*`.
- `ZensimError` variants `ImageTooLarge` and `FeatureWeightsLengthMismatch`.
  `ZensimError` is now `#[non_exhaustive]`.

### Fixed (zensim, unreleased)
- `compute_with_ref*` (including `compute_with_ref_and_diffmap` and
  `compute_with_ref_and_diffmap_linear_planar`) now rejects distorted
  images whose dimensions differ from the precomputed reference with
  `DimensionMismatch` instead of silently producing garbage scores or
  panicking on slice out-of-range.
- `RgbSlice` / `RgbaSlice` / `StridedBytes` now use `checked_mul` /
  `checked_add` for `width × height` and stride arithmetic, returning
  `ImageTooLarge` on overflow instead of wrapping silently on 32-bit /
  wasm32 targets.
- `simd_padded_width` saturates to `usize::MAX` instead of wrapping; every
  downstream allocation site is now guarded by `checked_padded_plane_len`.

## zensim

### [0.2.8] - 2026-05-04

### Added
- `Zensim::compute_extended_features()` — public method returning the full
  300-feature extended set (basic + peaks + masked) instead of the standard
  228 set. Score is identical to `compute()` (the extra 72 masked features
  have zero weight in the standard profiles); the extra features are useful
  inputs for downstream model training without re-running the multi-scale
  stats pass. Available without the `training` feature flag.

### [0.2.7] - 2026-04-27

### Added
- `ZensimScratch` reusable scratch buffer and `Zensim::compute_with_ref_into` for zero-allocation encoder loops with a precomputed reference (`71cb95c`).

### Changed
- Color conversion now uses magetypes `cbrt_midp` instead of the scalar-bounce + 2-iteration Halley path; score values shift by at most ~1e-2 absolute / ~2e-4 relative — downstream consumers tracking exact numeric scores should rebase their expectations (`0038bc3`).
- Bump archmage/magetypes minimums to 0.9.23 and switch the blur kernel to the two-block tier-natural-width pattern (`9a9f457`, `b88911d`).
- Bump `zenpixels` and `zenpixels-convert` minimums to 0.2.10 (`6836df6`).

### Fixed
- Cross-platform golden scores rebased to track the `cbrt_midp` swap so ARM, WASM, and AVX-512 tiers stay locked (`b3f7006`).
- `images_byte_identical` short-circuit now also requires matching color primaries, alpha mode, and pixel format before short-circuiting to score=100. Previously two byte-identical buffers labeled with different `ColorPrimaries` (e.g. BT.2020 vs sRGB) were collapsed to "identical" even though their actual displayed colors differ.

### Performance
- Multi-scale diffmap upsample fused into a single power-of-two pass: `diffmap_minimal` ≈ -7.7%, score bit-identical (`c2dd26a`).
- `PrecomputedReference::new` allocates all scales up front and downscales out-of-place: precompute ≈ -65% to -70% at 1080p / 4K (`05146dc`).
- Diffmap masking loop split with hoisted `inv_count` and reciprocal-multiply: `diffmap_full` ≈ -7.5% (`34648b8`).
- Synchronous drop path for small working sets reduces streaming-mode overhead on tiny inputs (`c9cf0ca`).
- Hand-tuned f32x8 v3 path for `downscale_2x_into` (`741bc0e`).

## zensim-regress

### [0.4.0] - 2026-04-27 _(unreleased)_

Breaking release (latest published is 0.3.1). Drops the `image` crate
from the runtime dependency tree, switches the public canvas type to a
new `Bitmap` (owned, packed RGBA8) plus `BitmapRef<'a>` (borrowed
view, stride-aware) for zero-copy interop with strided pixel sources
such as `zenpixels::PixelSlice`. Also makes `MontageOptions`
`#[non_exhaustive]` so subsequent field additions are additive.

#### Added
- `Bitmap`, `BitmapRef<'a>`, `PngError`, `BitmapError` — the public canvas surface (re-exported at crate root). `Bitmap` is owned + packed; `BitmapRef<'a>` borrows external buffers with arbitrary row stride. `BitmapRef::from_borrowed_rgba8_strided` and `from_borrowed_rgba8_packed` cover both common cases; `to_owned()` compacts strided into packed. `From<&Bitmap> for BitmapRef<'_>` provides ergonomic interop.
- `Bitmap::from_rgba_slice(rgba, width, height)` — owned-copy construction from `&[u8]` (one-line replacement for callers of the deleted `*_raw` functions).
- CI `no-leakage` job running `cargo public-api -p zensim-regress` and rejecting any public surface that names `zenpixels::`, `zenresize::`, `zenpng::`, `zenblend::`, `enough::`, `imgref::`, `bytemuck::`, `image::`, or `rgb::Rgb*`. `zensim::` is intentionally allowed.
- `MontageOptions::expected_label` and `actual_label` allow overriding the
  default `"EXPECTED"` / `"ACTUAL"` panel headers — useful for A/B
  comparisons where that framing doesn't fit (e.g. `"ORIG"` / `"DEFAULT"`)
  (`c1e2c38`).
- `MontageOptions::show_spatial_heatmap` opt-out for A/B comparisons over
  lossy encodings, where every region has full-magnitude differences and
  the 3×3 heatmap strip is uniformly red (`17f55e4`).

#### Removed
- The `image` crate is no longer a runtime dependency (now `dev-dependencies` only, used by tests/examples that decode JPEG fixtures).
- `diff_image::create_comparison_montage`, `create_comparison_montage_raw`, `create_annotated_montage`, `create_annotated_montage_raw`, `format_annotation`, `format_annotation_spatial` — deprecated since 0.2.3; use `MontageOptions::render` and `AnnotationText::from_report`.
- `diff_image::generate_diff_image_raw`, `generate_structural_diff_raw`, `create_structural_montage_raw` — replace with the typed equivalent and `Bitmap::from_rgba_slice` / `BitmapRef::from_borrowed_rgba8_packed` at the call site.
- `AnnotationText::spatial` field — deprecated since 0.2.3 (computed automatically by `MontageOptions::render`).
- `pub mod arch` demoted to `pub(crate)` — no external consumers.
- `pub use tolerance::ToleranceSpec as Tolerance` alias dropped — use `RegressionTolerance` (re-exported at crate root) or `tolerance::ToleranceSpec` directly.

#### Changed
- `MontageOptions` is now `#[non_exhaustive]`. Subsequent field additions
  will be additive (no further semver breaks). Callers must switch from
  struct-literal construction to `Default::default()` + field assignment.
- MSRV bumped to 1.93 (transitive minimum from `zenresize` / `zenpng` / `zenblend`).

#### Migration

```rust
// MontageOptions — before (0.3.x):
let opts = MontageOptions { amplification: 50, ..Default::default() };

// After (0.4.0):
let mut opts = MontageOptions::default();
opts.amplification = 50;
```

| Old | New |
|---|---|
| `generate_diff_image_raw(exp, act, w, h, amp)` | `generate_diff_image(&Bitmap::from_rgba_slice(exp, w, h)?, &Bitmap::from_rgba_slice(act, w, h)?, amp)` |
| `create_comparison_montage{,_raw}(...)` | `MontageOptions::default().render(...)` |
| `create_annotated_montage{,_raw}(...)` | `MontageOptions::default().render(...)` |
| `create_structural_montage_raw(...)` | `create_structural_montage(&Bitmap::from_rgba_slice(...)?, ...)` |
| `Tolerance` (alias) | `RegressionTolerance` |
| `AnnotationText { spatial: Some(...), .. }` | drop the field — `MontageOptions::render` computes it from pixels |

Known external migrations needed:
- `~/work/zen/zenjpeg/zenjpeg/tests/bundled/visual_diff_regression.rs` — uses `create_comparison_montage_raw` and `generate_diff_image_raw`.
- `~/work/zen/zenjpeg/zenjpeg/examples/mozjpeg_parity_regress.rs` — uses the `Tolerance` alias.

<details>
<summary>Replaced earlier 0.4.0 draft (never published) — see git log for original wording.</summary>

The original `[0.4.0]` draft covered only the `MontageOptions::#[non_exhaustive]` change. It was never tagged or pushed to crates.io (latest published: 0.3.1), so the breaking changes above ride on the same 0.4.0 bump.
</details>
