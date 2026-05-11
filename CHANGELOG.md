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
- `ZensimProfile::PreviewV0_4` — MLP-scored profile, behind the new
  `__experimental_versions` cargo feature (off by default; not part of
  the crates.io-published surface). Ships the 2026-04-30 trained
  228 → 64 LeakyReLU → 1 network (`zensim/weights/v0_4_2026-04-30.bin`,
  60 KB ZNPR v2) trained with synthetic + KADID_train + TID_train
  mixed supervision and validated on held-out KADID_val (SROCC=0.9417),
  TID_val (0.9414), CID22 (0.8928). Outputs raw distance (0..90 range)
  using the classic `100 - 18·d^0.7` score mapping shared with V0_1 /
  V0_2.
- `__experimental_versions` cargo feature — gates V0_4's profile,
  the `mlp` dispatch module, the `zenpredict` runtime dependency, and
  the bundled trained-weight `.bin`. The `weights/` directory is
  excluded from `cargo publish` artifacts (`package.exclude`), so
  default builds drop the AGPL-licensed `zenpredict` runtime entirely
  and remain MIT/Apache-2.0.
- `benchmarks/pareto_2026-05-11.md` — comprehensive Pareto-frontier
  summary from the 2026-05-11 training cycle. Documents post-bake
  binary eval numbers (`dataset_metric_baseline` full 4292-pair
  CID22): V0_4 lands at **KADID 0.8432 / TID 0.8401 / CID22 0.8893 /
  non-mono 4.57%**, distinct from the training-time held-out val
  SROCC numbers reported above. Per-band CID22 reveals V0_5 wins
  B0+B1 narrowly; KonJND-aligned recipes win B2 (q65-90) and B3
  (visually-lossless, by 2.8×). No bake in the recipe space
  dual-clears CID22 > 0.8934 and non-mono < 4.86%. Plots at
  `/mnt/v/output/zensim/cycle_2026-05-11/`; script archive at
  `benchmarks/make_cid22_*_2026-05-11.py`.

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

### Added (zensim, unreleased) — 2026-05-11 audit + parity cycle
- `zensim-validate/src/bin/check_holdout_overlap.rs` — stage-1
  dHash-64 perceptual overlap detector. Catches resize/exact-image
  leaks of CID22 holdout refs into the training corpus. Found 1
  strict (d≤8) + 66 relaxed (d≤16) hits on the safe-synthetic 218k
  CSV; 22 of 49 holdout refs were affected (`8d83f43e`,
  `fcc48941`).
- `zensim-validate/src/bin/check_holdout_overlap_stage2.rs` —
  stage-2 sliding-window cropped-variant detector. Found 425
  d≤10/window≥128 hits (25,674 training pairs / 11.77 %), with
  strongest matches at d=2 (effectively-identical crops of CID22
  ref `2887497.png`) (`0f019f99`, `dd4e9885`).
- `scripts/v_next/regen_tv_pairs.py` — rebuilds TV pairs file
  for the Rust trainer after a CSV is filtered. Used to produce
  the cleaned 216,151-pair TV file for V0_6 (`9faadca8`).
- `zensim-train-core` — new workspace member, WASM-compatible
  pure-Rust trainer core. Phase 1 of the WASM/CubeCL trainer plan
  (`docs/WASM_CUBECL_TRAINER_PLAN.md`). 15 unit tests, bit-exact
  ports of `SplitMix64`, `AdamState`, `pearson` / `ranks` /
  `spearman`, MLP `forward` / `backprop_step` / `predict_group`,
  `compute_scaler_from_groups`, `bake_two_layer_znpr_v2`,
  `TrainingGroup<'a>`, `TvRegularizer`, `MlpHyperparams`.
  (`49832a68`, `b1d190bf`, `ca7159e4`, `6db42725`, `dce062bf`)
- `docs/PARITY_AND_METHODOLOGY_PLAN_2026-05-11.md` — 6-goal
  parity-and-methodology plan covering trainer parity (Goal 1),
  paper page-by-page methodology (Goal 2), SSIM2 reproduction
  (Goal 3), balanced synth holdout (Goal 4), holdout-overlap
  detection (Goal 5, shipped), and an interactive GH Pages site
  (Goal 6, scaffolded) (`78392387`, `f7182c43`).
- `docs/CID22_PAPER_PAGE_BY_PAGE_2026-05-11.md` — 30-page-by-page
  methodology checklist (Goal 2, complete). Extracts Tables 3,
  4, 5, 6, 7 verbatim as Goal 3 reproduction targets. Confirms
  zensim's per-band cutoffs (50/65/90) match the paper's
  canonical scale (`24cbebec`, `23f3d4c4`, `3d513707`,
  `2797bbb4`, `1ba6bc20`, `d574979a`).
- `benchmarks/holdout_overlap_audit_2026-05-11.md` — full audit
  report with remediation plan (3 user-authorization questions).
- `benchmarks/v0_6_eval_2026-05-11.md` — V0_6 evaluation against
  KADID + TID + CID22 + KonJND. **Honest CID22 SROCC = 0.8839**
  (vs V0_5's leaked-training 0.8900, vs fast-ssim2's 0.8895).
  KonJND PJND reproduction matches paper Table 4 to 3-4 sig figs.
  (`0f8ceb8d`)
- `site/`, `scripts/v_next/build_site_data.py`,
  `.github/workflows/pages.yml` — Goal 6 GitHub Pages scaffold.
  Plotly.js-based per-band SROCC bars, per-bake comparison,
  paper Table 3 parity table. Local-preview-ready; GH Pages
  activation pending user authorization. (`0218a00b`, `aaf4cf0b`)

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
