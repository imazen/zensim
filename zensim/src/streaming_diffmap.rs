//! Streaming consumer for [`crate::DiffmapResult`].
//!
//! [`StreamingDiffmap`] wraps a [`PrecomputedReference`] and accepts strips of
//! distorted pixels in row order. After the final strip is pushed, [`finalize`]
//! produces a [`DiffmapResult`] numerically equivalent to
//! [`crate::Zensim::compute_with_ref_and_diffmap`] (within ≤1e-4).
//!
//! [`finalize`]: StreamingDiffmap::finalize
//!
//! # Why streaming, why not per-strip score
//!
//! The internal strip-band machinery in [`crate::streaming`] is band-parallel for
//! a single pyramid scale, but coarser scales (1/2x, 1/4x, 1/8x) require the
//! entire previous-scale plane to exist before they can be downscaled and
//! processed. Cross-scale fusion upsamples each coarser diffmap back to full
//! resolution and blends them together, so no row of the final diffmap is
//! "settled" until every scale has run.
//!
//! As a result, [`push_distorted_strip`] returns `None` for strips that aren't
//! the final one — there is no meaningful intermediate score or partial
//! diffmap that would round-trip to the full-image result. The final push
//! returns a [`StripContribution`] containing the full score and full
//! per-pixel diffmap, and [`finalize`] returns the same data wrapped in a
//! [`DiffmapResult`].
//!
//! [`push_distorted_strip`]: StreamingDiffmap::push_distorted_strip
//! [`finalize`]: StreamingDiffmap::finalize

use crate::diffmap::{
    DiffmapOptions, DiffmapResult, PixelFeatureWeights, apply_contrast_masking, sqrt_inplace,
};
use crate::metric::{ZensimConfig, ZensimResult, config_from_params};
use crate::profile::ZensimProfile;
use crate::source::{AlphaMode, ColorPrimaries, ImageSource, PixelFormat};
use crate::streaming::{
    PrecomputedReference, compute_diffmap_from_xyb, convert_linear_planar_to_xyb_into,
    convert_source_to_xyb_into,
};

/// Per-strip contribution returned by [`StreamingDiffmap::push_distorted_strip`].
///
/// A `Some(..)` result is produced only when the pushed strip completes the
/// image (`strip_y + rows == height`). Earlier strips return `None` because
/// the multi-scale fusion cannot finalize partial rows of the full diffmap.
///
/// `score_delta` is the **full** zensim score for the completed image (not an
/// incremental delta — the score is global and only meaningful once every
/// scale has been processed). `block_diffmap` is the trimmed `width × height`
/// per-pixel error map.
#[non_exhaustive]
pub struct StripContribution {
    /// Full image zensim score (final strip only).
    pub score_delta: f32,
    /// Full image diffmap, row-major `width × height` (final strip only).
    pub block_diffmap: Vec<f32>,
}

/// Row-window wrapper around an [`ImageSource`] so that the existing
/// `convert_source_to_xyb_into` helper can be used to convert just one strip.
///
/// Pretends the underlying source has only `rows` rows starting at `strip_y`.
struct RowWindowSource<'a, S: ImageSource> {
    inner: &'a S,
    strip_y: usize,
    rows: usize,
}

impl<S: ImageSource> ImageSource for RowWindowSource<'_, S> {
    #[inline]
    fn width(&self) -> usize {
        self.inner.width()
    }
    #[inline]
    fn height(&self) -> usize {
        self.rows
    }
    #[inline]
    fn pixel_format(&self) -> PixelFormat {
        self.inner.pixel_format()
    }
    #[inline]
    fn alpha_mode(&self) -> AlphaMode {
        self.inner.alpha_mode()
    }
    #[inline]
    fn color_primaries(&self) -> ColorPrimaries {
        self.inner.color_primaries()
    }
    #[inline]
    fn row_bytes(&self, y: usize) -> &[u8] {
        // Translate window-local y to source-global y.
        self.inner.row_bytes(self.strip_y + y)
    }
}

/// Streaming consumer producing a [`DiffmapResult`] from row-ordered distorted
/// strips compared against a [`PrecomputedReference`].
///
/// See module-level docs for streaming semantics and per-strip return value
/// rationale. Created via [`StreamingDiffmap::new`].
pub struct StreamingDiffmap<'a> {
    reference: &'a PrecomputedReference,
    options: DiffmapOptions,
    config: ZensimConfig,
    profile: ZensimProfile,
    weights: &'a [f64],
    per_scale_ch: Vec<[PixelFeatureWeights; 3]>,
    scale_blend: Vec<f32>,
    width: usize,
    height: usize,
    padded_width: usize,
    /// Distorted XYB plane progressively filled by `push_distorted_strip`.
    dst_planes: [Vec<f32>; 3],
    /// Strip-sized scratch planes (avoid per-call alloc when copying via the
    /// row-window XYB path).
    strip_planes: [Vec<f32>; 3],
    /// Lower bound on rows that have been pushed and converted; pushes are
    /// expected in row order, but tolerated out of order as long as no row
    /// gap exists at finalize time.
    rows_filled: usize,
    /// Cached final result, set on the final-strip push and reused by
    /// `finalize` / `current_score`.
    cached: Option<CachedResult>,
}

struct CachedResult {
    score: f32,
    diffmap: Vec<f32>,
    zresult: ZensimResult,
}

impl<'a> StreamingDiffmap<'a> {
    /// Create a new streaming diffmap consumer bound to `reference` and `options`.
    ///
    /// The image dimensions are taken from the reference's scale-0 plane, and
    /// per-scale feature weights are resolved up front (matching the work done
    /// inside [`crate::Zensim::compute_with_ref_and_diffmap`]).
    pub fn new(
        zensim: &crate::Zensim,
        reference: &'a PrecomputedReference,
        options: DiffmapOptions,
    ) -> Self {
        let profile = zensim.profile();
        let params = profile.params();
        let config = config_from_params(params, zensim.parallel());
        let (per_scale_ch, scale_blend) = options.weighting.resolve_multiscale(
            params.weights,
            config.num_scales,
            options.include_edge_mse,
            options.include_hf,
        );

        // Reference scale 0 dictates dims (PrecomputedReference is owned by caller).
        let (_planes, padded_width, height) = &reference.scales[0];
        let padded_width = *padded_width;
        let height = *height;
        // Width is whatever the reference was built from. The pad columns at the
        // right edge are mirror-padded; we need the actual width from somewhere.
        // PrecomputedReference doesn't store unpadded width, but
        // `simd_padded_width` is monotonic and inverting it would be guesswork
        // when padded_width == width (no padding), so we accept "width = the
        // distorted source's width()" lazily in push_distorted_strip and we
        // reconstruct a candidate width here from the padded_width == width path.
        // For now, store padded_width as both — it gets overwritten on first push.
        let width = padded_width;

        let n = padded_width * height;
        let dst_planes = std::array::from_fn(|_| vec![0.0f32; n]);
        let strip_planes = std::array::from_fn(|_| Vec::<f32>::new());

        Self {
            reference,
            options,
            config,
            profile,
            weights: params.weights,
            per_scale_ch,
            scale_blend,
            width,
            height,
            padded_width,
            dst_planes,
            strip_planes,
            rows_filled: 0,
            cached: None,
        }
    }

    /// Image width (actual, not SIMD-padded). Determined on first push or
    /// derived from the reference if equal to padded width.
    pub fn width(&self) -> usize {
        self.width
    }

    /// Image height (matches the reference).
    pub fn height(&self) -> usize {
        self.height
    }

    /// Push a strip of distorted pixels. Returns `None` for non-final strips
    /// and `Some(StripContribution)` containing the full image score and
    /// diffmap when the strip completes the image (`strip_y + rows == height`).
    ///
    /// # Panics
    ///
    /// Panics if `strip_y + rows > height`, if the source's `width()` does not
    /// match the reference's width, or if `rows == 0`.
    pub fn push_distorted_strip(
        &mut self,
        distorted: &impl ImageSource,
        strip_y: usize,
        rows: usize,
    ) -> Option<StripContribution> {
        assert!(rows > 0, "StreamingDiffmap::push_distorted_strip rows == 0");
        assert!(
            strip_y + rows <= self.height,
            "StreamingDiffmap::push_distorted_strip overruns height"
        );
        let strip_w = distorted.width();
        // Width discovery: the first push tells us the real width. Subsequent
        // pushes must match.
        if self.rows_filled == 0 && self.cached.is_none() {
            assert!(
                strip_w <= self.padded_width,
                "distorted width ({strip_w}) exceeds reference padded_width ({})",
                self.padded_width
            );
            self.width = strip_w;
        } else {
            assert_eq!(
                strip_w, self.width,
                "distorted width changed across pushes ({strip_w} vs {})",
                self.width
            );
        }

        // Resize strip scratch planes to exactly strip_n. The chunked XYB
        // converter uses `chunks_mut(chunk_rows * padded_width)` so the planes
        // must match the strip's row count exactly (not larger).
        let strip_n = self.padded_width * rows;
        for p in &mut self.strip_planes {
            p.resize(strip_n, 0.0);
        }

        // Convert the strip's rows to padded XYB into strip_planes.
        // RowWindowSource lets convert_source_to_xyb_into see only the strip.
        let window = RowWindowSource {
            inner: distorted,
            strip_y,
            rows,
        };
        convert_source_to_xyb_into(
            &window,
            &mut self.strip_planes,
            self.padded_width,
            self.config.allow_multithreading,
        );

        // Copy the converted XYB strip into the right offset of dst_planes.
        let dst_off = strip_y * self.padded_width;
        for c in 0..3 {
            self.dst_planes[c][dst_off..dst_off + strip_n]
                .copy_from_slice(&self.strip_planes[c][..strip_n]);
        }

        self.rows_filled = self.rows_filled.max(strip_y + rows);

        if self.rows_filled == self.height {
            self.run_final()
        } else {
            None
        }
    }

    /// Push a strip of distorted pixels in **planar linear f32 RGB** form.
    ///
    /// `planes` are `[R, G, B]` for the entire image; rows `strip_y..strip_y+rows`
    /// are read at `stride` f32 elements per row. This matches the layout used
    /// by [`crate::Zensim::compute_with_ref_and_diffmap_linear_planar`].
    ///
    /// # Panics
    ///
    /// Panics on the same conditions as [`Self::push_distorted_strip`].
    pub fn push_distorted_strip_linear_planar(
        &mut self,
        planes: [&[f32]; 3],
        strip_y: usize,
        rows: usize,
        stride: usize,
    ) -> Option<StripContribution> {
        assert!(rows > 0, "rows == 0");
        assert!(
            strip_y + rows <= self.height,
            "push_distorted_strip_linear_planar overruns height"
        );
        // Width is implicit in stride / caller intent. The linear-planar entry
        // point isn't given a width per row, so we adopt the width from the
        // reference's (assumed equal to padded_width when padding was 0). For
        // correctness we require the caller to have already declared width
        // matching the reference: if `self.width == self.padded_width`
        // (no padding), use that; otherwise the user must call
        // `with_distorted_width` first. We default to padded_width here.
        let width = self.width;

        let strip_n = self.padded_width * rows;
        for p in &mut self.strip_planes {
            p.resize(strip_n, 0.0);
        }

        // Slice each input plane to the strip's row range.
        let plane_slices: [&[f32]; 3] = [
            &planes[0][strip_y * stride..(strip_y + rows) * stride],
            &planes[1][strip_y * stride..(strip_y + rows) * stride],
            &planes[2][strip_y * stride..(strip_y + rows) * stride],
        ];
        convert_linear_planar_to_xyb_into(
            plane_slices,
            width,
            rows,
            stride,
            self.padded_width,
            &mut self.strip_planes,
        );

        let dst_off = strip_y * self.padded_width;
        for c in 0..3 {
            self.dst_planes[c][dst_off..dst_off + strip_n]
                .copy_from_slice(&self.strip_planes[c][..strip_n]);
        }

        self.rows_filled = self.rows_filled.max(strip_y + rows);

        if self.rows_filled == self.height {
            self.run_final()
        } else {
            None
        }
    }

    /// Set the actual (unpadded) image width when the reference was built from
    /// linear-planar input that produced a padded-width buffer. Only required
    /// before [`Self::push_distorted_strip_linear_planar`] when the SIMD
    /// padding actually pads (i.e., width is not a multiple of the SIMD lane
    /// width).
    pub fn with_distorted_width(mut self, width: usize) -> Self {
        assert!(
            width <= self.padded_width,
            "width ({width}) exceeds padded_width ({})",
            self.padded_width
        );
        self.width = width;
        self
    }

    /// Run the multi-scale pipeline now that all rows are filled.
    fn run_final(&mut self) -> Option<StripContribution> {
        // Move dst_planes out, run the existing pipeline, then cache.
        let dst_planes = std::mem::take(&mut self.dst_planes);
        let (zresult, mut diffmap_padded, padded_width) = compute_diffmap_from_xyb(
            self.reference,
            dst_planes,
            self.width,
            self.height,
            self.padded_width,
            &self.config,
            self.weights,
            &self.per_scale_ch,
            &self.scale_blend,
        );
        debug_assert_eq!(padded_width, self.padded_width);
        let zresult = zresult.with_profile(self.profile);

        // Trim padded width.
        let mut diffmap = if self.padded_width == self.width {
            std::mem::take(&mut diffmap_padded)
        } else {
            let mut out = Vec::with_capacity(self.width * self.height);
            for y in 0..self.height {
                let row_start = y * self.padded_width;
                out.extend_from_slice(&diffmap_padded[row_start..row_start + self.width]);
            }
            out
        };

        // Post-process: contrast masking + sqrt.
        if let Some(strength) = self.options.masking_strength {
            apply_contrast_masking(
                &mut diffmap,
                self.reference,
                self.width,
                self.height,
                self.padded_width,
                strength,
            );
        }
        if self.options.sqrt {
            sqrt_inplace(&mut diffmap);
        }

        let score = zresult.score() as f32;
        let block_diffmap = diffmap.clone();
        self.cached = Some(CachedResult {
            score,
            diffmap,
            zresult,
        });
        Some(StripContribution {
            score_delta: score,
            block_diffmap,
        })
    }

    /// Consume the streaming consumer and return the final [`DiffmapResult`].
    ///
    /// # Panics
    ///
    /// Panics if not all rows have been pushed (`rows_filled < height`).
    pub fn finalize(mut self) -> DiffmapResult {
        if self.cached.is_none() {
            assert_eq!(
                self.rows_filled, self.height,
                "StreamingDiffmap::finalize called before all rows pushed ({} of {})",
                self.rows_filled, self.height
            );
            // run_final returns Some here because rows_filled == height.
            let _ = self.run_final();
        }
        let cached = self.cached.take().expect("cached result set above");
        // Reconstruct a DiffmapResult via the public API surface. DiffmapResult
        // is #[non_exhaustive] in this crate — we own the fields, so use the
        // public constructor path.
        DiffmapResult::from_parts_internal(
            cached.zresult,
            cached.diffmap,
            self.width,
            self.height,
        )
    }

    /// Current zensim score, or `f32::NAN` if the final pipeline hasn't run yet.
    ///
    /// Only becomes meaningful once the final strip is pushed.
    pub fn current_score(&self) -> f32 {
        self.cached
            .as_ref()
            .map(|c| c.score)
            .unwrap_or(f32::NAN)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::RgbSlice;
    use crate::{DiffmapOptions, DiffmapWeighting, Zensim};

    fn checker(w: usize, h: usize) -> Vec<[u8; 3]> {
        (0..w * h)
            .map(|i| {
                let x = i % w;
                let y = i / w;
                let v: u8 = if (x / 4 + y / 4).is_multiple_of(2) { 200 } else { 60 };
                [v, v.saturating_add((x % 11) as u8), v.saturating_sub((y % 7) as u8)]
            })
            .collect()
    }

    #[test]
    fn streaming_matches_full_image_score_and_diffmap() {
        let w = 64;
        let h = 64;
        let src = checker(w, h);
        let dst: Vec<[u8; 3]> = src
            .iter()
            .map(|p| [p[0].saturating_add(8), p[1].saturating_sub(4), p[2]])
            .collect();
        let z = Zensim::new(crate::ZensimProfile::latest());
        let src_img = RgbSlice::new(&src, w, h);
        let dst_img = RgbSlice::new(&dst, w, h);
        let pre = z.precompute_reference(&src_img).unwrap();

        let opts = DiffmapOptions {
            weighting: DiffmapWeighting::Trained,
            ..Default::default()
        };
        let reference = z
            .compute_with_ref_and_diffmap(&pre, &dst_img, opts)
            .unwrap();

        let mut sd = StreamingDiffmap::new(&z, &pre, opts);
        let strip = 16;
        let mut last = None;
        let mut y = 0;
        while y < h {
            let r = strip.min(h - y);
            last = sd.push_distorted_strip(&dst_img, y, r);
            y += r;
        }
        let result = sd.finalize();
        assert!(last.is_some(), "final push should return Some");
        assert!(
            (result.score() - reference.score()).abs() < 1e-4,
            "score mismatch: {} vs {}",
            result.score(),
            reference.score()
        );
        let max_diff = reference
            .diffmap()
            .iter()
            .zip(result.diffmap())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_diff < 1e-4, "diffmap drift {max_diff}");
    }
}
