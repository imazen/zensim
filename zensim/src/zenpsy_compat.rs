//! `zenpsy-api` trait implementations for [`Zensim`].
//!
//! Implements [`zenpsy_api::Metric`] and [`zenpsy_api::PrepareReference`] over
//! colour-described `zenpixels 0.5` [`PixelView`] inputs:
//! - **Score-only by default:** [`EvalOpts::new()`] computes only the scalar
//!   [`ZensimResult`] and never allocates or computes a diffmap.
//! - **Opt-in diffmap:** [`MapRequest::Native`] and
//!   [`MapRequest::Named("diffmap")`] compute the full-resolution per-pixel
//!   SSIM error map (`Direction::Lower`, perfect `0.0`) paired with the exact
//!   same canonical headline score. Any unknown named map fails immediately
//!   with [`ErrorKind::NoMap`] before allocation.
//! - **Owned prepared reference:** [`PreparedZensim`] owns its configured
//!   [`Zensim`], [`PrecomputedReference`], and reusable [`ZensimScratch`], so
//!   it carries no borrow of the factory and can be boxed directly into
//!   job-local encoder verification loops.
//! - **SDR only:** HDR inputs, `DiffuseWhite`, `HdrMetadata`, pending
//!   orientation, premultiplied alpha, narrow range, and unsupported color
//!   spaces fail explicitly with [`ErrorKind::UnsupportedInput`].

use zenpixels::{PixelView, PlaneFormat, PlaneRole, PlaneView, SampleStride, SampleType};
use zenpsy_api::{
    Direction, ErrorKind, EvalOpts, Evaluation, MapGrid, MapRequest, MapView, Metric, MetricError,
    MetricId, MetricResult, PrepareReference, Score, ScoreSpec, WarmReference, fail, whereat,
};

use crate::diffmap::{DiffmapOptions, DiffmapResult};
use crate::error::ZensimError;
use crate::metric::{Zensim, ZensimResult};
use crate::profile::ZensimProfile;
use crate::streaming::{PrecomputedReference, ZensimScratch};
use crate::zenpixels_compat::{StrictIngressError, ZenpixelsSource};

static ZENSIM_SCORE_SPEC: ScoreSpec =
    ScoreSpec::new("zensim", Direction::Higher).with_perfect(100.0);
static ZENSIM_DIFFMAP_SPEC: ScoreSpec =
    ScoreSpec::new("diffmap", Direction::Lower).with_perfect(0.0);

/// Native `zenpsy-api` assessment returned by [`Zensim`].
///
/// Always carries the complete [`ZensimResult`]. The spatial [`DiffmapResult`]
/// is computed and retained **only** when explicitly requested via
/// [`EvalOpts::with_native_map`] or [`EvalOpts::with_map("diffmap")`].
pub struct ZensimEvaluation {
    result: ZensimResult,
    diffmap: Option<DiffmapResult>,
}

impl core::fmt::Debug for ZensimEvaluation {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ZensimEvaluation")
            .field("score", &self.result.score())
            .field("profile", &self.result.profile())
            .field("has_diffmap", &self.diffmap.is_some())
            .finish_non_exhaustive()
    }
}

impl ZensimEvaluation {
    /// Complete native [`ZensimResult`].
    #[must_use]
    pub fn result(&self) -> &ZensimResult {
        &self.result
    }

    /// Headline perceptual similarity score (`100.0` = identical, higher = better).
    #[must_use]
    pub fn score(&self) -> f64 {
        self.result.score()
    }

    /// Profile's native raw distance / MLP pre-disposition value.
    #[must_use]
    pub fn raw_distance(&self) -> f64 {
        self.result.raw_distance()
    }

    /// Opt-in per-pixel error map, present only when diagnostics were requested.
    #[must_use]
    pub fn diffmap(&self) -> Option<&DiffmapResult> {
        self.diffmap.as_ref()
    }

    /// Consume this evaluation and return its owned [`ZensimResult`].
    #[must_use]
    pub fn into_result(self) -> ZensimResult {
        self.result
    }
}

impl Evaluation for ZensimEvaluation {
    fn scores(&self) -> impl Iterator<Item = Score<'_>> {
        core::iter::once(Score::new(&ZENSIM_SCORE_SPEC, self.result.score()))
    }

    fn maps(&self) -> impl Iterator<Item = MapView<'_>> {
        self.diffmap.iter().filter_map(|dm| {
            let w = u32::try_from(dm.width()).ok()?;
            let h = u32::try_from(dm.height()).ok()?;
            let plane = PlaneView::from_samples_strided(
                dm.diffmap(),
                w,
                h,
                SampleStride(dm.width()),
                PlaneFormat::new(PlaneRole::Y, SampleType::F32),
            )
            .ok()?;
            let grid = MapGrid::new(w, h).ok()?;
            Some(MapView::new(&ZENSIM_DIFFMAP_SPEC, plane, grid))
        })
    }
}

/// Owned prepared reference work for repeated `zenpsy-api` comparisons against
/// one reference image.
///
/// Owns a cloned [`Zensim`] configuration, the precomputed multi-scale XYB
/// reference pyramid (plus exact reference pixels for byte-identical `100.0`
/// parity), and a reusable [`ZensimScratch`] buffer.
pub struct PreparedZensim {
    zensim: Zensim,
    precomputed: PrecomputedReference,
    scratch: ZensimScratch,
}

impl WarmReference for PreparedZensim {
    type Output = ZensimEvaluation;

    fn evaluate(
        &mut self,
        candidate: &PixelView<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Output> {
        let want_diffmap = validate_map_request(opts.map_request())?;
        check_active_max_pixels(candidate, opts.max_pixels())?;

        self.zensim.with_effective_stop(Some(opts.stop()), |stop| {
            check_effective_stop(stop)?;
            let dst = strict_source(candidate, stop)?;

            let out = if want_diffmap {
                let dm = self
                    .zensim
                    .compute_with_ref_and_diffmap_with_stop(
                        &self.precomputed,
                        &dst,
                        DiffmapOptions::default(),
                        stop,
                    )
                    .map_err(map_zensim_error)?;
                let result = dm.result().clone();
                ZensimEvaluation {
                    result,
                    diffmap: Some(dm),
                }
            } else {
                let result = self
                    .zensim
                    .compute_with_ref_into_with_stop(
                        &self.precomputed,
                        &dst,
                        &mut self.scratch,
                        stop,
                    )
                    .map_err(map_zensim_error)?;
                ZensimEvaluation {
                    result,
                    diffmap: None,
                }
            };
            check_effective_stop(stop)?;
            Ok(out)
        })
    }
}

impl Metric for Zensim {
    type Output = ZensimEvaluation;

    fn id(&self) -> MetricId {
        let (score_rev, formula_rev, bake) = profile_provenance(self.profile());
        let mut id = MetricId::new("zensim", zenpsy_api::implementation!(score_rev))
            .with_variant(self.profile().name())
            .with_config("profile", self.profile().name())
            .with_config("formula_revision", formula_rev)
            .with_config("parallel", self.parallel());
        if let Some(bake_name) = bake {
            id = id.with_config("bake", bake_name);
        }
        id
    }

    fn evaluate(
        &mut self,
        reference: &PixelView<'_>,
        candidate: &PixelView<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Output> {
        let want_diffmap = validate_map_request(opts.map_request())?;
        check_active_max_pixels(reference, opts.max_pixels())?;
        check_active_max_pixels(candidate, opts.max_pixels())?;

        self.with_effective_stop(Some(opts.stop()), |stop| {
            check_effective_stop(stop)?;
            let src = strict_source(reference, stop)?;
            let dst = strict_source(candidate, stop)?;

            let out = if want_diffmap {
                let dm = self
                    .compute_with_diffmap_with_stop(&src, &dst, DiffmapOptions::default(), stop)
                    .map_err(map_zensim_error)?;
                let result = dm.result().clone();
                ZensimEvaluation {
                    result,
                    diffmap: Some(dm),
                }
            } else {
                let result = self
                    .compute_with_stop(&src, &dst, stop)
                    .map_err(map_zensim_error)?;
                ZensimEvaluation {
                    result,
                    diffmap: None,
                }
            };
            check_effective_stop(stop)?;
            Ok(out)
        })
    }
}

impl PrepareReference for Zensim {
    type Prepared<'a> = PreparedZensim;

    fn warm_reference(
        &mut self,
        reference: &PixelView<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Prepared<'_>> {
        // Validate any map request up front so invalid names fail before reference prep.
        let _ = validate_map_request(opts.map_request())?;
        check_active_max_pixels(reference, opts.max_pixels())?;

        self.with_effective_stop(Some(opts.stop()), |stop| {
            check_effective_stop(stop)?;
            let src = strict_source(reference, stop)?;
            let precomputed = self
                .precompute_reference_with_stop(&src, stop)
                .map_err(map_zensim_error)?;
            check_effective_stop(stop)?;
            Ok(PreparedZensim {
                zensim: self.clone(),
                precomputed,
                scratch: ZensimScratch::new(),
            })
        })
    }
}

#[track_caller]
fn validate_map_request(req: MapRequest<'_>) -> MetricResult<bool> {
    match req {
        MapRequest::None => Ok(false),
        MapRequest::Native | MapRequest::Named("diffmap") => Ok(true),
        MapRequest::Named(_) => fail(
            MetricError::new(ErrorKind::NoMap)
                .with_metric("zensim")
                .with_message("zensim only publishes the 'diffmap' map"),
        ),
        _ => fail(
            MetricError::new(ErrorKind::UnsupportedRequest)
                .with_metric("zensim")
                .with_message("unsupported map request"),
        ),
    }
}

#[track_caller]
fn strict_source<'a>(
    view: &PixelView<'a>,
    stop: Option<&dyn enough::Stop>,
) -> MetricResult<ZenpixelsSource<'a>> {
    ZenpixelsSource::try_from_view_strict_stoppable(view, stop).map_err(|err| match err {
        StrictIngressError::Unsupported(msg) => whereat::at(
            MetricError::new(ErrorKind::UnsupportedInput)
                .with_metric("zensim")
                .with_message(msg),
        ),
        StrictIngressError::InvalidSamples(msg) => whereat::at(
            MetricError::new(ErrorKind::InvalidInput)
                .with_metric("zensim")
                .with_message(msg),
        ),
        StrictIngressError::Stopped(reason) => {
            whereat::at(MetricError::stopped(reason).with_metric("zensim"))
        }
    })
}

#[track_caller]
fn check_effective_stop(stop: Option<&dyn enough::Stop>) -> MetricResult<()> {
    if let Some(s) = stop {
        s.check()
            .map_err(|reason| whereat::at(MetricError::stopped(reason).with_metric("zensim")))?;
    }
    Ok(())
}

/// Enforce `EvalOpts::max_pixels()` strictly against active input pixels
/// (excluding internal reflect-padding scratch), per `zenpsy-api` contract.
/// Native `Zensim::with_max_pixels` remains the expanded-allocation guard.
#[track_caller]
fn check_active_max_pixels(view: &PixelView<'_>, max_pixels: Option<u64>) -> MetricResult<()> {
    if let Some(cap) = max_pixels {
        let active = u64::from(view.width())
            .checked_mul(u64::from(view.height()))
            .ok_or_else(|| {
                whereat::at(
                    MetricError::new(ErrorKind::Resource)
                        .with_metric("zensim")
                        .with_message("active pixel count overflow"),
                )
            })?;
        if active > cap {
            return fail(
                MetricError::new(ErrorKind::Resource)
                    .with_metric("zensim")
                    .with_message("image active pixel count exceeds EvalOpts::max_pixels"),
            );
        }
    }
    Ok(())
}

#[track_caller]
fn map_zensim_error(err: ZensimError) -> whereat::At<MetricError> {
    let kind = match err {
        ZensimError::DimensionMismatch => ErrorKind::DimensionMismatch,
        ZensimError::ImageTooSmall => ErrorKind::TooSmall,
        ZensimError::InvalidDataLength | ZensimError::InvalidStride => ErrorKind::InvalidInput,
        ZensimError::ImageTooLarge => ErrorKind::Resource,
        ZensimError::UnsupportedPixelFormat => ErrorKind::UnsupportedInput,
        ZensimError::FeatureWeightsLengthMismatch => ErrorKind::InvalidConfig,
        ZensimError::ModelLoadFailed { .. } | ZensimError::ModelForwardFailed { .. } => {
            ErrorKind::Internal
        }
        ZensimError::Cancelled { reason } => ErrorKind::Stopped(reason),
    };
    whereat::at(MetricError::from_detail(kind, err).with_metric("zensim"))
}

fn profile_provenance(profile: ZensimProfile) -> (u32, &'static str, Option<&'static str>) {
    match profile {
        ZensimProfile::A => (7, "rev7", Some("rev7_mix_h128_2026-11-03.bin")),
        #[cfg(feature = "legacy-models")]
        ZensimProfile::PreviewV0_1 => (1, "v0.1-linear", None),
        #[cfg(feature = "legacy-models")]
        ZensimProfile::PreviewV0_2 => (2, "v0.2-linear", None),
        #[cfg(all(feature = "legacy-models", any(feature = "training", test)))]
        ZensimProfile::LegacyLinearV0_2 => (2, "v0.2-legacy-linear", None),
    }
}
