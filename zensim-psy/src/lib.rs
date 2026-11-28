//! `zenpsy-api` adapter for [`zensim`], defaulting to Revision-7
//! [`ZensimProfile::A`] (`zensim-a`).
#![forbid(unsafe_code)]
#![warn(missing_docs)]

use std::borrow::Cow;
use zenpsy_api::{
    self as psy, Direction, ErrorKind, EvalOpts, Evaluation, EvaluationRef, MapGrid, MapRequest,
    MapView, Metric, MetricError, MetricId, MetricResult, PlanarMetric, PlanarWarmReference,
    PreparePlanarReference, PrepareReference, ResultMetricExt as _, Score, ScoreSpec,
    WarmReference, fail,
    zenpixels::{
        AlphaMode as ZpAlpha, ColorPrimaries as ZpPrimaries, PixelDescriptor,
        PixelFormat as ZpFormat, PixelSlice, SignalRange, TransferFunction,
        plane::{AnyPlane, FloatEncoding, PlanarSlice, PlaneEncoding, PlaneRef, PlaneRole},
    },
};
use zensim::{
    AlphaMode as ZsAlpha, ColorPrimaries as ZsPrimaries, DiffmapOptions, PixelFormat as ZsFormat,
    PrecomputedReference, StridedBytes, Zensim, ZensimError, ZensimProfile, ZensimResult,
    ZensimScratch,
};

pub use zenpsy_api;
pub use zensim;

/// Revision-7 `zensim-a` score unit string.
pub const ZENSIM_A_UNIT: &str = "zensim-a";
/// Spatial diffmap unit string.
pub const ZENSIM_DIFFMAP_UNIT: &str = "zensim-diffmap";
/// Stable headline score key.
pub const SCORE_KEY: &str = "score";
/// Stable steering diffmap key.
pub const DIFFMAP_KEY: &str = "diffmap";

fn score_spec_for_profile(profile: ZensimProfile) -> ScoreSpec {
    let unit = match profile {
        ZensimProfile::A => ZENSIM_A_UNIT,
        _ => "zensim",
    };
    ScoreSpec::new(SCORE_KEY, Direction::Higher)
        .with_perfect(100.0)
        .with_unit(unit)
}

fn diffmap_spec() -> ScoreSpec {
    ScoreSpec::new(DIFFMAP_KEY, Direction::Lower)
        .with_perfect(0.0)
        .with_unit(ZENSIM_DIFFMAP_UNIT)
}

fn map_zensim_error(err: &ZensimError) -> ErrorKind {
    match err {
        ZensimError::ImageTooSmall => ErrorKind::TooSmall,
        ZensimError::DimensionMismatch => ErrorKind::DimensionMismatch,
        ZensimError::InvalidStride | ZensimError::InvalidDataLength => ErrorKind::InvalidInput,
        ZensimError::ImageTooLarge => ErrorKind::Resource,
        ZensimError::Cancelled { reason } => ErrorKind::Stopped(*reason),
        ZensimError::UnsupportedPixelFormat => ErrorKind::UnsupportedInput,
        _ => ErrorKind::Backend,
    }
}

fn check_map_request(opts: EvalOpts<'_>) -> MetricResult<bool> {
    match opts.map_request() {
        MapRequest::None => Ok(false),
        MapRequest::Native => Ok(true),
        MapRequest::Named(DIFFMAP_KEY) => Ok(true),
        MapRequest::Named(_) => fail(
            MetricError::new(ErrorKind::NoMap)
                .with_metric("zensim")
                .with_message("zensim publishes only the 'diffmap' spatial map"),
        ),
        _ => fail(MetricError::new(ErrorKind::UnsupportedRequest).with_metric("zensim")),
    }
}

fn build_scorer(
    profile: ZensimProfile,
    parallel: bool,
    max_pixels: Option<usize>,
    opts: EvalOpts<'_>,
) -> Zensim {
    let mut z = Zensim::new(profile).with_parallel(parallel);
    let effective_cap = match (max_pixels, opts.max_pixels()) {
        (Some(a), Some(b)) => Some(a.min(usize::try_from(b).unwrap_or(usize::MAX))),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(usize::try_from(b).unwrap_or(usize::MAX)),
        (None, None) => None,
    };
    if let Some(cap) = effective_cap {
        z = z.with_max_pixels(cap);
    }
    z
}

struct OwnedImage<'a> {
    data: Cow<'a, [u8]>,
    width: usize,
    height: usize,
    stride: usize,
    pixel_format: ZsFormat,
    alpha_mode: ZsAlpha,
    primaries: ZsPrimaries,
}

impl OwnedImage<'_> {
    fn as_source(&self) -> MetricResult<StridedBytes<'_>> {
        StridedBytes::try_with_alpha_mode(
            &self.data,
            self.width,
            self.height,
            self.stride,
            self.pixel_format,
            self.alpha_mode,
        )
        .map(|s| s.with_color_primaries(self.primaries))
        .metric_err_with("zensim", map_zensim_error)
    }
}

fn unpack_pixel_slice<'a>(slice: &'a PixelSlice<'a>) -> MetricResult<OwnedImage<'a>> {
    if slice.color_context().is_some() {
        return fail(
            MetricError::new(ErrorKind::UnsupportedInput)
                .with_metric("zensim")
                .with_message("ICC/ColorContext override is not supported on packed PixelSlice"),
        );
    }
    let desc: PixelDescriptor = slice.descriptor();
    if desc.signal_range != SignalRange::Full {
        return fail(
            MetricError::new(ErrorKind::UnsupportedInput)
                .with_metric("zensim")
                .with_message("narrow signal range is not supported"),
        );
    }
    match desc.transfer {
        TransferFunction::Srgb | TransferFunction::Bt709 => {}
        TransferFunction::Linear if desc.format == ZpFormat::RgbaF32 => {}
        _ => {
            return fail(
                MetricError::new(ErrorKind::UnsupportedInput)
                    .with_metric("zensim")
                    .with_message("unsupported transfer function for SDR zensim"),
            );
        }
    }
    let pixel_format = match desc.format {
        ZpFormat::Rgb8 => ZsFormat::Srgb8Rgb,
        ZpFormat::Rgba8 | ZpFormat::Rgbx8 => ZsFormat::Srgb8Rgba,
        ZpFormat::Bgra8 | ZpFormat::Bgrx8 => ZsFormat::Srgb8Bgra,
        ZpFormat::Rgba16 => ZsFormat::Srgb16Rgba,
        ZpFormat::RgbaF32 => ZsFormat::LinearF32Rgba,
        _ => {
            return fail(
                MetricError::new(ErrorKind::UnsupportedInput)
                    .with_metric("zensim")
                    .with_message("unsupported pixel format"),
            );
        }
    };
    let alpha_mode = match desc.alpha {
        None | Some(ZpAlpha::Opaque) | Some(ZpAlpha::Undefined) => ZsAlpha::Opaque,
        Some(ZpAlpha::Straight) => ZsAlpha::Straight,
        _ => {
            return fail(
                MetricError::new(ErrorKind::UnsupportedInput)
                    .with_metric("zensim")
                    .with_message("premultiplied or unknown alpha mode is not supported"),
            );
        }
    };
    let primaries = match desc.primaries {
        ZpPrimaries::Bt709 => ZsPrimaries::Srgb,
        ZpPrimaries::DisplayP3 => ZsPrimaries::DisplayP3,
        ZpPrimaries::Bt2020 => ZsPrimaries::Bt2020,
        _ => {
            return fail(
                MetricError::new(ErrorKind::UnsupportedInput)
                    .with_metric("zensim")
                    .with_message("unsupported color primaries"),
            );
        }
    };
    let width = slice.width() as usize;
    let height = slice.rows() as usize;
    let stride = slice.stride();
    let raw = slice.as_strided_bytes();
    let needed_full = stride.checked_mul(height).ok_or_else(|| {
        psy::whereat::at(MetricError::new(ErrorKind::Resource).with_metric("zensim"))
    })?;
    let (data, out_stride) = if raw.len() >= needed_full {
        (Cow::Borrowed(raw), stride)
    } else {
        let row_bytes = width * pixel_format.bytes_per_pixel();
        let mut packed = Vec::new();
        packed
            .try_reserve_exact(row_bytes * height)
            .metric_err("zensim", ErrorKind::Resource)?;
        for y in 0..slice.rows() {
            packed.extend_from_slice(&slice.row(y)[..row_bytes]);
        }
        (Cow::Owned(packed), row_bytes)
    };
    Ok(OwnedImage {
        data,
        width,
        height,
        stride: out_stride,
        pixel_format,
        alpha_mode,
        primaries,
    })
}

type PlanarInput<'a> = ([&'a [f32]; 3], [usize; 3], ZsPrimaries);

fn unpack_planar_slice<'a>(slice: &PlanarSlice<'a>) -> MetricResult<PlanarInput<'a>> {
    let color = slice.color_context();
    if color.icc.is_some()
        || color.hdr_peak.is_some()
        || slice.alpha_mode().is_some()
        || slice.layout().planes().len() != 3
    {
        return fail(
            MetricError::new(ErrorKind::UnsupportedInput)
                .with_metric("zensim")
                .with_message("planar zensim requires 3 opaque linear RGB planes without ICC/HDR"),
        );
    }
    let Some(cicp) = color.cicp else {
        return fail(
            MetricError::new(ErrorKind::UnsupportedInput)
                .with_metric("zensim")
                .with_message("planar zensim requires explicit linear RGB CICP"),
        );
    };
    if cicp.transfer_characteristics != 8 || cicp.matrix_coefficients != 0 || !cicp.full_range {
        return fail(
            MetricError::new(ErrorKind::UnsupportedInput)
                .with_metric("zensim")
                .with_message(
                    "planar zensim requires full-range linear RGB (CICP transfer=8, matrix=0)",
                ),
        );
    }
    let primaries = match cicp.color_primaries {
        1 => ZsPrimaries::Srgb,
        12 => ZsPrimaries::DisplayP3,
        9 => ZsPrimaries::Bt2020,
        _ => {
            return fail(
                MetricError::new(ErrorKind::UnsupportedInput)
                    .with_metric("zensim")
                    .with_message(
                        "planar zensim supports BT.709, Display P3, and BT.2020 primaries",
                    ),
            );
        }
    };
    let mut planes = [None; 3];
    let mut strides = [0usize; 3];
    for (spec, plane) in slice.layout().planes().iter().zip(slice.planes()) {
        if spec.encoding() != PlaneEncoding::Float(FloatEncoding::NOMINAL) {
            return fail(
                MetricError::new(ErrorKind::UnsupportedInput)
                    .with_metric("zensim")
                    .with_message("planar zensim requires nominal f32 planes"),
            );
        }
        let ch = match spec.role() {
            PlaneRole::Red => 0,
            PlaneRole::Green => 1,
            PlaneRole::Blue => 2,
            _ => {
                return fail(
                    MetricError::new(ErrorKind::UnsupportedInput)
                        .with_metric("zensim")
                        .with_message("planar zensim requires Red, Green, and Blue planes"),
                );
            }
        };
        let AnyPlane::F32(p) = plane else {
            return fail(
                MetricError::new(ErrorKind::UnsupportedInput)
                    .with_metric("zensim")
                    .with_message("planar zensim requires f32 plane storage"),
            );
        };
        if p.width() != slice.width() as usize || p.height() != slice.height() as usize {
            return fail(MetricError::new(ErrorKind::DimensionMismatch).with_metric("zensim"));
        }
        planes[ch] = Some(p.into_strided_samples());
        strides[ch] = p.stride();
    }
    let [Some(r), Some(g), Some(b)] = planes else {
        return fail(
            MetricError::new(ErrorKind::UnsupportedInput)
                .with_metric("zensim")
                .with_message("planar zensim requires distinct Red, Green, and Blue planes"),
        );
    };
    Ok(([r, g, b], strides, primaries))
}

/// Owned assessment returned by [`ZensimMetric`].
#[derive(Debug)]
pub struct ZensimEvaluation {
    result: ZensimResult,
    diffmap: Option<Vec<f32>>,
    width: u32,
    height: u32,
    score_spec: ScoreSpec,
    map_spec: ScoreSpec,
}

impl ZensimEvaluation {
    /// Borrow the underlying [`ZensimResult`].
    pub fn native(&self) -> &ZensimResult {
        &self.result
    }
    /// Borrow the owned diffmap slice when diagnostics were requested.
    pub fn diffmap(&self) -> Option<&[f32]> {
        self.diffmap.as_deref()
    }
}

impl Evaluation for ZensimEvaluation {
    fn scores(&self) -> impl Iterator<Item = Score<'_>> {
        core::iter::once(Score::new(&self.score_spec, self.result.score()))
    }
    fn maps(&self) -> impl Iterator<Item = MapView<'_>> {
        self.diffmap.as_deref().into_iter().map(|buf| {
            let plane = PlaneRef::new(
                buf,
                self.width as usize,
                self.height as usize,
                self.width as usize,
            )
            .expect("diffmap matches image dimensions");
            MapView::new(
                &self.map_spec,
                AnyPlane::F32(plane),
                MapGrid::new(self.width, self.height).expect("non-empty image"),
            )
        })
    }
}

struct BorrowedEvaluation<'a> {
    score: f64,
    score_spec: &'a ScoreSpec,
    diffmap: Option<&'a [f32]>,
    map_spec: &'a ScoreSpec,
    width: u32,
    height: u32,
}

impl Evaluation for BorrowedEvaluation<'_> {
    fn scores(&self) -> impl Iterator<Item = Score<'_>> {
        core::iter::once(Score::new(self.score_spec, self.score))
    }
    fn maps(&self) -> impl Iterator<Item = MapView<'_>> {
        self.diffmap.into_iter().map(|buf| {
            let plane = PlaneRef::new(
                buf,
                self.width as usize,
                self.height as usize,
                self.width as usize,
            )
            .expect("diffmap matches image dimensions");
            MapView::new(
                self.map_spec,
                AnyPlane::F32(plane),
                MapGrid::new(self.width, self.height).expect("non-empty image"),
            )
        })
    }
}

/// Configured `zensim` scorer implementing [`zenpsy_api`].
///
/// Defaults to [`ZensimProfile::A`] (Revision-7 `zensim-a`).
#[derive(Clone, Debug)]
pub struct ZensimMetric {
    profile: ZensimProfile,
    diffmap_options: DiffmapOptions,
    parallel: bool,
    max_pixels: Option<usize>,
}

impl Default for ZensimMetric {
    fn default() -> Self {
        Self::profile_a()
    }
}

impl ZensimMetric {
    /// Construct a metric using Revision-7 [`ZensimProfile::A`] (`zensim-a`).
    pub fn profile_a() -> Self {
        Self::new(ZensimProfile::A)
    }

    /// Construct a metric for `profile`.
    pub fn new(profile: ZensimProfile) -> Self {
        Self {
            profile,
            diffmap_options: DiffmapOptions::default(),
            parallel: true,
            max_pixels: Some(120_000_000),
        }
    }

    /// Configure spatial diffmap options used when diagnostics are requested.
    #[must_use]
    pub fn with_diffmap_options(mut self, options: DiffmapOptions) -> Self {
        self.diffmap_options = options;
        self
    }

    /// Enable or disable internal rayon multithreading.
    #[must_use]
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Override the maximum pixel count accepted by this scorer.
    #[must_use]
    pub fn with_max_pixels(mut self, max_pixels: usize) -> Self {
        self.max_pixels = Some(max_pixels);
        self
    }

    /// Selected [`ZensimProfile`].
    pub const fn profile(&self) -> ZensimProfile {
        self.profile
    }

    /// Configured [`DiffmapOptions`].
    pub const fn diffmap_options(&self) -> DiffmapOptions {
        self.diffmap_options
    }
}

impl Metric for ZensimMetric {
    type Output = ZensimEvaluation;

    fn id(&self) -> MetricId {
        let mut id = MetricId::new("zensim", psy::implementation!(7));
        if self.profile == ZensimProfile::A {
            id = id.with_variant("a").with_reference("zensim-a rev7");
        }
        id.with_config("profile", self.profile)
            .with_config("parallel", self.parallel)
            .with_config("diffmap_edge_mse", self.diffmap_options.include_edge_mse)
            .with_config("diffmap_hf", self.diffmap_options.include_hf)
            .with_config("diffmap_sqrt", self.diffmap_options.sqrt)
    }

    fn headline_score_key(&self) -> Option<&str> {
        Some(SCORE_KEY)
    }

    fn steering_map_key(&self) -> Option<&str> {
        Some(DIFFMAP_KEY)
    }

    fn evaluate(
        &mut self,
        reference: &PixelSlice<'_>,
        candidate: &PixelSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Output> {
        let want_map = check_map_request(opts)?;
        let ref_img = unpack_pixel_slice(reference)?;
        let dst_img = unpack_pixel_slice(candidate)?;
        let ref_src = ref_img.as_source()?;
        let dst_src = dst_img.as_source()?;
        let scorer = build_scorer(self.profile, self.parallel, self.max_pixels, opts);
        opts.checkpoint()?;
        let (result, diffmap) = if want_map {
            let dm = scorer
                .compute_with_diffmap(&ref_src, &dst_src, self.diffmap_options)
                .metric_err_with("zensim", map_zensim_error)?;
            let (res, map, _, _) = dm.into_parts();
            (res, Some(map))
        } else {
            let res = scorer
                .compute(&ref_src, &dst_src)
                .metric_err_with("zensim", map_zensim_error)?;
            (res, None)
        };
        opts.checkpoint()?;
        Ok(ZensimEvaluation {
            result,
            diffmap,
            width: candidate.width(),
            height: candidate.rows(),
            score_spec: score_spec_for_profile(self.profile),
            map_spec: diffmap_spec(),
        })
    }
}

/// Prepared reference for packed [`PixelSlice`] comparisons.
pub struct ZensimPrepared {
    profile: ZensimProfile,
    parallel: bool,
    max_pixels: Option<usize>,
    diffmap_options: DiffmapOptions,
    precomputed: PrecomputedReference,
    scratch: ZensimScratch,
}

impl PrepareReference for ZensimMetric {
    type Prepared<'a> = ZensimPrepared;

    fn estimated_reference_bytes(&self, width: u32, height: u32) -> Option<u64> {
        PrecomputedReference::estimated_bytes_for_profile(
            width as usize,
            height as usize,
            self.profile,
        )
    }

    fn warm_reference(
        &mut self,
        reference: &PixelSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Prepared<'_>> {
        let _ = check_map_request(opts)?;
        let ref_img = unpack_pixel_slice(reference)?;
        let ref_src = ref_img.as_source()?;
        let scorer = build_scorer(self.profile, self.parallel, self.max_pixels, opts);
        opts.checkpoint()?;
        let precomputed = scorer
            .precompute_reference(&ref_src)
            .metric_err_with("zensim", map_zensim_error)?;
        opts.checkpoint()?;
        Ok(ZensimPrepared {
            profile: self.profile,
            parallel: self.parallel,
            max_pixels: self.max_pixels,
            diffmap_options: self.diffmap_options,
            precomputed,
            scratch: ZensimScratch::new(),
        })
    }
}

impl WarmReference for ZensimPrepared {
    type Output = ZensimEvaluation;

    fn is_deterministic(&self) -> bool {
        true
    }

    fn evaluate(
        &mut self,
        candidate: &PixelSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Output> {
        let want_map = check_map_request(opts)?;
        let dst_img = unpack_pixel_slice(candidate)?;
        let dst_src = dst_img.as_source()?;
        let scorer = build_scorer(self.profile, self.parallel, self.max_pixels, opts);
        opts.checkpoint()?;
        let (result, diffmap) = if want_map {
            let dm = scorer
                .compute_with_ref_and_diffmap(&self.precomputed, &dst_src, self.diffmap_options)
                .metric_err_with("zensim", map_zensim_error)?;
            let (res, map, _, _) = dm.into_parts();
            (res, Some(map))
        } else {
            let res = scorer
                .compute_with_ref_into(&self.precomputed, &dst_src, &mut self.scratch)
                .metric_err_with("zensim", map_zensim_error)?;
            (res, None)
        };
        opts.checkpoint()?;
        Ok(ZensimEvaluation {
            result,
            diffmap,
            width: candidate.width(),
            height: candidate.rows(),
            score_spec: score_spec_for_profile(self.profile),
            map_spec: diffmap_spec(),
        })
    }
}

/// Owned prepared planar reference for iterative encoder perceptual loops.
pub struct ZensimPreparedPlanar {
    profile: ZensimProfile,
    parallel: bool,
    max_pixels: Option<usize>,
    diffmap_options: DiffmapOptions,
    primaries: ZsPrimaries,
    precomputed: PrecomputedReference,
    scratch: ZensimScratch,
    score_spec: ScoreSpec,
    map_spec: ScoreSpec,
}

impl PlanarMetric for ZensimMetric {
    fn evaluate_planar(
        &mut self,
        reference: &PlanarSlice<'_>,
        candidate: &PlanarSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Output> {
        self.warm_planar_reference(reference, opts)?
            .evaluate_planar(candidate, opts)
    }
}

impl PreparePlanarReference for ZensimMetric {
    type PlanarPrepared = ZensimPreparedPlanar;

    fn estimated_reference_bytes(&self, width: u32, height: u32) -> Option<u64> {
        PrecomputedReference::estimated_bytes_for_profile(
            width as usize,
            height as usize,
            self.profile,
        )
    }

    fn warm_planar_reference(
        &mut self,
        reference: &PlanarSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::PlanarPrepared> {
        let _ = check_map_request(opts)?;
        let (planes, strides, primaries) = unpack_planar_slice(reference)?;
        let scorer = build_scorer(self.profile, self.parallel, self.max_pixels, opts);
        opts.checkpoint()?;
        let precomputed = scorer
            .precompute_reference_linear_planes(
                planes,
                strides,
                reference.width() as usize,
                reference.height() as usize,
                primaries,
            )
            .metric_err_with("zensim", map_zensim_error)?;
        opts.checkpoint()?;
        Ok(ZensimPreparedPlanar {
            profile: self.profile,
            parallel: self.parallel,
            max_pixels: self.max_pixels,
            diffmap_options: self.diffmap_options,
            primaries,
            precomputed,
            scratch: ZensimScratch::new(),
            score_spec: score_spec_for_profile(self.profile),
            map_spec: diffmap_spec(),
        })
    }
}

impl PlanarWarmReference for ZensimPreparedPlanar {
    type Output = ZensimEvaluation;

    fn is_deterministic(&self) -> bool {
        true
    }

    fn evaluate_planar(
        &mut self,
        candidate: &PlanarSlice<'_>,
        opts: EvalOpts<'_>,
    ) -> MetricResult<Self::Output> {
        let want_map = check_map_request(opts)?;
        let (planes, strides, primaries) = unpack_planar_slice(candidate)?;
        if primaries != self.primaries {
            return fail(
                MetricError::new(ErrorKind::UnsupportedInput)
                    .with_metric("zensim")
                    .with_message("planar candidate color primaries differ from reference"),
            );
        }
        let scorer = build_scorer(self.profile, self.parallel, self.max_pixels, opts);
        opts.checkpoint()?;
        let width = candidate.width();
        let height = candidate.height();
        if want_map {
            let (result, map) = scorer
                .compute_with_ref_and_diffmap_linear_planes_with(
                    &self.precomputed,
                    planes,
                    strides,
                    width as usize,
                    height as usize,
                    primaries,
                    self.diffmap_options,
                    &mut self.scratch,
                    Some(opts.stop()),
                    |res, dm| (res.clone(), dm.to_vec()),
                )
                .metric_err_with("zensim", map_zensim_error)?;
            Ok(ZensimEvaluation {
                result,
                diffmap: Some(map),
                width,
                height,
                score_spec: self.score_spec.clone(),
                map_spec: self.map_spec.clone(),
            })
        } else {
            let result = scorer
                .compute_with_ref_linear_planes_into(
                    &self.precomputed,
                    planes,
                    strides,
                    width as usize,
                    height as usize,
                    primaries,
                    &mut self.scratch,
                )
                .metric_err_with("zensim", map_zensim_error)?;
            opts.checkpoint()?;
            Ok(ZensimEvaluation {
                result,
                diffmap: None,
                width,
                height,
                score_spec: self.score_spec.clone(),
                map_spec: self.map_spec.clone(),
            })
        }
    }

    fn with_evaluation(
        &mut self,
        candidate: &PlanarSlice<'_>,
        opts: EvalOpts<'_>,
        consumer: &mut dyn FnMut(EvaluationRef<'_>),
    ) -> MetricResult<()> {
        let want_map = check_map_request(opts)?;
        let (planes, strides, primaries) = unpack_planar_slice(candidate)?;
        if primaries != self.primaries {
            return fail(
                MetricError::new(ErrorKind::UnsupportedInput)
                    .with_metric("zensim")
                    .with_message("planar candidate color primaries differ from reference"),
            );
        }
        let scorer = build_scorer(self.profile, self.parallel, self.max_pixels, opts);
        opts.checkpoint()?;
        let width = candidate.width();
        let height = candidate.height();
        if want_map {
            scorer
                .compute_with_ref_and_diffmap_linear_planes_with(
                    &self.precomputed,
                    planes,
                    strides,
                    width as usize,
                    height as usize,
                    primaries,
                    self.diffmap_options,
                    &mut self.scratch,
                    Some(opts.stop()),
                    |res, dm| {
                        let borrowed = BorrowedEvaluation {
                            score: res.score(),
                            score_spec: &self.score_spec,
                            diffmap: Some(dm),
                            map_spec: &self.map_spec,
                            width,
                            height,
                        };
                        consumer(EvaluationRef::new(&borrowed));
                    },
                )
                .metric_err_with("zensim", map_zensim_error)?;
        } else {
            let res = scorer
                .compute_with_ref_linear_planes_into(
                    &self.precomputed,
                    planes,
                    strides,
                    width as usize,
                    height as usize,
                    primaries,
                    &mut self.scratch,
                )
                .metric_err_with("zensim", map_zensim_error)?;
            opts.checkpoint()?;
            let borrowed = BorrowedEvaluation {
                score: res.score(),
                score_spec: &self.score_spec,
                diffmap: None,
                map_spec: &self.map_spec,
                width,
                height,
            };
            consumer(EvaluationRef::new(&borrowed));
        }
        Ok(())
    }
}
