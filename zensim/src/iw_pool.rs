//! Activity-weighted spatial pooling, motivated by Wang & Li 2011 IW-SSIM.
//! This module's estimators are proxies, not the paper's GSM information model.
//!
//! Replaces uniform spatial-mean pooling of per-pixel SSIM-like maps with
//! pooling weighted by the local information content of the reference
//! image. The motivation (from the source paper, IEEE TIP vol. 20 no. 5):
//! human observers concentrate attention on perceptually-informative
//! regions, so a metric that uniform-pools across the image dilutes
//! signal from salient regions with signal from flat / uninformative
//! regions. Weighting by per-region info content gives those regions more
//! influence on the aggregate score.
//!
//! ## Per-pixel weight estimator
//!
//! Wang 2011's exact form uses a Gaussian Scale Mixture model on per-band
//! wavelet coefficients. The standalone local-variance estimator below is a
//! cheaper activity proxy; it does not establish equivalent information content
//! or perceptual behavior. The production fused Rev3 944 path has its own
//! explicitly defined activity weights; these standalone defaults do not
//! describe that serving path.
//! Other candidate estimators (configurable via [`IwWeightKind`]):
//!
//! - [`IwWeightKind::LocalVariance`] (default) — variance in a 5×5 window.
//! - [`IwWeightKind::LocalGradL1`] — L1 norm of the gradient (∂x + ∂y).
//! - [`IwWeightKind::LocalGradL2`] — L2 norm of the gradient (√(∂x² + ∂y²)).
//!
//! All three return non-negative per-pixel weights. They are NOT
//! normalised here — the [`WeightedPool`] helper handles normalisation
//! at aggregate time.
//!
//! ## Status (2026-05-22)
//!
//! The IW-pool block shipped in the V_22-mix-LARGE+iwssim Balanced
//! recipe (PreviewV0_5Balanced) and contributes features
//! f300..f371. Implementation is fused into the SIMD streaming loop
//! via `streaming::process_scale_bands` and `process_strip_into_accum`;
//! the standalone helpers below are kept for offline experimentation
//! and for the legacy non-streaming code path.

use crate::det_math::DetRoots;

/// Choice of per-pixel info-content estimator.
// dead_code: the non-default estimators are research knobs selected through
// `IwWeightConfig` by offline/training experiments (see
// `benchmarks/iw_pyramid_spike_methodology_2026-05-15.md`); they are kept in
// every feature set so the config surface stays feature-stable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IwWeightKind {
    /// Local variance in a square window (kernel size set by config).
    /// Default. Closest to Wang 2011's GSM scale parameter direction.
    #[default]
    LocalVariance,
    /// L1 norm of the gradient — `|∂x I| + |∂y I|`. Cheaper than variance,
    /// emphasises edges.
    LocalGradL1,
    /// L2 norm of the gradient — `√((∂x I)² + (∂y I)²)`.
    LocalGradL2,
    /// **Directional-variance proxy inspired by steerable-pyramid GSM** (spike,
    /// 2026-05-15). Replaces the scalar spatial variance with a
    /// **directional max** across 4 oriented gradient orientations
    /// (0°, 45°, 90°, 135°). For each pixel, the local variance of
    /// each oriented gradient is computed in a Gaussian-weighted
    /// patch (default ~9×9 support); the per-pixel σ²_p is the max
    /// across orientations.
    ///
    /// This captures Wang & Li 2011's directional sensitivity at zeroth
    /// order — diagonal edges weigh differently from axis-aligned edges
    /// of the same total energy. It is an approximation to the paper's
    /// 5-level Simoncelli steerable pyramid; see
    /// `benchmarks/iw_pyramid_spike_methodology_2026-05-15.md` for the
    /// exact divergences.
    ///
    /// MUST be combined with `info_log_sigma_e_sq = Some(σ²_e)` to get
    /// the scalar `log₂(1 + σ²_p / σ²_e)` transform. The variant
    /// produces raw σ²_p; the log transform is applied at
    /// [`compute_iw_weights`] time.
    SteerablePyramidLogGsm,
}

/// IW weight computation parameters.
#[derive(Debug, Clone, Copy)]
pub struct IwWeightConfig {
    /// Which estimator to use.
    pub kind: IwWeightKind,
    /// Half-width of the local window for `LocalVariance`. Window size
    /// is `(2k+1) × (2k+1)` — `kernel_half = 2` gives a 5×5 window.
    /// Ignored for gradient estimators.
    pub kernel_half: usize,
    /// Floor added to all weights so flat regions still contribute
    /// non-zero signal. Without it the weighted mean is undefined
    /// when the reference is locally constant. Default: 1e-3 of the
    /// dynamic range.
    pub weight_floor: f32,
    /// When `Some(sigma_e_sq)`, apply Wang & Li 2011's information-
    /// content transform on the raw estimator output:
    ///
    ///   `w(x) ← log₂(1.0 + w_raw(x) / sigma_e_sq)`
    ///
    /// This scalar log-SNR transform is information-inspired. Substituting
    /// local variance for the paper's GSM/covariance model does not reproduce
    /// its full information estimator. The log reduces the influence of large
    /// activity values relative to direct variance weighting.
    ///
    /// `None` (default for back-compat with V_20a sweep) keeps the raw
    /// estimator output as the weight — variance / gradient magnitude
    /// directly. The default exists so existing experiments stay
    /// reproducible; new work should set `info_log_sigma_e_sq` for
    /// this explicitly different logarithmic activity proxy.
    ///
    /// Added 2026-05-15 after the V_20a IW-SSIM falsification revealed
    /// our implementation diverged from the paper's `log(1 + σ²_p/σ²_e)`
    /// weight formula. See `benchmarks/extended_iw_runtime_perf_2026-05-15.md`
    /// and the IW-paper-faithfulness analysis in the same message.
    pub info_log_sigma_e_sq: Option<f32>,
}

impl Default for IwWeightConfig {
    fn default() -> Self {
        Self {
            kind: IwWeightKind::LocalVariance,
            kernel_half: 2,
            weight_floor: 1.0e-3,
            info_log_sigma_e_sq: None,
        }
    }
}

enum GradNorm {
    L1,
    L2,
}

/// Weighted spatial pool. Mirrors the unweighted `mean`, `L2`, `L4`
/// pools used in the basic feature block, but each pixel's contribution
/// is multiplied by the corresponding weight from
/// [`compute_iw_weights`]. Normalisation divides by the sum of weights
/// (which equals N for unit weights, recovering the standard mean).
///
/// **`mean` established 2026-07-18 (iteration 1)** as THE canonical
/// `Σw·v/Σw` formula (Wang & Li 2011's IW-SSIM Eq.36 exactly) for v2
/// "bounded" masked/IW/soft-peak pooling — v1's shipped hot path
/// (`streaming.rs`, `1/n`-pooled) does NOT match this form and is
/// unaffected; see `benchmarks/iw_pooling_normalization_2026-07-15.md`
/// for that divergence.
///
/// **Iteration 2 update:** the v2 kernel (`feature_v2.rs`,
/// `compute_channel_scale_v2`) no longer CALLS `mean` directly — batching
/// per-pixel values into arrays to call it was measured to make v2 2-5x
/// slower than v1 (`docs/FEATURE_V2_SPEC_2026-07-18.md` §A.12). The kernel
/// now uses `feature_v2::WeightedSum`, an O(1)-space incremental
/// accumulator computing the IDENTICAL `Σw·v/Σw` formula online. `mean`
/// remains the canonical REFERENCE form — `feature_v2::tests::
/// weighted_sum_matches_weighted_pool_mean_exactly` pins the two bit-exact
/// on random data (the "gated mirror" exception to the no-duplication
/// policy: same formula, different computational strategy, tested
/// equivalent). `l2`/`l4` remain unused by v2 (kept for parity / future
/// use, individually `#[allow(dead_code)]` below).
pub(crate) struct WeightedPool;

impl WeightedPool {

}

/// Per-channel IW-pooled SSIM block — 6 features at
/// `f228+72*ch+0..f228+72*ch+5` for each scale (see `metric.rs`
/// `FEATURES_PER_CHANNEL_*_MASKED`). Mirrors the 6 "masked" features
/// in the existing extended profile but with the weight direction
/// inverted (high info content gets MORE weight, not less). Shipped
/// in the PreviewV0_5Balanced bake's f300..f371 input block.
///
/// **Crate-internal as of 0.3.0.** The streaming hot path emits the
/// same 6 numbers via `streaming::process_strip_into_accum`; this
/// type is kept for the offline-experiment entry point
/// [`Self::pool_from_maps`] and the iw_pool unit tests.
#[derive(Debug, Clone, Copy)]
pub(crate) struct IwSsimFeatures {
    /// Weighted mean of per-pixel SSIM(src, dst) at this scale & channel.
    pub iw_ssim_mean: f64,
    /// Weighted L2 of SSIM.
    pub iw_ssim_2nd: f64,
    /// Weighted L4 of SSIM (peak-emphasising, matches `ssim_4th`).
    pub iw_ssim_4th: f64,
    /// Weighted L4 of edge artifact map.
    pub iw_art_4th: f64,
    /// Weighted L4 of edge detail-lost map.
    pub iw_det_4th: f64,
    /// Weighted mean of (src-dst)².
    pub iw_mse: f64,
}

impl IwSsimFeatures {

}

#[cfg(test)]
mod tests {
    use super::*;

    // ─── Steerable-pyramid spike tests ──────────────────────────────────────

}
