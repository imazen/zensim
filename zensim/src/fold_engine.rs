//! The fold-backed scoring engine — "the fold becomes the engine".
//!
//! zensim ships two walks over the same statistic:
//!
//! * **buffered** ([`crate::streaming::compute_multiscale_stats_streaming`]):
//!   whole-image XYB pyramids for both sides, band-processed per scale. This
//!   is what [`crate::Zensim::compute`] has always run.
//! * **the fold** ([`crate::feature_v2`]'s `foldapp_streaming_walk`): rolling
//!   planes, no materialised pyramid, O(strip) plane residency.
//!
//! Since option C (`blur::pyramid_plane_stride(w) == w`, 2026-08-30) the
//! fold's `f0..372` is **bit-identical** to v1's 372-feature extraction at
//! every width — gated by
//! `feature_v2::tests::v1_372_bit_exact_to_fold_at_every_width` over 19
//! geometries. What was still missing was everything between a feature vector
//! and a [`ZensimResult`]: the fold is an extractor, and
//! `ZensimV2Result` has no `score()`.
//!
//! This module is that missing piece, and it is deliberately thin. It does
//! **not** re-implement any scoring:
//!
//! | job | owner |
//! |---|---|
//! | sub-64 reflect-pad, byte-identical short-circuit | [`crate::metric`], shared, unchanged |
//! | extraction | the fold (`compute_folded_v1_372_streaming_impl`) |
//! | `mean_offset` | the fold, via `feature_v2::MeanOffsetRows` |
//! | sanitize + weight dot + score mapping | [`crate::metric::score_v1_layout_features`], shared |
//! | bake forward, α/hybrid head, tanh pin, PCHIP output spline, per-codec affine, clamp disposition | [`crate::metric::apply_mlp_scoring_with_codec`], untouched |
//!
//! The bake/spline machinery attaches *under* this module, not into it:
//! `apply_mlp_scoring_with_codec` consumes a finished `ZensimResult` plus the
//! original `(width, height)` and does not care which walk produced it. That
//! is the whole reason the fold-backed engine is additive rather than a
//! rewrite.
//!
//! ## What the engine does NOT cover, and why
//!
//! * **`num_scales != crate::NUM_SCALES`.** The fold is hard-wired to 4
//!   scales; a `ZensimConfig` (the `training`-feature surface) asking for a
//!   different pyramid falls back to buffered rather than silently scoring a
//!   different pyramid. At 4, `min_pyramid_dim_for_scales(4) ==
//!   MIN_PYRAMID_DIM == 64`, so the two paths' reflect-pad decisions coincide
//!   exactly.
//! * **Declared-HDR / PU-linear input.** `compute_pu_linear*` runs the PU
//!   front-end; the fold has a PU front-end too, but its mean-offset path is
//!   not wired, so those entries keep buffered.
//! * **Non-default blur config.** The fold implements `blur_radius = 5`,
//!   `blur_passes = 1` — every shipped profile's values. Anything else falls
//!   back.
//!
//! [`is_fold_backable`] is the single owner of that predicate.

use crate::error::ZensimError;
use crate::feature_v2::V2Scratch;
use crate::metric::{ZensimConfig, ZensimResult};
use crate::source::ImageSource;

/// Which walk [`crate::Zensim`]'s scoring entries run.
///
/// `#[doc(hidden)]`: this is an internal engine selector for the parity gates
/// and the perf comparison, not a product knob. [`ScoringEngine::Buffered`]
/// is the default and a build without `feature-regime-v2` cannot name this
/// type at all — so a default build's behaviour and public surface are
/// unchanged by its existence.
///
/// It is `pub` rather than `pub(crate)` for the same reason
/// `V2NewFeatureToggles::v1_only` is: the parity gate lives in
/// `tests/fold_engine_parity.rs`, which is a separate crate and cannot reach
/// `pub(crate)`.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ScoringEngine {
    /// The buffered walk — today's shipped path.
    #[default]
    Buffered,
    /// The streaming fold, where it can serve the request (see
    /// [`is_fold_backable`]); buffered otherwise.
    Fold,
}

/// **THE single owner of "can the fold serve this scoring request?"**
///
/// Returns `false` for every case §"What the engine does NOT cover" lists.
/// Callers route on this rather than on `engine == Fold` alone, so an
/// out-of-domain request degrades to buffered instead of scoring a different
/// quantity.
pub(crate) fn is_fold_backable(config: &ZensimConfig) -> bool {
    config.num_scales == crate::NUM_SCALES
        && config.blur_radius == 5
        && config.blur_passes == 1
        // WEIGHT-SKIPPING is the subtle one. `streaming::active_channels`
        // drops a channel whose basic+peak weights are all ~0 unless
        // `compute_all_features` or `extended_features` is set, leaving those
        // slots at their `ScaleStats` default. The fold has no such notion —
        // it always computes all three channels — so on a weight-skipping
        // config the two paths would emit the same SCORE (the dropped slots
        // carry zero weight) but DIFFERENT feature vectors. Requiring one of
        // the two all-active flags keeps the parity claim about the whole
        // `ZensimResult`, not just its score. Every MLP-scored profile sets
        // `compute_all_features` (it is `mlp_bytes.is_some()`), and every
        // `compute_extended_features` call sets `extended_features`; the
        // plain linear `PreviewV0_1`/`PreviewV0_2` `compute()` path is the
        // one this excludes, and it stays on buffered.
        && (config.compute_all_features || config.extended_features)
}

/// v1's feature width for a config — the same `(extended, iw)` table
/// [`crate::metric::combine_scores`] assembles to, and the same one
/// `identical_result` uses.
///
/// v1's layout is `[0,156) basic · [156,228) peaks · [228,300) masked ·
/// [300,372) IW`, so a narrower config is a strict **prefix** of a wider one
/// and truncation is the whole conversion from the fold's 372. That claim is
/// gated (`fold_narrow_config_is_a_prefix_of_wide`), not assumed.
pub(crate) fn v1_feature_width(config: &ZensimConfig) -> usize {
    let fpc = match (config.extended_features, config.compute_iw_features) {
        (true, true) => {
            crate::metric::FEATURES_PER_CHANNEL_EXTENDED + crate::metric::FEATURES_PER_CHANNEL_IW
        }
        (true, false) => crate::metric::FEATURES_PER_CHANNEL_EXTENDED,
        (false, _) => crate::metric::FEATURES_PER_CHANNEL_WITH_PEAKS,
    };
    config.num_scales * 3 * fpc
}

/// Fold-backed replacement for
/// [`crate::streaming::compute_zensim_streaming_stoppable`]: same inputs,
/// same `ZensimResult`, different walk.
///
/// The caller ([`crate::metric::compute_with_config_inner`]) has already
/// applied the shared reflect-pad and byte-identical short-circuit, so this
/// sees exactly what the buffered walk would.
pub(crate) fn compute_fold_backed(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    config: &ZensimConfig,
    weights: &[f64],
    scratch: &mut V2Scratch,
) -> Result<ZensimResult, ZensimError> {
    let (mut features, mean_offset) = crate::feature_v2::compute_folded_v1_372_streaming_impl(
        source,
        distorted,
        None,
        config.allow_multithreading,
        scratch,
    )?;
    // The fold emits its regime's width (720) with `f372..` structurally
    // zero; the config's v1 width is a prefix of that.
    features.truncate(v1_feature_width(config));

    let (score, raw_distance) =
        crate::metric::score_v1_layout_features(&mut features, weights, config, config.num_scales);

    Ok(ZensimResult::new(
        score,
        raw_distance,
        features,
        // Placeholder profile tag, matching `combine_scores` — every
        // `Zensim::compute*` caller overrides it via `with_profile`.
        crate::profile::ZensimProfile::codec_target(),
        mean_offset,
    ))
}

/// Fold-backed replacement for
/// [`crate::streaming::compute_zensim_streaming_with_ref`] — the M1
/// precompute-once / compare-many shape, on the fold.
///
/// The reference's XYB pyramid IS what the fold's source side needs (the
/// producer fills scale 0 with `convert_source_to_xyb_into_slices` and
/// cascades `downscale_2x_into`, the same two functions that built the
/// cache), so no new reference type exists: `PrecomputedReference` serves
/// both engines. Returns `None` when the cache cannot feed the fold —
/// `feature_v2::cached_ref_feed_usable` owns that predicate — and the caller
/// falls back to the buffered `*_with_ref` walk rather than scoring a
/// mismatched pyramid.
pub(crate) fn compute_fold_backed_with_ref(
    precomputed: &crate::streaming::PrecomputedReference,
    distorted: &impl ImageSource,
    config: &ZensimConfig,
    weights: &[f64],
    scratch: &mut V2Scratch,
) -> Option<ZensimResult> {
    let (mut features, mean_offset) = crate::feature_v2::compute_folded_v1_372_with_ref_impl(
        precomputed,
        distorted,
        config.allow_multithreading,
        scratch,
    )?;
    features.truncate(v1_feature_width(config));
    let (score, raw_distance) =
        crate::metric::score_v1_layout_features(&mut features, weights, config, config.num_scales);
    Some(ZensimResult::new(
        score,
        raw_distance,
        features,
        crate::profile::ZensimProfile::codec_target(),
        mean_offset,
    ))
}
