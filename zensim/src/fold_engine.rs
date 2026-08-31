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
    // Per-profile weight-skipping; `None` = `V1PoolsMode::Full`, today's
    // unconditional behaviour. See `score_pool_mode`.
    pool_mode: Option<crate::feature_v2::V1PoolsMode>,
) -> Result<ZensimResult, ZensimError> {
    let (mut features, mean_offset) = crate::feature_v2::compute_folded_v1_372_streaming_impl(
        source,
        distorted,
        None,
        config.allow_multithreading,
        scratch,
        pool_mode,
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
    pool_mode: Option<crate::feature_v2::V1PoolsMode>,
) -> Option<ZensimResult> {
    let (mut features, mean_offset) = crate::feature_v2::compute_folded_v1_372_with_ref_impl(
        precomputed,
        distorted,
        config.allow_multithreading,
        scratch,
        pool_mode,
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

// ============================================================================
// Per-profile weight-skipping: computing only the v1 pool families a bake reads
// ============================================================================

/// Which of v1's three pool families (`f156..228` peaks, `f228..300` masked,
/// `f300..372` IW) a bake's layer 0 **structurally** reads — the compute-side
/// complement of `zensim-validate`'s `bake_block_profile`.
///
/// "Structurally" means the L∞ norm over that caller line's layer-0 output
/// weights is exactly zero, which is the same test the tooling uses and the
/// only one that is safe here: a zero weight makes the slot's value
/// unreachable by the forward pass for EVERY input, so leaving the slot at
/// 0.0 cannot move the score by a single ULP. A merely *small* weight is not
/// skippable and is never treated as one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct V1PoolNeed {
    pub(crate) peaks: bool,
    pub(crate) masked: bool,
    pub(crate) iw: bool,
}

impl V1PoolNeed {
    /// The conservative answer: assume every family is read. Returned for
    /// anything this module declines to analyse.
    pub(crate) const ALL: Self = Self {
        peaks: true,
        masked: true,
        iw: true,
    };
}

/// v1's 372 block boundaries, in caller feature indices.
const V1_PEAKS: (usize, usize) = (156, 228);
const V1_MASKED: (usize, usize) = (228, 300);
const V1_IW: (usize, usize) = (300, 372);

/// Layer-0 structural read-set of one bake, or [`V1PoolNeed::ALL`] when the
/// bake cannot be analysed safely.
///
/// **Declines (returns `ALL`) on transform-arity divergence.** When
/// `caller_input_width() != n_inputs()` the bake declares a variable-arity
/// [`zenpredict::FeatureTransform`] (`Drop` from dead-column pruning, or an
/// expander), so layer-0 column `k` is NOT caller line `k` and slicing the
/// columns at caller boundaries is the caller-width bug class
/// `zensim-validate::block_profile` documents. Rather than re-implement that
/// crate's arity walk here — one owner per task — this refuses the case. It
/// costs nothing today: pruning only ever removes columns that were already
/// exact zeros, so a pruned bake's parent gives the same answer, and no
/// shipped profile carries an expander.
fn bake_pool_need(bytes: &[u8]) -> V1PoolNeed {
    let Ok(model) = crate::mlp::Model::from_bytes(bytes) else {
        return V1PoolNeed::ALL;
    };
    let layer = model.layer(0);
    let (in_dim, out_dim) = (layer.in_dim, layer.out_dim);
    if model.caller_input_width() != model.n_inputs() || in_dim != model.n_inputs() {
        return V1PoolNeed::ALL;
    }
    // Any caller line at or beyond a family's start that carries a nonzero
    // weight marks that family read. A bake narrower than a family's start
    // cannot read it at all.
    let reads = |lo: usize, hi: usize| -> bool {
        let hi = hi.min(in_dim);
        if lo >= hi {
            return false;
        }
        (lo..hi).any(|i| match &layer.weights {
            crate::mlp::WeightStorage::F32(w) => {
                w[i * out_dim..(i + 1) * out_dim].iter().any(|&v| v != 0.0)
            }
            crate::mlp::WeightStorage::F16(w) => w[i * out_dim..(i + 1) * out_dim]
                .iter()
                .any(|&h| crate::mlp::f16_bits_to_f32(h) != 0.0),
            crate::mlp::WeightStorage::I8 { weights, scales } => weights
                [i * out_dim..(i + 1) * out_dim]
                .iter()
                .zip(scales.iter())
                .any(|(&q, &s)| q != 0 && s != 0.0),
        })
    };
    V1PoolNeed {
        peaks: reads(V1_PEAKS.0, V1_PEAKS.1),
        masked: reads(V1_MASKED.0, V1_MASKED.1),
        iw: reads(V1_IW.0, V1_IW.1),
    }
}

/// [`bake_pool_need`], interned by bake-bytes data pointer — the same cache
/// shape (and the same stability argument) as
/// `metric::cached_bake_metadata`: bake slices always come from `&'static`
/// slots via `ProfileParams::mlp_bytes`, so the pointer is a stable unique
/// key per slot. Parsing a bake per compare would defeat the point of the
/// skip.
pub(crate) fn cached_bake_pool_need(bytes: &[u8]) -> V1PoolNeed {
    use std::collections::HashMap;
    use std::sync::{OnceLock, RwLock};
    static CACHE: OnceLock<RwLock<HashMap<usize, V1PoolNeed>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    let key = bytes.as_ptr() as usize;
    if let Ok(read) = cache.read()
        && let Some(v) = read.get(&key)
    {
        return *v;
    }
    let need = bake_pool_need(bytes);
    if let Ok(mut write) = cache.write() {
        write.insert(key, need);
    }
    need
}

/// The pool mode a fold-backed SCORE may run for this profile.
///
/// Returns [`V1PoolsMode::Full`] — today's unconditional behaviour — unless
/// the caller opted into skipping AND **every** consumer of the feature
/// vector structurally ignores `f228..372`:
///
/// * the profile's linear `weights`, which `metric::score_v1_layout_features`
///   reads only over `[0, num_scales·3·FEATURES_PER_CHANNEL_WITH_PEAKS)` =
///   `f0..228` — masked/IW are structurally out of its reach, so it never
///   constrains the decision (asserted below rather than assumed);
/// * `mlp_bytes`, `mlp_bytes_b3` and `ensemble_classifier_bytes`, each read
///   through [`cached_bake_pool_need`].
///
/// The peaks family is never skipped, because skipping it saves nothing: its
/// accumulators are produced unconditionally by the fused V-blur kernel (see
/// [`V1PoolsMode::Peaks`]).
pub(crate) fn score_pool_mode(
    params: &crate::profile::ProfileParams,
    config: &crate::metric::ZensimConfig,
    skip_unread: bool,
) -> crate::feature_v2::V1PoolsMode {
    use crate::feature_v2::V1PoolsMode;
    if !skip_unread {
        return V1PoolsMode::Full;
    }
    // The linear tail's reach. If a future config ever widened it past the
    // peak block this decision would be unsound, so it is checked, not
    // assumed.
    if config.num_scales * 3 * crate::metric::FEATURES_PER_CHANNEL_WITH_PEAKS > V1_PEAKS.1 {
        return V1PoolsMode::Full;
    }
    let mut need = V1PoolNeed {
        peaks: false,
        masked: false,
        iw: false,
    };
    let mut any_bake = false;
    for bytes in params.scoring_bake_bytes() {
        any_bake = true;
        let n = cached_bake_pool_need(bytes);
        need.peaks |= n.peaks;
        need.masked |= n.masked;
        need.iw |= n.iw;
    }
    if !any_bake {
        // A purely linear profile reads f0..228 only, so masked/IW are
        // unread by construction.
        return V1PoolsMode::Peaks;
    }
    if need.masked || need.iw {
        V1PoolsMode::Full
    } else {
        V1PoolsMode::Peaks
    }
}

#[cfg(test)]
mod skip_policy_tests {
    use super::*;
    use crate::feature_v2::V1PoolsMode;
    use crate::profile::ZensimProfile;

    fn cfg_for(p: ZensimProfile) -> (&'static crate::profile::ProfileParams, ZensimConfig) {
        let params = p.params();
        let config = crate::metric::config_from_params(params, false);
        (params, config)
    }

    /// Every SHIPPED profile reads the masked/IW block, so the policy must
    /// resolve to `Full` — i.e. opting into skipping changes nothing for
    /// them. This is the "no silent behaviour change" half of the contract,
    /// stated as a fact about the bakes rather than as an intention.
    #[test]
    fn shipped_profiles_read_the_pool_block_so_skipping_is_a_no_op() {
        for p in [ZensimProfile::B] {
            let (params, config) = cfg_for(p);
            assert_eq!(
                score_pool_mode(params, &config, true),
                V1PoolsMode::Full,
                "{p} resolved to a skipping mode — its bake reads f228..372"
            );
            assert_eq!(score_pool_mode(params, &config, false), V1PoolsMode::Full);
        }
    }

    /// The skip is opt-in: with `skip_unread == false` the answer is `Full`
    /// no matter what the bake looks like.
    #[test]
    fn skipping_is_off_by_default() {
        let z = crate::Zensim::new(ZensimProfile::B);
        assert_eq!(z.score_pool_mode(), V1PoolsMode::Full);
        assert_eq!(
            z.clone().with_unread_feature_skipping(true).score_pool_mode(),
            V1PoolsMode::Full,
            "B reads the block, so even opted-in it must compute it"
        );
    }

    /// The read-set derivation itself, against the shipped B bake — the
    /// numbers `zensim-validate`'s `bake_block_profile` reports for the same
    /// bytes (v1_peaks 26 used / v1_masked 10 / v1_iw 13, 2026-08-31).
    #[test]
    fn bake_pool_need_matches_the_block_profile_tool_on_shipped_b() {
        let params = ZensimProfile::B.params();
        let bytes: Vec<&'static [u8]> = params.scoring_bake_bytes().collect();
        assert_eq!(bytes.len(), 1, "B forwards exactly one bake");
        let need = cached_bake_pool_need(bytes[0]);
        assert_eq!(
            need,
            V1PoolNeed {
                peaks: true,
                masked: true,
                iw: true
            }
        );
        // Interned: the second call must agree with the first.
        assert_eq!(cached_bake_pool_need(bytes[0]), need);
    }

    /// **The end-to-end score-safety gate.** When the skip DOES fire, the
    /// linear scoring tail must not notice: `metric::score_v1_layout_features`
    /// reads `features[..num_scales·3·FEATURES_PER_CHANNEL_WITH_PEAKS]` =
    /// `f0..228`, which `V1PoolsMode::Peaks` emits bit-identically. Forced
    /// here rather than driven through a profile, because no SHIPPED bake
    /// ignores the masked/IW block — so this is the only way to exercise the
    /// firing path until one does.
    #[cfg(all(feature = "training", feature = "threads"))]
    #[test]
    fn a_fired_skip_leaves_raw_distance_bit_identical() {
        use crate::feature_v2::V2Scratch;
        use crate::source::RgbSlice;
        let params = ZensimProfile::B.params();
        for &(w, h) in &[
            (96usize, 64usize),
            (200, 150),
            (127, 93),
            (256, 256),
            (577, 385),
        ] {
            for &parallel in &[false, true] {
                let mut config = crate::metric::config_from_params(params, parallel);
                config.allow_multithreading = parallel;
                let src = crate::feature_v2::tests::textured_image(w, h, 7);
                let dst = crate::feature_v2::tests::quantize_distort(&src, w, h);
                let (sref, dref) = (RgbSlice::new(&src, w, h), RgbSlice::new(&dst, w, h));
                let mut scratch = V2Scratch::new();
                let full = compute_fold_backed(
                    &sref,
                    &dref,
                    &config,
                    params.weights,
                    &mut scratch,
                    Some(V1PoolsMode::Full),
                )
                .expect("full");
                let peaks = compute_fold_backed(
                    &sref,
                    &dref,
                    &config,
                    params.weights,
                    &mut scratch,
                    Some(V1PoolsMode::Peaks),
                )
                .expect("peaks");
                assert_eq!(
                    full.raw_distance().to_bits(),
                    peaks.raw_distance().to_bits(),
                    "{w}x{h} par={parallel}: raw_distance moved ({:e} vs {:e})",
                    full.raw_distance(),
                    peaks.raw_distance()
                );
                assert_eq!(
                    full.score().to_bits(),
                    peaks.score().to_bits(),
                    "{w}x{h} par={parallel}: linear score moved"
                );
                assert_eq!(full.mean_offset(), peaks.mean_offset());
                let (ff, fp) = (full.features(), peaks.features());
                assert_eq!(ff.len(), fp.len());
                for i in 0..228 {
                    assert_eq!(
                        ff[i].to_bits(),
                        fp[i].to_bits(),
                        "{w}x{h} par={parallel}: scored slot {i} moved"
                    );
                }
                assert!(fp[228..372].iter().all(|&v| v == 0.0));
                // …and the skip is genuinely doing something: Full's block
                // is not all-zero on a distorted pair.
                assert!(ff[228..372].iter().any(|&v| v != 0.0));
            }
        }
    }

    /// A bake this module declines to analyse must come back as `ALL`, never
    /// as an optimistic "nothing is read" — the failure mode that would
    /// silently zero a live slot.
    #[test]
    fn an_unparseable_bake_needs_everything() {
        assert_eq!(bake_pool_need(&[0u8; 8]), V1PoolNeed::ALL);
        assert_eq!(bake_pool_need(&[]), V1PoolNeed::ALL);
    }
}
