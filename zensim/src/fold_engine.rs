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
//! * **Historical declared-HDR / PU-linear input.** Revisions 1–4 keep
//!   buffered. Rev5 `compute_pu_linear*` uses the PU fold with the declared
//!   bake plan and the revision's zero mean-offset metadata.
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
/// is the default for every profile except [`crate::profile::ZensimProfile::D`]
/// (`Zensim::new` opts `D` into `Fold` itself, since speed is its whole reason
/// to exist — `benchmarks/profile_d_notax_2026-09-01.md`). `feature-regime-v2`
/// is default-on as of 2026-09-01, so a plain `cargo add zensim` build CAN
/// name this type (it stays `#[doc(hidden)]`, not surfaced in rendered docs);
/// `--no-default-features` removes it and the module entirely, and every
/// profile including `D` then runs `Buffered` unconditionally — correctly,
/// just without the fold's speed.
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

pub(crate) fn compute_fold_backed_stoppable(
    source: &impl ImageSource,
    distorted: &impl ImageSource,
    config: &ZensimConfig,
    weights: &[f64],
    scratch: &mut V2Scratch,
    // The serving plan; `None` = today's unconditional v1 walk at
    // `V1PoolsMode::Full`. See `score_plan`.
    plan: Option<&crate::feature_plan::Plan>,
    stop: Option<&dyn enough::Stop>,
) -> Result<ZensimResult, ZensimError> {
    let (mut features, mean_offset) = crate::feature_v2::compute_folded_v1_372_streaming_stoppable(
        source,
        distorted,
        None,
        config.allow_multithreading,
        scratch,
        plan,
        stop,
    )?;
    // The fold emits its regime's width with `f372..` structurally zero. Keep
    // whichever is WIDER: the config's v1 width (a prefix, which the linear
    // tail reads) or the plan's WALK width (which a wide bake reads, and
    // which a DENSE layout's gather reads FROM — never the layout's own
    // packed width, which would truncate the walk before the gather could
    // reach the ids above it). Without a plan this is exactly today's
    // `truncate(v1_feature_width)`.
    //
    // REV4SERVE: `resize`, not `truncate` — a bake can declare an identity
    // input width beyond the plan's emitted regime width (the v2+basic
    // featpot cells are 1853-input rows whose live reads stop at f719).
    // Positions past the emit bound are uncomputed slots the bake provably
    // does not read (its live-read mask ends below them); zero is what the
    // extraction emits for unpopulated identity slots, so the score and the
    // visible row both stay honest.
    //
    // REV4SERVE review F1: before extending, verify the emitted vector
    // already covers everything the plan CLAIMS to populate — an emit the
    // walk did not materialize must surface loudly (debug panic, release
    // error), never as a zero-filled read slot.
    if let Some(p) = plan {
        debug_assert!(
            p.emit_covered(features.len()),
            "plan emit claims ids past the emitted vector (bound {} > {})",
            p.emit_bound(),
            features.len()
        );
        p.check_emit_covered(features.len())
            .map_err(|_| crate::ZensimError::ModelLoadFailed {
                reason: "bake plan emits features the fold walk did not materialize",
            })?;
    } else {
        debug_assert!(
            features.len() >= v1_feature_width(config),
            "v1 walk emitted fewer features than the config's width"
        );
        if features.len() < v1_feature_width(config) {
            return Err(crate::ZensimError::ModelForwardFailed {
                reason: "fold walk emitted fewer features than the config's v1 width",
            });
        }
    }
    let keep = plan
        .map_or(0, |p| p.walk_width())
        .max(v1_feature_width(config));
    features.resize(keep, 0.0);

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
    compute_fold_backed_with_ref_plan(
        precomputed,
        distorted,
        config,
        weights,
        scratch,
        pool_mode,
        None,
    )
}

pub(crate) fn compute_fold_backed_with_ref_plan(
    precomputed: &crate::streaming::PrecomputedReference,
    distorted: &impl ImageSource,
    config: &ZensimConfig,
    weights: &[f64],
    scratch: &mut V2Scratch,
    pool_mode: Option<crate::feature_v2::V1PoolsMode>,
    plan: Option<&crate::feature_plan::Plan>,
) -> Option<ZensimResult> {
    compute_fold_backed_with_ref_stoppable_plan(
        precomputed,
        distorted,
        config,
        weights,
        scratch,
        pool_mode,
        plan,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn compute_fold_backed_with_ref_stoppable_plan(
    precomputed: &crate::streaming::PrecomputedReference,
    distorted: &impl ImageSource,
    config: &ZensimConfig,
    weights: &[f64],
    scratch: &mut V2Scratch,
    pool_mode: Option<crate::feature_v2::V1PoolsMode>,
    plan: Option<&crate::feature_plan::Plan>,
    stop: Option<&dyn enough::Stop>,
) -> Option<ZensimResult> {
    let (mut features, mean_offset) = crate::feature_v2::compute_folded_v1_372_with_ref_stoppable(
        precomputed,
        distorted,
        config.allow_multithreading,
        scratch,
        pool_mode,
        plan,
        stop,
    )?;
    if let Some(plan) = plan {
        plan.check_emit_covered(features.len()).ok()?;
        features.resize(plan.walk_width().max(v1_feature_width(config)), 0.0);
    } else {
        features.truncate(v1_feature_width(config));
    }
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
}

/// The layer-0 column range each CALLER line occupies, in caller order —
/// `caller_col_spans()[k] = (start, end)` with `end == start` for a line the
/// bake dropped.
///
/// **Why this exists here and is not a duplicate.** A bake with a
/// variable-arity [`zenpredict::FeatureTransform`] has caller line `k` ≠
/// layer-0 column `k`: `Drop` (dead-column pruning, which
/// `bake_dial_refit pack` performs BY DEFAULT since 2026-08-04) consumes zero
/// columns, `Sinusoidal` consumes `2·N`. Slicing columns at caller boundaries
/// on such a bake is the caller-width bug class
/// `zensim-validate::block_profile` documents four instances of. That crate
/// folds per-column *values* back to caller lines; this needs only the
/// *spans*, and both are thin callers of
/// [`zenpredict::FeatureTransform::output_arity`], which is the single owner
/// of arity semantics. zensim cannot depend on zensim-validate (it is the
/// other way round), so the narrow primitive lives in the lower crate.
///
/// `None` when the arities do not tile layer 0 — a malformed bake, which the
/// caller must treat as "assume everything is read".
fn caller_col_spans(model: &crate::mlp::Model, in_dim: usize) -> Option<Vec<(usize, usize)>> {
    let Some(ts) = model.feature_transforms() else {
        // No transform table: caller line k IS column k.
        return (in_dim == model.n_inputs()).then(|| (0..in_dim).map(|i| (i, i + 1)).collect());
    };
    let params = model.feature_transform_params();
    if let Some(p) = params
        && p.len() != ts.len()
    {
        return None;
    }
    let mut out = Vec::with_capacity(ts.len());
    let mut cur = 0usize;
    for (k, t) in ts.iter().enumerate() {
        let pk: &[f32] = params.map(|p| p[k].as_slice()).unwrap_or(&[]);
        let end = cur + t.output_arity(pk);
        if end > in_dim {
            return None;
        }
        out.push((cur, end));
        cur = end;
    }
    (cur == in_dim).then_some(out)
}

/// **THE per-caller-line structural read predicate.** `caller_line_reads(m)[k]`
/// marks a nonzero layer-0 path, or a nonzero replacement min-max piece.
///
/// Extracted from [`bake_pool_need_from_model`]'s inner closure so that
/// [`crate::feature_plan::bake_read_slots`] reads the SAME predicate rather
/// than a second copy — the block-boundary question ("does it read the masked
/// family?") and the slot-set question ("exactly which lines does it read?")
/// are the same fold at two granularities, and a second copy is how the
/// caller-width bug class keeps recurring.
///
/// A dropped line spans no columns and so reads nothing, which is exactly
/// pruning's contract: a pruned column was already an exact zero, or a
/// transform-forced constant folded into the bias.
///
/// `None` for unreadable shapes or replacement-head contracts, which skip
/// callers must treat conservatively and serving admission must refuse.
pub(crate) fn caller_line_reads(model: &crate::mlp::Model) -> Option<Vec<bool>> {
    let layer = model.layer(0);
    let (in_dim, out_dim) = (layer.in_dim, layer.out_dim);
    let spans = caller_col_spans(model, in_dim)?;
    // This head replaces the network. Layer-zero zeros cannot prove its
    // inputs dead. Use the same parser as scoring, and require the scalar
    // head's one-to-one transform contract before making a skip decision.
    if let Some(entry) = model
        .metadata()
        .get(crate::bake_metadata::MINMAX_MONOTONE_HEAD_KEY)
    {
        let head = crate::bake_metadata::parse_minmax_head_meta(entry.value)?;
        if head.n != in_dim
            || spans.len() != head.n
            || spans
                .iter()
                .enumerate()
                .any(|(i, &span)| span != (i, i + 1))
        {
            return None;
        }
        return Some(
            (0..head.n)
                .map(|i| head.w.chunks_exact(head.n).any(|piece| piece[i] != 0.0))
                .collect(),
        );
    }
    Some(
        spans
            .iter()
            .map(|&(a, b)| {
                (a..b).any(|i| match &layer.weights {
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
            })
            .collect(),
    )
}

/// The **serving plan** for a profile: the pair `(compute, layout)` one walk
/// must run so that every one of the profile's bakes can be scored from it.
///
/// `None` means "today's path, unchanged" — the profile's widest bake fits the
/// v1 layout and the caller did not opt into pool skipping, so nothing about
/// the extraction needs to change. This is the byte-identity guarantee: the
/// narrow case does not merely produce the same numbers, it takes the same
/// code path.
///
/// `Some(plan)` is returned when the extraction must differ from that default
/// — because a bake declares a LAYOUT wider than the v1 walk can emit (the
/// servability case), or because pool skipping was requested (today's
/// `score_pool_mode` case, now expressed as one axis of a plan).
///
/// A profile can carry up to three scoring bakes and must serve all of them
/// from ONE extraction, so the plan is the UNION over them — never any single
/// bake's, which would under-compute for the others.
pub(crate) fn score_plan(
    params: &crate::profile::ProfileParams,
    config: &crate::metric::ZensimConfig,
    skip_unread: bool,
) -> Option<crate::feature_plan::Plan> {
    use crate::feature_plan::Plan;
    let v1_width = v1_feature_width(config);
    let mut widest = 0usize;
    let mut union: Option<Plan> = None;
    for bytes in params.scoring_bake_bytes() {
        let Ok(model) = crate::metric::cached_profile_model(bytes) else {
            // An unparseable bake is not a planning input: the scoring path
            // reports it on its own terms. Assume the widest safe plan.
            return None;
        };
        let Ok(p) = Plan::for_bake(&model) else {
            return None;
        };
        // The WALK width, not `caller_input_width()`. They are equal for
        // every identity-laid-out bake — i.e. every bake that exists today —
        // and differ for a DENSE one, whose 265 declared inputs still need a
        // walk that reaches f941. Comparing the caller width here sent a
        // dense bake down the "narrow, nothing to plan" shortcut below and
        // then gathered its ids out of a 372-wide vector, which is silently
        // wrong: measured, and the reason `dense_layout_round_trip` exists.
        widest = widest.max(p.walk_width());
        union = Some(match union {
            Some(u) => {
                // **REVISION IS PER BAKE, and one walk computes one era.** A
                // profile whose bakes declare different formula revisions
                // cannot be served from a single extraction; unioning them
                // would silently serve one bake the other's arithmetic, which
                // is exactly what the per-bake declaration exists to prevent.
                // Returning `None` sends the profile down the unplanned path,
                // which computes the process's revision for every bake and is
                // today's behaviour — and the loud half lives in
                // `mixed_revision_profiles_are_refused_by_the_planner`, which
                // pins that a mixed profile does NOT get a plan rather than
                // getting a quietly wrong one.
                if !u.revisions_agree(&p) {
                    return None;
                }
                u.union(&p)
            }
            None => p,
        });
    }
    // Nothing to plan for: a purely linear profile reads `f0..228` only, and
    // the existing `score_pool_mode` policy already covers it.
    let mut plan = union?;
    // A profile's linear tail reads the v1 prefix whatever its bakes declare, so
    // every v1 scale stays computed (the per-scale skip is for plans whose
    // readers are all declared, e.g. `BakeScorer`).
    if plan.compute.v1_scales != crate::feature_v2::ComputeSet::ALL_SCALES {
        plan.compute.v1_scales = crate::feature_v2::ComputeSet::ALL_SCALES;
        plan.emit = plan
            .compute
            .populated_slots(crate::NUM_SCALES, plan.layout_width());
    }
    if widest <= v1_width && !skip_unread {
        // The narrow, non-skipping case IS today's path. Returning `None`
        // keeps it on the identical code path rather than on an equivalent
        // one — which is a stronger guarantee than equal output.
        return None;
    }
    // Pool skipping is opt-in. Without it the walk computes the whole pool
    // block exactly as it does today, whatever the bakes read.
    if !skip_unread {
        plan.compute.coarse_y_only_scales = 0;
        plan.compute.local_only = false;
        plan.compute.omit_edges = false;
        plan.compute.v1_pools = crate::feature_v2::V1PoolsMode::Full;
        plan.compute.v1_full_scales = crate::feature_v2::ComputeSet::ALL_SCALES;
        plan.compute.v1_scales = crate::feature_v2::ComputeSet::ALL_SCALES;
        plan.emit = plan
            .compute
            .populated_slots(crate::NUM_SCALES, plan.layout_width());
    }
    // The emitted vector must be at least as wide as the linear tail reads.
    if plan.walk_width() < v1_width {
        // Widen the LAYOUT, not just a number: the linear tail reads the v1
        // prefix, so a plan narrower than it must be re-laid-out at that
        // width. Rebuilt through the constructor so `compute` and `emit`
        // stay the normalized fixed point rather than being patched in place.
        plan = crate::feature_plan::Plan::widened_to_identity(&plan, v1_width);
    }
    Some(plan)
}

#[cfg(test)]
mod skip_policy_tests {
    
    use crate::feature_v2::V1PoolsMode;
    

    /// **The `Off` is never served policy applies to the ID-SPACE branch too.**
    ///
    /// `pools_mode_for_need` documents (with a measured footprint argument)
    /// that `Off` is never the right answer for a SERVED v1 walk: it costs the
    /// same arithmetic as `Peaks` and grows the hot set by disabling the
    /// band-local self-blur. `Plan::for_bake` routes through that owner;
    /// `Plan::derive_with_layout` derives from touched families and will
    /// happily return `Off`. Nothing reached that branch until shipped `D`
    /// declared its 28 basic ids — at which point D's emitted vector went from
    /// real values at `f156..227` to zeros, score unmoved, footprint worse.
    #[test]
    fn a_dense_basic_only_bake_is_served_peaks_not_off() {
        let ids: Vec<usize> = (0..8).map(|i| i * 3).collect();
        let bytes = crate::feature_layout::tests::ids_bake_bytes(&ids);
        let m = crate::mlp::Model::from_bytes(&bytes).expect("synthetic dense bake parses");
        // The raw id-space derivation, with no policy applied: `Off`.
        let layout = crate::feature_layout::declared_layout(&m);
        let want = crate::feature_plan::bake_read_slots(&m).expect("read set");
        let raw = crate::feature_plan::Plan::derive_with_layout(&want, layout).expect("plan");
        assert_eq!(
            raw.compute.v1_pools,
            V1PoolsMode::Off,
            "the unfiltered id-space derivation must say Off, else the policy below is untested"
        );
        // The SERVING plan promotes it.
        let served = crate::feature_plan::Plan::for_bake(&m).expect("serving plan");
        assert_eq!(served.compute.v1_pools, V1PoolsMode::Peaks);
    }

}
