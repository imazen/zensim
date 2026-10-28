//! **The extraction PLAN** — one derivation from "what a consumer needs" to
//! "what the walk must run and how wide the answer is".
//!
//! Design: `docs/FEATURE_SYSTEM_DESIGN_2026-09-05.md` §5. Phases + gates:
//! `docs/PLAN_FEATURE_SYSTEM_2026-09-05.md`.
//!
//! Declared feature IDs and the formula revision determine one compute/layout
//! plan for dense and legacy identity bakes. The previous block-profile
//! fallback is retired; it enabled a BANDVIS variant absent from training.
//! The September 7 canonical-pixel and serving-census gates cover the change.
//!
//! ## Universal servability
//!
//! The contract (user directive, 2026-09-05: *"also make sure everything can
//! be served"*): **every bake whose read set consists of registered feature
//! ids at a supported revision is servable**, in any declared layout. There is
//! no "trains fine, cannot be served" class. A genuinely unregistered read is
//! refused LOUDLY, by [`PlanError`], naming the slots — never served as
//! silent zeros.

use crate::feature_layout::Layout;
use crate::feature_set_id::{ComputeToken, SlotSet};
use crate::feature_v2::{ComputeSet, V1FreeExtras, V1PoolsMode};

/// Why a request cannot be planned.
///
/// Always names the actionable detail. A refusal that says only "mismatch" is
/// what `metric::prep_bake_input_f32`'s `"bake declares more input features
/// than the caller supplied"` was, and it is why the servability gap read as
/// a wall rather than as a list of slots.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum PlanError {
    /// The request asks for slots no registered block populates at this
    /// layout width.
    Uncomputable {
        /// The slots that cannot be produced.
        missing: SlotSet,
        /// The layout width the request declared.
        layout_width: usize,
    },
    /// The bake's layer-0 arities do not tile its input width, so its read
    /// set cannot be derived. A malformed bake, not an unservable one.
    UnreadableBake,
    /// The request asks for slots outside the families
    /// [`crate::feature_defs::FormulaRevision::Rev5`] computes.
    ///
    /// Rev5's registered scope is `basic + peaks + v2`
    /// (`benchmarks/rev5_spec_2026-10-04.md` §1): a plan whose read set
    /// leaves them is refused at plan time rather than computed at a lower
    /// revision behind the caller's back.
    UnsupportedAtRev5 {
        /// The requested slots outside `basic + peaks + v2`.
        extra: SlotSet,
    },
}

impl core::fmt::Display for PlanError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            PlanError::Uncomputable {
                missing,
                layout_width,
            } => write!(
                f,
                "no registered block populates slot(s) {missing} at layout \
                 width {layout_width}"
            ),
            PlanError::UnreadableBake => {
                f.write_str("bake layer-0 arities do not tile its input width")
            }
            PlanError::UnsupportedAtRev5 { extra } => write!(
                f,
                "formula revisions 5 and later compute only the basic + peaks \
                 + v2 families (and globaff from revision 6); requested \
                 slot(s) {extra} are outside them"
            ),
        }
    }
}

/// The slots outside Rev5's two-family scope in `want`, if `revision` is
/// Rev5 or later — `None` below Rev5 and when the request is in scope.
///
/// ONE derivation for the refusal [`Plan::derive_with_layout`] and
/// [`Plan::for_bake`] share: the compute-side clamp that makes the same
/// scope hold for layouts wider than the request lives in
/// [`crate::feature_v2::ComputeSet::rev5_scope`].
fn rev5_unsupported(
    want: &SlotSet,
    layout_width: usize,
    revision: crate::feature_defs::FormulaRevision,
) -> Option<SlotSet> {
    if revision < crate::feature_defs::FormulaRevision::Rev5 {
        return None;
    }
    let ns = crate::NUM_SCALES;
    let mut supported = crate::feature_defs::family_slots(ComputeToken::Basic, ns)
        .union(&crate::feature_defs::family_slots(ComputeToken::Peaks, ns))
        .union(&crate::feature_defs::family_slots(ComputeToken::V2, ns));
    // Revision 6 adds the globaff family to the scope.
    if revision.computes_globaff() {
        supported = supported.union(&crate::feature_defs::family_slots(
            ComputeToken::GlobAff,
            ns,
        ));
    }
    let supported = supported.clipped_to(layout_width);
    // `a.missing_from(&b)` is b∖a — so this is want∖supported, the
    // requested slots Rev5 does not compute.
    let extra = supported.missing_from(want);
    (!extra.is_empty()).then_some(extra)
}

/// A resolved extraction plan: what to compute, how wide to emit, and which
/// slots will actually carry a value.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Plan {
    /// Which kernels/blocks run.
    pub compute: ComputeSet,
    /// **The LAYOUT** — the declared id→position mapping of the emitted
    /// vector, not merely its width. A `v1_only` request at `w944` is still a
    /// 944-wide row with `f372..` at the structural `0.0`; a `dense265`
    /// request over the same compute set is 265 wide with no gaps and the
    /// SAME ids.
    ///
    /// Was a bare `layout_width: usize` through phase 1, which recorded WHY
    /// (every registered layout was the identity mapping, so a map had no
    /// consumer). [`Layout::dense`] is that consumer.
    pub layout: Layout,
    /// The FEATURE IDS this plan populates — id space, not position space.
    /// Everything else in the layout is a structural fill.
    pub emit: SlotSet,
}

impl Plan {
    /// The emitted vector width — the LAYOUT's width.
    pub(crate) fn layout_width(&self) -> usize {
        self.layout.width()
    }

    /// The IDENTITY width the walk must emit for this plan's layout to be
    /// fillable: one past the highest id the layout carries.
    ///
    /// Equal to [`Self::layout_width`] for every identity layout — i.e. for
    /// every artifact that exists today — and strictly larger for a dense
    /// one: `dense265` over `basic+peaks+moments` is 265 wide and still needs
    /// a walk that reaches f941.
    ///
    /// **The scoring path must size its feature vector by THIS, not by
    /// `layout_width`.** The vector it produces is identity-laid-out (the
    /// linear tail reads the v1 prefix positionally); the LAYOUT is applied
    /// at the bake boundary, by `metric::forward_one_bake_with_codec`. Sizing
    /// by the layout width instead truncates the walk before the gather can
    /// reach the ids above it — measured, and the reason
    /// `dense_layout_round_trip` exists.
    pub(crate) fn walk_width(&self) -> usize {
        self.layout.walk_width()
    }

    /// Plan for an explicit slot request at an explicit layout width.
    ///
    /// The derivation is one rule applied per family: a family runs iff the
    /// request touches a slot it owns. The one subtlety is the free tranches —
    /// a request that touches the append or append2 block ONLY at tranche
    /// slots is served by the free accumulators instead of the (expensive)
    /// owning kernel, which is what makes `basic+peaks+moments` a cheap plan
    /// rather than a full 944 walk.
    pub(crate) fn derive(want: &SlotSet, layout_width: usize) -> Result<Plan, PlanError> {
        Plan::derive_with_layout(want, Layout::identity(layout_width))
    }

    /// Plan for an explicit slot request into an explicit [`Layout`] — the
    /// general form. [`Plan::derive`] is this with an identity layout, which
    /// is every artifact that exists today.
    pub(crate) fn derive_with_layout(want: &SlotSet, layout: Layout) -> Result<Plan, PlanError> {
        Plan::derive_with_layout_at(want, layout, crate::ssim_form::active_revision())
    }

    /// [`Plan::derive_with_layout`] at an explicit formula revision rather
    /// than the process's: the scope check, and the normalization that drops
    /// layout-only families, use `revision`. A revision-6 bake's read set
    /// reaches f1890, so its layout spans every optional block; deriving at
    /// the process revision would treat those blocks as computed and force the
    /// scale-zero chroma kernels on.
    pub(crate) fn derive_with_layout_at(
        want: &SlotSet,
        layout: Layout,
        revision: crate::feature_defs::FormulaRevision,
    ) -> Result<Plan, PlanError> {
        let ns = crate::NUM_SCALES;
        // COMPUTE is decided in ID space against the layout's WALK width (one
        // past the highest id it carries), never against its emitted width —
        // a `dense265` layout over `basic+peaks+moments` is 265 wide and
        // still needs a walk that reaches f941.
        let layout_width = layout.walk_width();
        let want = want.clipped_to(layout_width);
        // rev5 scope (spec §1): the request's slots must live inside
        // `basic + peaks + v2` — the families Rev5 computes. Anything else
        // is a loud plan-time error, never a silent lower-revision value.
        if let Some(extra) = rev5_unsupported(&want, layout_width, revision) {
            return Err(PlanError::UnsupportedAtRev5 { extra });
        }
        let touches = |t: ComputeToken| -> bool {
            !crate::feature_defs::family_slots(t, ns)
                .clipped_to(layout_width)
                .intersect(&want)
                .is_empty()
        };
        // v1 pool families: v1's pool modes turn masked and IW on together
        // (they share one activity chain, one sigma store and three
        // `*_inline_both` kernels), so the only compute boundary inside
        // `f156..372` is peaks vs masked-and-IW.
        let v1_pools = if touches(ComputeToken::Masked) || touches(ComputeToken::Iw) {
            V1PoolsMode::Full
        } else if touches(ComputeToken::Peaks) || touches(ComputeToken::Carriers) {
            V1PoolsMode::Peaks
        } else {
            V1PoolsMode::Off
        };

        // Does the request touch a block OUTSIDE its free tranche? If not,
        // the free accumulators serve it and the (expensive) owning kernel
        // stays off. This applies to the v2 dense block too, not just the
        // append pair: class C's 24 slots include the v2-348 `MSE` slot at
        // every (scale, channel), so a `+classC` request TOUCHES the v2 block
        // while needing none of its kernels.
        let outside_tranche = |t: ComputeToken| -> bool {
            let fam = crate::feature_defs::family_slots(t, ns).clipped_to(layout_width);
            let free = crate::feature_defs::family_slots(ComputeToken::Moments, ns)
                .union(&crate::feature_defs::family_slots(ComputeToken::ClassC, ns))
                .clipped_to(layout_width);
            let touched = fam.intersect(&want);
            !SlotSet::from_slots(touched.iter_slots().filter(|s| !free.contains(*s))).is_empty()
        };
        let append = touches(ComputeToken::Append) && outside_tranche(ComputeToken::Append);
        let append2 = touches(ComputeToken::Append2) && outside_tranche(ComputeToken::Append2);
        // append2 requires append (the block sits at f924+ and reuses append
        // accumulators — `V2NewFeatureToggles::append2_block` asserts it).
        let append = append || append2;
        let csfw = touches(ComputeToken::Csfw);
        let gridblk = touches(ComputeToken::Gridblk);
        let ringbasis = touches(ComputeToken::Ringbasis);
        let tailhist = touches(ComputeToken::Tailhist);
        let arttype = touches(ComputeToken::Arttype);
        let gmsbank = touches(ComputeToken::Gmsbank);
        let mapdev = touches(ComputeToken::Mapdev);
        let z1max = touches(ComputeToken::Z1max);
        let gmsnative = touches(ComputeToken::Gmsnative);
        // `globaff` reads every channel's planes at every scale but runs no
        // kernel: its slots are kept out of the channel and scale derivations
        // below (`kernel_want`), so asking for it never turns on X/B kernels.
        let globaff = touches(ComputeToken::GlobAff);
        let kernel_want =
            crate::feature_defs::family_slots(ComputeToken::GlobAff, ns).missing_from(&want);
        // `csfw_on` is `csfw_block && v2_blocks` in the walk, so a CSFW
        // request implies the v2-era pass regardless of what else is asked.
        // The four Rev4 feature-bank families are `rev4_* && v2_blocks` too.
        let v2_blocks = (touches(ComputeToken::V2) && outside_tranche(ComputeToken::V2))
            || append
            || append2
            || csfw
            || gridblk
            || ringbasis
            || tailhist
            || arttype
            || gmsbank
            || mapdev
            || z1max
            || gmsnative;

        // Free extras: only meaningful when the owning block is NOT running.
        let free_extras = if touches(ComputeToken::ClassC) && !append {
            V1FreeExtras::RawMomentsPlusBoundedErr
        } else if touches(ComputeToken::Moments) && !append {
            V1FreeExtras::RawMoments
        } else {
            V1FreeExtras::Off
        };

        let requested = ComputeSet {
            // A slot-set request carries no bake, so it takes the process's
            // revision. A BAKE-driven plan takes the bake's own — see
            // `Plan::for_bake`.
            formula_revision: revision,
            v1_basic: touches(ComputeToken::Basic) || v1_pools != V1PoolsMode::Off,
            full_res_xb: true,
            coarse_y_only_scales: 0,
            local_only: false,
            omit_edges: false,
            sampling: None,
            v1_pools,
            v1_full_scales: if v1_pools == V1PoolsMode::Full {
                let weighted = crate::feature_defs::family_slots(ComputeToken::Masked, ns)
                    .union(&crate::feature_defs::family_slots(ComputeToken::Iw, ns));
                weighted.intersect(&want).iter_slots().fold(0, |mask, id| {
                    mask | (1
                        << crate::feature_defs::def_at(id, ns)
                            .expect("registered pool ID")
                            .scale)
                })
            } else {
                ComputeSet::ALL_SCALES
            },
            // Scales the v1 fold must run at: every v1-era read, plus every read of the free tranche (raw moments,
            // class C), which the v1 fold's accumulators produce at their own scale.
            v1_scales: {
                let free = crate::feature_defs::family_slots(ComputeToken::Moments, ns)
                    .union(&crate::feature_defs::family_slots(ComputeToken::ClassC, ns));
                want.iter_slots()
                    .filter(|&id| id < 372 || free.contains(id))
                    .filter_map(|id| crate::feature_defs::def_at(id, ns))
                    .fold(0, |mask, d| mask | (1 << d.scale))
            },
            v2_scales: want
                .iter_slots()
                .filter_map(|id| {
                    let d = crate::feature_defs::def_at(id, ns)?;
                    (id >= 372).then_some(d)
                })
                .fold(0, |mask, d| {
                    let bit = 1 << d.scale;
                    // `edge_width_change` at scale s reads the gradient sums
                    // of s and s+1 — and arttype's `blur` multiplies that
                    // same finished slot, so it inherits the dependency.
                    if d.signal.name == "edge_width_change"
                        || (d.signal.family == ComputeToken::Arttype && d.signal.name == "blur")
                    {
                        let s = usize::from(d.scale).min(ns - 2);
                        mask | (1 << s) | (1 << (s + 1))
                    } else {
                        mask | bit
                    }
                }),
            v2_blocks,
            gradient: v2_blocks,
            blockiness: v2_blocks,
            transducer_bank: v2_blocks,
            transducers_luma_only: false,
            omit_pjnd_fragility: v2_blocks
                && !want.iter_slots().any(|id| {
                    crate::feature_defs::def_at(id, ns).is_some_and(|d| {
                        d.signal.family == ComputeToken::V2 && d.signal.name == "pjnd_fragility"
                    })
                }),
            free_extras,
        };
        let mut requested = requested;
        if !requested.v2_blocks {
            requested.v2_scales = ComputeSet::ALL_SCALES;
        }
        requested.full_res_xb = !requested.allows_full_res_y_subset()
            || kernel_want
                .iter_slots()
                .any(|id| ComputeSet::is_full_res_xb(id, ns));
        // v2-bearing requests over basic/peaks/v2 only (channel-local families): skip X/B work at every scale
        // whose X/B slots are never read. `edge_width_change` at scale s also reads s+1's gradient sums.
        let peaks_end = ns * 3 * crate::metric::FEATURES_PER_CHANNEL_WITH_PEAKS;
        let v2_start = ns * 3 * crate::metric::FEATURES_PER_CHANNEL_EXTENDED
            + ns * 3 * crate::metric::FEATURES_PER_CHANNEL_IW;
        let v2_end = v2_start + ns * 3 * crate::feature_v2::FEATURES_PER_CHANNEL_V2_TOTAL;
        let basic_peaks_v2_only = kernel_want
            .iter_slots()
            .all(|id| id < peaks_end || (v2_start..v2_end).contains(&id));
        // With the v2 block running over a request that reads nothing past f719, every class-C slot it touches is the v2
        // `MSE` slot, which the v2 kernel itself finalizes (the free path writes `MSE` only when `v2_blocks` is off and
        // the other free slots live in append/append2, which this request does not carry): the free-extras
        // accumulation would only add v1-kernel work. Turn it off so the channel mask can apply and the walk is cheaper.
        if requested.v2_blocks && basic_peaks_v2_only && requested.chroma_local_families() {
            requested.free_extras = V1FreeExtras::Off;
        }
        if requested.v2_blocks && basic_peaks_v2_only && requested.chroma_local_families() {
            let mut chroma = 0u8;
            for id in kernel_want.iter_slots() {
                if let Some(d) = crate::feature_defs::def_at(id, ns)
                    && matches!(
                        d.channel,
                        crate::feature_defs::Channel::X | crate::feature_defs::Channel::B
                    )
                {
                    chroma |= 1 << d.scale;
                    if id >= v2_start && d.signal.name == "edge_width_change" {
                        chroma |= 1 << (usize::from(d.scale).min(ns - 2) + 1);
                    }
                }
            }
            requested.full_res_xb = chroma & 1 != 0;
            requested.coarse_y_only_scales = (1..ns).fold(0u8, |mask, scale| {
                mask | if chroma & (1 << scale) == 0 {
                    1 << scale
                } else {
                    0
                }
            });
        } else if want.iter_slots().all(|id| id < 156 && id % 13 < 10) {
            requested.coarse_y_only_scales = (1..ns).fold(0, |mask, scale| {
                let reads_xb = want.iter_slots().any(|id| {
                    crate::feature_defs::def_at(id, ns).is_some_and(|d| {
                        usize::from(d.scale) == scale
                            && matches!(
                                d.channel,
                                crate::feature_defs::Channel::X | crate::feature_defs::Channel::B
                            )
                    })
                });
                mask | if reads_xb { 0 } else { 1 << scale }
            });
        }
        requested.local_only = want.iter_slots().all(|id| id < ns * 3 * 13 && id % 13 < 10);
        requested.omit_edges =
            requested.local_only && want.iter_slots().all(|id| matches!(id % 13, 0..=2 | 9));
        let plan = Plan::normalized(requested, layout);
        if !plan.emit.covers(&want) {
            return Err(PlanError::Uncomputable {
                missing: plan.emit.missing_from(&want),
                layout_width,
            });
        }
        Ok(plan)
    }

    /// Build a plan whose `compute` is **what the walk will actually run**,
    /// not merely what was asked for, and whose `emit` follows from it.
    ///
    /// ## Why this exists (found by phase 2's perturbation probe, 2026-09-05)
    ///
    /// `V2NewFeatureToggles` has exactly ONE layout/compute separation:
    /// `v1_only`, which turns every v2-era kernel off while leaving the
    /// declared width alone. There is no *per-block* layout-only flag —
    /// [`ComputeSet::from_toggles`] derives `append`/`append2`/`csfw` from the
    /// SAME `*_block` flags that decide the width (`append_block &&
    /// v2_blocks`, …), and it hard-sets `v1_basic: true` because no toggle can
    /// turn v1's basic block off.
    ///
    /// So a plan that said "compute the append block but not CSFW, at layout
    /// 956" described a walk that cannot exist: `toggles()` set `csfw_block`
    /// from the WIDTH, the walk computed CSFW, and `emit` — derived from the
    /// un-normalized request — said those twelve positions were structural
    /// zeros. They were not. The probe measured `f944` at `0.0678` on a plan
    /// that declared it unpopulated.
    ///
    /// The fix is a fixed point rather than a second rule: normalize through
    /// the toggles the plan would emit, so
    /// the family compute agrees with `ComputeSet::from_toggles(plan.toggles())`
    /// ([`toggle_gates::normalization_is_a_fixed_point`]). The later private
    /// full-resolution channel and per-scale restrictions are preserved
    /// separately below; the public toggles continue to describe all scales.
    /// `emit` only ever WIDENS, so no request that planned before stops
    /// planning, and nothing that was served changes.
    ///
    /// The missing capability — a per-block layout-only flag, so a 956-wide
    /// vector could carry a computed append block beside a zeroed CSFW one —
    /// is REGISTERED, not built: it needs a walk change, and this lane's
    /// scope is dispatch. Today the honest answer is that the walk computes
    /// every block its declared width reaches on each selected v2 scale.
    /// Declared IDs narrow those scales and retain adjacent gradient inputs;
    /// within-scale block separation remains conservative.
    fn normalized(requested: ComputeSet, layout: Layout) -> Plan {
        let ns = crate::NUM_SCALES;
        let probe = Plan {
            compute: requested,
            layout: layout.clone(),
            emit: SlotSet::from_slots([]),
        };
        let mut compute = ComputeSet::from_toggles(probe.toggles());
        // Channel selection is private plan data, separate from public family
        // toggles. Normalization preserves it only for supported families.
        compute.v1_full_scales = requested.v1_full_scales;
        compute.v1_scales = requested.v1_scales;
        compute.v2_scales = requested.v2_scales;
        compute.omit_pjnd_fragility = requested.omit_pjnd_fragility;
        // The globaff LAYOUT flag follows the width; the computation follows
        // the request (it is cheap, but a wide layout must not imply it).
        compute.full_res_xb = requested.full_res_xb || !compute.allows_full_res_y_subset();
        compute.sampling = requested.sampling;
        compute.coarse_y_only_scales = if (!compute.v2_blocks || compute.chroma_local_families())
            && compute.free_extras == V1FreeExtras::Off
            && compute.v1_pools != V1PoolsMode::Full
        {
            requested.coarse_y_only_scales
        } else {
            0
        };
        compute.local_only = requested.local_only;
        compute.omit_edges = requested.omit_edges;
        // `emit` is in ID space and is intersected with what the LAYOUT
        // carries: a dense layout that omits an id the walk computes does not
        // emit it, and saying otherwise would make `covers` lie.
        let emit = compute
            .populated_slots(ns, layout.walk_width())
            .intersect(&layout.ids());
        Plan {
            compute,
            layout,
            emit,
        }
    }

    /// Plan for a loaded bake — **the servability entry point**.
    ///
    /// Declared feature IDs determine compute for both dense and legacy
    /// identity layouts. The canonical BANDVIS activity variant is off,
    /// matching training extraction (September 7 correction).
    pub(crate) fn for_bake(model: &crate::mlp::Model) -> Result<Plan, PlanError> {
        let revision = crate::feature_layout::formula_revision(model)
            .map_err(|_| PlanError::UnreadableBake)?;
        let layout_width = model.caller_input_width();
        let layout = crate::feature_layout::declared_layout(model);
        let want = bake_read_slots(model).ok_or(PlanError::UnreadableBake)?;
        let walk_width = layout.walk_width();
        let palette = want.intersect(&crate::feature_defs::family_slots(
            crate::feature_defs::Family::Palette,
            crate::NUM_SCALES,
        ));
        if !palette.is_empty() {
            return Err(PlanError::Uncomputable {
                missing: palette,
                layout_width,
            });
        }
        let globaff = want.intersect(&crate::feature_defs::family_slots(
            ComputeToken::GlobAff,
            crate::NUM_SCALES,
        ));
        if !globaff.is_empty() && !revision.computes_globaff() {
            return Err(PlanError::Uncomputable {
                missing: globaff,
                layout_width,
            });
        }
        // Revision 6 derives at its own revision (see `derive_with_layout_at`);
        // earlier bakes keep the process-revision derivation they always had.
        let mut plan = if revision.computes_globaff() {
            Plan::derive_with_layout_at(&want, layout, revision)?
        } else {
            Plan::derive_with_layout(&want, layout)?
        };
        plan.compute.formula_revision = revision;
        // The derive checked the PROCESS revision; the bake's own
        // declaration is what this plan computes, so the rev5 scope check
        // runs again against it.
        if let Some(extra) = rev5_unsupported(&want, walk_width, revision) {
            return Err(PlanError::UnsupportedAtRev5 { extra });
        }
        let sampling =
            crate::sampling::Sampling::from_model(model).map_err(|_| PlanError::UnreadableBake)?;
        if let Some(sampling) = sampling {
            if !sampling.is_direct()
                && (!plan.compute.allows_full_res_y_subset()
                    || !matches!(plan.compute.v1_pools, V1PoolsMode::Off | V1PoolsMode::Peaks)
                    || (sampling.keep_y && plan.compute.full_res_xb))
            {
                return Err(PlanError::UnreadableBake);
            }
            plan.compute.sampling = Some(sampling);
        }
        // **The SERVING-plan footprint policy, applied to both branches.**
        // `fold_engine::pools_mode_for_need` owns the rule that `Off` is never
        // the right answer for a served v1 walk: `Off` and `Peaks` compute the
        // same sums (the peak accumulators are the fused V-blur's
        // unconditional L8/max tier), but `Off` hands the band no scratch,
        // which disables the band-local self-blur and falls back to phase A's
        // four STRIP-wide H planes — a LARGER hot set for no arithmetic
        // saving. `score_pool_mode` routes through that owner; the plan
        // derives `v1_pools` from the touched families and does not.
        //
        // Before the dense flip nothing reached the second branch, so the
        // divergence was invisible. MEASURED the moment shipped `D` declared
        // its 28 basic ids: the id-space plan chose `Off`, and D's emitted
        // vector went from real values at `f156..227` to zeros — the
        // footprint regression the policy exists to prevent, with the score
        // unmoved either way. Promoting here restores byte-identical
        // behaviour and puts the policy back in one place.
        let plan = if revision < crate::feature_defs::FormulaRevision::Rev5
            && plan.compute.v1_basic
            && plan.compute.v1_pools == V1PoolsMode::Off
        {
            let mut promoted = plan.compute;
            promoted.v1_pools = V1PoolsMode::Peaks;
            Plan::normalized(promoted, plan.layout)
        } else {
            plan
        };
        if !plan.emit.covers(&want) {
            return Err(PlanError::Uncomputable {
                missing: plan.emit.missing_from(&want),
                layout_width,
            });
        }
        Ok(plan)
    }

    /// The same COMPUTE, re-laid-out as the identity layout of at least
    /// `width`.
    ///
    /// The scoring path needs this because the linear tail reads the v1
    /// prefix: a plan narrower than that must widen, and widening a plan is a
    /// LAYOUT change, not a number to patch in place. Rebuilt through
    /// [`Plan::normalized`] so `compute` and `emit` stay the fixed point.
    pub(crate) fn widened_to_identity(plan: &Plan, width: usize) -> Plan {
        Plan::normalized(
            plan.compute,
            Layout::identity(plan.layout.width().max(width)),
        )
    }

    /// The extraction request this plan resolves to.
    ///
    /// **The layout flags and the compute flags are different things**, and
    /// the walk has always distinguished them — `layout_append =
    /// toggles.append_block` decides the emitted WIDTH while `append_on =
    /// compute.append` decides whether the kernel runs. That separation is
    /// what makes a 944-wide vector with `f372..` at the structural `0.0`
    /// expressible, and it is what the runtime never used: it hard-coded
    /// `v1_only: true, ..Default::default()`, which pins every layout flag off
    /// and so pins the emitted width to the v1 layout.
    pub(crate) fn toggles(&self) -> crate::feature_v2::V2NewFeatureToggles {
        let ns = crate::NUM_SCALES;
        let c = &self.compute;
        let layout = LayoutBlocks::for_width(self.layout.walk_width(), ns);
        crate::feature_v2::V2NewFeatureToggles {
            // THE revision this walk computes. Forwarded from the plan, so a
            // bake's declared revision reaches the finaliser rather than the
            // process-wide default.
            formula_revision: c.formula_revision,
            gradient_features: c.gradient,
            transducer_bank: c.transducer_bank,
            blockiness: c.blockiness,
            transducers_luma_only: c.transducers_luma_only,
            // LAYOUT: a block's flag is on when the declared width reaches
            // it, whether or not its kernel runs. The chain is NESTED, not
            // four independent tests — `append2_block` asserts
            // `append_block`, `csfw_block` asserts `append2_block`, and
            // previous), so a width that reaches
            // one necessarily reaches the ones below it. Written as a chain
            // rather than four `>` tests so a future non-contiguous width
            // cannot violate the assertion.
            // The Rev4 feature bank extends the nested chain: each family's
            // layout flag is on when the declared width reaches its base.
            // A sub-toggle that REFINES a block cannot outlive it: the walk
            // asserts `append2_dst_activity => append2_block`. `everything`
            // (the fallback compute set for a wide bake) turns it on
            // unconditionally, so a 720- or 924-wide bake reached that
            // assertion — found by the board census, not by reading.
            v1_pools: c.v1_pools,
            // COMPUTE: `v1_only` turns every v2-era kernel off while leaving
            // the layout flags — and therefore the width — untouched.
            v1_only: !c.v2_blocks,
            free_extras: c.free_extras,
        }
    }

    /// Do two plans compute the SAME arithmetic era?
    ///
    /// One walk computes one revision, so a profile whose bakes disagree
    /// cannot be served from a single extraction — and serving both from the
    /// winner's revision would silently re-price the loser's inputs, which is
    /// exactly the failure the per-bake declaration exists to prevent. The
    /// scoring path checks this and refuses.
    pub(crate) fn revisions_agree(&self, other: &Plan) -> bool {
        self.compute.formula_revision == other.compute.formula_revision
    }

    /// One past the highest id [`Self::emit`] claims to populate — the
    /// minimum length an identity-laid-out emitted vector must have for
    /// every promised slot to carry a computed value. `SlotSet` ranges
    /// are sorted, so this is one range read, not a scan.
    pub(crate) fn emit_bound(&self) -> usize {
        self.emit.ranges().last().map_or(0, |&(_, end)| end)
    }

    /// Is every id [`Self::emit`] claims already inside an emitted vector
    /// of `emitted_len`? The predicate form of [`Self::check_emit_covered`],
    /// for the `debug_assert!` the emit→plan boundary sites keep.
    pub(crate) fn emit_covered(&self, emitted_len: usize) -> bool {
        self.emit_bound() <= emitted_len
    }

    /// REV4SERVE review F1 — the emitted vector must cover [`Self::emit`].
    ///
    /// `emit` and the walk's emit bound are two expressions of the same
    /// compute set: `emit` is `populated_slots(compute) ∩ layout.ids()`,
    /// the walk emits a regime-width prefix of the identity layout. The
    /// emit→plan boundary sites `resize` emitted vectors up to
    /// [`Self::walk_width`] with `0.0`, which is correct for slots the
    /// layout carries but the compute set does not — the extraction
    /// contract's structural zeros — and WRONG for a slot `emit` claimed
    /// was populated: zero-filling it hands the score a fabricated
    /// feature where the pre-`resize` `truncate` left the vector short
    /// and the read panicked out-of-bounds. A planner/walk disagreement
    /// must stay loud, so this check runs in release builds too — the
    /// cost is one range read.
    pub(crate) fn check_emit_covered(&self, emitted_len: usize) -> Result<(), PlanError> {
        if self.emit_covered(emitted_len) {
            return Ok(());
        }
        Err(PlanError::Uncomputable {
            missing: SlotSet::from_slots(self.emit.iter_slots().filter(|&id| id >= emitted_len)),
            layout_width: self.walk_width(),
        })
    }

    /// The union of two plans — what one walk must run to serve both.
    ///
    /// A profile can carry up to three scoring bakes and must serve all of
    /// them from ONE extraction, so the plan the walk runs is the union, not
    /// any one bake's.
    pub(crate) fn union(&self, other: &Plan) -> Plan {
        let ns = crate::NUM_SCALES;
        let layout_width = self.layout.walk_width().max(other.layout.walk_width());
        let (a, b) = (&self.compute, &other.compute);
        let compute = ComputeSet {
            // **A union of two DIFFERENT revisions is not a plan.** One walk
            // computes ONE arithmetic era, so a profile whose bakes declare
            // different revisions cannot be served from one extraction. The
            // union takes the LEFT side's revision and `Plan::revisions_agree`
            // is the predicate a caller must check first — checking here would
            // mean either a silent pick or a `Result` on an infallible
            // operation, and both hide the question.
            formula_revision: a.formula_revision,
            v1_basic: a.v1_basic || b.v1_basic,
            full_res_xb: a.full_res_xb || b.full_res_xb,
            coarse_y_only_scales: a.coarse_y_only_scales & b.coarse_y_only_scales,
            local_only: a.local_only && b.local_only,
            omit_edges: a.omit_edges && b.omit_edges,
            sampling: a.sampling,
            v1_pools: pools_union(a.v1_pools, b.v1_pools),
            v1_full_scales: if pools_union(a.v1_pools, b.v1_pools) == V1PoolsMode::Full {
                a.full_pool_scales() | b.full_pool_scales()
            } else {
                ComputeSet::ALL_SCALES
            },
            v1_scales: (if a.v1_basic || a.v1_pools != V1PoolsMode::Off {
                a.v1_scales
            } else {
                0
            }) | (if b.v1_basic || b.v1_pools != V1PoolsMode::Off {
                b.v1_scales
            } else {
                0
            }),
            v2_scales: (if a.v2_blocks { a.v2_scales } else { 0 })
                | (if b.v2_blocks { b.v2_scales } else { 0 }),
            v2_blocks: a.v2_blocks || b.v2_blocks,
            gradient: a.gradient || b.gradient,
            blockiness: a.blockiness || b.blockiness,
            transducer_bank: a.transducer_bank || b.transducer_bank,
            // A luma-only request is a RESTRICTION; the union must not
            // restrict, so it is on only when both sides ask for it.
            transducers_luma_only: a.transducers_luma_only && b.transducers_luma_only,
            omit_pjnd_fragility: a.omit_pjnd_fragility && b.omit_pjnd_fragility,
            free_extras: free_union(a.free_extras, b.free_extras),
        };
        let _ = ns;
        // The union is only ever taken over a profile's own bakes, which are
        // all identity-laid-out today; a union of two DENSE layouts has no
        // caller and no meaning (whose positions would it use?), so it takes
        // the identity layout at the wider walk width.
        Plan::normalized(compute, Layout::identity(layout_width))
    }
}

/// Which optional blocks a declared layout width reaches.
///
/// The chain is nested by construction: `append2` implies `append`, `csfw`
/// these implications.
struct LayoutBlocks {
}

impl LayoutBlocks {
    fn for_width(width: usize, ns: usize) -> Self {
        let append = width > base_of(ComputeToken::Append, ns);
        let append2 = append && width > base_of(ComputeToken::Append2, ns);
        let csfw = append2 && width > base_of(ComputeToken::Csfw, ns);
        let gridblk = csfw && width > base_of(ComputeToken::Gridblk, ns);
        let ringbasis = gridblk && width > base_of(ComputeToken::Ringbasis, ns);
        let tailhist = ringbasis && width > base_of(ComputeToken::Tailhist, ns);
        let arttype = tailhist && width > base_of(ComputeToken::Arttype, ns);
        let gmsbank = arttype && width > base_of(ComputeToken::Gmsbank, ns);
        let mapdev = gmsbank && width > base_of(ComputeToken::Mapdev, ns);
        let z1max = mapdev && width > base_of(ComputeToken::Z1max, ns);
        let gmsnative = z1max && width > base_of(ComputeToken::Gmsnative, ns);
        let globaff = gmsnative && width > base_of(ComputeToken::GlobAff, ns);
        Self {
        }
    }
}

/// Base slot of a registered block at `n_scales`.
fn base_of(t: ComputeToken, ns: usize) -> usize {
    crate::feature_defs::block_base(t, ns).map_or(usize::MAX, |(b, _)| b)
}

/// The wider of two pool modes. `Carriers` is a strict subset of `Peaks`'
/// compute (it emits ten slots the peaks tier already produces), so the order
/// is `Off < Carriers < Peaks < Full`.
fn pools_union(a: V1PoolsMode, b: V1PoolsMode) -> V1PoolsMode {
    let rank = |m: V1PoolsMode| match m {
        V1PoolsMode::Off => 0u8,
        V1PoolsMode::Peaks => 2,
        V1PoolsMode::Full => 3,
    };
    if rank(a) >= rank(b) { a } else { b }
}

/// The wider of two free-extras modes.
fn free_union(a: V1FreeExtras, b: V1FreeExtras) -> V1FreeExtras {
    let rank = |m: V1FreeExtras| match m {
        V1FreeExtras::Off => 0u8,
        V1FreeExtras::RawMoments => 1,
        V1FreeExtras::RawMomentsPlusBoundedErr => 2,
    };
    if rank(a) >= rank(b) { a } else { b }
}

/// **THE owner of "which FEATURE IDS does this bake read".**
///
/// The in-crate twin of `zensim_validate::block_profile::used_caller_lines`,
/// and NOT a duplicate of it: both are thin callers of
/// [`crate::fold_engine::caller_line_reads`], which is the one owner of the
/// caller-space fold. zensim cannot depend on zensim-validate (the dependency
/// runs the other way), so the narrow primitive lives in the lower crate and
/// the validate-side function keeps its richer per-line norms.
///
/// **`caller_line_reads` returns layer-0 POSITIONS, and a position is only a
/// feature id under the identity layout.** That equality held for every bake
/// that shipped before 2026-09-06 and is exactly the assumption the dense
/// contract breaks: shipped `B` declares 95 ids spanning `f3..f369`, so its
/// live positions are `0..94` and reading them as ids says it touches nothing
/// above `f94` — which would tell the walk to skip the masked and IW pools
/// `B` demonstrably reads. Mapping through
/// [`crate::declared_feature_ids`] — the ONE owner of the declaration — is
/// what makes this function answer the question its name asks. An
/// identity-layout bake maps `i -> i`, so nothing shipped before the dense
/// contract moves.
pub(crate) fn bake_read_slots(model: &crate::mlp::Model) -> Option<SlotSet> {
    let reads = crate::fold_engine::caller_line_reads(model)?;
    // E33 `fx1`: a derived-input bake's model positions are entries over its
    // declared ids, not one id per position; a malformed declaration is
    // unreadable rather than read positionally.
    match crate::derived_inputs::from_model(model) {
        Err(_) => return None,
        Ok(Some(derived)) => {
            let declared: Vec<usize> = crate::feature_layout::declared_ids(model)?
                .iter_slots()
                .collect();
            return derived.read_slots(&declared, &reads);
        }
        Ok(None) => {}
    }
    let declared = crate::feature_layout::declared_feature_ids(model);
    Some(SlotSet::from_slots(
        reads
            .iter()
            .enumerate()
            .filter(|(_, live)| **live)
            .map(|(i, _)| match &declared {
                // A declared bake's position `i` carries feature id `ids[i]`.
                // A position past the declaration is a shape bug upstream; map
                // it to itself rather than dropping it, so a malformed bake
                // over-reports what it reads instead of under-reporting (the
                // safe direction for a SKIP decision).
                Some(ids) => ids.get(i).map_or(i, |&id| usize::from(id)),
                None => i,
            }),
    ))
}

#[cfg(test)]
mod tests {

    use super::*;

    fn slots(r: impl IntoIterator<Item = (usize, usize)>) -> SlotSet {
        SlotSet::from_ranges(r)
    }

    #[test]
    #[ignore]
    fn zz_probe_moments() {
        let m = crate::feature_defs::family_slots(
            crate::feature_set_id::ComputeToken::Moments,
            crate::NUM_SCALES,
        );
        println!("moments slots: {m}");
        let free = SlotSet::from_ranges([(0, 228)]).union(&m).clipped_to(944);
        println!("free set len {} : {free}", free.len());
        let in228_300: Vec<usize> = free
            .iter_slots()
            .filter(|s| (228..300).contains(s))
            .collect();
        let in300_372: Vec<usize> = free
            .iter_slots()
            .filter(|s| (300..372).contains(s))
            .collect();
        println!("in masked range: {in228_300:?}");
        println!("in iw range: {in300_372:?}");
    }

    /// A basic-only request skips the pool block entirely.
    #[test]
    fn a_basic_only_request_skips_the_pools() {
        let p = Plan::derive(&slots([(0, 156)]), 372).expect("plan");
        assert_eq!(p.compute.v1_pools, V1PoolsMode::Off);
        assert_eq!(p.emit, slots([(0, 156)]));
    }

    /// A request for a slot past every registered block is REFUSED, loudly,
    /// naming the slot — never served as a silent zero.
    #[test]
    fn an_unregistered_slot_is_refused_and_named() {
        let want = slots([(0, 156)]).union(&SlotSet::from_slots([5000]));
        let err = Plan::derive(&want, 6000).expect_err("must refuse");
        match err {
            PlanError::Uncomputable { ref missing, .. } => {
                assert!(missing.contains(5000), "the refusal must name slot 5000");
                assert!(!missing.contains(0), "it must not blame served slots");
            }
            other => panic!("wrong error: {other:?}"),
        }
        assert!(format!("{err}").contains("5000"), "Display names the slot");
    }

    /// REV4SERVE review F1: the emit→plan boundary `resize(walk_width, 0.0)`
    /// must never zero-fill a slot `emit` promised was computed. A
    /// planner/walk disagreement (an emit claim the emitted vector cannot
    /// reach) surfaces as `Uncomputable` naming the unmaterialized ids —
    /// not a silent structural zero a read then consumes.
    #[test]
    fn emit_coverage_check_refuses_ids_the_walk_did_not_materialize() {
        let want = SlotSet::from_slots(0..156);
        let mut plan = Plan::derive(&want, 1853).unwrap();
        // Disagreement: emit claims v2 ids at 372..800 while the emitted
        // vector only reaches the 720-wide regime bound.
        plan.emit = SlotSet::from_slots((0..156).chain(372..800));
        assert_eq!(plan.emit_bound(), 800);
        assert!(!plan.emit_covered(720));
        match plan.check_emit_covered(720) {
            Err(PlanError::Uncomputable {
                missing,
                layout_width,
            }) => {
                assert_eq!(missing, SlotSet::from_slots(720..800));
                assert_eq!(layout_width, 1853);
            }
            other => panic!("expected Uncomputable, got {other:?}"),
        }
        // The boundary value itself is covered, and a real plan (whose emit
        // is populated_slots ∩ layout.ids()) never trips the check against
        // the regime it would emit.
        assert!(plan.emit_covered(800));
        assert!(plan.check_emit_covered(800).is_ok());
        assert!(
            Plan::derive(&want, 1853)
                .unwrap()
                .check_emit_covered(720)
                .is_ok()
        );
    }
}

#[cfg(test)]
mod toggle_gates {
    use super::*;

    /// **The normalization is a FIXED POINT.** `Plan::normalized` resolves
    /// `compute` through the toggles it would emit; applying it again must
    /// change nothing, or "what the walk runs" would depend on how many times
    /// the plan was rebuilt.
    ///
    /// Also pins the WIDENING direction: normalization may only ADD emitted
    /// slots. A narrowing would mean a request that planned before stops
    /// planning — the servability regression this whole design exists to
    /// prevent.
    #[test]
    fn normalization_is_a_fixed_point() {
        let ns = crate::NUM_SCALES;
        let cases: [(SlotSet, usize); 5] = [
            (SlotSet::from_ranges([(0, 156)]), 372),
            (SlotSet::from_ranges([(0, 372)]), 372),
            (SlotSet::from_ranges([(0, 228)]), 944),
            (
                SlotSet::from_ranges([(0, 228)]).union(&crate::feature_defs::family_slots(
                    ComputeToken::Moments,
                    ns,
                )),
                944,
            ),
            (SlotSet::from_ranges([(0, 720)]), 720),
        ];
        for (want, width) in cases {
            let p = Plan::derive(&want, width).expect("plan");
            let again = Plan::normalized(p.compute, Layout::identity(width));
            assert_eq!(
                again.compute, p.compute,
                "normalization moved on the second pass at width {width}"
            );
            assert_eq!(again.emit, p.emit, "emit moved at width {width}");
            // `compute == from_toggles(toggles())`, by construction.
            assert_eq!(
                ComputeSet::from_toggles(p.toggles()),
                p.compute,
                "the walk would run something else at width {width}"
            );
            assert!(
                p.emit.covers(&want.clipped_to(width)),
                "normalization must never narrow below the request at width {width}"
            );
        }
    }

    /// The plan's toggles must round-trip through `ComputeSet::from_toggles`
    /// to the plan's own compute set — otherwise the walk would run something
    /// other than what was planned, silently.
    #[test]
    fn toggles_round_trip_to_the_planned_compute_set() {
        let ns = crate::NUM_SCALES;
        let cases = [
            (SlotSet::from_ranges([(0, 156)]), 372usize),
            (SlotSet::from_ranges([(0, 372)]), 372),
            (
                SlotSet::from_ranges([(0, 228)]).union(&crate::feature_defs::family_slots(
                    ComputeToken::Moments,
                    ns,
                )),
                944,
            ),
            (
                SlotSet::from_ranges([(0, 228)])
                    .union(&crate::feature_defs::family_slots(
                        ComputeToken::Moments,
                        ns,
                    ))
                    .union(&crate::feature_defs::family_slots(ComputeToken::ClassC, ns)),
                944,
            ),
        ];
        for (want, width) in cases {
            let p = Plan::derive(&want, width).expect("plan");
            let round = ComputeSet::from_toggles(p.toggles());
            assert_eq!(
                round, p.compute,
                "toggles for width {width} resolve to a different compute set"
            );
            // And the toggles' own populated set must equal the plan's.
            assert_eq!(
                round.populated_slots(crate::NUM_SCALES, width),
                p.emit,
                "toggles for width {width} populate a different slot set"
            );
        }
    }
}
