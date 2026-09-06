//! **The extraction PLAN** — one derivation from "what a consumer needs" to
//! "what the walk must run and how wide the answer is".
//!
//! Design: `docs/FEATURE_SYSTEM_DESIGN_2026-09-05.md` §5. Phases + gates:
//! `docs/PLAN_FEATURE_SYSTEM_2026-09-05.md`.
//!
//! ## What was missing, precisely
//!
//! [`crate::feature_v2::ComputeSet::from_block_profile`] already derived the
//! right COMPUTE set for any bake — including the cheap free-extras set for a
//! wide bake that only reads the raw-moment tranche. It was never on a runtime
//! path (`#[cfg_attr(not(test), allow(dead_code))]`), and promoting it alone
//! would not have helped, because the runtime had no notion of a LAYOUT: the
//! fold-backed score truncates to `fold_engine::v1_feature_width(config)` —
//! at most 372 — and `metric::prep_bake_input_f32` then refuses any bake
//! declaring more than `features.len() + 4`.
//!
//! So the missing piece was never the compute derivation. It was the pair
//! `(compute, layout)`. A [`Plan`] is exactly that pair, plus the slot set the
//! walk will actually populate, so "can this bake be served?" is a checked
//! question — [`Plan::covers`] — rather than three independent hard-codings.
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
        }
    }
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
        let ns = crate::NUM_SCALES;
        // COMPUTE is decided in ID space against the layout's WALK width (one
        // past the highest id it carries), never against its emitted width —
        // a `dense265` layout over `basic+peaks+moments` is 265 wide and
        // still needs a walk that reaches f941.
        let layout_width = layout.walk_width();
        let want = want.clipped_to(layout_width);
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
        // `csfw_on` is `csfw_block && v2_blocks` in the walk, so a CSFW
        // request implies the v2-era pass regardless of what else is asked.
        let v2_blocks = (touches(ComputeToken::V2) && outside_tranche(ComputeToken::V2))
            || append
            || append2
            || csfw;

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
            formula_revision: crate::ssim_form::active_revision(),
            v1_basic: touches(ComputeToken::Basic) || v1_pools != V1PoolsMode::Off,
            v1_pools,
            v2_blocks,
            gradient: v2_blocks,
            blockiness: v2_blocks,
            transducer_bank: v2_blocks,
            transducers_luma_only: false,
            append,
            append2,
            append2_dst_activity: false,
            csfw,
            free_extras,
        };
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
    /// `compute == ComputeSet::from_toggles(plan.toggles())` **by
    /// construction** ([`toggle_gates::normalization_is_a_fixed_point`]).
    /// `emit` only ever WIDENS, so no request that planned before stops
    /// planning, and nothing that was served changes.
    ///
    /// The missing capability — a per-block layout-only flag, so a 956-wide
    /// vector could carry a computed append block beside a zeroed CSFW one —
    /// is REGISTERED, not built: it needs a walk change, and this lane's
    /// scope is dispatch. Today the honest answer is that the walk computes
    /// every block its declared width reaches, and the plan now says so.
    fn normalized(requested: ComputeSet, layout: Layout) -> Plan {
        let ns = crate::NUM_SCALES;
        let probe = Plan {
            compute: requested,
            layout: layout.clone(),
            emit: SlotSet::from_slots([]),
        };
        let compute = ComputeSet::from_toggles(probe.toggles());
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
    /// COMPUTE comes from [`ComputeSet::from_block_profile`], the existing
    /// tested derivation, so this adds a layout rather than a second opinion.
    /// LAYOUT is `Model::caller_input_width()` — never `n_inputs()`, which is
    /// the pruned internal width and is a third, different number.
    pub(crate) fn for_bake(model: &crate::mlp::Model) -> Result<Plan, PlanError> {
        let ns = crate::NUM_SCALES;
        let layout_width = model.caller_input_width();
        let layout = crate::feature_layout::declared_layout(model);
        // The bake reads POSITIONS and the plan speaks IDS; for every identity
        // layout they are the same numbers and for a dense one they are not.
        // The translation lives in [`bake_read_slots`], which is THE owner of
        // "which ids does this bake read" — it used to happen here, which made
        // the owner return positions under an ids-shaped name and left
        // `fold_engine::bake_pool_need_from_model` folding raw positions
        // against the v1 family bounds (measured wrong on shipped B the moment
        // it went dense). One translation, one place.
        let want = bake_read_slots(model).ok_or(PlanError::UnreadableBake)?;
        // COMPUTE. `ComputeSet::from_block_profile` is the existing, tested
        // derivation and stays THE answer for an identity layout — which is
        // every bake that ships, so no served bake changes.
        //
        // It cannot serve a DENSE one, and the reason is structural rather
        // than a bug in it: it reads the bake's live layer-0 columns as v1
        // SLOT INDICES (`caller_input_width() <= v1_total` ⇒ the v1 branch,
        // then `bake_pool_need_from_model` on the same positions). Under a
        // dense layout position 228 is a raw-moment id, not a masked one, so
        // that reading is wrong by construction. MEASURED: a `dense265` bake
        // over `basic+peaks+moments` derived `v1_pools: Full, free_extras:
        // Off`, whose `emit` does not cover the moment ids, so
        // `Plan::for_bake` REFUSED its own bake and the profile silently fell
        // back to a 228-wide walk.
        //
        // So a non-identity layout is derived in ID space instead. Phase 5
        // unifies the two once `from_block_profile_agrees_with_the_id_space_
        // derivation` has held across the whole bake census.
        let plan = if layout.is_identity() {
            Plan::normalized(ComputeSet::from_block_profile(model), layout)
        } else {
            Plan::derive_with_layout(&want, layout)?
        };
        // **The SERVING-plan footprint policy, applied to both branches.**
        // `fold_engine::pools_mode_for_need` owns the rule that `Off` is never
        // the right answer for a served v1 walk: `Off` and `Peaks` compute the
        // same sums (the peak accumulators are the fused V-blur's
        // unconditional L8/max tier), but `Off` hands the band no scratch,
        // which disables the band-local self-blur and falls back to phase A's
        // four STRIP-wide H planes — a LARGER hot set for no arithmetic
        // saving. `from_block_profile` routes through that owner; the id-space
        // branch derives `v1_pools` from the touched families and does not.
        //
        // Before the dense flip nothing reached the second branch, so the
        // divergence was invisible. MEASURED the moment shipped `D` declared
        // its 28 basic ids: the id-space plan chose `Off`, and D's emitted
        // vector went from real values at `f156..227` to zeros — the
        // footprint regression the policy exists to prevent, with the score
        // unmoved either way. Promoting here restores byte-identical
        // behaviour and puts the policy back in one place.
        let plan = if plan.compute.v1_basic && plan.compute.v1_pools == V1PoolsMode::Off {
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
        let _ = ns;
        Ok(plan)
    }

    /// A v1-layout plan at an explicit pool mode — the shape every pre-plan
    /// caller expressed as a bare `Option<V1PoolsMode>`.
    pub(crate) fn v1(pools: V1PoolsMode, layout_width: usize) -> Plan {
        let compute = ComputeSet {
            formula_revision: crate::ssim_form::active_revision(),
            v1_basic: true,
            v1_pools: pools,
            v2_blocks: false,
            gradient: false,
            blockiness: false,
            transducer_bank: false,
            transducers_luma_only: false,
            append: false,
            append2: false,
            append2_dst_activity: false,
            csfw: false,
            free_extras: V1FreeExtras::Off,
        };
        Plan::normalized(compute, Layout::identity(layout_width))
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
            // three independent tests — `append2_block` asserts `append_block`
            // and `csfw_block` asserts `append2_block` (each sits above the
            // previous and reuses its accumulators), so a width that reaches
            // one necessarily reaches the ones below it. Written as a chain
            // rather than three `>` tests so a future non-contiguous width
            // cannot violate the assertion.
            append_block: layout.append,
            append2_block: layout.append2,
            csfw_block: layout.csfw,
            // A sub-toggle that REFINES a block cannot outlive it: the walk
            // asserts `append2_dst_activity => append2_block`. `everything`
            // (the fallback compute set for a wide bake) turns it on
            // unconditionally, so a 720- or 924-wide bake reached that
            // assertion — found by the board census, not by reading.
            append2_dst_activity: c.append2_dst_activity && layout.append2,
            v1_pools: c.v1_pools,
            // COMPUTE: `v1_only` turns every v2-era kernel off while leaving
            // the layout flags — and therefore the width — untouched.
            v1_only: !c.v2_blocks,
            free_extras: c.free_extras,
        }
    }

    /// The formula revision this plan computes.
    pub(crate) fn formula_revision(&self) -> crate::feature_defs::FormulaRevision {
        self.compute.formula_revision
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

    /// Does this plan populate every slot of `want`?
    pub(crate) fn covers(&self, want: &SlotSet) -> bool {
        self.emit.covers(want)
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
            v1_pools: pools_union(a.v1_pools, b.v1_pools),
            v2_blocks: a.v2_blocks || b.v2_blocks,
            gradient: a.gradient || b.gradient,
            blockiness: a.blockiness || b.blockiness,
            transducer_bank: a.transducer_bank || b.transducer_bank,
            // A luma-only request is a RESTRICTION; the union must not
            // restrict, so it is on only when both sides ask for it.
            transducers_luma_only: a.transducers_luma_only && b.transducers_luma_only,
            append: a.append || b.append,
            append2: a.append2 || b.append2,
            append2_dst_activity: a.append2_dst_activity || b.append2_dst_activity,
            csfw: a.csfw || b.csfw,
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
/// implies `append2`. The walk asserts exactly these implications.
struct LayoutBlocks {
    append: bool,
    append2: bool,
    csfw: bool,
}

impl LayoutBlocks {
    fn for_width(width: usize, ns: usize) -> Self {
        let append = width > base_of(ComputeToken::Append, ns);
        let append2 = append && width > base_of(ComputeToken::Append2, ns);
        let csfw = append2 && width > base_of(ComputeToken::Csfw, ns);
        Self {
            append,
            append2,
            csfw,
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
        V1PoolsMode::Carriers => 1,
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
    let declared = crate::declared_feature_ids(model);
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

    /// **G1.7** — the v1 request plans to exactly today's v1 walk.
    #[test]
    fn a_v1_request_plans_the_v1_only_walk() {
        let p = Plan::derive(&slots([(0, 372)]), 372).expect("plan");
        assert!(p.compute.v1_basic);
        assert_eq!(p.compute.v1_pools, V1PoolsMode::Full);
        assert!(!p.compute.v2_blocks);
        assert!(!p.compute.append);
        assert_eq!(p.compute.free_extras, V1FreeExtras::Off);
        assert_eq!(p.layout_width(), 372);
        assert_eq!(p.emit, slots([(0, 372)]));
    }

    /// A basic-only request skips the pool block entirely.
    #[test]
    fn a_basic_only_request_skips_the_pools() {
        let p = Plan::derive(&slots([(0, 156)]), 372).expect("plan");
        assert_eq!(p.compute.v1_pools, V1PoolsMode::Off);
        assert_eq!(p.emit, slots([(0, 156)]));
    }

    /// **THE cheap wide plan.** A 944-layout request that touches the append
    /// block only at raw-moment slots must NOT schedule the append kernel.
    #[test]
    fn a_free_set_request_plans_the_cheap_walk_not_the_944_walk() {
        let ns = crate::NUM_SCALES;
        let want = slots([(0, 228)]).union(&crate::feature_defs::family_slots(
            ComputeToken::Moments,
            ns,
        ));
        let p = Plan::derive(&want, 944).expect("plan");
        assert!(!p.compute.v2_blocks, "v2 blocks must stay off");
        assert!(!p.compute.append, "the append kernel must not run");
        assert_eq!(p.compute.free_extras, V1FreeExtras::RawMoments);
        assert_eq!(p.compute.v1_pools, V1PoolsMode::Peaks);
        assert_eq!(p.layout_width(), 944);
        assert!(p.covers(&want), "plan must cover what was asked for");
        // 156 basic + 72 peaks + 37 raw-moment slots.
        assert_eq!(p.emit.len(), 265);
    }

    /// The class-C arm adds its 24 slots without waking the append kernel.
    #[test]
    fn a_class_c_request_plans_the_bounded_error_accumulator() {
        let ns = crate::NUM_SCALES;
        let want = slots([(0, 228)])
            .union(&crate::feature_defs::family_slots(
                ComputeToken::Moments,
                ns,
            ))
            .union(&crate::feature_defs::family_slots(ComputeToken::ClassC, ns));
        let p = Plan::derive(&want, 944).expect("plan");
        assert!(!p.compute.append);
        assert!(!p.compute.v2_blocks);
        assert_eq!(
            p.compute.free_extras,
            V1FreeExtras::RawMomentsPlusBoundedErr
        );
        assert_eq!(p.emit.len(), 289);
        assert!(p.covers(&want));
    }

    /// A request touching the append block OUTSIDE its tranche schedules the
    /// real kernel — the negative control for the cheap-plan rule.
    #[test]
    fn an_append_request_outside_the_tranche_schedules_the_kernel() {
        // `xmask_transducer` at (scale 0, X) — append block, not in a tranche.
        let slot = crate::feature_defs::slot_id(ComputeToken::Append, 0, 0, 0, crate::NUM_SCALES)
            .expect("slot");
        let want = slots([(0, 156)]).union(&SlotSet::from_slots([slot]));
        let p = Plan::derive(&want, 944).expect("plan");
        assert!(p.compute.append, "the append kernel must run");
        assert!(p.compute.v2_blocks);
        assert!(p.covers(&want));
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

    /// Union takes the wider of every axis, and never RESTRICTS.
    #[test]
    fn union_widens_and_never_restricts() {
        let a = Plan::derive(&slots([(0, 156)]), 372).expect("plan");
        let b = Plan::derive(&slots([(0, 372)]), 372).expect("plan");
        let u = a.union(&b);
        assert_eq!(u.compute.v1_pools, V1PoolsMode::Full);
        assert_eq!(u.layout_width(), 372);
        assert!(u.covers(&slots([(0, 372)])));
        // A wide cheap plan unioned with a narrow one keeps the wide layout.
        let ns = crate::NUM_SCALES;
        let wide = Plan::derive(
            &slots([(0, 228)]).union(&crate::feature_defs::family_slots(
                ComputeToken::Moments,
                ns,
            )),
            944,
        )
        .expect("plan");
        let u2 = a.union(&wide);
        assert_eq!(u2.layout_width(), 944);
        assert_eq!(u2.compute.free_extras, V1FreeExtras::RawMoments);
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
        let cases: [(SlotSet, usize); 8] = [
            (SlotSet::from_ranges([(0, 156)]), 372),
            (SlotSet::from_ranges([(0, 372)]), 372),
            (SlotSet::from_ranges([(0, 228)]), 944),
            (SlotSet::from_ranges([(0, 944)]), 944),
            (SlotSet::from_ranges([(0, 956)]), 956),
            // The case that FOUND the defect: a wide layout whose request
            // deliberately skips the top block.
            (
                SlotSet::from_slots((0..956).filter(|s| {
                    !crate::feature_defs::family_slots(ComputeToken::Csfw, ns).contains(*s)
                })),
                956,
            ),
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

    /// A wide layout computes every block it reaches — the honest statement
    /// of the capability the toggles do NOT have.
    ///
    /// This is a NEGATIVE gate: it pins a limitation so that the day a
    /// per-block layout-only flag lands, this test fails and forces the plan
    /// to stop over-claiming.
    #[test]
    fn a_wide_layout_computes_every_block_it_reaches() {
        let ns = crate::NUM_SCALES;
        let csfw = crate::feature_defs::family_slots(ComputeToken::Csfw, ns);
        let want = SlotSet::from_slots((0..956).filter(|s| !csfw.contains(*s)));
        let p = Plan::derive(&want, 956).expect("plan");
        assert!(
            p.compute.csfw,
            "at layout 956 with the v2 blocks on, the walk computes CSFW \
             whether or not it was asked for — `csfw_block` is both the \
             layout flag and the compute flag"
        );
        assert!(
            p.emit.covers(&csfw),
            "and the plan must SAY those slots are populated"
        );
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
            (SlotSet::from_ranges([(0, 944)]), 944),
            (SlotSet::from_ranges([(0, 956)]), 956),
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

    /// A wide LAYOUT with a v1-only COMPUTE is expressible — the shape the
    /// runtime could not ask for.
    #[test]
    fn a_944_layout_with_a_v1_only_compute_is_expressible() {
        let ns = crate::NUM_SCALES;
        let want = SlotSet::from_ranges([(0, 228)]).union(&crate::feature_defs::family_slots(
            ComputeToken::Moments,
            ns,
        ));
        let t = Plan::derive(&want, 944).expect("plan").toggles();
        assert!(t.v1_only, "compute stays v1-only");
        assert!(t.append_block, "layout reaches the append block");
        assert!(t.append2_block, "layout reaches append2");
        assert!(!t.csfw_block, "944 does not reach csfw");
        assert_eq!(t.free_extras, V1FreeExtras::RawMoments);
    }
}

/// **The SERVABILITY CENSUS** — the hard contract gate.
///
/// User directive (2026-09-05): *"also make sure everything can be served"*.
/// The contract is that there is no "trains fine, cannot be served" class:
/// every bake whose read set is registered feature ids at a supported
/// revision is servable through `Zensim::compute`, in the layout it declares.
///
/// This module enumerates every SHIPPED profile and every registered producer
/// set and proves it, on real pixels, with no filesystem access — so the gate
/// runs everywhere rather than only where `/mnt/v` is mounted. The
/// filesystem tier (board bakes, stored-table parity) is
/// `zensim/examples/serve_custom_bake.rs --census`, which drives the same
/// `Zensim::compute` entry.
#[cfg(test)]
pub(crate) mod servability_census {
    use super::*;
    use crate::feature_set_id::ComputeToken as T;
    use crate::{RgbSlice, Zensim, ZensimProfile};

    /// A deterministic non-identical SDR pair. 64×64 is the pyramid minimum,
    /// so it exercises the real 4-scale walk with no reflect-pad.
    fn pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
        let mut src = vec![[0u8; 3]; w * h];
        let mut dst = vec![[0u8; 3]; w * h];
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                let v = ((x * 7 + y * 13) % 251) as u8;
                src[i] = [v, v.wrapping_add(40), v.wrapping_mul(3)];
                // A structured, non-trivial distortion: quantize + shift.
                dst[i] = [
                    v & 0xF0,
                    v.wrapping_add(37),
                    v.wrapping_mul(3).wrapping_sub(9),
                ];
            }
        }
        (src, dst)
    }

    /// Every profile this build ships, with its name.
    ///
    /// `pub(crate)` so the layout census (`feature_layout::tests`) enumerates
    /// the SAME roster rather than keeping a second list that could drift
    /// past a feature flag — the roster is `#[cfg]`-dependent, which is
    /// exactly the kind of list a copy gets wrong.
    pub(crate) fn shipped_profiles() -> Vec<(&'static str, ZensimProfile)> {
        let mut v: Vec<(&'static str, ZensimProfile)> = vec![
            ("B", ZensimProfile::B),
            ("BHdr", ZensimProfile::BHdr),
            ("PreviewV0_1", ZensimProfile::PreviewV0_1),
            ("PreviewV0_2", ZensimProfile::PreviewV0_2),
        ];
        #[cfg(feature = "deprecated-profiles")]
        {
            // `A` is deprecated but SHIPPED, and the contract is about what
            // ships. Censusing it is the point.
            #[allow(deprecated)]
            v.push(("A", ZensimProfile::A));
        }
        #[cfg(feature = "candidate-profiles")]
        {
            v.push(("C", ZensimProfile::C));
            v.push(("CHdr", ZensimProfile::CHdr));
            v.push(("D", ZensimProfile::D));
        }
        v
    }

    /// The number of (profile, bake) pairs [`shipped_profiles`] contributes a
    /// SCORING bake for, under the ACTIVE feature set — the floor
    /// `every_shipped_bake_resolves_to_its_own_declared_width` and
    /// `from_block_profile_agrees_with_the_id_space_derivation` check against
    /// so a census that silently sees nothing still fails loud.
    ///
    /// `PreviewV0_1` / `PreviewV0_2` are always in the roster but carry no
    /// MLP bake (`scoring_bake_bytes()` yields nothing for them), so the
    /// floor tracks the SAME `#[cfg]` gates as `shipped_profiles` itself —
    /// `A` (1 bake) behind `deprecated-profiles`, `C`+`CHdr`+`D` (3 bakes)
    /// behind `candidate-profiles`. A bare constant here is precisely the
    /// "second list that could drift past a feature flag" this module's own
    /// doc on `shipped_profiles` warns about: found 2026-09-06 when the CI
    /// permutation matrix's `--features feature-regime-v2` cell (neither
    /// extra feature on) hit a hardcoded `>= 5` that only B+BHdr (2 bakes)
    /// can ever satisfy without them.
    pub(crate) fn expected_min_bake_count() -> usize {
        2 // B, BHdr — unconditional
            + usize::from(cfg!(feature = "deprecated-profiles")) // A
            + 3 * usize::from(cfg!(feature = "candidate-profiles")) // C, CHdr, D
    }

    /// How many of [`expected_min_bake_count`]'s bakes are DENSE
    /// (`zentrain.feature_ids`-declared) under the active feature set.
    ///
    /// `B` and `BHdr` are dense unconditionally; `A` and `D` are dense when
    /// their gating feature is on; `C` / `CHdr` are DELIBERATELY never dense
    /// — see `profile::mlp_bake_c_purity944`'s doc comment, a registered,
    /// pending user decision, not an oversight.
    pub(crate) fn expected_min_dense_count() -> usize {
        2 // B, BHdr
            + usize::from(cfg!(feature = "deprecated-profiles")) // A
            + usize::from(cfg!(feature = "candidate-profiles")) // D only — not C/CHdr
    }

    /// One census row.
    #[derive(Debug)]
    struct Row {
        name: String,
        declared_width: usize,
        outcome: Result<usize, String>,
    }

    fn census_profile(name: &str, p: ZensimProfile) -> Row {
        let (w, h) = (64usize, 64usize);
        let (src, dst) = pair(w, h);
        let declared_width = p
            .params()
            .scoring_bake_bytes()
            .filter_map(|b| crate::mlp::Model::from_bytes(b).ok())
            .map(|m| m.caller_input_width())
            .max()
            .unwrap_or(0);
        let z = Zensim::new(p);
        let outcome = z
            .compute(&RgbSlice::new(&src, w, h), &RgbSlice::new(&dst, w, h))
            .map(|r| r.features().len())
            .map_err(|e| format!("{e:?}"));
        Row {
            name: name.to_string(),
            declared_width,
            outcome,
        }
    }

    /// **Per-bake revision, and the one place its limitation is
    /// load-bearing.** A profile whose bakes declare DIFFERENT formula
    /// revisions gets NO plan, because one walk computes one arithmetic era —
    /// unioning them would silently serve one bake the other's arithmetic,
    /// which is the failure the per-bake declaration exists to prevent.
    ///
    /// Every shipped profile is single-revision today (no bake carries a
    /// `zentrain.formula_revision` stamp, so all resolve to the shipped
    /// revision), so this gate asserts both halves: the real profiles agree,
    /// and `revisions_agree` actually distinguishes when they would not.
    #[test]
    fn mixed_revision_profiles_get_no_plan() {
        use crate::feature_defs::FormulaRevision;
        // Every shipped profile's bakes agree on a revision.
        for (name, p) in shipped_profiles() {
            let revs: Vec<FormulaRevision> = p
                .params()
                .scoring_bake_bytes()
                .filter_map(|b| crate::mlp::Model::from_bytes(b).ok())
                .map(|m| crate::feature_v2::bake_formula_revision(&m))
                .collect();
            assert!(
                revs.windows(2).all(|w| w[0] == w[1]),
                "{name}: bakes declare different revisions {revs:?} — this \
                 profile cannot be served from one extraction"
            );
        }
        // And the predicate is not vacuously true: two plans at different
        // revisions must NOT agree.
        let mut a = Plan::derive(&SlotSet::from_ranges([(0, 372)]), 372).expect("plan");
        let b = Plan::derive(&SlotSet::from_ranges([(0, 372)]), 372).expect("plan");
        assert!(a.revisions_agree(&b), "same revision must agree");
        a.compute.formula_revision = match a.compute.formula_revision {
            FormulaRevision::Rev1 => FormulaRevision::Rev2,
            FormulaRevision::Rev2 => FormulaRevision::Rev1,
        };
        assert!(
            !a.revisions_agree(&b),
            "different revisions must NOT agree — otherwise the refusal in \
             `fold_engine::score_plan` can never fire"
        );
    }

    /// **Phase 5 evidence** — `ComputeSet::from_block_profile` and the
    /// ID-SPACE derivation (`Plan::derive_with_layout`) agree on every bake
    /// this build ships.
    ///
    /// `for_bake` still routes identity layouts through `from_block_profile`,
    /// because that is the tested derivation and keeping it keeps the 445-bake
    /// census on the SAME code rather than an equivalent one. This gate is
    /// what would let phase 5 collapse the two: while it holds, the id-space
    /// derivation is a drop-in, and `fold_engine::wide_bake_v2_read` (which
    /// exists only to serve `from_block_profile`'s wide branch) loses its last
    /// caller.
    ///
    /// Reported per bake on failure, so a disagreement names WHICH bake and
    /// on which axis rather than just failing.
    #[test]
    fn from_block_profile_agrees_with_the_id_space_derivation() {
        let mut checked = 0usize;
        let mut disagreements = Vec::new();
        for (name, p) in shipped_profiles() {
            for bytes in p.params().scoring_bake_bytes() {
                let Ok(m) = crate::mlp::Model::from_bytes(bytes) else {
                    continue;
                };
                let layout = crate::feature_layout::declared_layout(&m);
                // `bake_read_slots` already answers in ID space.
                let Some(want) = bake_read_slots(&m) else {
                    continue;
                };
                let legacy = Plan::normalized(ComputeSet::from_block_profile(&m), layout.clone());
                let derived = Plan::derive_with_layout(&want, layout);
                checked += 1;
                match derived {
                    Ok(d) => {
                        // The derived plan may be a STRICT SUBSET of the
                        // legacy one (it computes only what the bake reads);
                        // what it may never be is short of the read set, and
                        // its emit must be contained in the legacy emit.
                        if !d.emit.covers(&want) {
                            disagreements.push(format!(
                                "{name}: id-space derivation does not cover the read set \
                                 (missing {})",
                                d.emit.missing_from(&want)
                            ));
                        }
                        if !legacy.emit.covers(&d.emit) {
                            disagreements.push(format!(
                                "{name}: id-space emit is NOT a subset of from_block_profile's \
                                 (extra {})",
                                d.emit.missing_from(&legacy.emit)
                            ));
                        }
                    }
                    Err(e) => disagreements.push(format!(
                        "{name}: id-space derivation REFUSED a bake from_block_profile \
                         serves: {e}"
                    )),
                }
            }
        }
        assert!(
            checked >= expected_min_bake_count(),
            "the gate must see bakes, saw {checked}, expected >= {}",
            expected_min_bake_count()
        );
        assert!(
            disagreements.is_empty(),
            "the two derivations disagree on {} of {checked} bakes:\n  {}",
            disagreements.len(),
            disagreements.join("\n  ")
        );
    }

    /// **THE contract gate.** Every shipped profile serves a non-identical
    /// pair, and emits at least the width its widest bake declares.
    ///
    /// The census REPORT is printed on every run (`--nocapture` to see it) so
    /// "what cannot be served today, and why" is a measurement rather than an
    /// inference from reading `profile.rs`.
    #[test]
    fn every_shipped_profile_is_servable() {
        let rows: Vec<Row> = shipped_profiles()
            .into_iter()
            .map(|(n, p)| census_profile(n, p))
            .collect();
        // The PRE-increment-1 rule, stated exactly as `prep_bake_input_f32`
        // enforced it: the extraction emitted at most the v1 width, so a bake
        // was servable iff `caller_input_width() <= v1_width + 4` (the `+4` is
        // the optional size-axis augmentation). Reproduced here rather than
        // recalled, so the BEFORE column of the census is derived from the
        // removed condition rather than from memory.
        let v1_width = crate::NUM_SCALES
            * 3
            * (crate::metric::FEATURES_PER_CHANNEL_EXTENDED
                + crate::metric::FEATURES_PER_CHANNEL_IW);
        let old_rule = |declared: usize| declared <= v1_width + 4;
        let mut report = String::from(
            "\nSERVABILITY CENSUS — shipped profiles\n\
             (BEFORE = the removed `prep_bake_input_f32` rule: declared <= v1_width + 4)\n\
             profile      declared  emitted  BEFORE  NOW\n",
        );
        let (mut before_ok, mut after_ok) = (0usize, 0usize);
        let mut unservable = Vec::new();
        for r in &rows {
            if old_rule(r.declared_width) {
                before_ok += 1;
            }
            match &r.outcome {
                Ok(emitted) => {
                    after_ok += 1;
                    report.push_str(&format!(
                        "  {:<10} {:>8}  {:>7}  {:>6}  SERVED\n",
                        r.name,
                        r.declared_width,
                        emitted,
                        if old_rule(r.declared_width) {
                            "served"
                        } else {
                            "REFUSED"
                        }
                    ));
                    assert!(
                        *emitted >= r.declared_width,
                        "{}: emitted {emitted} < declared {}",
                        r.name,
                        r.declared_width
                    );
                }
                Err(e) => {
                    report.push_str(&format!(
                        "  {:<10} {:>8}  {:>7}  {:>6}  NOT SERVED: {e}\n",
                        r.name,
                        r.declared_width,
                        "-",
                        if old_rule(r.declared_width) {
                            "served"
                        } else {
                            "REFUSED"
                        }
                    ));
                    unservable.push(format!("{} ({e})", r.name));
                }
            }
        }
        report.push_str(&format!(
            "  ---- servable: {before_ok}/{} BEFORE, {after_ok}/{} NOW\n",
            rows.len(),
            rows.len()
        ));
        println!("{report}");
        assert!(
            unservable.is_empty(),
            "{} shipped profile(s) cannot be served: {}\n{report}",
            unservable.len(),
            unservable.join(", ")
        );
    }

    /// Every registered PRODUCER set is plannable, and the plan populates
    /// exactly the slots the registry records for it.
    ///
    /// A registered set that cannot be planned is the same defect class as an
    /// unservable bake, one step earlier: it means a name in the identity
    /// registry has no walk behind it.
    #[test]
    fn every_registered_producer_set_is_plannable() {
        let ns = crate::NUM_SCALES;
        let mut checked = 0usize;
        for (compute, width, expect) in registered_producer_sets() {
            let Some(parts) = crate::feature_set_id::ComputeParts::parse(&compute) else {
                panic!("unparseable compute {compute:?}");
            };
            let mut want = SlotSet::default();
            for t in parts.iter() {
                want = want.union(&crate::feature_defs::family_slots(t, ns));
            }
            let want = want.clipped_to(width);
            assert_eq!(want, expect, "{compute}@w{width}: registry slots");
            let plan = Plan::derive(&want, width)
                .unwrap_or_else(|e| panic!("{compute}@w{width} is not plannable: {e}"));
            assert!(
                plan.covers(&want),
                "{compute}@w{width}: plan does not cover the registered slots"
            );
            checked += 1;
        }
        assert!(checked >= 10, "expected the registered producer sets");
    }

    /// The registry's producer entries as `(compute, layout_width, slots)`.
    fn registered_producer_sets() -> Vec<(String, usize, SlotSet)> {
        let json = include_str!("../../benchmarks/feature_sets_registry.json");
        let mut out = Vec::new();
        for chunk in json.split("\"compute\":").skip(1) {
            let Some(compute) = between_quotes(chunk) else {
                continue;
            };
            let Some(width) = chunk
                .split("\"layout\":")
                .nth(1)
                .and_then(|t| t.split(',').next())
                .and_then(|t| t.trim().parse::<usize>().ok())
            else {
                continue;
            };
            let role = chunk
                .split("\"role\":")
                .nth(1)
                .and_then(between_quotes)
                .unwrap_or_default();
            if role != "producer" {
                continue;
            }
            let Some(slots) = chunk
                .split("\"slots\":")
                .nth(1)
                .and_then(between_quotes)
                .and_then(|s| SlotSet::parse(&s))
            else {
                continue;
            };
            out.push((compute, width, slots));
        }
        out
    }

    fn between_quotes(s: &str) -> Option<String> {
        let a = s.find('"')? + 1;
        let rest = &s[a..];
        let b = rest.find('"')?;
        Some(rest[..b].to_string())
    }

    /// The two campaign arms the contract names as its first concrete
    /// targets: the 265-wide (`+raw moments`) and 289-wide (`+class-C`) sets.
    /// Both must plan to the CHEAP walk — if either woke the append kernel the
    /// plan would be correct and pointless.
    #[test]
    fn the_campaign_free_set_arms_plan_to_the_cheap_walk() {
        let ns = crate::NUM_SCALES;
        let base = SlotSet::from_ranges([(0, 228)]);
        let m = base.union(&crate::feature_defs::family_slots(T::Moments, ns));
        let c = m.union(&crate::feature_defs::family_slots(T::ClassC, ns));
        for (label, want, n) in [("265 (+moments)", m, 265usize), ("289 (+classC)", c, 289)] {
            let p = Plan::derive(&want, 944).unwrap_or_else(|e| panic!("{label}: {e}"));
            assert_eq!(p.emit.len(), n, "{label}: populated slot count");
            assert_eq!(p.layout_width(), 944, "{label}: layout width");
            assert!(!p.compute.append, "{label}: append kernel must stay off");
            assert!(!p.compute.v2_blocks, "{label}: v2 blocks must stay off");
            assert!(p.covers(&want), "{label}: plan must cover the request");
        }
    }
}
