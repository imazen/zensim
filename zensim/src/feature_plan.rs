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
    /// The emitted vector width — the LAYOUT, not the compute set. A
    /// `v1_only` request at 944 is still a 944-wide row with `f372..` at the
    /// structural `0.0`.
    pub layout_width: usize,
    /// The slots this plan populates. Everything else in the layout is a
    /// structural zero.
    pub emit: SlotSet,
}

impl Plan {
    /// Plan for an explicit slot request at an explicit layout width.
    ///
    /// The derivation is one rule applied per family: a family runs iff the
    /// request touches a slot it owns. The one subtlety is the free tranches —
    /// a request that touches the append or append2 block ONLY at tranche
    /// slots is served by the free accumulators instead of the (expensive)
    /// owning kernel, which is what makes `basic+peaks+moments` a cheap plan
    /// rather than a full 944 walk.
    pub(crate) fn derive(want: &SlotSet, layout_width: usize) -> Result<Plan, PlanError> {
        let ns = crate::NUM_SCALES;
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
        let plan = Plan::normalized(requested, layout_width);
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
    fn normalized(requested: ComputeSet, layout_width: usize) -> Plan {
        let ns = crate::NUM_SCALES;
        let probe = Plan {
            compute: requested,
            layout_width,
            emit: SlotSet::from_slots([]),
        };
        let compute = ComputeSet::from_toggles(probe.toggles());
        let emit = compute.populated_slots(ns, layout_width);
        Plan {
            compute,
            layout_width,
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
        let plan = Plan::normalized(ComputeSet::from_block_profile(model), layout_width);
        let want = bake_read_slots(model).ok_or(PlanError::UnreadableBake)?;
        let want = want.clipped_to(layout_width);
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
        Plan::normalized(compute, layout_width)
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
        let layout = LayoutBlocks::for_width(self.layout_width, ns);
        crate::feature_v2::V2NewFeatureToggles {
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
        let layout_width = self.layout_width.max(other.layout_width);
        let (a, b) = (&self.compute, &other.compute);
        let compute = ComputeSet {
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
        Plan::normalized(compute, layout_width)
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

/// The caller lines a bake structurally reads, as a [`SlotSet`].
///
/// The in-crate twin of `zensim_validate::block_profile::used_caller_lines`,
/// and NOT a duplicate of it: both are thin callers of
/// [`crate::fold_engine::caller_line_reads`], which is the one owner of the
/// caller-space fold. zensim cannot depend on zensim-validate (the dependency
/// runs the other way), so the narrow primitive lives in the lower crate and
/// the validate-side function keeps its richer per-line norms.
pub(crate) fn bake_read_slots(model: &crate::mlp::Model) -> Option<SlotSet> {
    let reads = crate::fold_engine::caller_line_reads(model)?;
    Some(SlotSet::from_slots(
        reads
            .iter()
            .enumerate()
            .filter(|(_, live)| **live)
            .map(|(i, _)| i),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slots(r: impl IntoIterator<Item = (usize, usize)>) -> SlotSet {
        SlotSet::from_ranges(r)
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
        assert_eq!(p.layout_width, 372);
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
        assert_eq!(p.layout_width, 944);
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
        assert_eq!(u.layout_width, 372);
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
        assert_eq!(u2.layout_width, 944);
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
            let again = Plan::normalized(p.compute, width);
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
mod servability_census {
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
    fn shipped_profiles() -> Vec<(&'static str, ZensimProfile)> {
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
            assert_eq!(p.layout_width, 944, "{label}: layout width");
            assert!(!p.compute.append, "{label}: append kernel must stay off");
            assert!(!p.compute.v2_blocks, "{label}: v2 blocks must stay off");
            assert!(p.covers(&want), "{label}: plan must cover the request");
        }
    }
}
