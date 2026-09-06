//! **The feature DEFINITION registry** — what each slot IS, one entry per
//! signal, and the layout arithmetic that expands signals into slot ids.
//!
//! Design: `docs/FEATURE_SYSTEM_DESIGN_2026-09-05.md` §2. Phases + gates:
//! `docs/PLAN_FEATURE_SYSTEM_2026-09-05.md`.
//!
//! [`crate::feature_set_id`] answers *"are these two vectors the same kind of
//! thing?"* at the granularity of a slot FAMILY. This module answers *"what IS
//! slot 353?"* — which, before it landed, nothing in-tree could. Per-slot
//! naming existed only as a 13-entry array inside a `#[cfg(test)]` diagnostic
//! (`streaming.rs`) and four `pub mod idx*` constant blocks that give a LOCAL
//! offset inside one block.
//!
//! ## Signals, not slots
//!
//! A 944-wide vector holds 944 slots but only **85 distinct signals**: the
//! layout replicates most of them across 4 pyramid scales × 3 channels. The
//! registry stores the 85 and expands them, so declaring a new v2 signal is
//! ONE entry rather than twelve, and the (scale, channel) arithmetic has one
//! owner instead of being written out at every emit site.
//!
//! ## The layout arithmetic, in one place
//!
//! Block-major, exactly as `feature_v2`'s emit sites document it:
//!
//! ```text
//!   [0, 156)    basic    scale-major, then channel:  scale*39 + ch*13 + local
//!   [156, 228)  peaks    (scale*3 + ch)*6 + local, based at n_scales*39
//!   [228, 300)  masked   same shape,               based at n_scales*57
//!   [300, 372)  iw       same shape,               based at n_scales*75
//!   [372, 720)  v2       scale*3*29 + ch*29 + local
//!   [720, 924)  append   scale*3*17 + ch*17 + local
//!   [924, 944)  append2  scale*5 + local     (Y-only: one cell per scale)
//!   [944, 956)  csfw     scale*3 + local     (Y-only)
//! ```
//!
//! ## Why the geometry constants are DERIVED here, not copied
//!
//! `feature_v2::FEATURES_PER_CHANNEL_APPEND` is `17` because there are 17
//! append signals. This module knows that number by COUNTING its own append
//! entries, which is a derivation rather than a second copy — and
//! [`tests::geometry_matches_the_block_constants`] proves the two agree, so a
//! new signal that forgets to bump the constant fails the build rather than
//! silently shifting every slot above it.
//!
//! ## Append-only
//!
//! Ids are never renumbered (user directive 2026-07-19: *"new v2 features use
//! indices after all v1 features; we deprecate not renumber"*). A retired
//! signal keeps its `block_local` forever and gets [`SignalDef::deprecated`];
//! a revision APPENDS to [`SignalDef::revisions`] and never edits history.

use crate::feature_set_id::{ComputeToken, SlotSet};

/// Which colour plane a slot is pooled over.
///
/// [`Channel::Scalar`] is not "no channel" — it is the per-SCALE blocks
/// (append2, csfw) whose signals exist once per scale rather than once per
/// (scale, channel). They are computed on Y (`feature_v2::APPEND2_CHANNEL`)
/// but occupy one slot, not three.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Channel {
    /// XYB X (red-green chroma).
    X,
    /// XYB Y (luma).
    Y,
    /// XYB B (blue-yellow chroma).
    B,
    /// A per-scale slot with no channel axis.
    Scalar,
}

impl Channel {
    /// The per-(scale, channel) blocks index channels in this order.
    pub(crate) const TRIPLE: [Channel; 3] = [Channel::X, Channel::Y, Channel::B];

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Channel::X => "x",
            Channel::Y => "y",
            Channel::B => "b",
            Channel::Scalar => "s",
        }
    }
}

/// How a signal pools its per-pixel term over the plane.
///
/// Recorded because it is the difference between two slots that share a name
/// stem and are otherwise indistinguishable in the vector (`ssim_mean` vs
/// `ssim_4th` vs `ssim_2nd` are one per-pixel term under three pooling
/// exponents), and because the pooling exponent is what a future revision is
/// most likely to change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Statistic {
    /// Arithmetic mean of the per-pixel term.
    Mean,
    /// `(Σx²/n)^½`.
    L2,
    /// `(Σx⁴/n)^¼`.
    L4,
    /// `(Σx⁸/n)^⅛`.
    L8,
    /// Plane maximum.
    Max,
    /// `Σw·v / Σw` — the canonical weighted pooling the masked/IW blocks use.
    WeightedMean,
    /// A ratio of two pooled quantities (the HF gain/loss/mag family).
    Ratio,
    /// A mean restricted to a luminance bin.
    Bin,
    /// A whole-plane scalar that is not a pooled per-pixel term (the
    /// `GLOBAL_*` / `W_GLOBAL_*` families, computed from raw moments).
    Global,
}

impl Statistic {
    /// Stable lower-case token, for provenance output.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Statistic::Mean => "mean",
            Statistic::L2 => "l2",
            Statistic::L4 => "l4",
            Statistic::L8 => "l8",
            Statistic::Max => "max",
            Statistic::WeightedMean => "weighted_mean",
            Statistic::Ratio => "ratio",
            Statistic::Bin => "bin",
            Statistic::Global => "global",
        }
    }
}

/// What a slot costs when its OWNING BLOCK runs.
///
/// This is the signal's base cost. It is NOT the whole cost story: a slot can
/// also be [`Tranche`]-harvestable, i.e. obtainable for free by a walk that
/// never runs its owning block at all. See [`cost_of`], which is the honest
/// per-SLOT answer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CostClass {
    /// Falls out of an accumulator the plan already runs for another slot.
    Free,
    /// Shares an existing sweep; no new pass over the plane.
    Cheap,
    /// Needs its own pass, plane, or kernel.
    Expensive,
}

impl CostClass {
    /// Stable lower-case token, for provenance output.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            CostClass::Free => "free",
            CostClass::Cheap => "cheap",
            CostClass::Expensive => "expensive",
        }
    }
}

/// Whether a slot can be harvested by a v1-only walk that never runs the
/// block the slot lives in — `feature_v2::V1FreeExtras`'s two tranches.
///
/// **This is why cost is a property of a SLOT, not of a SIGNAL.** The
/// registry's first draft declared it per signal and the cross-check gate
/// caught it: `lum_dark_err` is harvestable at Y and not at X/B;
/// `global_dmean` is harvestable everywhere EXCEPT the `(B, scale 0)` append
/// cell the 944 walk skips entirely; and the v2-348 `mse` slot is harvestable
/// despite living in the most expensive block in the layout. A per-signal
/// cost field is wrong by 16 slots on the raw-moment tranche alone.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tranche {
    /// Not harvestable — needs its owning block.
    None,
    /// `V1FreeExtras::RawMoments`: derivable from the fused kernel's raw
    /// `Σs`, `Σd`, `Σs²`, `Σd²`.
    RawMoments,
    /// `V1FreeExtras::RawMomentsPlusBoundedErr`'s addition: derivable from
    /// the fused kernel's saturating-MSE and luminance-weighted sums.
    ClassC,
}

impl Tranche {
    /// Stable lower-case token, for provenance output.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Tranche::None => "none",
            Tranche::RawMoments => "raw_moments",
            Tranche::ClassC => "class_c",
        }
    }
}

/// Which (scale, channel) placements of a signal its [`Tranche`] can serve.
///
/// Mirrors the two real restrictions in `feature_v2`'s owner functions:
/// `APPEND_SKIP_B_SCALE0` (the 944 walk never computes that append cell, so
/// its slots are structural zeros on both sides) and the Y-only register-carry
/// constraint on the luminance bins.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Placement {
    /// Every cell the block occupies.
    AllCells,
    /// Every cell except the append block's `(B, scale 0)`.
    SkipBScale0,
    /// The Y channel only, and only where the append cell is computed.
    YOnlyActive,
}

/// Whether a slot is a difference between reference and distorted, and so
/// whether an identity pair must drive it to zero.
///
/// **This is why the 372 identity probe is the zero vector and the 944 one is
/// not.** `append2::LUMA_MEAN_REF` is [`Form::ReferenceOnly`] — it reports a
/// property of the reference alone, so a perfect copy leaves it at its
/// reference value, not at 0. That fact cost a lane a day and lived in one
/// benchmark doc; as a registry field it is a query.
///
/// [`Form::Undeclared`] is an HONEST state, not a default to fill in later: a
/// signal whose identity behaviour has not been established reports that,
/// rather than claiming a form nobody measured. The declared ones are pinned
/// by [`tests::declared_difference_forms_are_zero_on_an_identity_pair`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Form {
    /// A difference/dissimilarity: exactly `0.0` on an identity pair.
    Difference,
    /// A similarity: `1.0`-ish on an identity pair, higher is better.
    Similarity,
    /// Computed from the reference alone; an identity pair does not zero it.
    ReferenceOnly,
    /// Not yet established. Never treated as any of the above.
    ///
    /// Reserved for the Phase-2 research engine's provenance output
    /// (`docs/PLAN_FEATURE_SYSTEM_2026-09-05.md`); no signal in the committed
    /// registry currently declares it, so nothing constructs it yet.
    #[allow(dead_code)]
    Undeclared,
}

impl Form {
    /// Stable lower-case token, for provenance output.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Form::Difference => "difference",
            Form::Similarity => "similarity",
            Form::ReferenceOnly => "reference_only",
            Form::Undeclared => "undeclared",
        }
    }
}

/// The monotone expectation a dial gate may rely on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Direction {
    /// Larger value = more distortion.
    HigherIsWorse,
    /// Larger value = better fidelity.
    HigherIsBetter,
    /// No signed expectation (reference-only conditioners, raw statistics).
    Unsigned,
    /// Not yet established.
    ///
    /// Reserved for the Phase-2 research engine's provenance output
    /// (`docs/PLAN_FEATURE_SYSTEM_2026-09-05.md`); no signal in the committed
    /// registry currently declares it, so nothing constructs it yet.
    #[allow(dead_code)]
    Undeclared,
}

impl Direction {
    /// Stable lower-case token, for provenance output.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Direction::HigherIsWorse => "higher_is_worse",
            Direction::HigherIsBetter => "higher_is_better",
            Direction::Unsigned => "unsigned",
            Direction::Undeclared => "undeclared",
        }
    }
}

/// The kernel that owns a signal's accumulation — the "who computes this"
/// link the plan follows to decide what must run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum KernelId {
    /// v1's fused basic/edge/HF accumulation (`fused.rs`, replayed by the
    /// fold's `fold_v1_basic_bands`).
    V1Fused,
    /// v1's peak tier (max + L8), unconditional inside the fused kernel.
    V1Peaks,
    /// v1's shared activity chain feeding masked and IW together.
    V1MaskIw,
    /// The v2-348 dense block kernel.
    V2Dense,
    /// The v2 gradient kernel (GMS, ringing, banding, edge width).
    V2Gradient,
    /// The append-204 kernel.
    Append,
    /// The append2 / BANDVIS kernel.
    Append2,
    /// The CSFW tier-1 kernel.
    Csfw,
    /// The free raw-moment accumulator a v1-only walk finalizes.
    FreeRawMoments,
    /// The free bounded-error (class C) accumulator.
    FreeBoundedErr,
}

impl KernelId {
    /// Stable lower-case token, for provenance output.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            KernelId::V1Fused => "v1_fused",
            KernelId::V1Peaks => "v1_peaks",
            KernelId::V1MaskIw => "v1_mask_iw",
            KernelId::V2Dense => "v2_dense",
            KernelId::V2Gradient => "v2_gradient",
            KernelId::Append => "append",
            KernelId::Append2 => "append2",
            KernelId::Csfw => "csfw",
            KernelId::FreeRawMoments => "free_raw_moments",
            KernelId::FreeBoundedErr => "free_bounded_err",
        }
    }
}

/// Whether a [`Revision`] has LANDED (the values moved) or is PROPOSED (the
/// defect is known and modelled, the fix is not applied).
///
/// Modelling a proposed revision is not the same as making one. The
/// architecture's job is to make "which slots would this change, and what
/// would it cost to re-extract?" a lookup — so a fix can be scheduled with its
/// blast radius known, instead of discovered after it lands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RevisionStatus {
    /// The values moved; tables built before it read a different quantity.
    Landed,
    /// Registered, NOT applied. Shipped bytes are unchanged.
    Proposed,
}

impl RevisionStatus {
    /// Stable lower-case token, for provenance output.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            RevisionStatus::Landed => "landed",
            RevisionStatus::Proposed => "proposed",
        }
    }
}

/// A known defect in a signal's current definition.
///
/// Sourced from `docs/FEATURE_DEFECTS_AUDIT_2026-09-05.md`, whose ids
/// (`F4`, `F5`, `F15`, …) are the keys. A defect is a property of the SIGNAL,
/// so the registry is where it belongs: "does anything I read have a live
/// defect?" becomes a query over a bake's read set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Defect {
    /// The audit's id.
    pub id: &'static str,
    /// What is wrong, and what currently absorbs it.
    pub note: &'static str,
}

/// One registered change to how a signal's value is computed.
///
/// The era axis the identity layer records per TABLE, recorded here per
/// SIGNAL — so *"is my number affected by era X?"* is a lookup instead of a
/// re-extraction. Append-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Revision {
    /// The `feature_set_id` era token this landed in, or the token a proposed
    /// revision would create.
    pub era: &'static str,
    /// The byte-changing commit, or `"-"` for a proposal.
    pub commit: &'static str,
    /// Landed, or registered-not-applied.
    pub status: RevisionStatus,
    /// What changed (or would), and what it means for numbers read before it.
    pub note: &'static str,
}

/// WHICH registered revision's semantics a computation uses.
///
/// The [`Revision`] entries above record, per signal, which eras MOVED it.
/// This is the knob that SELECTS one — phase 3 of
/// `docs/PLAN_FEATURE_SYSTEM_2026-09-05.md`, executed by
/// `docs/PLAN_FEATURE_REV2_2026-09-05.md`.
///
/// It is deliberately a small closed enum rather than a set of era tokens:
/// eras in this crate have only ever moved forward together, and a research
/// extraction needs to reproduce "the semantics of era X", not an arbitrary
/// per-slot mixture. [`Self::era_tokens`] is the bridge back to the registry,
/// so "which slots does this revision move?" stays a lookup over the signal
/// table rather than a second list that can drift.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash, PartialOrd, Ord)]
pub enum FormulaRevision {
    /// Everything shipped through era `v1postc` — the semantics every stored
    /// table and every published verdict was read at.
    #[default]
    Rev1,
    /// The 2026-09-05 batched arithmetic-fix era: the audit's F4, F5 and F17
    /// fixes landing together, so exactly one era boundary exists rather than
    /// one per defect — one recalculation, not three.
    Rev2,
}

impl FormulaRevision {
    /// The era tokens this revision introduces, relative to [`Self::Rev1`].
    ///
    /// Empty for `Rev1`, which is the baseline rather than a change.
    pub(crate) const fn era_tokens(self) -> &'static [&'static str] {
        match self {
            Self::Rev1 => &[],
            Self::Rev2 => &["v1ssimcap", "freecomp", "v1hfgain"],
        }
    }

    /// Whether `GLOBAL_CGAIN`/`GLOBAL_CLOSS` are formed from the PAIRED
    /// second moment `Σ(d−s)(d+s)` instead of the difference of two raw
    /// variances — the `freecomp` half of revision 2.
    ///
    /// F5: `fused.rs`'s own comment claims the band-batched reorder "costs
    /// negligible precision: worst |Δ| vs the 944 append block is 5.35e-6".
    /// That was measured through `free_extras_match_the_944_append_block`,
    /// which is SYNTHETIC-IMAGE-ONLY. On real pairs the audit measures 2,607
    /// of 28,601 cells (9.12 %) past the 2e-5 bar, worst 3.63e-3 — because
    /// `Σs²/n − mean²` amplifies the accumulated f32 error by `mean²/var`,
    /// which is unbounded as a region flattens, and synthetic images do not
    /// produce the flat-but-not-constant regions that make it large.
    ///
    /// **Granularity was tried first and MEASURED not to be the lever**:
    /// matching the append kernel's per-row reduction moves the paired
    /// disagreement only 4.47 % -> 3.90 % past the bar and leaves the worst
    /// cell worse, because the error lives in the f32 accumulation WITHIN one
    /// row. Compensation removes the depth dependence instead. See
    /// [`crate::fused`]'s `kahan_add16` for the full measurement.
    ///
    /// It changes **no shipped byte**: only the free walk is compensated, the
    /// append route is untouched, so no 944 table moves. That also makes the
    /// free route the more accurate of the two, which is what turns the
    /// remaining free-vs-append gap into a MEASUREMENT of the append route's
    /// own error (plan R4) rather than an unattributed disagreement.
    pub(crate) const fn paired_global_contrast(self) -> bool {
        matches!(self, Self::Rev2)
    }

    /// Every slot id this revision moves, derived from the signal table's own
    /// [`Revision`] entries — never from a separate list. The union over
    /// [`Self::era_tokens`], so a revision that batches three defects returns
    /// all three defects' slots.
    ///
    /// This is the predicate gate **G3.1** checks against a re-extraction: a
    /// slot that moves and is not returned here is a FAILURE.
    pub(crate) fn moved_slots(self, width: u16, n_scales: usize) -> Vec<u16> {
        let tokens = self.era_tokens();
        (0..width as usize)
            .filter_map(|id| def_at(id, n_scales))
            .filter(|d| d.signal.revisions.iter().any(|r| tokens.contains(&r.era)))
            .map(|d| d.id)
            .collect()
    }
}

/// Every slot id ONE registered era moves.
///
/// [`FormulaRevision::moved_slots`] is the union over a revision's eras, which
/// is what a recalculation needs — but a G3.1 gate is a claim about ONE defect
/// ("F4 moves these 132"), and once a revision batches three of them the union
/// can no longer express that claim. Added when `v1hfgain` (F17) joined
/// revision 2 and the F4 gate's 132 became 144: the gate was asserting an
/// era-level fact through a revision-level instrument, which was correct only
/// while the revision had one v1 era in it.
pub(crate) fn era_moved_slots(era: &str, width: u16, n_scales: usize) -> Vec<u16> {
    (0..width as usize)
        .filter_map(|id| def_at(id, n_scales))
        .filter(|d| d.signal.revisions.iter().any(|r| r.era == era))
        .map(|d| d.id)
        .collect()
}

/// One registered signal: a feature definition before the layout replicates
/// it across scales and channels.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SignalDef {
    /// The slot family — the [`ComputeToken`] vocabulary, one owner.
    pub family: ComputeToken,
    /// Index within the family's per-cell block (the `idx*` constant).
    pub block_local: u16,
    /// Name stem, `[a-z0-9_]+`, unique within the family.
    pub name: &'static str,
    // `statistic`, `form`, `direction`, `kernel`, and `deprecated` are all
    // constructed (every signal declaration sets them) but not yet read back
    // anywhere — they are Phase 1a's declarative data, awaiting the Phase-2
    // research engine's provenance consumer
    // (`docs/PLAN_FEATURE_SYSTEM_2026-09-05.md`). Not dead: reserved.
    #[allow(dead_code)]
    pub statistic: Statistic,
    /// Cost when the signal's OWNING BLOCK runs. See [`cost_of`] for the
    /// per-slot answer, which also accounts for [`SignalDef::tranche`].
    pub cost: CostClass,
    /// Whether a v1-only walk can harvest this signal without its block.
    pub tranche: Tranche,
    /// Which placements the tranche can serve.
    pub placement: Placement,
    #[allow(dead_code)]
    pub form: Form,
    #[allow(dead_code)]
    pub direction: Direction,
    #[allow(dead_code)]
    pub kernel: KernelId,
    /// Retired: the slot keeps its id and its structural zero forever.
    #[allow(dead_code)]
    pub deprecated: bool,
    /// A live defect in the current definition, from the defect audit.
    pub defect: Option<Defect>,
    pub revisions: &'static [Revision],
}

/// One slot: a [`SignalDef`] placed at a (scale, channel) by the layout.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FeatureDef {
    /// The stable slot id. Append-only, never renumbered.
    ///
    /// Constructed by every layout expansion but not yet read back anywhere
    /// — reserved for the Phase-2 provenance consumer
    /// (`docs/PLAN_FEATURE_SYSTEM_2026-09-05.md`).
    #[allow(dead_code)]
    pub id: u16,
    pub signal: &'static SignalDef,
    pub scale: u8,
    pub channel: Channel,
}

impl FeatureDef {
    /// The slot's unique name: `<family>_<signal>_s<scale>_<channel>`.
    ///
    /// The family prefix is what makes it unique — `basic`'s `ssim_mean` and
    /// `v2`'s `ssim_mean` are different quantities under the same stem.
    /// Uniqueness over every registered layout is gated by
    /// [`tests::names_are_unique_and_valid`].
    pub(crate) fn name(&self) -> String {
        format!(
            "{}_{}_s{}_{}",
            self.signal.family.as_str(),
            self.signal.name,
            self.scale,
            self.channel.as_str()
        )
    }
}

// ============================================================================
// The signal table
// ============================================================================

const NO_REV: &[Revision] = &[];

/// The v1 option-C revision: v1 stopped pooling mirror-padded phantom
/// columns, which moves every pooled v1 slot at any non-tight width.
const REV_OPTION_C: &[Revision] = &[Revision {
    era: "v1postc",
    commit: "56bbcda2",
    status: RevisionStatus::Landed,
    note: "option C: v1 stopped pooling mirror-padded phantom columns. Moves \
           every pooled v1 slot at widths where simd_padded_width(w) != w \
           (512/576/1152/2304 among them). Numbers read on v1pre/v1cur cannot \
           be corrected into v1postc, only re-extracted.",
}];

/// **F4** — the one LIVE arithmetic defect
/// (`docs/FEATURE_DEFECTS_AUDIT_2026-09-05.md`).
const DEFECT_F4: Defect = Defect {
    id: "F4",
    note: "v1's SSIM per-pixel dissimilarity has a `.max(0)` floor and NO upper \
           cap, and `num_m = 1 - (mu1-mu2)^2` carries no C1. L4 pooling then \
           amplifies it: f313 reads 5,814,302 on scanned rows against a \
           photographic p99.9 of 0.48. Absorbed today by the bake-side winsor \
           guard, not by the feature. BLAST RADIUS CORRECTED 2026-09-05: the \
           audit attached this to the masked and IW blocks only (72 slots); a \
           bounded-form re-extraction MEASURES 132 — basic 36, peaks 24, \
           masked 36, IW 36 — because every one of them is pooled from the \
           SAME per-pixel `d`. The peaks pair is `ssim_max`/`ssim_l8`, which a \
           per-block reading of the audit would not have predicted.",
};

/// The PROPOSED fix for [`DEFECT_F4`] — registered, deliberately NOT applied.
const REV_F4_PROPOSED: &[Revision] = &[
    Revision {
        era: "v1postc",
        commit: "56bbcda2",
        status: RevisionStatus::Landed,
        note: "option C: v1 stopped pooling mirror-padded phantom columns.",
    },
    Revision {
        era: "v1ssimcap",
        commit: "-",
        status: RevisionStatus::Proposed,
        note: "F4 fix: bound the per-pixel dissimilarity's luminance term. The \
               ARM IS DECIDED — `SsimLumaForm::Clamp`, i.e. `max(0, 1 - D^2)` \
               — by measurement over 217,756 rows \
               (benchmarks/f4_arm_decision_2026-09-05.md). Clamp is \
               BIT-IDENTICAL to revision 1 wherever `D^2 <= 1`, and no corpus \
               with local pixels reaches past that, so this era moves NOTHING \
               on content resembling the shipped corpora; the 5.8e6 outlier \
               that motivates it lives in the bigcodec sweep. STILL PROPOSED, \
               not landed: `ssim_form::SHIPPED_REVISION` is `Rev1` until the \
               recalculation and refit land, and Proposed is what `shipped \
               bytes are unchanged` means.",
    },
];

/// **F5** — the free-40 raw-moment route-parity skew.
const DEFECT_F5: Defect = Defect {
    id: "F5",
    note: "`global_stats_from_raw_moments` uses the catastrophic-cancellation \
           form `sum(s^2)/n - (sum(s)/n)^2`, and the two routes reduce at \
           different granularities (append kernel per ROW, fused kernel per \
           BAND). 2,607 of 28,601 cells (9.12 %) exceed the 2e-5 parity bar on \
           real pairs, worst 3.63e-3. The class-C 24 slots are clean (0/18,552).",
};

/// The PROPOSED fix for [`DEFECT_F5`] — the CHEAPEST window, because no
/// shipped bake reads these slots yet.
const REV_F5_PROPOSED: &[Revision] = &[Revision {
    era: "freecomp",
    commit: "-",
    status: RevisionStatus::Proposed,
    note: "F5 fix: compensated (or two-pass) accumulation for the raw second \
           moment, so the free route and the append kernel agree. Migration \
           cost is ZERO shipped bytes TODAY — no shipped bake reads the \
           raw-moment tranche — and rises the moment one does. Land it before \
           a bake reads these slots, not after.",
}];

/// **F17** — the v1 HF-energy GAIN member is unbounded above.
///
/// Not in the 2026-09-05 audit's F1..F16 inventory; found by R6 while reading
/// its own "max over ALL slots" column, and registered here by R6b.
const DEFECT_F17: Defect = Defect {
    id: "F17",
    note: "v1's `contrast_inc` = `max(0, var_dst/var_src - 1)` divides by the \
           SOURCE term, so its numerator is unbounded by its denominator. Its \
           two siblings in the same family divide by the source term too, but \
           their numerators are `max(0, src - dst)`, which the denominator \
           bounds — measured max exactly 1.000000 for both. The `> 1e-10` gate \
           is a threshold, not a stabiliser. MEASURED over 216,756 real pairs \
           on 8 corpora: the twelve `contrast_inc` slots are the TOP TWELVE of \
           all 372 by maximum (worst 36,465.74) and the thirteenth is 1.972 — \
           x105,127 the gold holdout's own p99.9 of 0.34687. Unlike F4 it \
           fires on real corpora, on 0.0198 % of cells. Absorbed by a \
           bake-side winsor guard on Profiles A and B ONLY: Profile D (the SDR \
           default) has no transform block at all and reads f116 (max 1,380) \
           and f155 (max 2,127) raw, and CHdr reads all twelve at `identity`.",
};

/// The PROPOSED fix for [`DEFECT_F17`] — registered, deliberately NOT applied.
const REV_F17_PROPOSED: &[Revision] = &[
    Revision {
        era: "v1postc",
        commit: "56bbcda2",
        status: RevisionStatus::Landed,
        note: "option C: v1 stopped pooling mirror-padded phantom columns.",
    },
    Revision {
        era: "v1hfgain",
        commit: "-",
        status: RevisionStatus::Proposed,
        note: "F17 fix: bound the HF-energy gain member. The arm is selected \
               at `hf_gain_form::HfGainForm::REV2_HFGAIN` and decided by \
               measurement over 216,756 rows \
               (docs/PLAN_FEATURE_REV2_2026-09-05.md section 11, \
               benchmarks/feature_rev2_2026-09-05.md section 11). Unlike \
               `v1ssimcap` this era moves values on EVERY corpus with pixels — \
               `contrast_inc` is nonzero on 12 % (CID22) to 52 % (KADID) of \
               cells — so a rev2 flip re-extracts these twelve slots for real. \
               It rides in revision 2 with F4 and F5 so there is ONE era \
               boundary and one recalculation. STILL PROPOSED: \
               `ssim_form::SHIPPED_REVISION` is `Rev1`.",
    },
];

/// **F15** — `PJND_FRAGILITY` is nonzero on an identity pair.
const DEFECT_F15: Defect = Defect {
    id: "F15",
    note: "A fragility measure of an undistorted pair should be 0. It reads \
           exactly 1.0 on a v1-only 944 walk (from zeroed accumulators) and \
           0.395 on the full walk — the same slot, two artifacts. It is one of \
           the two reasons the 944 identity vector is not the zero vector; the \
           other 15 nonzero slots are correctly reference-only.",
};

const fn v1(
    family: ComputeToken,
    block_local: u16,
    name: &'static str,
    statistic: Statistic,
    kernel: KernelId,
) -> SignalDef {
    SignalDef {
        family,
        block_local,
        name,
        statistic,
        cost: CostClass::Cheap,
        tranche: Tranche::None,
        placement: Placement::AllCells,
        // Every v1 slot is a difference form: the 372 identity probe is the
        // ZERO vector on all 38 dial-grid references, measured
        // (benchmarks/dial_addressability_gate_2026-09-04.md §15).
        form: Form::Difference,
        direction: Direction::HigherIsWorse,
        kernel,
        deprecated: false,
        defect: None,
        revisions: REV_OPTION_C,
    }
}

/// A v1 signal carrying a named defect and its proposed revision.
const fn v1_with_defect(
    family: ComputeToken,
    block_local: u16,
    name: &'static str,
    statistic: Statistic,
    kernel: KernelId,
    defect: Defect,
    revisions: &'static [Revision],
) -> SignalDef {
    SignalDef {
        family,
        block_local,
        name,
        statistic,
        cost: CostClass::Cheap,
        tranche: Tranche::None,
        placement: Placement::AllCells,
        form: Form::Difference,
        direction: Direction::HigherIsWorse,
        kernel,
        deprecated: false,
        defect: Some(defect),
        revisions,
    }
}

/// A v1 signal carrying the F4 defect and its proposed revision.
const fn v1_defect(
    family: ComputeToken,
    block_local: u16,
    name: &'static str,
    statistic: Statistic,
    kernel: KernelId,
) -> SignalDef {
    v1_with_defect(
        family,
        block_local,
        name,
        statistic,
        kernel,
        DEFECT_F4,
        REV_F4_PROPOSED,
    )
}

const fn v2sig(
    block_local: u16,
    name: &'static str,
    statistic: Statistic,
    form: Form,
    direction: Direction,
    kernel: KernelId,
    tranche: Tranche,
    defect: Option<Defect>,
) -> SignalDef {
    SignalDef {
        family: ComputeToken::V2,
        block_local,
        name,
        statistic,
        cost: CostClass::Expensive,
        tranche,
        placement: Placement::AllCells,
        form,
        direction,
        kernel,
        deprecated: false,
        defect,
        revisions: NO_REV,
    }
}

const fn app(
    block_local: u16,
    name: &'static str,
    statistic: Statistic,
    form: Form,
    direction: Direction,
    kernel: KernelId,
    tranche: Tranche,
    placement: Placement,
    defect: Option<Defect>,
    revisions: &'static [Revision],
) -> SignalDef {
    SignalDef {
        family: ComputeToken::Append,
        block_local,
        name,
        statistic,
        cost: CostClass::Expensive,
        tranche,
        placement,
        form,
        direction,
        kernel,
        deprecated: false,
        defect,
        revisions,
    }
}

/// The 13 v1 basic signals per (scale, channel), in `V1BasicSums::finalize_into`
/// order. Names follow the in-tree diagnostic array in `streaming.rs`.
pub(crate) static BASIC: [SignalDef; 13] = {
    use ComputeToken::Basic as F;
    use KernelId::V1Fused as K;
    use Statistic::{L2, L4, Mean, Ratio};
    [
        // F4 reaches EVERY slot built from the per-pixel SSIM dissimilarity,
        // not only the masked and IW blocks the audit named — MEASURED, see
        // `f4_moves_exactly_the_registered_slots`.
        v1_defect(F, 0, "ssim_mean", Mean, K),
        v1_defect(F, 1, "ssim_4th", L4, K),
        v1_defect(F, 2, "ssim_2nd", L2, K),
        v1(F, 3, "edge_art_mean", Mean, K),
        v1(F, 4, "edge_art_4th", L4, K),
        v1(F, 5, "edge_art_2nd", L2, K),
        v1(F, 6, "edge_det_mean", Mean, K),
        v1(F, 7, "edge_det_4th", L4, K),
        v1(F, 8, "edge_det_2nd", L2, K),
        v1(F, 9, "mse", Mean, K),
        // The HF-energy ratio family. The two `loss` members are bounded by
        // construction (their numerator is `max(0, src - dst)`, which their
        // own denominator bounds); the `gain` member's is `max(0, dst - src)`,
        // which it does not — that asymmetry IS F17, and it is why only the
        // third of these three carries a defect.
        v1(F, 10, "var_loss", Ratio, K),
        v1(F, 11, "tex_loss", Ratio, K),
        v1_with_defect(
            F,
            12,
            "contrast_inc",
            Ratio,
            K,
            DEFECT_F17,
            REV_F17_PROPOSED,
        ),
    ]
};

/// The 6 v1 peak signals per (scale, channel), in `finalize_pools_into` order.
pub(crate) static PEAKS: [SignalDef; 6] = {
    use ComputeToken::Peaks as F;
    use KernelId::V1Peaks as K;
    use Statistic::{L8, Max};
    [
        // Same source expression as BASIC's ssim_* — see the F4 note there.
        v1_defect(F, 0, "ssim_max", Max, K),
        v1(F, 1, "edge_art_max", Max, K),
        v1(F, 2, "edge_det_max", Max, K),
        v1_defect(F, 3, "ssim_l8", L8, K),
        v1(F, 4, "edge_art_l8", L8, K),
        v1(F, 5, "edge_det_l8", L8, K),
    ]
};

/// The 6 v1 masked signals per (scale, channel).
pub(crate) static MASKED: [SignalDef; 6] = {
    use ComputeToken::Masked as F;
    use KernelId::V1MaskIw as K;
    use Statistic::{L2, L4, Mean};
    [
        // F4: the per-pixel SSIM dissimilarity these three pool is uncapped
        // and carries no C1. f241 (masked ssim_4th) and f313 (iw ssim_4th)
        // are the two worst slots the defect audit scanned.
        v1_defect(F, 0, "ssim_mean", Mean, K),
        v1_defect(F, 1, "ssim_4th", L4, K),
        v1_defect(F, 2, "ssim_2nd", L2, K),
        v1(F, 3, "edge_art_4th", L4, K),
        v1(F, 4, "edge_det_4th", L4, K),
        v1(F, 5, "mse", Mean, K),
    ]
};

/// The 6 v1 IW signals per (scale, channel) — same shape as [`MASKED`],
/// opposite pooling polarity.
pub(crate) static IW: [SignalDef; 6] = {
    use ComputeToken::Iw as F;
    use KernelId::V1MaskIw as K;
    use Statistic::{L2, L4, Mean};
    [
        // F4: the per-pixel SSIM dissimilarity these three pool is uncapped
        // and carries no C1. f241 (masked ssim_4th) and f313 (iw ssim_4th)
        // are the two worst slots the defect audit scanned.
        v1_defect(F, 0, "ssim_mean", Mean, K),
        v1_defect(F, 1, "ssim_4th", L4, K),
        v1_defect(F, 2, "ssim_2nd", L2, K),
        v1(F, 3, "edge_art_4th", L4, K),
        v1(F, 4, "edge_det_4th", L4, K),
        v1(F, 5, "mse", Mean, K),
    ]
};

/// The 29 v2 signals per (scale, channel), index-aligned with
/// `feature_v2::idx`.
pub(crate) static V2: [SignalDef; 29] = {
    use Direction::{HigherIsBetter, HigherIsWorse, Unsigned};
    use Form::{Difference, ReferenceOnly, Similarity};
    use KernelId::{V2Dense, V2Gradient};
    use Statistic::{L2, L4, Mean, WeightedMean};
    [
        // Bounded-basic block (idx 0..8).
        v2sig(
            0,
            "ssim_mean",
            Mean,
            Similarity,
            HigherIsBetter,
            V2Dense,
            Tranche::None,
            None,
        ),
        v2sig(
            1,
            "ssim_dev2",
            L2,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        v2sig(
            2,
            "ssim_dev4",
            L4,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        v2sig(
            3,
            "art",
            Mean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        v2sig(
            4,
            "det",
            Mean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        // Class C: harvestable from the fused kernel's saturating-MSE sum,
        // despite living in the most expensive block in the layout.
        v2sig(
            5,
            "mse",
            Mean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::ClassC,
            None,
        ),
        v2sig(
            6,
            "hf_gain",
            Mean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        v2sig(
            7,
            "hf_loss",
            Mean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        v2sig(
            8,
            "hf_mag_loss",
            Mean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        // Soft-saliency peak block (idx 9..11).
        v2sig(
            9,
            "ssim_soft_peak",
            WeightedMean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        v2sig(
            10,
            "art_soft_peak",
            WeightedMean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        v2sig(
            11,
            "det_soft_peak",
            WeightedMean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        // Masked block (idx 12..15).
        v2sig(
            12,
            "masked_ssim",
            WeightedMean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        v2sig(
            13,
            "masked_art",
            WeightedMean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        v2sig(
            14,
            "masked_det",
            WeightedMean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        v2sig(
            15,
            "masked_mse",
            WeightedMean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        // IW block (idx 16..19).
        v2sig(
            16,
            "iw_ssim",
            WeightedMean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        v2sig(
            17,
            "iw_art",
            WeightedMean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        v2sig(
            18,
            "iw_det",
            WeightedMean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        v2sig(
            19,
            "iw_mse",
            WeightedMean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        // Near-threshold core (idx 20..21). FRAGILITY is reference-only
        // masking-susceptibility — see `idx::PJND_FRAGILITY`'s own doc.
        v2sig(
            20,
            "pjnd_transducer",
            Mean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        // F15: nonzero on an identity pair (1.0 on a v1-only walk, 0.395 on
        // the full one). Declared ReferenceOnly because it IS computed from
        // the reference — the defect is the VALUE, not the form.
        v2sig(
            21,
            "pjnd_fragility",
            Mean,
            ReferenceOnly,
            Unsigned,
            V2Dense,
            Tranche::None,
            Some(DEFECT_F15),
        ),
        // Phase-2 additions (idx 22..28).
        v2sig(
            22,
            "gms",
            Mean,
            Difference,
            HigherIsWorse,
            V2Gradient,
            Tranche::None,
            None,
        ),
        v2sig(
            23,
            "pjnd_transducer_low_k",
            Mean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        v2sig(
            24,
            "pjnd_transducer_high_k",
            Mean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
        ),
        v2sig(
            25,
            "blockiness",
            Mean,
            Difference,
            HigherIsWorse,
            V2Gradient,
            Tranche::None,
            None,
        ),
        v2sig(
            26,
            "ringing",
            Mean,
            Difference,
            HigherIsWorse,
            V2Gradient,
            Tranche::None,
            None,
        ),
        v2sig(
            27,
            "banding",
            Mean,
            Difference,
            HigherIsWorse,
            V2Gradient,
            Tranche::None,
            None,
        ),
        // The one scale-level (not per-pixel) signal in the set — see
        // `idx::EDGE_WIDTH_CHANGE`'s doc and the v2 spec.
        v2sig(
            28,
            "edge_width_change",
            Mean,
            Difference,
            HigherIsWorse,
            V2Gradient,
            Tranche::None,
            None,
        ),
    ]
};

/// The 17 append signals per (scale, channel), index-aligned with
/// `feature_v2::idx_append`.
///
/// The three `GLOBAL_*` and `GRAD_SRC_MEAN` are the tranche a v1-only walk can
/// finalize from raw moments — [`CostClass::Free`] — which is what makes the
/// `basic+peaks+moments` plan reachable without the append kernel.
pub(crate) static APPEND: [SignalDef; 17] = {
    use Direction::{HigherIsWorse, Unsigned};
    use Form::{Difference, ReferenceOnly};
    use KernelId::{Append as KA, FreeBoundedErr as KB, FreeRawMoments as KR};
    use Placement::{AllCells, SkipBScale0, YOnlyActive};
    use Statistic::{Bin, Global, L2, Mean};
    use Tranche::{ClassC, None as NoTranche, RawMoments};
    [
        app(
            0,
            "xmask_transducer",
            Mean,
            Difference,
            HigherIsWorse,
            KA,
            NoTranche,
            AllCells,
            None,
            NO_REV,
        ),
        app(
            1,
            "lum_transducer",
            Mean,
            Difference,
            HigherIsWorse,
            KA,
            NoTranche,
            AllCells,
            None,
            NO_REV,
        ),
        // The three luminance-binned error means are the class-C tranche, and
        // ONLY at Y: the register-carry constraint the free accumulator works
        // under has no room for the chroma pair.
        app(
            2,
            "lum_dark_err",
            Bin,
            Difference,
            HigherIsWorse,
            KB,
            ClassC,
            YOnlyActive,
            None,
            NO_REV,
        ),
        app(
            3,
            "lum_mid_err",
            Bin,
            Difference,
            HigherIsWorse,
            KB,
            ClassC,
            YOnlyActive,
            None,
            NO_REV,
        ),
        app(
            4,
            "lum_bright_err",
            Bin,
            Difference,
            HigherIsWorse,
            KB,
            ClassC,
            YOnlyActive,
            None,
            NO_REV,
        ),
        app(
            5,
            "mscn_diff_mean",
            Mean,
            Difference,
            HigherIsWorse,
            KA,
            NoTranche,
            AllCells,
            None,
            NO_REV,
        ),
        app(
            6,
            "mscn_diff_l2",
            L2,
            Difference,
            HigherIsWorse,
            KA,
            NoTranche,
            AllCells,
            None,
            NO_REV,
        ),
        app(
            7,
            "contrast_gain",
            Mean,
            Difference,
            HigherIsWorse,
            KA,
            NoTranche,
            AllCells,
            None,
            NO_REV,
        ),
        app(
            8,
            "contrast_loss",
            Mean,
            Difference,
            HigherIsWorse,
            KA,
            NoTranche,
            AllCells,
            None,
            NO_REV,
        ),
        app(
            9,
            "texture_dissim",
            Mean,
            Difference,
            HigherIsWorse,
            KA,
            NoTranche,
            AllCells,
            None,
            NO_REV,
        ),
        app(
            10,
            "gms_dev2",
            L2,
            Difference,
            HigherIsWorse,
            KA,
            NoTranche,
            AllCells,
            None,
            NO_REV,
        ),
        app(
            11,
            "art_dev2",
            L2,
            Difference,
            HigherIsWorse,
            KA,
            NoTranche,
            AllCells,
            None,
            NO_REV,
        ),
        app(
            12,
            "det_dev2",
            L2,
            Difference,
            HigherIsWorse,
            KA,
            NoTranche,
            AllCells,
            None,
            NO_REV,
        ),
        // The raw-moment tranche — every append cell the 944 walk computes,
        // i.e. all but `(B, scale 0)` (`APPEND_SKIP_B_SCALE0`).
        app(
            13,
            "global_dmean",
            Global,
            Difference,
            HigherIsWorse,
            KR,
            RawMoments,
            SkipBScale0,
            Some(DEFECT_F5),
            REV_F5_PROPOSED,
        ),
        app(
            14,
            "global_cgain",
            Global,
            Difference,
            HigherIsWorse,
            KR,
            RawMoments,
            SkipBScale0,
            Some(DEFECT_F5),
            REV_F5_PROPOSED,
        ),
        app(
            15,
            "global_closs",
            Global,
            Difference,
            HigherIsWorse,
            KR,
            RawMoments,
            SkipBScale0,
            Some(DEFECT_F5),
            REV_F5_PROPOSED,
        ),
        app(
            16,
            "grad_src_mean",
            Mean,
            ReferenceOnly,
            Unsigned,
            KA,
            NoTranche,
            AllCells,
            None,
            NO_REV,
        ),
    ]
};

/// The 5 append2 signals per SCALE (Y-only), index-aligned with
/// `feature_v2::idx_append2`.
pub(crate) static APPEND2: [SignalDef; 5] = {
    use ComputeToken::Append2 as F;
    use Direction::{HigherIsWorse, Unsigned};
    use Form::{Difference, ReferenceOnly};
    use KernelId::{Append2 as KA, FreeRawMoments as KR};
    use Statistic::{Bin, Mean};
    const fn a2(
        block_local: u16,
        name: &'static str,
        statistic: Statistic,
        form: Form,
        direction: Direction,
        kernel: KernelId,
        tranche: Tranche,
        defect: Option<Defect>,
    ) -> SignalDef {
        SignalDef {
            family: F,
            block_local,
            name,
            statistic,
            cost: CostClass::Expensive,
            tranche,
            // append2 is a per-SCALE block, so there is no channel axis to
            // restrict: every cell it has is a Y cell.
            placement: Placement::AllCells,
            form,
            direction,
            kernel,
            deprecated: false,
            defect,
            revisions: NO_REV,
        }
    }
    [
        a2(
            0,
            "bandvis_gain",
            Mean,
            Difference,
            HigherIsWorse,
            KA,
            Tranche::None,
            None,
        ),
        a2(
            1,
            "bandvis_loss",
            Mean,
            Difference,
            HigherIsWorse,
            KA,
            Tranche::None,
            None,
        ),
        // THE slot that makes the 944 identity probe non-zero: a property of
        // the reference alone, so a perfect copy leaves it at its reference
        // value. Free from the raw moments a v1-only walk already has.
        a2(
            2,
            "luma_mean_ref",
            Mean,
            ReferenceOnly,
            Unsigned,
            KR,
            Tranche::RawMoments,
            Some(DEFECT_F5),
        ),
        a2(
            3,
            "hl_bin1",
            Bin,
            Difference,
            HigherIsWorse,
            KA,
            Tranche::None,
            None,
        ),
        a2(
            4,
            "hl_bin2",
            Bin,
            Difference,
            HigherIsWorse,
            KA,
            Tranche::None,
            None,
        ),
    ]
};

/// The 3 CSFW tier-1 signals per SCALE (Y-only), index-aligned with
/// `feature_v2::idx_csfw`.
pub(crate) static CSFW: [SignalDef; 3] = {
    use ComputeToken::Csfw as F;
    use Direction::HigherIsWorse;
    use Form::Difference;
    use KernelId::Csfw as K;
    use Statistic::Global;
    const fn cs(block_local: u16, name: &'static str) -> SignalDef {
        SignalDef {
            family: F,
            block_local,
            name,
            statistic: Global,
            cost: CostClass::Expensive,
            tranche: Tranche::None,
            placement: Placement::AllCells,
            form: Difference,
            direction: HigherIsWorse,
            kernel: K,
            deprecated: false,
            defect: None,
            revisions: NO_REV,
        }
    }
    [
        cs(0, "w_global_dmean"),
        cs(1, "w_global_cgain"),
        cs(2, "w_global_closs"),
    ]
};

// ============================================================================
// Layout arithmetic — THE owner
// ============================================================================

/// How a family is replicated across the (scale, channel) axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Replication {
    /// One cell per (scale, channel): 3 cells per scale.
    PerChannel,
    /// One cell per scale (the Y-only blocks).
    PerScale,
}

/// A registered block: a signal table plus where it sits in the layout.
pub(crate) struct BlockDef {
    pub family: ComputeToken,
    pub signals: &'static [SignalDef],
    pub replication: Replication,
}

/// Every block, in LAYOUT order. The base offset of each is the sum of the
/// widths of the ones before it, so inserting a block anywhere but the end
/// would renumber — which the append-only directive forbids.
pub(crate) static BLOCKS: &[BlockDef] = &[
    BlockDef {
        family: ComputeToken::Basic,
        signals: &BASIC,
        replication: Replication::PerChannel,
    },
    BlockDef {
        family: ComputeToken::Peaks,
        signals: &PEAKS,
        replication: Replication::PerChannel,
    },
    BlockDef {
        family: ComputeToken::Masked,
        signals: &MASKED,
        replication: Replication::PerChannel,
    },
    BlockDef {
        family: ComputeToken::Iw,
        signals: &IW,
        replication: Replication::PerChannel,
    },
    BlockDef {
        family: ComputeToken::V2,
        signals: &V2,
        replication: Replication::PerChannel,
    },
    BlockDef {
        family: ComputeToken::Append,
        signals: &APPEND,
        replication: Replication::PerChannel,
    },
    BlockDef {
        family: ComputeToken::Append2,
        signals: &APPEND2,
        replication: Replication::PerScale,
    },
    BlockDef {
        family: ComputeToken::Csfw,
        signals: &CSFW,
        replication: Replication::PerScale,
    },
];

impl BlockDef {
    /// Slots this block occupies at `n_scales`.
    pub(crate) fn width(&self, n_scales: usize) -> usize {
        let cells = match self.replication {
            Replication::PerChannel => n_scales * 3,
            Replication::PerScale => n_scales,
        };
        cells * self.signals.len()
    }
}

/// The base slot id of `family` at `n_scales`, and its block.
pub(crate) fn block_base(
    family: ComputeToken,
    n_scales: usize,
) -> Option<(usize, &'static BlockDef)> {
    let mut base = 0usize;
    for b in BLOCKS {
        if b.family == family {
            return Some((base, b));
        }
        base += b.width(n_scales);
    }
    None
}

/// Total layout width at `n_scales` with every registered block present.
pub(crate) fn full_width(n_scales: usize) -> usize {
    BLOCKS.iter().map(|b| b.width(n_scales)).sum()
}

/// The slot id of one signal placement.
///
/// `channel` is ignored for [`Replication::PerScale`] blocks.
pub(crate) fn slot_id(
    family: ComputeToken,
    block_local: usize,
    scale: usize,
    channel: usize,
    n_scales: usize,
) -> Option<usize> {
    let (base, block) = block_base(family, n_scales)?;
    if scale >= n_scales || block_local >= block.signals.len() {
        return None;
    }
    let per = block.signals.len();
    Some(match block.replication {
        Replication::PerChannel => {
            if channel >= 3 {
                return None;
            }
            base + (scale * 3 + channel) * per + block_local
        }
        Replication::PerScale => base + scale * per + block_local,
    })
}

/// The definition of slot `id` at `n_scales`, or `None` past the full width.
///
/// The inverse of [`slot_id`], and gated to round-trip against it.
pub(crate) fn def_at(id: usize, n_scales: usize) -> Option<FeatureDef> {
    let mut base = 0usize;
    for block in BLOCKS {
        let w = block.width(n_scales);
        if id < base + w {
            let off = id - base;
            let per = block.signals.len();
            let (scale, channel, local) = match block.replication {
                Replication::PerChannel => {
                    let cell = off / per;
                    (cell / 3, Channel::TRIPLE[cell % 3], off % per)
                }
                Replication::PerScale => (off / per, Channel::Scalar, off % per),
            };
            return Some(FeatureDef {
                id: u16::try_from(id).ok()?,
                signal: &block.signals[local],
                scale: u8::try_from(scale).ok()?,
                channel,
            });
        }
        base += w;
    }
    None
}

/// The ten `fused944native` CARRIER slots.
///
/// A hand-picked subset of peaks/masked/iw rather than a derivable rule — the
/// slots the `fused944native` tables actually carry (peak `art_l8`, and the
/// masked/IW `art_4th` at the scales the fused edge kernel reaches). Declared
/// here so the registry is complete on a `--no-default-features` build, and
/// gated equal to `V1PoolsMode::CARRIER_SLOTS` — the owner — by
/// [`tests::carrier_slots_match_the_feature_v2_owner`].
pub(crate) const CARRIER_SLOTS: [usize; 10] = [178, 190, 196, 226, 231, 237, 243, 303, 321, 333];

/// Every slot of `family` at `n_scales`, as a [`SlotSet`].
///
/// THE family→slots derivation. Covers all twelve [`ComputeToken`]s: the
/// eight contiguous blocks, the two scattered free tranches, and the carrier
/// subset. Gated equal to `ComputeSet::populated_slots` for every token by
/// [`tests::family_slots_match_compute_set_populated_slots`].
pub(crate) fn family_slots(family: ComputeToken, n_scales: usize) -> SlotSet {
    match family {
        ComputeToken::Moments => tranche_slots(Tranche::RawMoments, n_scales),
        ComputeToken::ClassC => tranche_slots(Tranche::ClassC, n_scales),
        ComputeToken::Carriers => SlotSet::from_slots(CARRIER_SLOTS),
        // `Hdr` is RESERVED — registered in the vocabulary, emitted by
        // nothing, so its slot set is legitimately empty rather than absent.
        ComputeToken::Hdr => SlotSet::default(),
        other => match block_base(other, n_scales) {
            Some((base, block)) => SlotSet::from_ranges([(base, base + block.width(n_scales))]),
            None => SlotSet::default(),
        },
    }
}

/// Does this signal's [`Placement`] admit `(scale, channel)`?
fn placement_admits(placement: Placement, scale: usize, channel: Channel) -> bool {
    match placement {
        Placement::AllCells => true,
        // `APPEND_SKIP_B_SCALE0`: the 944 walk never computes that append
        // cell, so its slots are structural zeros on BOTH sides — including
        // them would make a parity gate weaker, not stronger.
        Placement::SkipBScale0 => !(channel == Channel::B && scale == 0),
        Placement::YOnlyActive => channel == Channel::Y,
    }
}

/// The slots a v1-only walk can harvest at `tranche`, at `n_scales`.
///
/// **Derived, never listed.** Gated bit-for-bit against `feature_v2`'s owner
/// functions (`free_slot_indices` / `class_c_slot_indices`) by
/// [`tests::tranche_slots_match_the_feature_v2_owners`], so the registry
/// carries the same answer without the registry being a second copy of the
/// arithmetic: it derives the set from per-signal declarations, and the gate
/// is what makes the two provably equal.
///
/// [`Tranche::ClassC`] is the ADDITION over [`Tranche::RawMoments`], matching
/// `V1FreeExtras::RawMomentsPlusBoundedErr`'s own definition — not a superset.
pub(crate) fn tranche_slots(tranche: Tranche, n_scales: usize) -> SlotSet {
    let mut out = Vec::new();
    for id in 0..full_width(n_scales) {
        let Some(d) = def_at(id, n_scales) else {
            continue;
        };
        if d.signal.tranche == tranche
            && placement_admits(d.signal.placement, usize::from(d.scale), d.channel)
        {
            out.push(id);
        }
    }
    SlotSet::from_slots(out)
}

/// The honest per-SLOT cost: [`CostClass::Free`] when a tranche can harvest
/// this placement, else the signal's own base cost.
///
/// A per-signal cost field cannot express this — see [`Tranche`]'s doc for
/// the three ways it is wrong.
pub(crate) fn cost_of(id: usize, n_scales: usize) -> Option<CostClass> {
    let d = def_at(id, n_scales)?;
    let harvestable = d.signal.tranche != Tranche::None
        && placement_admits(d.signal.placement, usize::from(d.scale), d.channel);
    Some(if harvestable {
        CostClass::Free
    } else {
        d.signal.cost
    })
}

/// Every registered signal, in layout order — the registry's iteration entry.
///
/// Not yet called (reserved for the Phase-2 research engine's iteration over
/// the registry, `docs/PLAN_FEATURE_SYSTEM_2026-09-05.md`); kept because it is
/// the documented entry point for that consumer, not because it runs today.
#[allow(dead_code)]
pub(crate) fn signals() -> impl Iterator<Item = &'static SignalDef> {
    BLOCKS.iter().flat_map(|b| b.signals.iter())
}

#[cfg(test)]
mod tests {
    /// **G3.1 for revision 2's `v1ssimcap` era** — the registry's claim about
    /// WHICH slots the F4 fix moves, checked against a re-extraction.
    ///
    /// MEASURED 2026-09-05 (`/mnt/v/output/zensim/rev2-2026-09-05/`): the
    /// `feature_invariant_probe dump` set, 22,396 `to_bits()` cells over 3
    /// routes, re-run with `ZENSIM_SSIM_LUMA=c1` and diffed against the
    /// shipped form, moves exactly **132** slots — basic 36, peaks 24, masked
    /// 36, IW 36.
    ///
    /// **This gate FAILED when it was first written, and that is the finding.**
    /// `docs/FEATURE_DEFECTS_AUDIT_2026-09-05.md` scopes F4 to "the `ssim_*`
    /// family in `f228..299` + `f300..371`" — 72 slots — because that is where
    /// the 5.8e6 outliers were observed. But the basic block's
    /// `ssim_mean/4th/2nd` and the peaks block's `ssim_max`/`ssim_l8` are
    /// pooled from the SAME per-pixel `d`, so a bounded form moves them too.
    /// Observing where a defect's SYMPTOM is largest is not the same as
    /// deriving where its CAUSE reaches; the registry now carries the derived
    /// answer.
    ///
    /// The peaks pair is the part no per-block reading would have predicted:
    /// `ssim_max` and `ssim_l8` are local 0 and local 3 of a 6-signal block,
    /// so the moved ids are an every-third-slot comb, not a contiguous run.
    ///
    /// ⚠ The instrument changed on 2026-09-05 when F17's `v1hfgain` joined
    /// revision 2: this asserts an ERA's slot set, so it now reads
    /// [`super::era_moved_slots`]`("v1ssimcap", …)` rather than
    /// `Rev2.moved_slots`, whose union is 144. The CLAIM (132, and which 132)
    /// is unchanged and is still checked exactly — see `era_moved_slots`'s own
    /// doc for why the revision-level form could no longer express it.
    #[test]
    fn f4_moves_exactly_the_registered_slots() {
        use crate::NUM_SCALES;
        let got = super::era_moved_slots("v1ssimcap", 372, NUM_SCALES);

        // Derived independently of the signal table, from the layout the
        // measurement exposed: (block base, per-cell width, local ids).
        let mut want: Vec<u16> = Vec::new();
        for (base, per_cell, locals) in [
            (0u16, 13u16, &[0u16, 1, 2][..]), // basic  ssim_mean/4th/2nd
            (156, 6, &[0, 3][..]),            // peaks  ssim_max / ssim_l8
            (228, 6, &[0, 1, 2][..]),         // masked ssim_mean/4th/2nd
            (300, 6, &[0, 1, 2][..]),         // IW     ssim_mean/4th/2nd
        ] {
            for cell in 0..(NUM_SCALES as u16 * 3) {
                for &l in locals {
                    want.push(base + cell * per_cell + l);
                }
            }
        }
        want.sort_unstable();

        assert_eq!(
            got, want,
            "the registry's v1ssimcap slot set disagrees with the measured one"
        );
        assert_eq!(
            got.len(),
            132,
            "expected 132 moved slots, got {}",
            got.len()
        );
        // The audit's two worst columns must be in it.
        for id in [241u16, 313] {
            assert!(
                got.contains(&id),
                "f{id} (an audit-named worst column) is missing"
            );
        }
    }

    /// Revision 1 is the baseline, so it moves nothing by definition.
    #[test]
    fn rev1_moves_no_slots() {
        use super::FormulaRevision;
        use crate::NUM_SCALES;
        assert!(
            FormulaRevision::Rev1
                .moved_slots(944, NUM_SCALES)
                .is_empty()
        );
        assert!(FormulaRevision::Rev1.era_tokens().is_empty());
    }

    /// **G3.1 for revision 2's `v1hfgain` era** — F17's twelve slots, derived
    /// from the layout rather than copied from a list.
    ///
    /// `contrast_inc` is block-local 12 of the 13-signal basic block, so the
    /// moved ids are `13·cell + 12` for the twelve (scale, channel) cells.
    /// Unlike `v1ssimcap`, this set does NOT change with pool state: the basic
    /// block is present in every registered layout, so a 944 read moves the
    /// same twelve and only those twelve.
    #[test]
    fn f17_moves_exactly_the_twelve_contrast_inc_slots() {
        use crate::NUM_SCALES;
        let want: Vec<u16> = (0..NUM_SCALES as u16 * 3).map(|c| c * 13 + 12).collect();
        assert_eq!(want.len(), 12);
        for width in [372u16, 944] {
            let got = super::era_moved_slots("v1hfgain", width, NUM_SCALES);
            assert_eq!(
                got, want,
                "v1hfgain slot set at width {width} disagrees with the layout"
            );
        }
        // The four slots the R6 record names as the corpus-wise worst, and the
        // two the SDR default reads unguarded.
        for id in [12u16, 38, 129, 116, 155] {
            assert!(want.contains(&id), "f{id} missing from the F17 set");
        }
        // Its two bounded siblings must NOT be in it — F17 is the asymmetry,
        // and a fix that moved `var_loss` would be a different, larger change.
        for id in [10u16, 11, 23, 24] {
            assert!(!want.contains(&id), "f{id} (a loss member) must not move");
        }
    }

    /// Revision 2 carries all THREE registered eras, and `freecomp`'s slots are
    /// reachable only at a width that has an append block — which is exactly
    /// why F5 is free today and stops being free the moment a bake declares
    /// one.
    #[test]
    fn rev2_carries_all_eras_and_f5_needs_the_append_block() {
        use super::FormulaRevision;
        use crate::NUM_SCALES;
        let toks = FormulaRevision::Rev2.era_tokens();
        for want in ["v1ssimcap", "freecomp", "v1hfgain"] {
            assert!(toks.contains(&want), "{want} missing from {toks:?}");
        }

        let at372 = FormulaRevision::Rev2.moved_slots(372, NUM_SCALES);
        let at944 = FormulaRevision::Rev2.moved_slots(944, NUM_SCALES);
        // 132 (v1ssimcap) + 12 (v1hfgain); the two sets are disjoint, which is
        // itself the check that batching three defects did not double-count.
        assert_eq!(at372.len(), 144, "372 sees the two v1 eras' slots");
        assert_eq!(
            super::era_moved_slots("v1ssimcap", 372, NUM_SCALES).len()
                + super::era_moved_slots("v1hfgain", 372, NUM_SCALES).len(),
            at372.len(),
            "the two v1 eras must move disjoint slot sets"
        );
        assert!(
            at944.len() > at372.len(),
            "944 must additionally reach freecomp's raw-moment slots: {} vs {}",
            at944.len(),
            at372.len()
        );
        // Every 372 slot is still in the 944 set — a revision appends, it
        // never drops.
        for id in &at372 {
            assert!(at944.contains(id), "f{id} vanished at 944");
        }
    }

    use super::*;

    const NS: usize = crate::NUM_SCALES;

    /// **G1.4** — `def_at` and `slot_id` are inverses on every slot of the
    /// full layout. A layout-arithmetic typo cannot survive this.
    #[test]
    fn id_arithmetic_round_trips_on_every_slot() {
        let w = full_width(NS);
        assert_eq!(w, 956, "full registered width at 4 scales");
        for id in 0..w {
            let d = def_at(id, NS).unwrap_or_else(|| panic!("no def for slot {id}"));
            let ch = match d.channel {
                Channel::X => 0,
                Channel::Y => 1,
                Channel::B => 2,
                Channel::Scalar => 0,
            };
            let back = slot_id(
                d.signal.family,
                usize::from(d.signal.block_local),
                usize::from(d.scale),
                ch,
                NS,
            )
            .unwrap_or_else(|| panic!("slot_id failed for slot {id}"));
            assert_eq!(back, id, "round-trip for slot {id} ({})", d.name());
        }
        assert!(def_at(w, NS).is_none(), "past the full width");
    }

    /// The block bases are the numbers every emit site in `feature_v2` writes
    /// out by hand. Pinned here so the registry cannot drift from them.
    #[test]
    fn block_bases_match_the_documented_layout() {
        let expect = [
            (ComputeToken::Basic, 0usize, 156usize),
            (ComputeToken::Peaks, 156, 72),
            (ComputeToken::Masked, 228, 72),
            (ComputeToken::Iw, 300, 72),
            (ComputeToken::V2, 372, 348),
            (ComputeToken::Append, 720, 204),
            (ComputeToken::Append2, 924, 20),
            (ComputeToken::Csfw, 944, 12),
        ];
        for (family, base, width) in expect {
            let (b, blk) = block_base(family, NS).expect("registered family");
            assert_eq!(b, base, "{family} base");
            assert_eq!(blk.width(NS), width, "{family} width");
        }
    }

    /// **G1.3** — every slot name is unique and obeys the `zenanalyze-api`
    /// name charset (`[a-z0-9_]+`), which the feature-set id grammar reuses.
    #[test]
    fn names_are_unique_and_valid() {
        let mut seen = std::collections::BTreeSet::new();
        for id in 0..full_width(NS) {
            let n = def_at(id, NS).expect("def").name();
            assert!(
                crate::feature_set_id::is_valid_token(&n),
                "slot {id} name {n:?} is not [a-z0-9_]+"
            );
            assert!(seen.insert(n.clone()), "duplicate slot name {n:?} at {id}");
        }
        assert_eq!(seen.len(), 956);
    }

    /// Signal names are unique WITHIN a family (the family prefix is what
    /// disambiguates `basic`'s `ssim_mean` from `v2`'s).
    #[test]
    fn signal_names_are_unique_within_each_family() {
        for block in BLOCKS {
            let mut seen = std::collections::BTreeSet::new();
            for s in block.signals {
                assert!(
                    seen.insert(s.name),
                    "{} has two signals named {:?}",
                    block.family,
                    s.name
                );
            }
        }
    }

    /// `block_local` is the array index — i.e. the registry's order IS the
    /// `idx*` constant order, which is what index-aligns it with the emit
    /// sites.
    #[test]
    fn block_local_equals_the_array_index() {
        for block in BLOCKS {
            for (i, s) in block.signals.iter().enumerate() {
                assert_eq!(usize::from(s.block_local), i, "{} signal {i}", block.family);
                assert_eq!(s.family, block.family, "{} signal {i} family", block.family);
            }
        }
    }

    /// The free tranches are DERIVED from the per-signal declarations, and
    /// the counts are the ones `feature_v2`'s own docs state.
    #[test]
    fn free_tranches_are_derivable_from_the_declarations() {
        let raw = tranche_slots(Tranche::RawMoments, NS);
        let bounded = tranche_slots(Tranche::ClassC, NS);
        // 3 GLOBAL_* over the 11 append cells the 944 walk computes (all but
        // `(B, scale 0)`) = 33, plus LUMA_MEAN_REF per scale = 4.
        assert_eq!(raw.len(), 37, "raw-moment tranche");
        // The v2-348 MSE slot per (scale, channel) = 12, plus the three
        // luminance bins at Y only = 12.
        assert_eq!(bounded.len(), 24, "class-C tranche");
        for id in raw.iter_slots().chain(bounded.iter_slots()) {
            assert_eq!(
                cost_of(id, NS).expect("cost"),
                CostClass::Free,
                "slot {id} is in a tranche but not costed Free"
            );
        }
    }

    /// The two tranches are DISJOINT — class C is the ADDITION over the raw
    /// moments, matching `V1FreeExtras::RawMomentsPlusBoundedErr`'s own
    /// definition rather than being a superset.
    #[test]
    fn the_two_tranches_are_disjoint() {
        let raw = tranche_slots(Tranche::RawMoments, NS);
        let bounded = tranche_slots(Tranche::ClassC, NS);
        assert!(
            raw.intersect(&bounded).is_empty(),
            "tranches overlap: {}",
            raw.intersect(&bounded)
        );
    }

    /// A slot outside every tranche keeps its block's base cost — the
    /// negative control for `cost_of`.
    #[test]
    fn slots_outside_a_tranche_keep_their_block_cost() {
        // `(B, scale 0)` global_dmean is in the tranche's SIGNAL but not its
        // placement: the 944 walk never computes that append cell.
        let skipped = slot_id(ComputeToken::Append, 13, 0, 2, NS).expect("slot");
        assert_eq!(cost_of(skipped, NS), Some(CostClass::Expensive));
        assert!(!tranche_slots(Tranche::RawMoments, NS).contains(skipped));
        // The same signal at `(X, scale 0)` IS harvestable.
        let kept = slot_id(ComputeToken::Append, 13, 0, 0, NS).expect("slot");
        assert_eq!(cost_of(kept, NS), Some(CostClass::Free));
        // A luminance bin at X is not harvestable; at Y it is.
        let lum_x = slot_id(ComputeToken::Append, 2, 1, 0, NS).expect("slot");
        let lum_y = slot_id(ComputeToken::Append, 2, 1, 1, NS).expect("slot");
        assert_eq!(cost_of(lum_x, NS), Some(CostClass::Expensive));
        assert_eq!(cost_of(lum_y, NS), Some(CostClass::Free));
    }

    /// **The identity decomposition, pinned to the registry.**
    ///
    /// The audit measured that a 944-wide identity vector resolves into
    /// exactly three populations: 15 reference-only slots (correct by design),
    /// 12 `PJND_FRAGILITY` slots (defect F15), and fp residue <= 1.12e-3. The
    /// registry's `Form` flag must reproduce the first two counts from its own
    /// declarations, or it does not encode what the identity probe measures.
    ///
    /// 15 = `grad_src_mean` at the 11 append cells the 944 walk computes
    /// (12 minus the `(B, scale 0)` cell) + `luma_mean_ref` once per scale.
    #[test]
    fn identity_nonzero_slots_decompose_exactly_as_the_audit_measured() {
        let w944 = 944usize;
        let mut ref_only = Vec::new();
        let mut fragility = Vec::new();
        for id in 0..w944 {
            let d = def_at(id, NS).expect("def");
            if d.signal.name == "pjnd_fragility" {
                fragility.push(id);
                continue;
            }
            if d.signal.form == Form::ReferenceOnly
                && placement_admits(d.signal.placement, usize::from(d.scale), d.channel)
                // `grad_src_mean` lives in the append block, which the 944
                // walk skips at `(B, scale 0)` — the same cell the raw-moment
                // tranche skips, for the same reason.
                && !(d.signal.family == ComputeToken::Append
                    && d.channel == Channel::B
                    && d.scale == 0)
            {
                ref_only.push(id);
            }
        }
        assert_eq!(
            ref_only.len(),
            15,
            "reference-only slots at 944 (audit measured 15): {ref_only:?}"
        );
        assert_eq!(
            fragility.len(),
            12,
            "PJND_FRAGILITY slots at 944 (audit measured 12): {fragility:?}"
        );
        // And the artifact is registered as a defect, not silently accepted.
        let f = def_at(fragility[0], NS).expect("def");
        assert_eq!(
            f.signal.defect.map(|d| d.id),
            Some("F15"),
            "PJND_FRAGILITY's identity artifact must be a REGISTERED defect"
        );
    }

    /// Every audit defect the registry models is attached to the slots the
    /// audit named, and to no others.
    #[test]
    fn registered_defects_cover_exactly_the_audited_slots() {
        let mut by_id: std::collections::BTreeMap<&str, Vec<usize>> = Default::default();
        for id in 0..full_width(NS) {
            if let Some(d) = def_at(id, NS).expect("def").signal.defect {
                by_id.entry(d.id).or_default().push(id);
            }
        }
        // F4: every slot pooled from the per-pixel SSIM dissimilarity.
        //
        // WIDENED 2026-09-05 from 72 to 132, by MEASUREMENT rather than by
        // re-reading the audit. This assertion previously encoded the audit's
        // own scoping — "the `ssim_*` family in `f228..299` + `f300..371`",
        // i.e. masked + IW only — which is where the 5.8e6 SYMPTOM was
        // observed. Re-extracting the probe's 22,396-cell dump under a bounded
        // luminance form moves 132 slots: basic 36 (`ssim_mean/4th/2nd`),
        // peaks 24 (`ssim_max`, `ssim_l8`), masked 36, IW 36. They share one
        // per-pixel `d`, so they share its defect.
        //
        // This is a WIDENING of the claim, not a relaxation: the registry now
        // asserts the defect reaches 60 MORE slots than before, and
        // `f4_moves_exactly_the_registered_slots` holds that set against a
        // re-extraction so it cannot drift back to the smaller number.
        assert_eq!(by_id.get("F4").map(Vec::len), Some(132), "F4 slot count");
        // The audit's two worst scanned slots must be inside it.
        for named in [241usize, 313] {
            assert!(
                by_id["F4"].contains(&named),
                "F4 must cover f{named}, the audit's worst scanned slot"
            );
        }
        // F5: the three GLOBAL_* signals over the 11 computed append cells,
        // plus `luma_mean_ref` per scale — the raw-moment tranche exactly.
        let f5: std::collections::BTreeSet<usize> = by_id["F5"].iter().copied().collect();
        let tranche: std::collections::BTreeSet<usize> = tranche_slots(Tranche::RawMoments, NS)
            .iter_slots()
            .collect();
        assert!(
            tranche.is_subset(&f5),
            "F5 must cover the whole raw-moment tranche"
        );
        // F15: 12 slots.
        assert_eq!(by_id.get("F15").map(Vec::len), Some(12), "F15 slot count");
        // F17: the twelve `contrast_inc` slots — and ONLY those. The two
        // bounded members of the same family must stay defect-free, because
        // the asymmetry is the defect: attaching F17 to `var_loss` would
        // triple a 12-slot recalculation to fix nothing.
        assert_eq!(by_id.get("F17").map(Vec::len), Some(12), "F17 slot count");
        for id in [12usize, 38, 116, 129, 155] {
            assert!(by_id["F17"].contains(&id), "F17 must cover f{id}");
        }
        for id in 0..full_width(NS) {
            let d = def_at(id, NS).expect("def");
            if d.signal.name == "var_loss" || d.signal.name == "tex_loss" {
                assert!(
                    d.signal.defect.is_none(),
                    "f{id} ({}) is bounded by construction and must carry no defect",
                    d.signal.name
                );
            }
        }
    }

    /// A PROPOSED revision must not claim to have landed, and every LANDED
    /// revision must name a real commit. This is what keeps "modelled" and
    /// "applied" from blurring: F4's fix is registered and deliberately NOT
    /// applied, and the registry has to say so in a machine-checkable way.
    #[test]
    fn proposed_revisions_are_distinguishable_from_landed_ones() {
        let (mut landed, mut proposed) = (0usize, 0usize);
        for sig in signals() {
            for r in sig.revisions {
                match r.status {
                    RevisionStatus::Landed => {
                        assert_ne!(
                            r.commit, "-",
                            "{}: landed revision needs a commit",
                            sig.name
                        );
                        landed += 1;
                    }
                    RevisionStatus::Proposed => {
                        assert_eq!(
                            r.commit, "-",
                            "{}: a PROPOSED revision must not name a commit",
                            sig.name
                        );
                        proposed += 1;
                    }
                }
            }
        }
        assert!(landed > 0 && proposed > 0, "expected both kinds");
        // The three proposals revision 2 batches.
        let eras: std::collections::BTreeSet<&str> = signals()
            .flat_map(|s| s.revisions.iter())
            .filter(|r| r.status == RevisionStatus::Proposed)
            .map(|r| r.era)
            .collect();
        assert!(eras.contains("v1ssimcap"), "F4's proposed revision");
        assert!(eras.contains("freecomp"), "F5's proposed revision");
        assert!(eras.contains("v1hfgain"), "F17's proposed revision");
    }

    /// Every v1 slot carries the option-C revision; nothing else claims it.
    #[test]
    fn option_c_revision_is_scoped_to_the_v1_families() {
        for block in BLOCKS {
            let v1_family = matches!(
                block.family,
                ComputeToken::Basic | ComputeToken::Peaks | ComputeToken::Masked | ComputeToken::Iw
            );
            for s in block.signals {
                let has_c = s.revisions.iter().any(|r| r.commit == "56bbcda2");
                assert_eq!(
                    has_c, v1_family,
                    "{}::{} option-C revision presence",
                    block.family, s.name
                );
            }
        }
    }
}

/// Cross-checks against the `feature_v2` owners — the gates that make the
/// registry's independent derivations provably equal to the code that
/// actually emits the rows, rather than a plausible second opinion.
///
/// Gated on `feature-regime-v2` because the owners are: the registry itself
/// is pure data and builds without it.
#[cfg(all(test, feature = "feature-regime-v2"))]
mod owner_gates {
    use super::*;
    use crate::feature_set_id::ComputeToken as T;
    use crate::feature_v2;

    const NS: usize = crate::NUM_SCALES;

    /// The block widths this registry derives BY COUNTING its own signals must
    /// equal the geometry constants `feature_v2` and `metric` declare. A new
    /// signal that forgets to bump its block's constant fails here rather than
    /// silently shifting every slot above it.
    #[test]
    fn geometry_matches_the_block_constants() {
        assert_eq!(BASIC.len(), crate::metric::FEATURES_PER_CHANNEL_BASIC);
        assert_eq!(
            BASIC.len() + PEAKS.len(),
            crate::metric::FEATURES_PER_CHANNEL_WITH_PEAKS
        );
        assert_eq!(
            BASIC.len() + PEAKS.len() + MASKED.len(),
            crate::metric::FEATURES_PER_CHANNEL_EXTENDED
        );
        assert_eq!(IW.len(), crate::metric::FEATURES_PER_CHANNEL_IW);
        assert_eq!(V2.len(), feature_v2::FEATURES_PER_CHANNEL_V2_TOTAL);
        assert_eq!(APPEND.len(), feature_v2::FEATURES_PER_CHANNEL_APPEND);
        assert_eq!(APPEND2.len(), feature_v2::APPEND2_PER_SCALE);
        assert_eq!(CSFW.len(), feature_v2::CSFW_PER_SCALE);
    }

    /// The registry's `block_local` ordering IS the `idx*` constant ordering —
    /// which is what index-aligns a `FeatureDef` with the value at that slot.
    /// Spot-checked at every block's named boundaries.
    #[test]
    fn block_local_indices_match_the_idx_modules() {
        use feature_v2::{idx, idx_append, idx_append2, idx_csfw};
        let find = |t: &[SignalDef], name: &str| -> usize {
            usize::from(
                t.iter()
                    .find(|s| s.name == name)
                    .unwrap_or_else(|| panic!("no signal {name}"))
                    .block_local,
            )
        };
        assert_eq!(find(&V2, "ssim_mean"), idx::SSIM_MEAN);
        assert_eq!(find(&V2, "mse"), idx::MSE);
        assert_eq!(find(&V2, "hf_gain"), idx::HF_GAIN);
        assert_eq!(find(&V2, "pjnd_fragility"), idx::PJND_FRAGILITY);
        assert_eq!(find(&V2, "gms"), idx::GMS);
        assert_eq!(find(&V2, "edge_width_change"), idx::EDGE_WIDTH_CHANGE);
        assert_eq!(
            find(&APPEND, "xmask_transducer"),
            idx_append::XMASK_TRANSDUCER
        );
        assert_eq!(find(&APPEND, "lum_dark_err"), idx_append::LUM_DARK_ERR);
        assert_eq!(find(&APPEND, "lum_bright_err"), idx_append::LUM_BRIGHT_ERR);
        assert_eq!(find(&APPEND, "global_dmean"), idx_append::GLOBAL_DMEAN);
        assert_eq!(find(&APPEND, "global_closs"), idx_append::GLOBAL_CLOSS);
        assert_eq!(find(&APPEND, "grad_src_mean"), idx_append::GRAD_SRC_MEAN);
        assert_eq!(find(&APPEND2, "bandvis_gain"), idx_append2::BANDVIS_GAIN);
        assert_eq!(find(&APPEND2, "luma_mean_ref"), idx_append2::LUMA_MEAN_REF);
        assert_eq!(find(&APPEND2, "hl_bin2"), idx_append2::HL_BIN2);
        assert_eq!(find(&CSFW, "w_global_dmean"), idx_csfw::W_GLOBAL_DMEAN);
        assert_eq!(find(&CSFW, "w_global_closs"), idx_csfw::W_GLOBAL_CLOSS);
    }

    /// **The tranche gate.** The registry derives the free tranches from
    /// per-signal `Tranche` + `Placement` declarations; `feature_v2` derives
    /// them from hand-written loops. They must be the SAME SET.
    ///
    /// This gate is what caught the registry's first draft, which declared
    /// cost per SIGNAL and was wrong by 16 slots (it gave the luminance bins
    /// to all three channels and the `GLOBAL_*` family the `(B, scale 0)` cell
    /// the 944 walk never computes).
    #[test]
    fn tranche_slots_match_the_feature_v2_owners() {
        let raw_owner = SlotSet::from_slots(feature_v2::free_slot_indices(NS));
        assert_eq!(
            tranche_slots(Tranche::RawMoments, NS),
            raw_owner,
            "raw-moment tranche disagrees with feature_v2::free_slot_indices"
        );
        let cc_owner = SlotSet::from_slots(feature_v2::class_c_slot_indices(NS));
        assert_eq!(
            tranche_slots(Tranche::ClassC, NS),
            cc_owner,
            "class-C tranche disagrees with feature_v2::class_c_slot_indices"
        );
    }

    /// The carrier list is the owner's list.
    #[test]
    fn carrier_slots_match_the_feature_v2_owner() {
        assert_eq!(
            CARRIER_SLOTS,
            feature_v2::V1PoolsMode::CARRIER_SLOTS,
            "carrier slot list disagrees with V1PoolsMode::CARRIER_SLOTS"
        );
    }

    /// **G1.1** — for every registered `ComputeToken`, the registry's slot set
    /// equals what a `ComputeSet` with exactly that family on POPULATES.
    ///
    /// This is the load-bearing one: it makes the registry's family→slots
    /// derivation and `ComputeSet::populated_slots` provably the same
    /// function, so the plan can be built from either without them drifting.
    #[test]
    fn family_slots_match_compute_set_populated_slots() {
        use feature_v2::{ComputeSet, V1FreeExtras, V1PoolsMode};
        let width = full_width(NS);
        let off = ComputeSet {
            formula_revision: crate::ssim_form::active_revision(),
            v1_basic: false,
            v1_pools: V1PoolsMode::Off,
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
        // One `ComputeSet` per token that turns on EXACTLY that family.
        let cases: [(T, ComputeSet); 9] = [
            (
                T::Basic,
                ComputeSet {
                    v1_basic: true,
                    ..off
                },
            ),
            (
                T::Carriers,
                ComputeSet {
                    v1_pools: V1PoolsMode::Carriers,
                    ..off
                },
            ),
            (
                T::V2,
                ComputeSet {
                    v2_blocks: true,
                    ..off
                },
            ),
            (
                T::Append,
                ComputeSet {
                    append: true,
                    ..off
                },
            ),
            (
                T::Append2,
                ComputeSet {
                    append2: true,
                    ..off
                },
            ),
            (T::Csfw, ComputeSet { csfw: true, ..off }),
            // `Peaks` alone is expressible; `Masked`/`Iw` are not (v1's pool
            // modes turn the two on together), so they are checked as the
            // DIFFERENCE between `Full` and `Peaks` below.
            (
                T::Peaks,
                ComputeSet {
                    v1_pools: V1PoolsMode::Peaks,
                    ..off
                },
            ),
            (
                T::Moments,
                ComputeSet {
                    free_extras: V1FreeExtras::RawMoments,
                    ..off
                },
            ),
            (
                T::ClassC,
                ComputeSet {
                    free_extras: V1FreeExtras::RawMomentsPlusBoundedErr,
                    ..off
                },
            ),
        ];
        for (token, cs) in cases {
            let owner = cs.populated_slots(NS, width);
            let mine = match token {
                // `RawMomentsPlusBoundedErr` populates BOTH tranches, so the
                // registry's comparison set is the union — matching the
                // owner's own `raw_moments() || bounded_err()` shape.
                T::ClassC => family_slots(T::Moments, NS).union(&family_slots(T::ClassC, NS)),
                t => family_slots(t, NS),
            };
            assert_eq!(mine, owner, "family {token}: registry vs populated_slots");
        }
        // `Masked` + `Iw` = `Full` minus `Peaks`.
        let full = ComputeSet {
            v1_pools: V1PoolsMode::Full,
            ..off
        }
        .populated_slots(NS, width);
        let peaks = ComputeSet {
            v1_pools: V1PoolsMode::Peaks,
            ..off
        }
        .populated_slots(NS, width);
        let mask_iw = SlotSet::from_slots(full.iter_slots().filter(|s| !peaks.contains(*s)));
        assert_eq!(
            family_slots(T::Masked, NS).union(&family_slots(T::Iw, NS)),
            mask_iw,
            "masked+iw: registry vs (Full minus Peaks)"
        );
    }

    /// A minimal field reader for the committed registry JSON — this crate
    /// has no serde dep and is not getting one for a test.
    fn json_str_field(obj: &str, key: &str) -> Option<String> {
        let pat = format!("\"{key}\"");
        let at = obj.find(&pat)? + pat.len();
        let rest = &obj[at..];
        let colon = rest.find(':')? + 1;
        let rest = rest[colon..].trim_start();
        // A JSON `null` is a real value here (a `kind: "class"` entry pins the
        // class and deliberately records no slot set), not a missing field —
        // returning the NEXT field's string for it is how a naive scan lies.
        if rest.starts_with("null") {
            return None;
        }
        let open = rest.find('"')? + 1;
        let rest = &rest[open..];
        let close = rest.find('"')?;
        Some(rest[..close].to_string())
    }

    /// Each `sets` entry as `(key, object-text)`.
    fn registry_sets(json: &str) -> Vec<(String, String)> {
        let sets_at = json.find("\"sets\"").expect("registry has a sets section");
        let body = &json[sets_at..];
        let mut out = Vec::new();
        let bytes = body.as_bytes();
        let mut i = 0usize;
        while i < bytes.len() {
            // A key is a quoted string containing "@w" followed by `: {`.
            if bytes[i] == b'"' {
                let Some(close) = body[i + 1..].find('"') else {
                    break;
                };
                let key = &body[i + 1..i + 1 + close];
                let after = &body[i + 1 + close + 1..];
                let trimmed = after.trim_start();
                if key.contains("@w") && trimmed.starts_with(':') {
                    let obj_rel = after.find('{');
                    if let Some(rel) = obj_rel {
                        let obj_start = i + 1 + close + 1 + rel;
                        let mut depth = 0i32;
                        let mut j = obj_start;
                        while j < bytes.len() {
                            match bytes[j] {
                                b'{' => depth += 1,
                                b'}' => {
                                    depth -= 1;
                                    if depth == 0 {
                                        break;
                                    }
                                }
                                _ => {}
                            }
                            j += 1;
                        }
                        out.push((
                            key.to_string(),
                            body[obj_start..=j.min(bytes.len() - 1)].to_string(),
                        ));
                        i = j + 1;
                        continue;
                    }
                }
                i += 1 + close + 1;
                continue;
            }
            i += 1;
        }
        out
    }

    /// **G1.2** — every feature-set id registered in
    /// `benchmarks/feature_sets_registry.json` is reproducible from THIS
    /// registry, at the right strength for its role.
    ///
    /// The two roles are checked differently, and the difference is the
    /// finding that shaped this gate. A **producer** id (an extractor, a
    /// features root) names what it POPULATES, which IS derivable from its
    /// compute tokens and layout width — so it is checked for EQUALITY. A
    /// **consumer** id (a bake) names what it READS, which is an arbitrary
    /// subset of some producer's set and is NOT derivable from any name — so
    /// it is checked for COVERAGE against its producer.
    ///
    /// The registry carries a live instance of exactly that distinction:
    /// `basic+peaks+moments@w944/era2r4` (producer, 265 slots, `#4fcef1d6`)
    /// and `…#0b476506` (consumer, 261 slots — the same set minus the four
    /// `LUMA_MEAN_REF` slots, which are the free set's only reference-absolute
    /// statistic and so its only slots that do not vanish at identity). Same
    /// handle, different hash, and the hash is what tells them apart — which
    /// is what `docs/FEATURE_SET_IDS.md` §2.4 means by "the name is a handle;
    /// the hash is the identity".
    #[test]
    fn registered_sets_are_reproducible_from_the_definition_registry() {
        let json = include_str!("../../benchmarks/feature_sets_registry.json");
        let (mut producers, mut consumers, mut classes) = (0usize, 0usize, 0usize);
        for (key, obj) in registry_sets(json) {
            let compute = json_str_field(&obj, "compute").expect("compute");
            let era = json_str_field(&obj, "era").expect("era");
            let role = json_str_field(&obj, "role").unwrap_or_default();
            let kind = json_str_field(&obj, "kind").unwrap_or_default();
            // A `kind: "class"` entry pins the handle without an exact slot
            // set (a bake's exact read set is derived from its bytes). There
            // is nothing to check, and inventing a set for it would be a
            // fabrication rather than a gate.
            let (Some(slots_s), Some(hash_s)) = (
                json_str_field(&obj, "slots"),
                json_str_field(&obj, "slots_hash8"),
            ) else {
                assert_eq!(
                    kind, "class",
                    "{key}: no slots/hash recorded but kind is {kind:?}, not \"class\""
                );
                classes += 1;
                continue;
            };
            let width: usize = obj
                .split("\"layout\"")
                .nth(1)
                .and_then(|t| t.split(':').nth(1))
                .and_then(|t| t.trim().trim_end_matches(',').split(',').next())
                .and_then(|t| t.trim().parse().ok())
                .expect("layout width");
            let parts = crate::feature_set_id::ComputeParts::parse(&compute)
                .unwrap_or_else(|| panic!("{key}: unparseable compute {compute:?}"));
            let recorded =
                SlotSet::parse(&slots_s).unwrap_or_else(|| panic!("{key}: unparseable slots"));
            let expect_hash = u32::from_str_radix(&hash_s, 16).expect("hex hash");

            // (a) The recorded slots hash to the recorded hash — the identity
            //     owner (`slots_hash8`) agrees with the committed value.
            assert_eq!(
                recorded.hash8(),
                expect_hash,
                "{key}: recorded slots hash {:08x} != recorded slots_hash8 {hash_s}",
                recorded.hash8()
            );

            // (b) The token-derived PRODUCER set for this compute + width.
            let mut derived = SlotSet::default();
            for t in parts.iter() {
                derived = derived.union(&family_slots(t, NS));
            }
            let derived = derived.clipped_to(width);

            if role == "consumer" {
                // A read set is a subset of what its producer populates.
                assert!(
                    derived.covers(&recorded),
                    "{key}: consumer reads slots no producer with compute                      {compute:?} populates: {}",
                    derived.missing_from(&recorded)
                );
                consumers += 1;
            } else {
                assert_eq!(
                    derived, recorded,
                    "{key}: registry-derived producer slots != recorded slots"
                );
                producers += 1;
            }
            assert!(
                crate::feature_set_id::is_valid_token(&era),
                "{key}: era {era:?} is not [a-z0-9_]+"
            );
        }
        assert!(
            producers >= 10 && consumers >= 1,
            "expected the registered sets; saw {producers} producers, \
             {consumers} consumers, {classes} classes"
        );
    }
}
