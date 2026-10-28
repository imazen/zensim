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
//!   [956, 986)  reserved; later IDs retain their original positions.
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
}

impl Statistic {
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
}

/// Whether a slot is a difference between reference and distorted, and so
/// whether an identity pair must drive it to zero.
///
/// Reference-only slots can remain nonzero on a perfect copy. For example,
/// `append2::LUMA_MEAN_REF` reports the reference's luminance, and Rev5
/// `PJND_FRAGILITY` reports reference structure. The historical 372 all-zero
/// identity probe captured the product shortcut's placeholder features;
/// it does not establish that canonical extracted identity features are zero.
///
/// [`Form::Undeclared`] is an HONEST state, not a default to fill in later: a
/// signal whose identity behaviour has not been established (or, for the
/// restored-cut slots, depends on the formula revision) reports that, rather
/// than claiming a form nobody measured. The declared ones are pinned
/// by [`tests::declared_difference_forms_are_zero_on_an_identity_pair`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Form {
    /// A difference/dissimilarity: exactly `0.0` on an identity pair.
    Difference,
    /// Computed from the reference alone; an identity pair does not zero it.
    ReferenceOnly,
}

impl Form {
}

/// The monotone expectation a dial gate may rely on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Direction {
    /// Larger value = more distortion.
    HigherIsWorse,
    /// No signed expectation (reference-only conditioners, raw statistics).
    Unsigned,
}

impl Direction {
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
}

impl KernelId {
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
#[non_exhaustive]
pub enum FormulaRevision {
    /// Everything shipped through era `v1postc` — the semantics every stored
    /// table and every published verdict was read at.
    #[default]
    Rev1,
    /// Revision 4's formulas under **local-window arithmetic** (the
    /// `localwin` era, `benchmarks/rev5_spec_2026-10-04.md`), computed for
    /// the `basic` (f0..227), `peaks` (f156..227) and `v2` (f372..719)
    /// families only.
    ///
    /// Rev5 replaces the sliding-sum blur recurrences with a per-output
    /// 11-tap f32 pair tree over the window's own inputs (production's
    /// mirror padding), replaces the 8-lane pools with 16 fixed virtual
    /// lanes and one fixed pairwise reduce, allows fused multiply-add
    /// wherever the same fused expression is written, and carries the
    /// stable central-moment finalizers. Bit-identity across tiers is a
    /// property of the single fixed order, not a hope.
    ///
    /// The two-family scope is deliberate: a plan or bake whose read set
    /// leaves `basic + peaks + v2` is refused at plan time
    /// ([`crate::feature_plan::PlanError::UnsupportedAtRev5`]) rather than
    /// computed at a lower revision behind the caller's back. The process
    /// model is Rev4's: `ZENSIM_FORMULA_REV=5` pins it and
    /// `ssim_form::refuse_rev4_mix` refuses every cross-boundary mix.
    Rev5,
    /// Revision 5's measurements, raw ringing included, with revision 6's
    /// `globaff` family and faster, cancellation-free pooled arithmetic (the
    /// `rowlanes` era, `benchmarks/byconstruction_2026-10-22.md`, "Speed pass
    /// before revision 7"). Revision 6's high-pass ringing form is not part
    /// of it (owner, 2026-10-25: for the monotone models the raw form scores
    /// better on codec uniformity and robustness; "Before the revision-7
    /// cut" in the record):
    ///
    /// * **f32 row lanes.** The dense, fused-SSIM and gradient pools add
    ///   the columns of each row's complete 8-column chunks to sixteen f32
    ///   lanes (`x & 15`) and fold every lane into its f64 lane at the end
    ///   of the chunks and after every 4,096th column, instead of adding
    ///   every column in f64; the remaining columns add in f64 as before.
    ///   The f64 lanes live for a band of 32 rows (dense) or for the call
    ///   (fused SSIM, gradient: a 128-row strip) and are finished once there
    ///   instead of once per row. Pools whose features difference or ratio
    ///   two large sums (v1
    ///   variance, texture and contrast terms, the edge-width change, the
    ///   band-lifetime raw moments) keep f64 lanes.
    /// * **Blockiness lattice lanes.** The vertical lattice step at column
    ///   `8j` adds to f64 lane `(j − 1) & 7` instead of lane 0, so the eight
    ///   lanes take eight steps at once; each 128-row kernel strip sums into
    ///   its own lanes, added to the running lanes in strip order (strips run
    ///   in parallel, with the same bits at any thread count).
    /// * **Central moments per row.** The SSIM deviation moments are
    ///   two-pass per row (a sixteen-lane f64 mean, then the centred sums)
    ///   with one Chan/Pébay merge per row instead of one per 16 columns.
    /// * **Cancellation-free forms.** The v2 SSIM luminance term is
    ///   `(m1 − m2)²`, the bounded dissimilarities `(a − b)²/(a² + b² + c)`
    ///   and the v1 edge ratio `(x − y)/(1 + y)`: the exact values the old
    ///   forms computed by subtracting nearly equal numbers.
    /// * **Opsin cube root with one division**
    ///   ([`crate::det_math::cbrt_newton_f32`]): within 3 ulp like
    ///   `cbrt_midp`, correctly rounded more often (72.7% against 56%), with
    ///   one division instead of two.
    ///
    /// Any live slot may move (an arithmetic era); scope and process model
    /// are Rev6's (`ZENSIM_FORMULA_REV=7`, or a bake declaring revision 7).
    /// `ZENSIM_RING_FORM=hpm` still selects the high-pass ringing form as a
    /// per-process research arm.
    Rev7,
}

impl FormulaRevision {

    /// Whether the `globaff` family (f1867..1890) is computed at this revision.
    pub(crate) const fn computes_globaff(self) -> bool {
        matches!(self, Self::Rev7)
    }

    /// Whether the pools add columns in f32 row lanes and the central
    /// moments are two-pass per row ([`Self::Rev7`]'s `rowlanes` era).
    pub(crate) const fn row_lanes(self) -> bool {
        matches!(self, Self::Rev7)
    }

    /// Whether the opsin conversion takes its cube root from
    /// [`crate::det_math::cbrt_newton_f32`] (a Newton then a Halley step, one
    /// division) instead of `cbrt_midp` ([`Self::Rev7`]).
    pub(crate) const fn cbrt_newton(self) -> bool {
        matches!(self, Self::Rev7)
    }

    /// Whether the cancellation-free forms replace the subtractive ones:
    /// `(m1 − m2)²` for the v2 SSIM luminance term, `(a − b)²/(a² + b² + c)`
    /// for the bounded dissimilarities and `(x − y)/(1 + y)` for the v1 edge
    /// ratio ([`Self::Rev7`]).
    pub(crate) const fn stable_forms(self) -> bool {
        matches!(self, Self::Rev7)
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
        matches!(
            self,
            Self::Rev5 | Self::Rev7
        )
    }

}

/// Internal dispatch family. Palette has no supported public token.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Family {
    Supported(ComputeToken),
    Palette,
}

impl From<ComputeToken> for Family {
    fn from(token: ComputeToken) -> Self {
        Self::Supported(token)
    }
}

impl PartialEq<ComputeToken> for Family {
    fn eq(&self, token: &ComputeToken) -> bool {
        *self == Self::Supported(*token)
    }
}

impl Family {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Supported(token) => token.as_str(),
            Self::Palette => "palette",
        }
    }
}

impl core::fmt::Display for Family {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// One registered signal: a feature definition before the layout replicates
/// it across scales and channels.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SignalDef {
    /// The slot family — the [`ComputeToken`] vocabulary, one owner.
    pub family: Family,
    /// Name stem, `[a-z0-9_]+`, unique within the family.
    pub name: &'static str,
    /// Whether a v1-only walk can harvest this signal without its block.
    pub tranche: Tranche,
    /// Which placements the tranche can serve.
    pub placement: Placement,
    pub form: Form,
}

/// One slot: a [`SignalDef`] placed at a (scale, channel) by the layout.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FeatureDef {
    pub signal: &'static SignalDef,
    pub scale: u8,
    pub channel: Channel,
}

impl FeatureDef {
}

// ============================================================================
// The signal table
// ============================================================================

/// The defect behind the `ringhpm` era: the ringing error is the raw pixel
/// error, so a global edit reads as ringing.
const DEFECT_RING_GLOBAL: Defect = Defect {
    id: "RINGGLOBAL",
    note: "v2 `ringing` saturates the raw error `|src − dst|`, which a global \
           offset, contrast or saturation change makes uniformly large away \
           from edges. MEASURED on 24 references (feature_probe.py, \
           benchmarks/byconstruction_2026-10-22.md): a +8 luma shift was \
           ringing's largest response, 2.4x a visible blur sigma 1.2 \
           (median 0.334 vs 0.139). imazen/zensimx#72.",
};

/// The `ringhpm` era: ringing measures the error after removing each image's
/// SSIM local mean ([`FormulaRevision::Rev6`]).
const REV_RINGHPM: Revision = Revision {
    era: "ringhpm",
    commit: "-",
    status: RevisionStatus::Proposed,
    note: "ringing's per-pixel error becomes `(src − mu1) − (dst − mu2)` with \
           the SSIM local means the dense kernel already holds; the three \
           saturations are unchanged. Moves only the 12 ringing slots \
           (+8 shift median 0.334 -> 0.017, blur unchanged 0.139 -> 0.143). \
           Rev5 keeps the raw form byte for byte, and Rev7 returns to it \
           (owner, 2026-10-25); `ZENSIM_RING_FORM=hpm` is \
           the per-process research arm of the same arithmetic. Record: \
           benchmarks/byconstruction_2026-10-22.md.",
};
const RINGING_REVS: &[Revision] = &[REV_RINGHPM];

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
    REV_F4_ENTRY,
    REV_SSIM_STABLE,
];

/// The PROPOSED fix for [`DEFECT_F18`] on a slot that ALSO carries F4 — the
/// SSIM-derived `L4`/`L8` pools, which are in both eras.
const REV_F4_AND_DETROOT: &[Revision] = &[
    Revision {
        era: "v1postc",
        commit: "56bbcda2",
        status: RevisionStatus::Landed,
        note: "option C: v1 stopped pooling mirror-padded phantom columns.",
    },
    REV_F4_ENTRY,
    REV_SSIM_STABLE,
    REV_DETROOT,
];

/// The v2 block's SSIM-derived signals under revision 3. They read the same
/// `sigma12` plane the v1 signal does, and under revision 3 that plane holds
/// the direct error moment — so `feature_v2::ssim_d_local` forms the v2
/// dissimilarity from it directly rather than as `cov = s12 - mu1*mu2`. A
/// separate era from `v1ssimstable` because it names a different kernel and a
/// different slot family; both are in revision 3's list.
const REV_EXT_FUSED: Revision = Revision {
    era: "v1extfused",
    commit: "-",
    status: RevisionStatus::Proposed,
    note: "Rev3 (issue #61 perf campaign): the masked/IW extension is FUSED \
           into the SSIM V sweep (`fused::ExtPoolsWork`). The activity is \
           `blur(|src - H(src)|)` exactly as before — the H-only mu plane the \
           H pass already wrote, H-blurred once, V-blurred inside the sweep \
           by the same `sum + add - rem` recurrence — and every masked/IW \
           pool is formed from the same per-pixel values the separate passes \
           formed. What moves is the ORDER the f64 chunk sums are added in \
           (column-group-major inside the sweep, row-major in the passes), so \
           these slots move at the last f64 bits on every image. The masked/\
           IW SSIM pools are already `v1ssimstable` movers; this era adds \
           the v1 MASKED/IW `edge_art_4th`, `edge_det_4th` and `mse` slots. STILL PROPOSED: `SHIPPED_REVISION` \
           is `Rev1`.",
};
/// `v1()`'s default list (`v1postc`) plus the fused-extension era: the v1
/// MASKED/IW `mse` slots.
const REV_OPTION_C_AND_EXT_FUSED: &[Revision] = &[
    Revision {
        era: "v1postc",
        commit: "56bbcda2",
        status: RevisionStatus::Landed,
        note: "option C: v1 stopped pooling mirror-padded phantom columns.",
    },
    REV_EXT_FUSED,
];
/// `v1()`'s pooled-root list (`v1postc` + F18) plus the fused-extension era:
/// the v1 MASKED/IW `edge_art_4th` / `edge_det_4th` slots.
const REV_DETROOT_AND_EXT_FUSED: &[Revision] = &[
    Revision {
        era: "v1postc",
        commit: "56bbcda2",
        status: RevisionStatus::Landed,
        note: "option C: v1 stopped pooling mirror-padded phantom columns.",
    },
    REV_DETROOT,
    REV_EXT_FUSED,
];

const REV_V2_SSIM_STABLE: Revision = Revision {
    era: "v2ssimstable",
    commit: "-",
    status: RevisionStatus::Proposed,
    note: "Rev3 (issue #61): the v2 dense kernel's SSIM dissimilarity \
           (`ssim_d_local` / `ssim_d_local_v`) reads the fourth moment plane \
           as the DIRECT error `E[(a-b)^2]` and forms \
           `(d(b-a) + a*err_var)/(b*d)` instead of recovering the covariance \
           by subtraction. Same bounded-error contract as `v1ssimstable`. \
           Registered after the first fused build was caught (by reading, \
           not by a gate) still computing `cov = s12 - mu1*mu2` on the \
           redefined plane; `rev3_moves_exactly_the_registered_slots_on_the_944_layout` \
           now covers the v2 block. Proposed: nothing is trained against it.",
};
const REV_V2_SSIM: &[Revision] = &[REV_V2_SSIM_STABLE];
/// `ssim_dev4` carries F18's pooled-root era AND the v2 SSIM era.
const REV_V2_SSIM_AND_DETROOT: &[Revision] = &[REV_DETROOT, REV_V2_SSIM_STABLE];

/// The one text of the `v1detroot` era, so the three lists above cannot drift.
const REV_DETROOT: Revision = Revision {
    era: "v1detroot",
    commit: "-",
    status: RevisionStatus::Proposed,
    note: "F18 fix: evaluate the pooled 4th/8th roots as `sqrt.sqrt` and \
           `sqrt.sqrt.sqrt`. THE ARM IS DERIVED, NOT CHOSEN - IEEE-754 \
           requires `sqrt` to be correctly rounded, so for these two \
           exponents the replacement is unique, bit-identical on every libc, \
           and cheaper. NOT more accurate: `sqrt.sqrt` double-rounds where \
           `powf` rounds once, and MEASURED over 4,000 log-uniform doubles \
           against a 60-digit reference the two agree on 3,455 while glibc's \
           `pow` is nearer the truth in 544 of the 545 that differ - so the \
           case is determinism and a BOUNDED error, not accuracy (the \
           discovery record's contrary claim is falsified in \
           benchmarks/libc_determinism_2026-09-06.md section 3). Owner: \
           `crate::det_math::RootForm`. Its cost is that it moves revision-1 \
           bytes by up to one ULP - which is why it is an era item and not a \
           plain fix. It rides in revision 2 with F4, F5 and F17 so \
           there is ONE era boundary and one recalculation. MIGRATION: this \
           era invalidates any `ZENSIM_FORMULA_REV=2` table extracted BEFORE \
           it was registered (the R6b lane has some); \
           `ZENSIM_ROOT_FORM=libm` reproduces those exactly. STILL PROPOSED: \
           `ssim_form::SHIPPED_REVISION` is `Rev1`, so no shipped byte moves.",
};

/// **The one text of the `v1ssimcap` era.** Both [`REV_F4_PROPOSED`] (a slot
/// in F4 alone) and [`REV_F4_AND_DETROOT`] (a slot in F4 *and* F18) name this
/// constant, so the two lists cannot state the same era differently — the
/// exact drift a second copy of the text would invite.
const REV_SSIM_STABLE: Revision = Revision {
    era: "v1ssimstable",
    commit: "-",
    status: RevisionStatus::Proposed,
    note: "Rev3 (issue #61), FUSED form: the v1 SSIM dissimilarity is formed \
           from a DIRECT error moment. `blur::fused_blur_h_ssim` accumulates \
           `sum (a-b)^2` in the fourth plane in place of `sum a*b` (same two \
           FMAs), and `fused::fused_vblur_ssim_inner` forms \
           `loss + (1-loss)*E_err/(var1+var2+C2)` from the same four f32 \
           planes it always V-blurred — no second traversal. Basic, peak, \
           masked, IW and the attribution planes all read that one retained \
           value. BOUNDED, not exact, by user directive 2026-09-09 (\"bounded \
           error is fine, speed above minor flaws\"): registered on the \
           locality fixture, peak out-of-support movement <= 2e-5 (measured \
           4.277e-6; shipped rev 1 measures 4.886e-4), max abs error vs the \
           exact f64 reference kernel <= 1e-3, all-equal-window residue \
           <= 1e-5 (measured 3.689e-6). The exact f64 second-pass kernel \
           `ssim_form::stable_ssim_plane` measured 0 movement but cost \
           +27..87% of extraction (22% of the walk in perf) and is retained \
           only as the reference the bounds are measured against. This era \
           moves every SSIM-derived slot on every image, unlike `v1ssimcap`. \
           STILL PROPOSED: `ssim_form::SHIPPED_REVISION` is `Rev1`, no table \
           has been re-extracted and no bake refit against it.",
};

const REV_F4_ENTRY: Revision = Revision {
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
};

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
        note: "F17 fix: bound the HF-energy gain member. THE ARM IS DECIDED — \
               `HfGainForm::SaturatingExcess`, i.e. `g/(g+1)` — by measurement \
               over 216,756 rows, and it is the ONLY arm passing H3+H4+H5 \
               (docs/PLAN_FEATURE_REV2_2026-09-05.md section 11, \
               benchmarks/feature_rev2_2026-09-05.md section 11). Unlike \
               `v1ssimcap` this era moves values on EVERY corpus with pixels — \
               `contrast_inc` is nonzero on 12 % (CID22) to 52 % (KADID) of \
               cells — so a rev2 flip re-extracts these twelve slots for real. \
               MEASURED benefit and cost: LIVE 0.7357 -> 0.9500 (+0.214), TID \
               +0.033, KADID +0.021, CID22 +0.0027..+0.0090 CI-excluding; and \
               KonJND REGRESSES -0.013..-0.080 on EVERY arm, which makes it a \
               property of bounding rather than of this arm. It rides in \
               revision 2 with F4 and F5 so there is ONE era boundary and one \
               recalculation. STILL PROPOSED: \
               `ssim_form::SHIPPED_REVISION` is `Rev1`.",
    },
];

/// **F15** — `PJND_FRAGILITY` is nonzero on an identity pair.
const DEFECT_F15: Defect = Defect {
    id: "F15",
    note: "The identity-pair value is NOT a defect: the slot is \
           1 − saturate(mean grad_src_mag), a property of the reference, and on \
           the Rev4 bank's 88 identical keys it equals every same-reference \
           sibling bit for bit (2,376 cells; REVIEW_PARTB 2026-09-25). The earlier \
           premise that it should read 0 there was wrong. The defect is the other \
           artifact: a v1-only 944 walk reads exactly 1.0 from zeroed \
           accumulators, a value emitted without its inputs.",
};

/// **THE one owner of "does this slot use a pooled 4th/8th root?"** — derived
/// from the slot's own [`Statistic`], never from a hand-kept list.
///
/// `L4` is `(sum x^4/n)^(1/4)` and `L8` is `(sum x^8/n)^(1/8)`; every other v1
/// statistic is a mean, a max, a ratio or an `L2` (`sqrt`, correctly rounded
/// by IEEE-754). So the registry's answer to "which slots does `v1detroot`
/// move?" is a property of the statistic and cannot drift away from the
/// kernels — which is the failure mode a second list would have.
const fn uses_pooled_root(statistic: Statistic) -> bool {
    matches!(statistic, Statistic::L4 | Statistic::L8)
}

const fn v1(
    family: ComputeToken,
    block_local: u16,
    name: &'static str,
    statistic: Statistic,
    kernel: KernelId,
) -> SignalDef {
    SignalDef {
        family: Family::Supported(family),
        name,
        tranche: Tranche::None,
        placement: Placement::AllCells,
        // Every v1 slot is a difference form: the 372 identity probe is the
        // ZERO vector on all 38 dial-grid references, measured
        // (benchmarks/dial_addressability_gate_2026-09-04.md §15).
        form: Form::Difference,
        // F18: an `L4`/`L8` slot with no other defect carries the libc
        // exposure alone. F4 slots are handled by `v1_defect`, which keeps
        // `Defect::F4` (the larger one) in the single `defect` field and
        // records F18 through the revision list — `SignalDef` holds ONE
        // defect, so a slot in two defects names the bigger and registers
        // both eras.
    }
}

/// [`v1`] with an explicit revision list — for slots whose era membership is
/// not implied by their statistic (the fused-extension pools). The F18
/// defect is still derived from the statistic exactly as [`v1`] derives it.
const fn v1_rev(
    family: ComputeToken,
    block_local: u16,
    name: &'static str,
    statistic: Statistic,
    kernel: KernelId,
    revisions: &'static [Revision],
) -> SignalDef {
    SignalDef {
        family: Family::Supported(family),
        name,
        tranche: Tranche::None,
        placement: Placement::AllCells,
        form: Form::Difference,
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
        family: Family::Supported(family),
        name,
        tranche: Tranche::None,
        placement: Placement::AllCells,
        form: Form::Difference,
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
        // An SSIM-derived `L4`/`L8` pool is in BOTH `v1ssimcap` (F4) and
        // `v1detroot` (F18): the same slot, two independent arithmetic
        // problems. The eras therefore OVERLAP, which is why
        // `rev2_carries_all_eras_*` checks a union and not a sum.
        if uses_pooled_root(statistic) {
            REV_F4_AND_DETROOT
        } else {
            REV_F4_PROPOSED
        },
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
        family: Family::Supported(ComputeToken::V2),
        name,
        tranche,
        placement: Placement::AllCells,
        form,
        // F18 reaches the v2 block too: `ssim_dev4` is `(M4/n)^(1/4)`, the
        // same `powf(0.25)` in three finalizers (`OnlineMoments::finish`, the
        // dense kernel, the append kernel). The caller's explicit `defect`
        // wins when it has one — a v2 slot with a named defect keeps it.
    }
}

/// [`v2sig`] with an explicit `revisions` list, for the v2 signals an era
/// names by KERNEL rather than by statistic (the SSIM-derived six).
#[allow(clippy::too_many_arguments)]
const fn v2sig_rev(
    block_local: u16,
    name: &'static str,
    statistic: Statistic,
    form: Form,
    direction: Direction,
    kernel: KernelId,
    tranche: Tranche,
    defect: Option<Defect>,
    revisions: &'static [Revision],
) -> SignalDef {
    let base = v2sig(
        block_local,
        name,
        statistic,
        form,
        direction,
        kernel,
        tranche,
        defect,
    );
    SignalDef {  ..base }
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
        v1_rev(F, 3, "edge_art_4th", L4, K, REV_DETROOT_AND_EXT_FUSED),
        v1_rev(F, 4, "edge_det_4th", L4, K, REV_DETROOT_AND_EXT_FUSED),
        v1_rev(F, 5, "mse", Mean, K, REV_OPTION_C_AND_EXT_FUSED),
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
        v1_rev(F, 3, "edge_art_4th", L4, K, REV_DETROOT_AND_EXT_FUSED),
        v1_rev(F, 4, "edge_det_4th", L4, K, REV_DETROOT_AND_EXT_FUSED),
        v1_rev(F, 5, "mse", Mean, K, REV_OPTION_C_AND_EXT_FUSED),
    ]
};

/// The 29 v2 signals per (scale, channel), index-aligned with
/// `feature_v2::idx`.
pub(crate) static V2: [SignalDef; 29] = {
    use Direction::{HigherIsWorse, Unsigned};
    use Form::{Difference, ReferenceOnly};
    use KernelId::{V2Dense, V2Gradient};
    use Statistic::{L2, L4, Mean, WeightedMean};
    [
        // Bounded-basic block (idx 0..8).
        v2sig_rev(
            0,
            "ssim_mean",
            Mean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
            REV_V2_SSIM,
        ),
        v2sig_rev(
            1,
            "ssim_dev2",
            L2,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
            REV_V2_SSIM,
        ),
        v2sig_rev(
            2,
            "ssim_dev4",
            L4,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
            REV_V2_SSIM_AND_DETROOT,
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
        v2sig_rev(
            9,
            "ssim_soft_peak",
            WeightedMean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
            REV_V2_SSIM,
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
        v2sig_rev(
            12,
            "masked_ssim",
            WeightedMean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
            REV_V2_SSIM,
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
        v2sig_rev(
            16,
            "iw_ssim",
            WeightedMean,
            Difference,
            HigherIsWorse,
            V2Dense,
            Tranche::None,
            None,
            REV_V2_SSIM,
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
        // F15: ReferenceOnly, and its nonzero identity-pair value is correct
        // (it equals the same-reference siblings). The defect is the 1.0 a
        // v1-only walk reads from zeroed accumulators.
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
        v2sig_rev(
            26,
            "ringing",
            Mean,
            Difference,
            HigherIsWorse,
            V2Gradient,
            Tranche::None,
            Some(DEFECT_RING_GLOBAL),
            RINGING_REVS,
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

// ============================================================================
// Rev4 candidate feature bank (f986..1321 at 4 scales)
// ============================================================================
//
// Spec: `docs/REV4_FEATURE_BANK_PLAN_2026-09-23.md` §1.1 / §2.4–2.5; pinned
// geometry: `benchmarks/rev4_featbank_impl_2026-09-23.md`. All four families
// are difference-form by construction (identity pair → 0.0), carry no
// tranche (a v1-only walk cannot harvest any of them — they live on the
// v2-era passes), and declare `Placement::AllCells`: gridblk's `(Y, scale 3)`
// cell is registered-with-stated-zero (period 1 has no phase contrast), not
// skipped, so the slot exists and emits 0.0 deliberately.

// ============================================================================
// Layout arithmetic — THE owner
// ============================================================================

/// How a family is replicated across the (scale, channel) axes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Replication {
    /// One cell per (scale, channel): 3 cells per scale.
    PerChannel,
}

/// A registered block: a signal table plus where it sits in the layout.
pub(crate) struct BlockDef {
    pub family: Family,
    pub signals: &'static [SignalDef],
    pub replication: Replication,
}

/// Every block, in LAYOUT order. The base offset of each is the sum of the
/// widths of the ones before it, so inserting a block anywhere but the end
/// would renumber — which the append-only directive forbids.
pub(crate) static BLOCKS: &[BlockDef] = &[
    BlockDef {
        family: Family::Supported(ComputeToken::Basic),
        signals: &BASIC,
        replication: Replication::PerChannel,
    },
    BlockDef {
        family: Family::Supported(ComputeToken::Peaks),
        signals: &PEAKS,
        replication: Replication::PerChannel,
    },
    BlockDef {
        family: Family::Supported(ComputeToken::Masked),
        signals: &MASKED,
        replication: Replication::PerChannel,
    },
    BlockDef {
        family: Family::Supported(ComputeToken::Iw),
        signals: &IW,
        replication: Replication::PerChannel,
    },
    BlockDef {
        family: Family::Supported(ComputeToken::V2),
        signals: &V2,
        replication: Replication::PerChannel,
    },
];

impl BlockDef {
    /// Slots this block occupies at `n_scales`.
    pub(crate) fn width(&self, n_scales: usize) -> usize {
        let cells = match self.replication {
            Replication::PerChannel => n_scales * 3,
        };
        cells * self.signals.len()
    }
}

/// Retired slots stay unassigned so later feature IDs keep their meaning.
fn reserved_before(_family: Family) -> usize {
    0
}

/// The base slot id of `family` at `n_scales`, and its block.
pub(crate) fn block_base(
    family: impl Into<Family>,
    n_scales: usize,
) -> Option<(usize, &'static BlockDef)> {
    let family = family.into();
    let mut base = 0usize;
    for b in BLOCKS {
        base += reserved_before(b.family);
        if b.family == family {
            return Some((base, b));
        }
        base += b.width(n_scales);
    }
    None
}

/// Total layout width at `n_scales` with every registered block present.
pub(crate) fn full_width(n_scales: usize) -> usize {
    BLOCKS
        .iter()
        .map(|b| reserved_before(b.family) + b.width(n_scales))
        .sum()
}

/// The definition of slot `id` at `n_scales`, or `None` past the full width.
///
/// The inverse of [`slot_id`], and gated to round-trip against it.
pub(crate) fn def_at(id: usize, n_scales: usize) -> Option<FeatureDef> {
    let mut base = 0usize;
    for block in BLOCKS {
        base += reserved_before(block.family);
        if id < base {
            return None;
        }
        let w = block.width(n_scales);
        if id < base + w {
            let off = id - base;
            let per = block.signals.len();
            let (scale, channel, local) = match block.replication {
                Replication::PerChannel => {
                    let cell = off / per;
                    (cell / 3, Channel::TRIPLE[cell % 3], off % per)
                }
            };
            return Some(FeatureDef {
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
pub(crate) fn family_slots(family: impl Into<Family>, n_scales: usize) -> SlotSet {
    match family.into() {
        Family::Supported(ComputeToken::Moments) => tranche_slots(Tranche::RawMoments, n_scales),
        Family::Supported(ComputeToken::ClassC) => tranche_slots(Tranche::ClassC, n_scales),
        Family::Supported(ComputeToken::Carriers) => SlotSet::from_slots(CARRIER_SLOTS),
        // `Hdr` is RESERVED — registered in the vocabulary, emitted by
        // nothing, so its slot set is legitimately empty rather than absent.
        Family::Supported(ComputeToken::Hdr) => SlotSet::default(),
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

#[cfg(test)]
mod tests {

    use super::*;

    const NS: usize = crate::NUM_SCALES;

    /// The block bases are the numbers every emit site in `feature_v2` writes
    /// out by hand. Pinned here so the registry cannot drift from them.
    #[test]
    fn block_bases_match_the_documented_layout() {
        let expect = [
            (Family::Supported(ComputeToken::Basic), 0usize, 156usize),
            (Family::Supported(ComputeToken::Peaks), 156, 72),
            (Family::Supported(ComputeToken::Masked), 228, 72),
            (Family::Supported(ComputeToken::Iw), 300, 72),
            (Family::Supported(ComputeToken::V2), 372, 348),
        ];
        for (family, base, width) in expect {
            let (b, blk) = block_base(family, NS).expect("registered family");
            assert_eq!(b, base, "{family} base");
            assert_eq!(blk.width(NS), width, "{family} width");
        }
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

}

/// Cross-checks against the `feature_v2` owners — the gates that make the
/// registry's independent derivations provably equal to the code that
/// actually emits the rows, rather than a plausible second opinion.
///
/// Gated on `feature-regime-v2` because the owners are: the registry itself
/// is pure data and builds without it.
#[cfg(test)]
mod owner_gates {
    use super::*;
    use crate::feature_set_id::ComputeToken as T;
    use crate::feature_v2;

    const NS: usize = crate::NUM_SCALES;

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
        // The owners also list slots in the research blocks after the v2 block, which
        // this registry does not carry; compare within the registered width.
        let raw_owner =
            SlotSet::from_slots(feature_v2::free_slot_indices(NS)).clipped_to(full_width(NS));
        assert_eq!(
            tranche_slots(Tranche::RawMoments, NS),
            raw_owner,
            "raw-moment tranche disagrees with feature_v2::free_slot_indices"
        );
        let cc_owner =
            SlotSet::from_slots(feature_v2::class_c_slot_indices(NS)).clipped_to(full_width(NS));
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
}
