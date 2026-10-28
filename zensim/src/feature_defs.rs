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
}

impl Statistic {}

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

impl Form {}

/// The monotone expectation a dial gate may rely on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Direction {
    /// Larger value = more distortion.
    HigherIsWorse,
}

impl Direction {}

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

impl KernelId {}

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
    pub form: Form,
}

/// One slot: a [`SignalDef`] placed at a (scale, channel) by the layout.
#[derive(Debug, Clone, Copy)]
pub(crate) struct FeatureDef {
    pub signal: &'static SignalDef,
    pub scale: u8,
    pub channel: Channel,
}

impl FeatureDef {}

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

const fn v1(family: ComputeToken, name: &'static str) -> SignalDef {
    SignalDef {
        family: Family::Supported(family),
        name,
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
const fn v1_rev(family: ComputeToken, name: &'static str) -> SignalDef {
    SignalDef {
        family: Family::Supported(family),
        name,
        form: Form::Difference,
    }
}

/// A v1 signal carrying a named defect and its proposed revision.
const fn v1_with_defect(family: ComputeToken, name: &'static str) -> SignalDef {
    SignalDef {
        family: Family::Supported(family),
        name,
        form: Form::Difference,
    }
}

/// A v1 signal carrying the F4 defect and its proposed revision.
const fn v1_defect(
    family: ComputeToken,
    _block_local: u16,
    name: &'static str,
    _statistic: Statistic,
    _kernel: KernelId,
) -> SignalDef {
    v1_with_defect(family, name)
}

const fn v2sig(name: &'static str, form: Form) -> SignalDef {
    SignalDef {
        family: Family::Supported(ComputeToken::V2),
        name,
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
    _block_local: u16,
    name: &'static str,
    _statistic: Statistic,
    form: Form,
    _direction: Direction,
    _kernel: KernelId,
    _defect: Option<Defect>,
) -> SignalDef {
    let base = v2sig(name, form);
    SignalDef { ..base }
}

/// The 13 v1 basic signals per (scale, channel), in `V1BasicSums::finalize_into`
/// order. Names follow the in-tree diagnostic array in `streaming.rs`.
pub(crate) static BASIC: [SignalDef; 13] = {
    use ComputeToken::Basic as F;
    use KernelId::V1Fused as K;
    use Statistic::{L2, L4, Mean};
    [
        // F4 reaches EVERY slot built from the per-pixel SSIM dissimilarity,
        // not only the masked and IW blocks the audit named — MEASURED, see
        // `f4_moves_exactly_the_registered_slots`.
        v1_defect(F, 0, "ssim_mean", Mean, K),
        v1_defect(F, 1, "ssim_4th", L4, K),
        v1_defect(F, 2, "ssim_2nd", L2, K),
        v1(F, "edge_art_mean"),
        v1(F, "edge_art_4th"),
        v1(F, "edge_art_2nd"),
        v1(F, "edge_det_mean"),
        v1(F, "edge_det_4th"),
        v1(F, "edge_det_2nd"),
        v1(F, "mse"),
        // The HF-energy ratio family. The two `loss` members are bounded by
        // construction (their numerator is `max(0, src - dst)`, which their
        // own denominator bounds); the `gain` member's is `max(0, dst - src)`,
        // which it does not — that asymmetry IS F17, and it is why only the
        // third of these three carries a defect.
        v1(F, "var_loss"),
        v1(F, "tex_loss"),
        v1_with_defect(F, "contrast_inc"),
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
        v1(F, "edge_art_max"),
        v1(F, "edge_det_max"),
        v1_defect(F, 3, "ssim_l8", L8, K),
        v1(F, "edge_art_l8"),
        v1(F, "edge_det_l8"),
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
        v1_rev(F, "edge_art_4th"),
        v1_rev(F, "edge_det_4th"),
        v1_rev(F, "mse"),
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
        v1_rev(F, "edge_art_4th"),
        v1_rev(F, "edge_det_4th"),
        v1_rev(F, "mse"),
    ]
};

/// The 29 v2 signals per (scale, channel), index-aligned with
/// `feature_v2::idx`.
pub(crate) static V2: [SignalDef; 29] = {
    use Direction::HigherIsWorse;
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
            None,
        ),
        v2sig_rev(1, "ssim_dev2", L2, Difference, HigherIsWorse, V2Dense, None),
        v2sig_rev(2, "ssim_dev4", L4, Difference, HigherIsWorse, V2Dense, None),
        v2sig("art", Difference),
        v2sig("det", Difference),
        // Class C: harvestable from the fused kernel's saturating-MSE sum,
        // despite living in the most expensive block in the layout.
        v2sig("mse", Difference),
        v2sig("hf_gain", Difference),
        v2sig("hf_loss", Difference),
        v2sig("hf_mag_loss", Difference),
        // Soft-saliency peak block (idx 9..11).
        v2sig_rev(
            9,
            "ssim_soft_peak",
            WeightedMean,
            Difference,
            HigherIsWorse,
            V2Dense,
            None,
        ),
        v2sig("art_soft_peak", Difference),
        v2sig("det_soft_peak", Difference),
        // Masked block (idx 12..15).
        v2sig_rev(
            12,
            "masked_ssim",
            WeightedMean,
            Difference,
            HigherIsWorse,
            V2Dense,
            None,
        ),
        v2sig("masked_art", Difference),
        v2sig("masked_det", Difference),
        v2sig("masked_mse", Difference),
        // IW block (idx 16..19).
        v2sig_rev(
            16,
            "iw_ssim",
            WeightedMean,
            Difference,
            HigherIsWorse,
            V2Dense,
            None,
        ),
        v2sig("iw_art", Difference),
        v2sig("iw_det", Difference),
        v2sig("iw_mse", Difference),
        // Near-threshold core (idx 20..21). FRAGILITY is reference-only
        // masking-susceptibility — see `idx::PJND_FRAGILITY`'s own doc.
        v2sig("pjnd_transducer", Difference),
        // F15: ReferenceOnly, and its nonzero identity-pair value is correct
        // (it equals the same-reference siblings). The defect is the 1.0 a
        // v1-only walk reads from zeroed accumulators.
        v2sig("pjnd_fragility", ReferenceOnly),
        // Phase-2 additions (idx 22..28).
        v2sig("gms", Difference),
        v2sig("pjnd_transducer_low_k", Difference),
        v2sig("pjnd_transducer_high_k", Difference),
        v2sig("blockiness", Difference),
        v2sig_rev(
            26,
            "ringing",
            Mean,
            Difference,
            HigherIsWorse,
            V2Gradient,
            Some(DEFECT_RING_GLOBAL),
        ),
        v2sig("banding", Difference),
        // The one scale-level (not per-pixel) signal in the set — see
        // `idx::EDGE_WIDTH_CHANGE`'s doc and the v2 spec.
        v2sig("edge_width_change", Difference),
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
    use crate::feature_v2;

    /// The carrier list is the owner's list.
    #[test]
    fn carrier_slots_match_the_feature_v2_owner() {
        assert_eq!(
            CARRIER_SLOTS,
            feature_v2::V1PoolsMode::CARRIER_SLOTS,
            "carrier slot list disagrees with V1PoolsMode::CARRIER_SLOTS"
        );
    }
}
