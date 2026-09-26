//! Featcanon (2026-09-25): canonical feature arithmetic behind
//! [`crate::feature_defs::FormulaRevision::Rev4`], plus the measurement
//! substrate that chose it (`benchmarks/featcanon_WORKLOG.md`).
//!
//! The problem: several extraction kernels emit different bits per SIMD tier
//! because (a) `mul_add` is fused on v3/v4/neon but plain `a*b+c` on the
//! magetypes scalar and wasm128 backends, (b) `reduce_add`'s tree is
//! backend-defined (16-lane native on v4, 2×8-lane on v3, array-order on
//! scalar), and (c) per-tier tail/chunk geometry differs (width-16 vs width-8
//! vs remainder paths). [`era2_reduce8`] established the
//! fix pattern for the dense block kernel; this module generalises it: one
//! body, fixed virtual-lane accumulation, a written-out reduction tree, and
//! `f32::mul_add` (the fused intrinsic on every target) wherever the formula
//! is a product-sum.
//!
//! **The arithmetic is chosen PER COMPUTATION, never per process.** Every
//! canonical leaf receives the revision of the computation it serves and asks
//! [`mode`] with it. `ZENSIM_FORMULA_REV` does not reach a leaf through this
//! module: it only chooses which revision a caller that did not name one
//! gets, and the entries refuse every request whose revision disagrees with
//! the process revision when either is Rev4 (see
//! [`crate::ssim_form::refuse_rev4_mix`]), because other formula gates still
//! read the process switch.
//!
//! **Rev4 is research-extraction-only.** The served and HDR paths still run
//! tier-dispatched leaves (`color::linear_to_pu_xyb_planar_into`, the
//! edge-only `blur::fused_blur_h_mu` route, `attribution::attr_pass_b_*`), so
//! every `Zensim`, `BakeScorer`, HDR and diffmap entry refuses Rev4
//! ([`crate::ssim_form::refuse_rev4_served`]). Only `research::extract` (and
//! the crate-internal walks it drives) computes Rev4.
//!
//! **Measurement modes are not in a product build.** `ZENSIM_FEATCANON`
//! (`exact`/`c32`/`c64`/`neum`/`off`) and the f64 exact-oracle bodies exist
//! only with the `oracle` cargo feature, the crate's precedent for a ruler
//! that is not semantics. A product build knows two arithmetics: production
//! (Rev1–Rev3) and [`Mode::Canon32`] (Rev4).

use crate::feature_defs::FormulaRevision;

/// Arithmetic mode of the canonical leaves for ONE computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    /// Production dispatch (no canonical routing): Rev1–Rev3.
    Off,
    /// Candidate (d), the Rev4 arithmetic: fused `mul_add` elements + fixed
    /// 8-virtual-lane f32 partials + fixed pairwise reduction (era-2 shape).
    Canon32,
    /// f64 element evaluation + compensated f64 accumulation — the exact
    /// oracle arm. Measurement only.
    #[cfg(feature = "oracle")]
    Exact,
    /// Candidate (e): fused `mul_add` elements + fixed 8-virtual-lane f64
    /// partials + fixed pairwise reduction. Measurement only.
    #[cfg(feature = "oracle")]
    Canon64,
    /// Candidate (f): fused `mul_add` elements + Neumaier-compensated f64
    /// accumulation. Measurement only.
    #[cfg(feature = "oracle")]
    CanonNeum,
}

impl Mode {
    /// `true` when the canonical/candidate path should run (any non-Off mode).
    pub(crate) const fn active(self) -> bool {
        !matches!(self, Mode::Off)
    }

    /// `true` when this is the exact-oracle arm (`oracle` builds only; a
    /// product build has no such arm).
    #[cfg(feature = "oracle")]
    #[inline(always)]
    pub(crate) const fn exact(self) -> bool {
        matches!(self, Mode::Exact)
    }
}

/// The canonical-leaf arithmetic for a computation at `revision`.
///
/// `revision` is the revision the leaf's CALLER is computing — the one it
/// already carries for its other formula gates (`fused_blur_h_ssim_at_revision`'s
/// `revision`, `FreeExtrasWork::revision()`, `V2NewFeatureToggles::formula_revision`,
/// `ZensimConfig::formula_revision`). `Rev4` selects [`Mode::Canon32`], the
/// measured winner; every earlier revision selects [`Mode::Off`], so Rev1–Rev3
/// bytes cannot depend on this module.
///
/// With the `oracle` feature, `ZENSIM_FEATCANON` overrides the result for
/// measurement (`exact`/`c32`/`c64`/`neum`, or `off` to force production even
/// at Rev4). The override is read once per process; a product build does not
/// contain it.
#[inline]
pub(crate) fn mode(revision: FormulaRevision) -> Mode {
    #[cfg(feature = "oracle")]
    if let Some(m) = measurement_override() {
        return m;
    }
    if revision >= FormulaRevision::Rev4 {
        Mode::Canon32
    } else {
        Mode::Off
    }
}

/// `ZENSIM_FEATCANON`, read once. `None` when unset or unrecognised.
#[cfg(feature = "oracle")]
fn measurement_override() -> Option<Mode> {
    use std::sync::OnceLock;
    static M: OnceLock<Option<Mode>> = OnceLock::new();
    *M.get_or_init(|| match std::env::var("ZENSIM_FEATCANON").as_deref() {
        Ok("exact") => Some(Mode::Exact),
        Ok("c32") => Some(Mode::Canon32),
        Ok("c64") => Some(Mode::Canon64),
        Ok("neum") => Some(Mode::CanonNeum),
        Ok("off") => Some(Mode::Off),
        _ => None,
    })
}

// ============================================================================
// Canonical accumulators — the three measured candidates
// ============================================================================

/// The era-2 horizontal reduction — **part of the semantics**.
///
/// Pairwise rather than sequential (tighter error, equally fixed), and written
/// out rather than delegated: `GenericF32x8::reduce_add` resolves to a
/// per-backend order (§14.2 of `benchmarks/era2_perf_break_2026-08-31.md`),
/// so calling it would make the reduction tree an unspecified, tier-dependent
/// operation — precisely what the era-2 identity theorem forbids.
///
/// Owned here, not in `feature_v2` (which re-exports it for the era-2 dense
/// kernel), because the canonical leaves compile in every build and
/// `feature_v2` exists only with `feature-regime-v2`.
#[inline(always)]
pub(crate) fn era2_reduce8(a: [f32; 8]) -> f64 {
    (((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))) as f64
}

/// The canonical lane count. 8 because it is the largest power of two that
/// every production tier maps onto without remainder (the era-2 choice,
/// `benchmarks/era2_perf_break_2026-08-31.md` §2.0).
pub(crate) const CANON_LANES: usize = 8;

/// Candidate (d): 8 f32 virtual lanes, term j → lane `j % 8`, tail folds into
/// the same lanes, closed by the fixed [`era2_reduce8`]
/// pairwise tree. Width-independent by construction.
#[derive(Clone, Copy, Default)]
pub(crate) struct LanesF32(pub [f32; CANON_LANES]);

impl LanesF32 {
    #[inline(always)]
    pub(crate) fn zero() -> Self {
        Self([0.0; 8])
    }
    /// Add one full 8-wide chunk lane-wise.
    // dead_code until the chunked-pool kernels land; kept — `Pool::add`
    // covers the per-element callers first.
    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) fn add_chunk(&mut self, c: &[f32; 8]) {
        for (l, &v) in self.0.iter_mut().zip(c.iter()) {
            *l += v;
        }
    }
    /// Add one element to lane `i` — the explicit-lane form for kernels whose
    /// terms arrive per-element rather than per-chunk.
    #[inline(always)]
    pub(crate) fn add_lane(&mut self, i: usize, v: f32) {
        self.0[i & (CANON_LANES - 1)] += v;
    }
    /// Fold a partial tail into the SAME lanes (element k → lane k).
    #[allow(dead_code)]
    #[inline(always)]
    pub(crate) fn add_tail(&mut self, t: &[f32]) {
        for (l, &v) in self.0.iter_mut().zip(t.iter()) {
            *l += v;
        }
    }
    /// The fixed pairwise reduction. `f32` adds in the written order, then one
    /// exact `f64` widening.
    #[inline(always)]
    pub(crate) fn reduce(self) -> f64 {
        era2_reduce8(self.0)
    }
}

/// Candidate (e): 8 f64 virtual lanes — same lane mapping and the same
/// adjacent-pairwise tree shape as [`LanesF32`], but every partial is f64, so
/// per-term error is only the term's own f32 rounding. Measurement only.
#[cfg(feature = "oracle")]
#[derive(Clone, Copy, Default)]
pub(crate) struct LanesF64(pub [f64; CANON_LANES]);

/// dead_code until chunked-pool kernels land; see LanesF32::add_chunk.
#[cfg(feature = "oracle")]
#[allow(dead_code)]
impl LanesF64 {
    #[inline(always)]
    pub(crate) fn zero() -> Self {
        Self([0.0; 8])
    }
    #[inline(always)]
    pub(crate) fn add_chunk(&mut self, c: &[f32; 8]) {
        for (l, &v) in self.0.iter_mut().zip(c.iter()) {
            *l += v as f64;
        }
    }
    #[inline(always)]
    pub(crate) fn add_chunk_f64(&mut self, c: &[f64; 8]) {
        for (l, &v) in self.0.iter_mut().zip(c.iter()) {
            *l += v;
        }
    }
    #[inline(always)]
    pub(crate) fn add_lane(&mut self, i: usize, v: f64) {
        self.0[i & (CANON_LANES - 1)] += v;
    }
    #[inline(always)]
    pub(crate) fn add_tail(&mut self, t: &[f32]) {
        for (l, &v) in self.0.iter_mut().zip(t.iter()) {
            *l += v as f64;
        }
    }
    #[inline(always)]
    pub(crate) fn add_tail_f64(&mut self, t: &[f64]) {
        for (l, &v) in self.0.iter_mut().zip(t.iter()) {
            *l += v;
        }
    }
    /// The same adjacent-pairwise tree as `era2_reduce8`, in f64.
    #[inline(always)]
    pub(crate) fn reduce(self) -> f64 {
        (self.0[0] + self.0[1] + (self.0[2] + self.0[3]))
            + ((self.0[4] + self.0[5]) + (self.0[6] + self.0[7]))
    }
}

/// Candidate (f): Neumaier-compensated f64 running sum — the compensated form
/// already proven in `feature_v2::oracle` (`Neumaier`), reused so the lane's
/// compensation semantics have exactly one definition. Measurement only (the
/// candidate and the exact-oracle bodies).
#[cfg(feature = "oracle")]
#[derive(Clone, Copy, Default)]
pub(crate) struct Neum64 {
    sum: f64,
    comp: f64,
}

#[cfg(feature = "oracle")]
impl Neum64 {
    #[inline(always)]
    pub(crate) fn zero() -> Self {
        Self {
            sum: 0.0,
            comp: 0.0,
        }
    }
    #[inline(always)]
    pub(crate) fn add(&mut self, v: f64) {
        let t = self.sum + v;
        if self.sum.abs() >= v.abs() {
            self.comp += (self.sum - t) + v;
        } else {
            self.comp += (v - t) + self.sum;
        }
        self.sum = t;
    }
    #[inline(always)]
    pub(crate) fn reduce(self) -> f64 {
        self.sum + self.comp
    }
}

// ============================================================================
// The `Pool` abstraction — one body drives all three candidate accumulators
// ============================================================================

/// One canonical accumulator slot in a kernel body.
///
/// `add(lane, v)` puts term `v` into canonical lane `lane` (the term's
/// position mod 8 — kernels pass the column index). `fin` applies the fixed
/// reduction and returns f64. The three impls are the brief's measured
/// candidates: (d) f32 virtual lanes + fixed pairwise tree, (e) f64 virtual
/// lanes + the same tree, (f) sequential Neumaier compensation.
pub(crate) trait Pool: Copy {
    fn zero() -> Self;
    fn add(&mut self, lane: usize, v: f32);
    /// f64 term form — used when the term was itself computed in f64
    /// (the measurement accumulators and the exact bodies).
    #[cfg(feature = "oracle")]
    fn add64(&mut self, lane: usize, v: f64) {
        self.add(lane, v as f32);
    }
    fn fin(self) -> f64;
}

impl Pool for LanesF32 {
    #[inline(always)]
    fn zero() -> Self {
        Self::zero()
    }
    #[inline(always)]
    fn add(&mut self, lane: usize, v: f32) {
        self.add_lane(lane, v);
    }
    #[inline(always)]
    fn fin(self) -> f64 {
        self.reduce()
    }
}

#[cfg(feature = "oracle")]
impl Pool for LanesF64 {
    #[inline(always)]
    fn zero() -> Self {
        Self::zero()
    }
    #[inline(always)]
    fn add(&mut self, lane: usize, v: f32) {
        self.add_lane(lane, v as f64);
    }
    #[inline(always)]
    fn add64(&mut self, lane: usize, v: f64) {
        self.add_lane(lane, v);
    }
    #[inline(always)]
    fn fin(self) -> f64 {
        self.reduce()
    }
}

#[cfg(feature = "oracle")]
impl Pool for Neum64 {
    #[inline(always)]
    fn zero() -> Self {
        Self::zero()
    }
    #[inline(always)]
    fn add(&mut self, _lane: usize, v: f32) {
        Neum64::add(self, v as f64);
    }
    #[inline(always)]
    fn add64(&mut self, _lane: usize, v: f64) {
        Neum64::add(self, v);
    }
    #[inline(always)]
    fn fin(self) -> f64 {
        self.reduce()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// featcanon D1/D6: the mode is a function of the computation's revision
    /// alone — the process revision does not enter it — and a product build
    /// (no `oracle`) knows exactly two arithmetics.
    #[test]
    fn mode_is_a_function_of_the_computation_revision() {
        #[cfg(feature = "oracle")]
        assert!(
            std::env::var_os("ZENSIM_FEATCANON").is_none(),
            "this gate reads the unoverridden mapping; unset ZENSIM_FEATCANON"
        );
        for rev in [
            FormulaRevision::Rev1,
            FormulaRevision::Rev2,
            FormulaRevision::Rev3,
        ] {
            assert_eq!(mode(rev), Mode::Off, "{rev:?}");
        }
        assert_eq!(mode(FormulaRevision::Rev4), Mode::Canon32);
        // Same answers whatever ZENSIM_FORMULA_REV this process runs at.
        let _ = crate::ssim_form::active_revision();
        #[cfg(not(feature = "oracle"))]
        {
            // Exhaustive in a product build: adding a third arithmetic
            // without the `oracle` gate fails to compile here.
            let _: fn(Mode) = |m| match m {
                Mode::Off | Mode::Canon32 => {}
            };
        }
    }
}
