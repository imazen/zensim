//! Featcanon (2026-09-25): measurement + canonical-arithmetic substrate for the
//! feature-extraction SIMD tier-parity lane (`BRIEF_featcanon.md`).
//!
//! The problem this module exists to measure and then fix: several extraction
//! kernels emit different bits per SIMD tier because (a) `mul_add` is fused on
//! v3/v4/neon but plain `a*b+c` on the magetypes scalar and wasm128 backends,
//! (b) `reduce_add`'s tree is backend-defined (16-lane native on v4, 2×8-lane
//! on v3, array-order on scalar), and (c) per-tier tail/chunk geometry differs
//! (width-16 vs width-8 vs remainder paths). [`crate::feature_v2::era2_reduce8`]
//! established the fix pattern for the dense block kernel; this module
//! generalises it: one body, fixed virtual-lane accumulation, a written-out
//! reduction tree, and `f32::mul_add` (the fused intrinsic on every target)
//! wherever the formula is a product-sum.
//!
//! **Two switches, deliberately separate:**
//!
//! * `ZENSIM_FEATCANON` — measurement mode for this lane. `exact` routes the
//!   canonicalised kernels through their f64 reference bodies (the exact
//!   oracle); `c32` / `c64` / `neum` select the candidate canonical
//!   accumulation (f32 virtual lanes, f64 virtual lanes, Neumaier-compensated).
//!   `off` (default) leaves production dispatch untouched.
//! * `ZENSIM_FORMULA_REV=4` — the eventual canonical arithmetic revision.
//!   Until the measurement decides the winner, revision 4 is not a routable
//!   value; the mode switch above is what the candidate kernels consult.

/// Candidate/exact arithmetic mode for the featcanon measurement lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    /// Production dispatch (no canonical routing).
    Off,
    /// f64 element evaluation + compensated f64 accumulation — the exact
    /// oracle arm.
    Exact,
    /// Candidate (d): fused `mul_add` elements + fixed 8-virtual-lane f32
    /// partials + fixed pairwise reduction (era-2 shape).
    Canon32,
    /// Candidate (e): fused `mul_add` elements + fixed 8-virtual-lane f64
    /// partials + fixed pairwise reduction.
    Canon64,
    /// Candidate (f): fused `mul_add` elements + Neumaier-compensated f64
    /// accumulation.
    CanonNeum,
}

impl Mode {
    /// `true` when the canonical/candidate path should run (any non-Off mode).
    pub(crate) const fn active(self) -> bool {
        !matches!(self, Mode::Off)
    }
}

/// The lane's process-global mode, read once from `ZENSIM_FEATCANON`, or
/// selected by the formula revision.
///
/// `ZENSIM_FEATCANON` (`exact`/`c32`/`c64`/`neum`/`off`) is the MEASUREMENT
/// override — it exists so the audit binary can pin a candidate without
/// moving revisions. Unset, the mode follows
/// [`crate::feature_defs::FormulaRevision`]: `Rev4` selects
/// [`Mode::Canon32`] — the measured winner (see
/// `benchmarks/featcanon_WORKLOG.md`) — and everything else selects `Off`.
/// `ZENSIM_FEATCANON=off` forces production kernels even under Rev4 (A/B
/// escape hatch); `exact` keeps the oracle arm reachable.
///
/// A `OnceLock` is the right shape: the mode must not move between calls or a
/// feature vector could mix two arithmetics.
pub(crate) fn mode() -> Mode {
    use std::sync::OnceLock;
    static M: OnceLock<Mode> = OnceLock::new();
    *M.get_or_init(|| match std::env::var("ZENSIM_FEATCANON").as_deref() {
        Ok("exact") => Mode::Exact,
        Ok("c32") => Mode::Canon32,
        Ok("c64") => Mode::Canon64,
        Ok("neum") => Mode::CanonNeum,
        Ok("off") => Mode::Off,
        _ => {
            if crate::ssim_form::active_revision() >= crate::feature_defs::FormulaRevision::Rev4 {
                Mode::Canon32
            } else {
                Mode::Off
            }
        }
    })
}

/// `true` when the featcanon measurement/canonical path is active.
#[inline(always)]
pub(crate) fn active() -> bool {
    mode().active()
}

// ============================================================================
// Canonical accumulators — the three measured candidates
// ============================================================================

/// The canonical lane count. 8 because it is the largest power of two that
/// every production tier maps onto without remainder (the era-2 choice,
/// `benchmarks/era2_perf_break_2026-08-31.md` §2.0).
pub(crate) const CANON_LANES: usize = 8;

/// Candidate (d): 8 f32 virtual lanes, term j → lane `j % 8`, tail folds into
/// the same lanes, closed by the fixed [`crate::feature_v2::era2_reduce8`]
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
        crate::feature_v2::era2_reduce8(self.0)
    }
}

/// Candidate (e): 8 f64 virtual lanes — same lane mapping and the same
/// adjacent-pairwise tree shape as [`LanesF32`], but every partial is f64, so
/// per-term error is only the term's own f32 rounding.
#[derive(Clone, Copy, Default)]
pub(crate) struct LanesF64(pub [f64; CANON_LANES]);

/// dead_code until chunked-pool kernels land; see LanesF32::add_chunk.
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
/// compensation semantics have exactly one definition.
#[derive(Clone, Copy, Default)]
pub(crate) struct Neum64 {
    sum: f64,
    comp: f64,
}

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

/// Which candidate accumulation a `*_canon` body should use, from [`mode`].
///
/// `Exact` maps to `LanesF64`-order compensated behaviour inside the exact
/// bodies, which evaluate elements in f64; this type covers only the
/// production-arithmetic candidates.
#[derive(Clone, Copy)]
pub(crate) enum CanonAcc {
    F32Lanes,
    F64Lanes,
    Neumaier,
}

#[inline(always)]
pub(crate) fn canon_acc() -> Option<CanonAcc> {
    match mode() {
        Mode::Canon32 => Some(CanonAcc::F32Lanes),
        Mode::Canon64 => Some(CanonAcc::F64Lanes),
        Mode::CanonNeum => Some(CanonAcc::Neumaier),
        _ => None,
    }
}

/// `true` when the exact-oracle arm is active.
#[inline(always)]
pub(crate) fn exact() -> bool {
    mode() == Mode::Exact
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
    /// f64 term form — used when the term was itself computed in f64.
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
