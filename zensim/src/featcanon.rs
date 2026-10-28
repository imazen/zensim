//! Canonical feature accumulation and blur arithmetic.
//!
//! Revision 1 retains legacy arithmetic. Revisions 5 and 7 use canonical
//! accumulation; revision 7 also uses row lanes and stable difference forms.

use crate::feature_defs::FormulaRevision;

/// Arithmetic mode of the canonical leaves for ONE computation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    Off,
    Canon64,
}

impl Mode {
    /// `true` when the canonical/candidate path should run (any non-Off mode).
    pub(crate) const fn active(self) -> bool {
        !matches!(self, Mode::Off)
    }
}

#[inline]
pub(crate) fn mode(revision: FormulaRevision) -> Mode {
    if revision >= FormulaRevision::Rev5 {
        Mode::Canon64
    } else {
        Mode::Off
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlurMode {
    Rec64,
    /// THE REV5 CANON's blur axis (`localwin`): each output's 11-tap window
    /// is summed independently in f32 over the spec's fixed pair tree —
    /// `s2 = x[i]+x[i+1]; s4 = s2[i]+s2[i+2]; s8 = s4[i]+s4[i+4];
    /// w = s8 + s2[8] + x[10]` — with [`tap_mirror`] padding. No recurrence,
    /// so rounding cannot travel along a row or strip.
    Local,
}

/// Non-oracle sibling: the only canon mode a product build can select is
/// `Canon64`, whose blur is the f64 recurrence — a constant here.
#[inline(always)]
pub(crate) const fn canon_blur_axis(_mode: Mode) -> BlurMode {
    BlurMode::Rec64
}

#[inline(always)]
pub(crate) fn canon_blur_axis_for(
    _mode: Mode,
    revision: crate::feature_defs::FormulaRevision,
) -> BlurMode {
    if revision >= crate::feature_defs::FormulaRevision::Rev5 {
        BlurMode::Local
    } else {
        BlurMode::Rec64
    }
}

/// The boundary map the blur recurrences implement, for the `Fresh` arm's
/// per-position window: reflect once toward the image, clamp if still out
/// (the kernels' `mirror_idx`/`vblur_add_idx`/`vblur_rem_idx` all reduce to
/// this: `j < 0 → -j`, `j ≥ n → 2(n−1) − j`, both clamped into `[0, n−1]`).
/// NOT the crate's periodic [`crate::feature_v2::reflect_101`] — the sliding
/// kernels' convention is what `fresh` must replay exactly, so a `rec` vs
/// `fresh` diff measures recurrence drift, not a boundary convention. Rev5's
/// `Local` windows pad with the same map, so a `local` vs `fresh` diff also
/// measures the f32 pair tree alone, not a boundary convention.
#[inline(always)]
pub(crate) fn tap_mirror(j: isize, n: usize) -> usize {
    let n1 = n - 1;
    if j < 0 {
        j.unsigned_abs().min(n1)
    } else if j as usize > n1 {
        (2 * n1).saturating_sub(j as usize).min(n1)
    } else {
        j as usize
    }
}

// ============================================================================
// Canonical accumulators — the three measured candidates
// ============================================================================

#[inline(always)]
pub(crate) fn era2_reduce8(a: [f32; 8]) -> f64 {
    (((a[0] + a[1]) + (a[2] + a[3])) + ((a[4] + a[5]) + (a[6] + a[7]))) as f64
}

/// The canonical lane count. 8 because it is the largest power of two that
/// every production tier maps onto without remainder (the era-2 choice,
/// `benchmarks/era2_perf_break_2026-08-31.md` §2.0).
pub(crate) const CANON_LANES: usize = 8;

#[derive(Debug, Clone, Copy, Default)]
#[repr(align(64))]
pub(crate) struct LanesF64(pub [f64; CANON_LANES]);

/// dead_code until chunked-pool kernels land; see LanesF32::add_chunk.
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
    pub(crate) fn add_lane(&mut self, i: usize, v: f64) {
        self.0[i & (CANON_LANES - 1)] += v;
    }
    /// The same adjacent-pairwise tree as `era2_reduce8`, in f64.
    #[inline(always)]
    pub(crate) fn reduce(self) -> f64 {
        (self.0[0] + self.0[1] + (self.0[2] + self.0[3]))
            + ((self.0[4] + self.0[5]) + (self.0[6] + self.0[7]))
    }
}

/// Rev5's sixteen virtual f64 lanes, independent of the hardware width.
///
/// 64-byte aligned: the canonical kernels add whole 8-lane chunks (`chunk8`, 64 bytes) every few pixels, and
/// at the default 8-byte alignment those accesses split cache lines whenever the caller's frame puts the
/// array off a line. Measured (benchmarks/byconstruction_2026-10-22.md): the Rev5 walk at 2048², one thread,
/// ran 9.3% faster with this alignment, and without it an unrelated call-site change cost 3.5% in the
/// unchanged dense kernel. Values are unchanged; this is memory alignment only.
#[derive(Debug, Clone, Copy, Default)]
#[repr(align(64))]
pub(crate) struct Lanes16F64(pub [f64; 16]);

impl Pool for Lanes16F64 {
    const LANES: usize = 16;
    #[inline(always)]
    fn zero() -> Self {
        Self([0.0; 16])
    }
    #[inline(always)]
    fn add(&mut self, lane: usize, v: f32) {
        self.0[lane & 15] += v as f64;
    }
    #[inline(always)]
    fn fin(self) -> f64 {
        let mut a = self.0;
        for width in [8, 4, 2, 1] {
            for i in 0..width {
                a[i] = a[2 * i] + a[2 * i + 1];
            }
        }
        a[0]
    }
}

/// Revision 7's sixteen f32 row lanes (the `rowlanes` era,
/// [`crate::feature_defs::FormulaRevision::Rev7`]).
///
/// A pool's terms for the columns of a row's complete eight-column chunks
/// (`x < 8·⌊width/8⌋`) add here, lane `x & 15`, ascending `x`;
/// [`Self::fold_into`] then adds each lane to the matching f64 lane of the
/// pool's [`Lanes16F64`] and clears it, after every column `x` with
/// `(x + 1) % ROW_LANES_FOLD == 0` and at the end of the chunks. The row's
/// tail columns add to the f64 lanes directly. One f32 add per term replaces a
/// conversion and an f64 add, and no f32 lane takes more than 256 terms
/// between folds. The f64 lanes live for a dense band of 32 rows, or for a
/// fused-SSIM or gradient call, and are finished once there.
#[derive(Debug, Clone, Copy, Default)]
#[repr(align(64))]
pub(crate) struct Lanes16F32(pub [f32; 16]);

/// Columns between two folds of [`Lanes16F32`] into the f64 lanes.
pub(crate) const ROW_LANES_FOLD: usize = 4096;

impl Lanes16F32 {
    /// Lanes `offset..offset + 8` (`offset` is 0 or 8; indexed by its bit 3,
    /// so the access needs no range check).
    #[inline(always)]
    pub(crate) fn chunk8(&mut self, offset: usize) -> &mut [f32; 8] {
        debug_assert!(offset == 0 || offset == 8);
        &mut self.0.as_chunks_mut::<8>().0[(offset >> 3) & 1]
    }

    /// Add every lane to `pool`'s matching lane, keeping the lanes (the
    /// first half of a difference pair's fold, see the gradient kernel).
    #[inline(always)]
    pub(crate) fn add_into<P: Pool>(&self, pool: &mut P) {
        for l in 0..16 {
            pool.add(l, self.0[l]);
        }
    }

    /// Add every lane to `pool`'s matching lane (for [`Lanes16F64`],
    /// `lane += v as f64`), then clear.
    #[inline(always)]
    pub(crate) fn fold_into<P: Pool>(&mut self, pool: &mut P) {
        for l in 0..16 {
            pool.add(l, self.0[l]);
        }
        self.0 = [0.0; 16];
    }
}

impl Pool for Lanes16F32 {
    const LANES: usize = 16;
    #[inline(always)]
    fn zero() -> Self {
        Self([0.0; 16])
    }
    #[inline(always)]
    fn add(&mut self, lane: usize, v: f32) {
        self.0[lane & 15] += v;
    }
    /// Row lanes are folded into [`Lanes16F64`], never finished themselves;
    /// this reduces them by its tree for completeness.
    #[inline(always)]
    fn fin(self) -> f64 {
        Lanes16F64(self.0.map(f64::from)).fin()
    }
}

/// Fixed f64 lane storage used by the existing vector kernels.
pub(crate) trait Pool64: Pool {
    fn chunk8(&mut self, offset: usize) -> &mut [f64; 8];
}
impl Pool64 for LanesF64 {
    #[inline(always)]
    fn chunk8(&mut self, _offset: usize) -> &mut [f64; 8] {
        &mut self.0
    }
}
impl Pool64 for Lanes16F64 {
    /// Lanes `offset..offset + 8` (`offset` is 0 or 8; indexed by its bit 3,
    /// so the access needs no range check).
    #[inline(always)]
    fn chunk8(&mut self, offset: usize) -> &mut [f64; 8] {
        debug_assert!(offset == 0 || offset == 8);
        &mut self.0.as_chunks_mut::<8>().0[(offset >> 3) & 1]
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
    const LANES: usize = 8;
    fn zero() -> Self;
    fn add(&mut self, lane: usize, v: f32);
    fn fin(self) -> f64;
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
    fn fin(self) -> f64 {
        self.reduce()
    }
}

/// Sequential f64 accumulation as a `Pool` — the production shape of the
/// pools). `add(lane, v)` ignores `lane` and adds in arrival order, so a
/// `Pool`-generic kernel body instantiated at `f64` replays the shipped
/// sequence exactly.
impl Pool for f64 {
    #[inline(always)]
    fn zero() -> Self {
        0.0
    }
    #[inline(always)]
    fn add(&mut self, _lane: usize, v: f32) {
        *self += v as f64;
    }
    #[inline(always)]
    fn fin(self) -> f64 {
        self
    }
}

// ============================================================================
// `SumVar` / `WelfordVar` — runtime-mode accumulators for kernels whose
// production state is scalar f64 (a `Pool`-generic body would put a type
// parameter on structs like `LevelSums` that live inside long-lived,
// non-generic accumulator objects).
// ============================================================================

/// One accumulator slot whose arithmetic is chosen by [`measurement_mode`].
///
/// `Seq` is the shipped form (a plain `f64 += ` per term). Which candidate
/// `c32`/`c64`/`neum` maps to depends on the ELEMENTS the kernel sums —
/// [`SumVar::f32_elems`] for terms that are f32-representable (f32-computed
/// or f32-stored: an `as f32` narrowing of `f64::from(v)` is exact), and
/// [`SumVar::f64_elems`] for f64-native terms — for those, `c32` keeps the
/// sequential form since f32 lanes would change the ELEMENT precision, not
/// just the accumulation order (that would conflate the two axes the lane
/// exists to separate).
#[derive(Debug, Clone, Copy)]
pub(crate) enum SumVar {
    /// Sequential f64 — production accumulation.
    Seq(f64),
    L64(LanesF64),
}

impl SumVar {
    pub(crate) fn f64_elems(mode: Mode) -> Self {
        match mode {
            Mode::Off => Self::Seq(0.0),
            Mode::Canon64 => Self::L64(LanesF64::zero()),
        }
    }

    /// Accumulate one f64 term — `Seq`/`L64`/`Neum` keep it exact; `L32`
    /// narrows to f32 first (the caller-chosen variant encodes whether that
    /// narrowing is value-exact — it is whenever the term is f32-stored).
    #[inline(always)]
    pub(crate) fn add64(&mut self, lane: usize, v: f64) {
        match self {
            Self::Seq(s) => *s += v,
            Self::L64(p) => p.add_lane(lane, v),
        }
    }

    /// The slot's running value (fold-time readout).
    #[inline]
    pub(crate) fn fin(&self) -> f64 {
        match self {
            Self::Seq(s) => *s,
            Self::L64(p) => (*p).fin(),
        }
    }

    /// Fold a same-mode sibling's accumulated value into `self` —
    /// lane-aligned for the lane variants (the per-row cell → strip cell
    /// merge in the var-typed kernels).
    pub(crate) fn merge_from(&mut self, o: &Self) {
        match (self, o) {
            (Self::Seq(a), Self::Seq(b)) => *a += *b,
            (Self::L64(a), Self::L64(b)) => {
                for j in 0..CANON_LANES {
                    a.0[j] += b.0[j];
                }
            }
            _ => panic!("SumVar::merge_from across modes"),
        }
    }
}
