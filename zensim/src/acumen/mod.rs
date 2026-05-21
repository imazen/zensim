//! `acumen` — zensim's perceptual front-end primitives.
//!
//! **Naming intent.** "Acumen" is Latin for keen perception. The
//! name deliberately *does not* overload existing terms like
//! castleCSF, ColorVideoVDP, VDP, JND, or JOD, which are calibrated
//! reference models from published research. What we ship here is
//! a *honed approximation* — fast enough for zensim's per-pair
//! scoring envelope, accurate enough to drive a learned MLP head,
//! but not a claim of parity with any one reference metric.
//!
//! What *is* faithfully implemented from published research is
//! the underlying primitives, e.g. [`castle_csf`] sources its LUT
//! from gfxdisp/castleCSF (Ashraf, Mantiuk, Chapiro, Wuerger 2024;
//! MIT) sampled through cvvdp v0.5.4's pipeline. Where we deviate
//! from published math (e.g. per-pixel L_adapt vs. per-image mean),
//! the deviation is documented per submodule.
//!
//! ## Scope (v0)
//!
//! - [`castle_csf`] — contrast sensitivity LUT, bilinear interp.
//! - More submodules added as the algorithm slate from the
//!   tracking issue lands. Priority 1: DKL color transform,
//!   per-band CSF modulation. Priority 2: contrast masking,
//!   distortion-tolerance map. Priority 3: HDR enable.
//!
//! ## Output units
//!
//! Internal computations produce relative perceptual scalars. We
//! *do not* claim equivalence with JOD (CVVDP's unit), JND
//! (HDR-VDP / AIC-3 unit), or any other calibrated unit. Where an
//! output type exists, it's named [`Acu`] (acumen unit) — a
//! deliberately unspecified scalar that downstream MLPs interpret.
//!
//! ## Stability
//!
//! Everything in this module is `#[doc(hidden)]` and considered
//! internal until the algorithm slate stabilises. The tracking
//! issue (`imazen/zensim#40`) governs the public API surface.

#[doc(hidden)]
pub mod band_weights;
#[doc(hidden)]
pub mod castle_csf;
#[doc(hidden)]
pub mod viewing;

/// Relative perceptual scalar emitted by [`acumen`](crate::acumen)
/// primitives. Not equivalent to JOD, JND, or any calibrated
/// reference unit — downstream consumers interpret via the trained
/// profile MLP. Provided for API readability; conversions to/from
/// `f32` are free.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Acu(pub f32);

impl From<f32> for Acu {
    #[inline]
    fn from(v: f32) -> Self {
        Acu(v)
    }
}

impl From<Acu> for f32 {
    #[inline]
    fn from(a: Acu) -> Self {
        a.0
    }
}
