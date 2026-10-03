//! Transfer functions and the display model: mapping pixel code values to
//! **absolute emitted display luminance** (cd/m²).
//!
//! This is the enabling front-end for HDR-aware scoring. Today zensim scores
//! in relative linear-sRGB `[0, 1]`; an HDR-correct metric must first map code
//! values to absolute luminance so that luminance-dependent contrast
//! sensitivity (and, on the HDR path, PU encoding) can be applied. The full
//! roadmap and citations live in `docs/HDR_PLAN.md`.
//!
//! Every constant here is a public ITU-R / SMPTE / IEC specification value
//! (ITU-R BT.2100, SMPTE ST 2084, IEC 61966-2-1); the functions are
//! reimplemented from those specs. The reference-parity tests at the bottom
//! pin each function to published golden values and to the display models in
//! the locally-verified `pycvvdp` 0.5.4 reference.
//!
//! This module is internal (`pub(crate)`); a public HDR API is added in a
//! later chunk once the scoring path that consumes it lands.
//!
//! Foundation-ahead-of-consumer: the scoring path that uses these does not
//! land until the PU front-end chunk, so the items are `allow(dead_code)` for
//! now. They are fully exercised by the reference-parity tests below.
#![allow(dead_code)]

use archmage::incant;
use archmage::magetypes;
use magetypes::simd::backends::F32x16Convert;
use magetypes::simd::generic::f32x16 as GenericF32x16;

/// IEC 61966-2-1 sRGB EOTF: sRGB-encoded `v ∈ [0, 1]` → relative linear
/// light `[0, 1]`.
#[inline]
pub(crate) fn srgb_eotf(v: f32) -> f32 {
    if v <= 0.040_449_936 {
        v / 12.92
    } else {
        ((v + 0.055) / 1.055).powf(2.4)
    }
}

/// SMPTE ST 2084 (PQ) EOTF: PQ-encoded `v ∈ [0, 1]` → **absolute** luminance
/// in cd/m² over `[0, 10000]`.
///
/// Constants are the ST 2084 rational-polynomial coefficients
/// (`m1 = 2610/16384`, `m2 = 2523/4096·128`, `c1 = 3424/4096`,
/// `c2 = 2413/4096·32`, `c3 = 2392/4096·32`).
#[inline]
pub(crate) fn pq_eotf(v: f32) -> f32 {
    const L_MAX: f32 = 10000.0;
    const M1: f32 = 0.159_301_75; // 2610 / 16384
    const M2: f32 = 78.843_75; // 2523 / 4096 * 128
    const C1: f32 = 0.835_937_5; // 3424 / 4096
    const C2: f32 = 18.851_562; // 2413 / 4096 * 32
    const C3: f32 = 18.687_5; // 2392 / 4096 * 32

    let im = v.powf(1.0 / M2);
    let num = (im - C1).max(0.0);
    let den = C2 - C3 * im;
    L_MAX * (num / den).powf(1.0 / M1)
}

/// Canonical (Rev4) [`pq_eotf`]: identical structure, every transcendental
/// replaced by [`crate::det_math`]'s midp replication — tier- and
/// libc-independent bits.
#[inline]
fn pq_eotf_canon(v: f32) -> f32 {
    use crate::det_math::pow_midp_f32;
    const L_MAX: f32 = 10000.0;
    const M1: f32 = 0.159_301_75; // 2610 / 16384
    const M2: f32 = 78.843_75; // 2523 / 4096 * 128
    const C1: f32 = 0.835_937_5; // 3424 / 4096
    const C2: f32 = 18.851_562; // 2413 / 4096 * 32
    const C3: f32 = 18.687_5; // 2392 / 4096 * 32

    let im = pow_midp_f32(v, 1.0 / M2);
    let num = (im - C1).max(0.0);
    let den = C2 - C3 * im;
    L_MAX * pow_midp_f32(num / den, 1.0 / M1)
}

/// [`pq_eotf`] at an explicit formula revision: the libm body through
/// Rev3, the canonical midp body at Rev4 (featcanon D1).
#[inline]
pub(crate) fn pq_eotf_at_revision(v: f32, revision: crate::feature_defs::FormulaRevision) -> f32 {
    if crate::featcanon::mode(revision).active() {
        pq_eotf_canon(v)
    } else {
        pq_eotf(v)
    }
}

/// ITU-R BT.2100 HLG inverse-OETF: HLG-encoded `v ∈ [0, 1]` → scene-relative
/// linear `[0, 12]` **per channel**.
///
/// The HLG OOTF (system gamma, which converts scene-relative to
/// display-referred light) depends on the luminance of the whole RGB triple,
/// so it is applied at the color stage, not here. See [`hlg_system_gamma`].
#[inline]
pub(crate) fn hlg_inverse_oetf(v: f32) -> f32 {
    const A: f32 = 0.178_832_77;
    const B: f32 = 1.0 - 4.0 * A; // 0.28466892
    // c = 0.5 − a·ln(4a) ≈ 0.55991073
    const C: f32 = 0.559_910_7;
    if v <= 0.5 {
        (v * v) / 3.0
    } else {
        (((v - C) / A).exp() + B) / 12.0
    }
}

/// [`hlg_inverse_oetf`] at an explicit formula revision: libm `exp` through
/// Rev3, the canonical midp `exp` at Rev4.
#[inline]
pub(crate) fn hlg_inverse_oetf_at_revision(
    v: f32,
    revision: crate::feature_defs::FormulaRevision,
) -> f32 {
    if !crate::featcanon::mode(revision).active() {
        return hlg_inverse_oetf(v);
    }
    hlg_inverse_oetf_canon(v)
}

/// The Rev4 canonical HLG inverse-OETF — [`hlg_inverse_oetf_at_revision`]'s
/// active-mode arm — as a per-element scalar body. Shared by the
/// lane-parallel row decode's remainder and stubs (REV4VEC).
#[inline(always)]
fn hlg_inverse_oetf_canon(v: f32) -> f32 {
    const A: f32 = 0.178_832_77;
    const B: f32 = 1.0 - 4.0 * A; // 0.28466892
    const C: f32 = 0.559_910_7;
    if v <= 0.5 {
        (v * v) / 3.0
    } else {
        (crate::det_math::exp_midp_f32((v - C) / A) + B) / 12.0
    }
}

/// HLG system gamma (ITU-R BT.2100 / BBC WHP 369): `1.2` at a 1000 cd/m² peak,
/// with a luminance term and an ambient-light correction above that.
#[inline]
pub(crate) fn hlg_system_gamma(y_peak: f32, e_ambient_lux: f32) -> f32 {
    if y_peak <= 1000.0 {
        1.2
    } else {
        let amb = if e_ambient_lux > 0.0 {
            e_ambient_lux
        } else {
            5.0
        };
        1.2 + 0.42 * (y_peak / 1000.0).log10() - 0.076_23 * (amb / 5.0).log10()
    }
}

/// [`hlg_system_gamma`] at an explicit formula revision: libm `log10`
/// through Rev3, the canonical midp `log10` at Rev4.
#[inline]
pub(crate) fn hlg_system_gamma_at_revision(
    y_peak: f32,
    e_ambient_lux: f32,
    revision: crate::feature_defs::FormulaRevision,
) -> f32 {
    if !crate::featcanon::mode(revision).active() || y_peak <= 1000.0 {
        return hlg_system_gamma(y_peak, e_ambient_lux);
    }
    let amb = if e_ambient_lux > 0.0 {
        e_ambient_lux
    } else {
        5.0
    };
    1.2 + 0.42 * crate::det_math::log10_midp_f32(y_peak / 1000.0)
        - 0.076_23 * crate::det_math::log10_midp_f32(amb / 5.0)
}

/// The physical display the metric assumes the image is shown on: peak and
/// black emitted luminance plus the ambient light reflected off the screen,
/// all in cd/m². These three numbers turn relative pixel values into the
/// absolute luminance the eye actually adapts to.
///
/// The presets mirror `pycvvdp` 0.5.4 display models; `STANDARD_4K` is the
/// same SDR display zensim's CVVDP feature path already assumes
/// (`cvvdp_features.rs`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DisplayModel {
    /// Peak emitted luminance (white), cd/m².
    pub y_peak: f32,
    /// Black emitted luminance (leakage), cd/m².
    pub y_black: f32,
    /// Ambient light reflected off the screen, cd/m² (`E_ambient · k / π`).
    pub y_refl: f32,
}

impl DisplayModel {
    /// `pycvvdp` `standard_4k`: 200 cd/m² peak, 0.2 black, 250 lux ambient
    /// (`250 · 0.005 / π ≈ 0.3979`). The SDR reference display.
    pub(crate) const STANDARD_4K: Self = Self {
        y_peak: 200.0,
        y_black: 0.2,
        y_refl: 0.397_887_36,
    };

    /// A 1000 cd/m² PQ HDR reference display (BT.2100 grade-1000).
    pub(crate) const STANDARD_HDR_PQ_1000: Self = Self {
        y_peak: 1000.0,
        y_black: 0.005,
        y_refl: 0.397_887_36,
    };

    /// Map relative linear light `[0, 1]` (e.g. the output of [`srgb_eotf`]) to
    /// absolute emitted luminance:
    /// `L = (peak − black)·lin + black + reflection`.
    #[inline]
    pub(crate) fn sdr_linear_to_luminance(&self, lin: f32) -> f32 {
        (self.y_peak - self.y_black) * lin + self.y_black + self.y_refl
    }

    /// Map a PQ-encoded code value to absolute emitted luminance, clamped to
    /// what the display can physically reproduce and lifted by black +
    /// reflected ambient. A 1000 cd/m² display cannot show PQ's 10000 cd/m²
    /// peak, so the highlight clamps to `y_peak`.
    #[inline]
    pub(crate) fn pq_to_luminance(&self, v: f32) -> f32 {
        self.pq_nits_to_display(pq_eotf(v))
    }

    /// [`pq_to_luminance`](Self::pq_to_luminance) at an explicit formula
    /// revision — [`pq_eotf_at_revision`] underneath.
    #[inline]
    pub(crate) fn pq_to_luminance_at_revision(
        &self,
        v: f32,
        revision: crate::feature_defs::FormulaRevision,
    ) -> f32 {
        self.pq_nits_to_display(pq_eotf_at_revision(v, revision))
    }

    #[inline]
    fn pq_nits_to_display(&self, nits: f32) -> f32 {
        nits.min(self.y_peak) + self.y_black + self.y_refl
    }
}

/// BT.2100 luminance coefficients (`Y_s = 0.2627 R + 0.6780 G + 0.0593 B`)
/// — the scene-luminance weights the HLG OOTF applies its system gamma to.
/// Public ITU-R BT.2100-2 Table 6 values.
pub(crate) const BT2100_LUMA: [f32; 3] = [0.2627, 0.6780, 0.0593];

/// Decode one row of PQ code-value RGB triples (`[0, 1]`) IN PLACE to
/// absolute display-light cd/m², per channel:
/// `F_D = min(EOTF_PQ(v), peak) + black + reflection` — the
/// [`DisplayModel::pq_to_luminance`] display model applied per channel
/// (ST 2084 is defined per color component). `peak_nits` caps what the
/// display physically emits (pass `10000.0` for a spec-peak/mastering
/// decode with no display clamp).
pub(crate) fn decode_pq_row(row: &mut [[f32; 3]], peak_nits: f32) {
    let dm = DisplayModel {
        y_peak: peak_nits,
        y_black: DisplayModel::STANDARD_HDR_PQ_1000.y_black,
        y_refl: DisplayModel::STANDARD_HDR_PQ_1000.y_refl,
    };
    for px in row.iter_mut() {
        px[0] = dm.pq_to_luminance(px[0]);
        px[1] = dm.pq_to_luminance(px[1]);
        px[2] = dm.pq_to_luminance(px[2]);
    }
}

/// [`decode_pq_row`] at an explicit formula revision — the same display
/// model over [`pq_eotf_at_revision`]. REV4VEC: at Rev4 the per-channel
/// canonical body (`pq_eotf_canon` + the display clamp/lift) runs
/// lane-parallel over flat channel chunks on the fused-FMA tiers; the
/// scalar canonical body covers the remainder and the non-fused tiers.
pub(crate) fn decode_pq_row_at_revision(
    row: &mut [[f32; 3]],
    peak_nits: f32,
    revision: crate::feature_defs::FormulaRevision,
) {
    if !crate::featcanon::mode(revision).active() {
        decode_pq_row(row, peak_nits);
        return;
    }
    let dm = DisplayModel {
        y_peak: peak_nits,
        y_black: DisplayModel::STANDARD_HDR_PQ_1000.y_black,
        y_refl: DisplayModel::STANDARD_HDR_PQ_1000.y_refl,
    };
    // The EOTF + display map is elementwise — the channel-major layout of
    // `[[f32; 3]]` needs no transpose, the flat channel sequence is the
    // chunk sequence.
    let flat: &mut [f32] = bytemuck::cast_slice_mut(row);
    incant!(
        decode_pq_row_canon_vec(flat, dm),
        [v4x, v4, v3, neon, wasm128, scalar]
    );
}

/// The per-element scalar canonical PQ decode body — what the fused
/// tiers run for the `n mod 16` remainder and the non-fused tiers run
/// for every channel (REV4VEC).
fn decode_pq_row_canon_body(flat: &mut [f32], dm: DisplayModel) {
    for v in flat.iter_mut() {
        *v = dm.pq_nits_to_display(pq_eotf_canon(*v));
    }
}

/// [`pq_eotf_canon`] + [`DisplayModel::pq_nits_to_display`], lane-parallel
/// over 16 channels per chunk. Bit-identical to
/// [`decode_pq_row_canon_body`] per element on a fused-FMA tier: the
/// `f32x16` `pow_midp` is the same op sequence as `pow_midp_f32`
/// (`exp2_midp(log2_midp · n)`), and `mul_add`-free tail ops are plain
/// IEEE. `-scalar` keeps `#[magetypes]` from emitting a generic `_scalar`
/// variant (its `mul_add` is unfused — not canonical).
#[magetypes(define(f32x16), v4x, v4, v3, -scalar)]
fn decode_pq_row_canon_vec(token: Token, flat: &mut [f32], dm: DisplayModel) {
    const L_MAX: f32 = 10000.0;
    const M1: f32 = 0.159_301_75; // 2610 / 16384
    const M2: f32 = 78.843_75; // 2523 / 4096 * 128
    const C1: f32 = 0.835_937_5; // 3424 / 4096
    const C2: f32 = 18.851_562; // 2413 / 4096 * 32
    const C3: f32 = 18.687_5; // 2392 / 4096 * 32

    let c1 = f32x16::splat(token, C1);
    let c2 = f32x16::splat(token, C2);
    let c3 = f32x16::splat(token, C3);
    let l_max = f32x16::splat(token, L_MAX);
    let zero = f32x16::zero(token);
    let peak = f32x16::splat(token, dm.y_peak);
    let black = f32x16::splat(token, dm.y_black);
    let refl = f32x16::splat(token, dm.y_refl);

    let n = flat.len();
    let chunks = n / 16;
    for c in 0..chunks {
        let base = c * 16;
        // FIXED-SIZE ARRAY PATTERN: one range check at the boundary.
        let chunk: &[f32; 16] = flat[base..base + 16]
            .try_into()
            .expect("16 channels per chunk");
        let v = f32x16::from_array(token, *chunk);
        let im = v.pow_midp(1.0 / M2);
        let num = (im - c1).max(zero);
        let den = c2 - c3 * im;
        let nits = l_max * (num / den).pow_midp(1.0 / M1);
        let out = nits.min(peak) + black + refl;
        out.store((&mut flat[base..base + 16]).try_into().unwrap());
    }
    for v in flat[chunks * 16..].iter_mut() {
        *v = dm.pq_nits_to_display(pq_eotf_canon(*v));
    }
}

/// The scalar tier's canonical PQ decode: the per-element scalar body —
/// magetypes' generic `mul_add` is unfused on the scalar backend.
fn decode_pq_row_canon_vec_scalar(
    _token: archmage::ScalarToken,
    flat: &mut [f32],
    dm: DisplayModel,
) {
    decode_pq_row_canon_body(flat, dm);
}

/// wasm128 has no hardware FMA — same scalar canonical body.
#[cfg(target_arch = "wasm32")]
fn decode_pq_row_canon_vec_wasm128(
    _token: archmage::Wasm128Token,
    flat: &mut [f32],
    dm: DisplayModel,
) {
    decode_pq_row_canon_body(flat, dm);
}

/// NEON runs the scalar canonical body (2026-10-03): its vector `max`/`min` propagate NaN, while the canon clamps with
/// `f32::max`/`min`, which drop it, so a lane-parallel NEON chunk differed from the scalar canon on NaN inputs (CI aarch64).
#[cfg(target_arch = "aarch64")]
fn decode_pq_row_canon_vec_neon(_token: archmage::NeonToken, flat: &mut [f32], dm: DisplayModel) {
    decode_pq_row_canon_body(flat, dm);
}

/// Native PQ16 uses the exact same normalized f32 codes/EOTF as the float
/// route. Cache only the display-independent EOTF, so peak/black/reflection
/// remain per comparison. The shared table is 256 KiB, initialized once.
/// RGBA16 source layout is validated by the HDR entry; alpha is opaque.
pub(crate) fn decode_pq_u16_rgba_row(row: &[u8], out: &mut [[f32; 3]], peak_nits: f32) {
    static LUT: std::sync::OnceLock<Box<[f32]>> = std::sync::OnceLock::new();
    let lut = LUT.get_or_init(|| {
        (0..=u16::MAX)
            .map(|code| pq_eotf(f32::from(code) * (1.0 / 65535.0)))
            .collect::<Vec<_>>()
            .into_boxed_slice()
    });
    let dm = DisplayModel {
        y_peak: peak_nits,
        ..DisplayModel::STANDARD_HDR_PQ_1000
    };
    for (px, source) in out.iter_mut().zip(row.as_chunks::<8>().0) {
        for c in 0..3 {
            let code = u16::from_ne_bytes([source[c * 2], source[c * 2 + 1]]);
            px[c] = dm.pq_nits_to_display(lut[usize::from(code)]);
        }
    }
}

/// [`decode_pq_u16_rgba_row`] at an explicit formula revision. The Rev4
/// table is a second cache: the canonical EOTF differs in bits from the
/// libm one, and a Rev4 consumer must not read production-table values
/// (nor poison the shared production table).
pub(crate) fn decode_pq_u16_rgba_row_at_revision(
    row: &[u8],
    out: &mut [[f32; 3]],
    peak_nits: f32,
    revision: crate::feature_defs::FormulaRevision,
) {
    if !crate::featcanon::mode(revision).active() {
        decode_pq_u16_rgba_row(row, out, peak_nits);
        return;
    }
    static LUT_CANON: std::sync::OnceLock<Box<[f32]>> = std::sync::OnceLock::new();
    let lut = LUT_CANON.get_or_init(|| {
        (0..=u16::MAX)
            .map(|code| pq_eotf_canon(f32::from(code) * (1.0 / 65535.0)))
            .collect::<Vec<_>>()
            .into_boxed_slice()
    });
    let dm = DisplayModel {
        y_peak: peak_nits,
        ..DisplayModel::STANDARD_HDR_PQ_1000
    };
    for (px, source) in out.iter_mut().zip(row.as_chunks::<8>().0) {
        for c in 0..3 {
            let code = u16::from_ne_bytes([source[c * 2], source[c * 2 + 1]]);
            px[c] = dm.pq_nits_to_display(lut[usize::from(code)]);
        }
    }
}

/// Decode one row of HLG signal-value RGB triples (`[0, 1]`) IN PLACE to
/// absolute display-light cd/m² per BT.2100's reference OOTF:
/// per-channel scene light `E_s = OETF⁻¹(E')`, scene luminance
/// `Y_s = Σ BT2100_LUMA·E_s`, then
/// `F_D = peak · Y_s^(γ−1) · E_s + black + reflection` with
/// `γ = hlg_system_gamma(peak, ambient)`. Black/reflection lift matches
/// the PQ decode's display model for cross-transfer consistency.
pub(crate) fn decode_hlg_row(row: &mut [[f32; 3]], peak_nits: f32, ambient_lux: f32) {
    decode_hlg_row_in_primaries(
        row,
        peak_nits,
        ambient_lux,
        crate::source::ColorPrimaries::Bt2020,
    );
}

/// HLG OOTF in the declared source basis, before the linear gamut transform.
/// D65 RGB-to-XYZ Y rows; BT.2020 keeps the published BT.2100 coefficients.
pub(crate) fn decode_hlg_row_in_primaries(
    row: &mut [[f32; 3]],
    peak_nits: f32,
    ambient_lux: f32,
    primaries: crate::source::ColorPrimaries,
) {
    use crate::source::ColorPrimaries;
    let luma = match primaries {
        ColorPrimaries::Srgb => [0.212_639, 0.715_168_7, 0.072_192_32],
        ColorPrimaries::DisplayP3 => [0.228_974_57, 0.691_738_55, 0.079_286_91],
        ColorPrimaries::Bt2020 => BT2100_LUMA,
    };
    let gamma = hlg_system_gamma(peak_nits, ambient_lux);
    let lift =
        DisplayModel::STANDARD_HDR_PQ_1000.y_black + DisplayModel::STANDARD_HDR_PQ_1000.y_refl;
    for px in row.iter_mut() {
        let rs = hlg_inverse_oetf(px[0].clamp(0.0, 1.0));
        let gs = hlg_inverse_oetf(px[1].clamp(0.0, 1.0));
        let bs = hlg_inverse_oetf(px[2].clamp(0.0, 1.0));
        let ys = luma[0] * rs + luma[1] * gs + luma[2] * bs;
        // Y_s = 0 ⇒ 0^(γ−1) with γ > 1 is 0; the multiply keeps it 0.
        let scale = peak_nits * ys.max(0.0).powf(gamma - 1.0);
        px[0] = scale * rs + lift;
        px[1] = scale * gs + lift;
        px[2] = scale * bs + lift;
    }
}

/// [`decode_hlg_row_in_primaries`] at an explicit formula revision — the
/// canonical midp transcendentals at Rev4, the libm bodies below it.
/// REV4VEC: at Rev4 the per-pixel canonical body runs lane-parallel over
/// 16-pixel chunks on the fused-FMA tiers; the scalar canonical body
/// covers the remainder and the non-fused tiers.
pub(crate) fn decode_hlg_row_in_primaries_at_revision(
    row: &mut [[f32; 3]],
    peak_nits: f32,
    ambient_lux: f32,
    primaries: crate::source::ColorPrimaries,
    revision: crate::feature_defs::FormulaRevision,
) {
    if !crate::featcanon::mode(revision).active() {
        decode_hlg_row_in_primaries(row, peak_nits, ambient_lux, primaries);
        return;
    }
    use crate::source::ColorPrimaries;
    let luma = match primaries {
        ColorPrimaries::Srgb => [0.212_639, 0.715_168_7, 0.072_192_32],
        ColorPrimaries::DisplayP3 => [0.228_974_57, 0.691_738_55, 0.079_286_91],
        ColorPrimaries::Bt2020 => BT2100_LUMA,
    };
    let gamma = hlg_system_gamma_at_revision(peak_nits, ambient_lux, revision);
    let lift =
        DisplayModel::STANDARD_HDR_PQ_1000.y_black + DisplayModel::STANDARD_HDR_PQ_1000.y_refl;
    incant!(
        decode_hlg_row_canon_vec(row, luma, gamma, peak_nits, lift),
        [v4x, v4, v3, neon, wasm128, scalar]
    );
}

/// `f32::clamp(v, 0.0, 1.0)` lane-parallel — `if v < 0 {0} else if v > 1
/// {1} else {v}`: NaN passes through and `-0.0` keeps its sign, where
/// `.max(0).min(1)` would give `+0` for both.
#[inline(always)]
fn clamp01_canon<T: F32x16Convert>(
    v: GenericF32x16<T>,
    zero: GenericF32x16<T>,
    one: GenericF32x16<T>,
) -> GenericF32x16<T> {
    GenericF32x16::blend(
        v.simd_lt(zero),
        zero,
        GenericF32x16::blend(v.simd_gt(one), one, v),
    )
}

/// The per-pixel scalar canonical HLG row body — fused tiers' remainder
/// and the non-fused tiers' whole input (REV4VEC).
fn decode_hlg_row_canon_body(
    row: &mut [[f32; 3]],
    luma: [f32; 3],
    gamma: f32,
    peak_nits: f32,
    lift: f32,
) {
    for px in row.iter_mut() {
        let rs = hlg_inverse_oetf_canon(px[0].clamp(0.0, 1.0));
        let gs = hlg_inverse_oetf_canon(px[1].clamp(0.0, 1.0));
        let bs = hlg_inverse_oetf_canon(px[2].clamp(0.0, 1.0));
        let ys = luma[0] * rs + luma[1] * gs + luma[2] * bs;
        // Y_s = 0 ⇒ 0^(γ−1) with γ > 1 is 0; the multiply keeps it 0.
        let scale = peak_nits * crate::det_math::pow_midp_f32(ys.max(0.0), gamma - 1.0);
        px[0] = scale * rs + lift;
        px[1] = scale * gs + lift;
        px[2] = scale * bs + lift;
    }
}

/// [`decode_hlg_row_canon_body`] lane-parallel over 16 pixels per chunk.
/// Every lane computes the scalar body's op sequence: the `clamp(0, 1)`
/// branch structure as nested `blend`s (NaN/−0.0 semantics preserved),
/// the `v <= 0.5` select as `blend`, `ys`'s unfused `Σ luma·e`, and
/// `pow_midp`/`exp_midp` through the generated `f32x16` forms — the same
/// op sequence as `det_math`'s scalar bodies on a fused tier.
/// `-scalar` as for [`decode_pq_row_canon_vec`].
#[magetypes(define(f32x16), v4x, v4, v3, -scalar)]
fn decode_hlg_row_canon_vec(
    token: Token,
    row: &mut [[f32; 3]],
    luma: [f32; 3],
    gamma: f32,
    peak_nits: f32,
    lift: f32,
) {
    const A: f32 = 0.178_832_77;
    const B: f32 = 1.0 - 4.0 * A; // 0.28466892
    const C: f32 = 0.559_910_7;

    let a = f32x16::splat(token, A);
    let bb = f32x16::splat(token, B);
    let cc = f32x16::splat(token, C);
    let zero = f32x16::zero(token);
    let one = f32x16::splat(token, 1.0);
    let half = f32x16::splat(token, 0.5);
    let three = f32x16::splat(token, 3.0);
    let twelve = f32x16::splat(token, 12.0);
    let l0 = f32x16::splat(token, luma[0]);
    let l1 = f32x16::splat(token, luma[1]);
    let l2 = f32x16::splat(token, luma[2]);
    let gm1 = gamma - 1.0;
    let peak = f32x16::splat(token, peak_nits);
    let lifts = f32x16::splat(token, lift);

    /// The branch structure of `v <= 0.5 ? (v·v)/3 : (exp_midp((v−C)/A)+B)/12`
    /// lane-parallel — both arms are evaluated and `blend`ed, like the
    /// scalar `if` selects.
    #[inline(always)]
    fn hlg_e<T: F32x16Convert>(
        v: GenericF32x16<T>,
        a: GenericF32x16<T>,
        bb: GenericF32x16<T>,
        cc: GenericF32x16<T>,
        half: GenericF32x16<T>,
        three: GenericF32x16<T>,
        twelve: GenericF32x16<T>,
    ) -> GenericF32x16<T> {
        let lo = (v * v) / three;
        let hi = (((v - cc) / a).exp_midp() + bb) / twelve;
        GenericF32x16::blend(v.simd_le(half), lo, hi)
    }

    let n = row.len();
    let chunks = n / 16;
    for c in 0..chunks {
        let base = c * 16;
        // FIXED-SIZE ARRAY PATTERN: one range check at the boundary.
        let px: &[[f32; 3]; 16] = row[base..base + 16]
            .try_into()
            .expect("16 pixels per chunk");
        let mut r_arr = [0.0f32; 16];
        let mut g_arr = [0.0f32; 16];
        let mut b_arr = [0.0f32; 16];
        for i in 0..16 {
            let p = px[i];
            r_arr[i] = p[0];
            g_arr[i] = p[1];
            b_arr[i] = p[2];
        }
        let r = clamp01_canon(GenericF32x16::from_array(token, r_arr), zero, one);
        let g = clamp01_canon(GenericF32x16::from_array(token, g_arr), zero, one);
        let b = clamp01_canon(GenericF32x16::from_array(token, b_arr), zero, one);
        let rs = hlg_e(r, a, bb, cc, half, three, twelve);
        let gs = hlg_e(g, a, bb, cc, half, three, twelve);
        let bs = hlg_e(b, a, bb, cc, half, three, twelve);
        let ys = l0 * rs + l1 * gs + l2 * bs;
        let scale = peak * (ys.max(zero)).pow_midp(gm1);
        let ro = (scale * rs + lifts).to_array();
        let go = (scale * gs + lifts).to_array();
        let bo = (scale * bs + lifts).to_array();
        for i in 0..16 {
            row[base + i] = [ro[i], go[i], bo[i]];
        }
    }
    decode_hlg_row_canon_body(&mut row[chunks * 16..], luma, gamma, peak_nits, lift);
}

/// The scalar tier's canonical HLG decode: the per-element scalar body —
/// magetypes' generic `mul_add` is unfused on the scalar backend.
fn decode_hlg_row_canon_vec_scalar(
    _token: archmage::ScalarToken,
    row: &mut [[f32; 3]],
    luma: [f32; 3],
    gamma: f32,
    peak_nits: f32,
    lift: f32,
) {
    decode_hlg_row_canon_body(row, luma, gamma, peak_nits, lift);
}

/// wasm128 has no hardware FMA — same scalar canonical body.
#[cfg(target_arch = "wasm32")]
fn decode_hlg_row_canon_vec_wasm128(
    _token: archmage::Wasm128Token,
    row: &mut [[f32; 3]],
    luma: [f32; 3],
    gamma: f32,
    peak_nits: f32,
    lift: f32,
) {
    decode_hlg_row_canon_body(row, luma, gamma, peak_nits, lift);
}

/// NEON runs the scalar canonical body (2026-10-03): its vector `max`/`min` propagate NaN, while the canon clamps with
/// `f32::max`/`min`, which drop it, so a lane-parallel NEON chunk differed from the scalar canon on NaN inputs (CI aarch64).
#[cfg(target_arch = "aarch64")]
fn decode_hlg_row_canon_vec_neon(
    _token: archmage::NeonToken,
    row: &mut [[f32; 3]],
    luma: [f32; 3],
    gamma: f32,
    peak_nits: f32,
    lift: f32,
) {
    decode_hlg_row_canon_body(row, luma, gamma, peak_nits, lift);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pq16_lookup_matches_float_decode_for_every_code_and_display_peak() {
        let codes: Vec<[u16; 4]> = (0..=u16::MAX)
            .map(|c| [c, c.wrapping_add(21845), c.wrapping_add(43690), u16::MAX])
            .collect();
        for peak in [100.0, 1000.0, 4000.0, 10000.0] {
            let mut expected: Vec<[f32; 3]> = codes
                .iter()
                .map(|p| core::array::from_fn(|c| f32::from(p[c]) * (1.0 / 65535.0)))
                .collect();
            decode_pq_row(&mut expected, peak);
            let mut actual = vec![[0.0; 3]; codes.len()];
            decode_pq_u16_rgba_row(bytemuck::cast_slice(&codes), &mut actual, peak);
            for (code, (a, b)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(
                    a.map(f32::to_bits),
                    b.map(f32::to_bits),
                    "code={code} peak={peak}"
                );
            }
        }
    }

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    // ── sRGB EOTF (IEC 61966-2-1) ──────────────────────────────────────────
    #[test]
    fn srgb_eotf_endpoints_and_midpoint() {
        assert_eq!(srgb_eotf(0.0), 0.0);
        assert!(close(srgb_eotf(1.0), 1.0, 1e-6));
        // 128/255 = 0.501961 → ~0.2159 linear (the classic "mid-gray" value).
        assert!(close(srgb_eotf(128.0 / 255.0), 0.215_861, 1e-4));
        // Linear segment below the 0.04045 break.
        assert!(close(srgb_eotf(0.02), 0.02 / 12.92, 1e-7));
    }

    // ── PQ EOTF (SMPTE ST 2084) — golden cd/m² values ──────────────────────
    #[test]
    fn pq_eotf_reference_values() {
        assert_eq!(pq_eotf(0.0), 0.0);
        // Published: PQ code 0.5 → ~92.25 cd/m².
        assert!(
            close(pq_eotf(0.5), 92.2466, 0.05),
            "pq_eotf(0.5) = {}",
            pq_eotf(0.5)
        );
        // Peak: code 1.0 → 10000 cd/m².
        assert!(
            close(pq_eotf(1.0), 10000.0, 1.0),
            "pq_eotf(1.0) = {}",
            pq_eotf(1.0)
        );
        // The code that encodes exactly 100 cd/m² (sRGB-ish reference white).
        assert!(
            close(pq_eotf(0.508_078), 100.0, 0.1),
            "pq_eotf(0.508078) = {}",
            pq_eotf(0.508_078)
        );
        // Monotone non-decreasing across the range.
        let mut prev = -1.0;
        for i in 0..=100 {
            let l = pq_eotf(i as f32 / 100.0);
            assert!(l >= prev, "PQ EOTF not monotone at v={}", i);
            prev = l;
        }
    }

    // ── HLG inverse-OETF (BT.2100) ─────────────────────────────────────────
    #[test]
    fn hlg_inverse_oetf_reference_values() {
        assert_eq!(hlg_inverse_oetf(0.0), 0.0);
        // The lower-segment join at v = 0.5 is 1/12.
        assert!(
            close(hlg_inverse_oetf(0.5), 1.0 / 12.0, 1e-5),
            "hlg(0.5) = {}",
            hlg_inverse_oetf(0.5)
        );
        // Upper segment reaches 1.0 (scene reference white) at v = 1.0.
        assert!(
            close(hlg_inverse_oetf(1.0), 1.0, 1e-4),
            "hlg(1.0) = {}",
            hlg_inverse_oetf(1.0)
        );
        // Continuity across the 0.5 segment break.
        let lo = hlg_inverse_oetf(0.499_999);
        let hi = hlg_inverse_oetf(0.500_001);
        assert!(
            close(lo, hi, 1e-4),
            "HLG discontinuous at 0.5: {lo} vs {hi}"
        );
    }

    #[test]
    fn hlg_system_gamma_values() {
        // 1.2 at and below a 1000 cd/m² peak.
        assert_eq!(hlg_system_gamma(1000.0, 200.0), 1.2);
        assert_eq!(hlg_system_gamma(600.0, 200.0), 1.2);
        // Brighter peak raises the gamma.
        assert!(hlg_system_gamma(4000.0, 200.0) > 1.2);
    }

    // ── REV4VEC: lane-parallel canon == scalar canon body, per tier ──────
    //
    // Variants are invoked directly with their own summoned token — no
    // global dispatch state, safe under the parallel test harness.

    /// Deterministic xorshift64 — reproducible inputs, no new deps.
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            let mut x = self.0;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.0 = x;
            x.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn f32_bits(&mut self) -> f32 {
            f32::from_bits(self.next() as u32)
        }
    }

    fn special_channels() -> Vec<f32> {
        vec![
            0.0,
            -0.0,
            0.5,
            0.499_999_9,
            0.500_000_1,
            1.0,
            -1.0,
            2.0,
            f32::MIN_POSITIVE,
            f32::from_bits(1),
            f32::from_bits(0x007F_FFFF),
            f32::INFINITY,
            f32::NEG_INFINITY,
            f32::NAN,
            f32::from_bits(0x7FC0_0001),
            1e-30,
            1e30,
            -1e30,
            0.04045,
            0.055,
        ]
    }

    /// PQ row canon: the flat-channel EOTF+display body, every compiled
    /// tier variant vs the scalar body on 10^7 random channels plus every
    /// special class (n ≡ 15 mod 16 covers the remainder).
    #[test]
    fn pq_row_canon_vec_matches_scalar_body_on_every_compiled_tier() {
        use archmage::SimdToken as _;
        let mut rng = Rng(0xD1B5_4A32_D192_ED03);
        let mut flat: Vec<f32> = special_channels();
        let target = flat.len() + 10_000_000;
        while flat.len() < target {
            flat.push(rng.f32_bits());
        }
        while flat.len() % 16 != 15 {
            flat.push(rng.f32_bits());
        }
        let n = flat.len();

        for peak in [100.0f32, 1000.0, 4000.0, 10000.0] {
            let dm = DisplayModel {
                y_peak: peak,
                y_black: DisplayModel::STANDARD_HDR_PQ_1000.y_black,
                y_refl: DisplayModel::STANDARD_HDR_PQ_1000.y_refl,
            };
            let mut expect = flat.clone();
            decode_pq_row_canon_body(&mut expect, dm);
            let run = |name: &str, f: &mut dyn FnMut(&mut [f32])| {
                let mut got = flat.clone();
                f(&mut got);
                for i in 0..n {
                    assert_eq!(
                        got[i].to_bits(),
                        expect[i].to_bits(),
                        "{name} peak={peak}: differs at channel {i} ({:?})",
                        flat[i]
                    );
                }
            };
            run("scalar", &mut |f| {
                decode_pq_row_canon_vec_scalar(archmage::ScalarToken::summon().unwrap(), f, dm)
            });
            #[cfg(target_arch = "x86_64")]
            {
                if let Some(t) = archmage::X64V3Token::summon() {
                    run("v3", &mut |f| decode_pq_row_canon_vec_v3(t, f, dm));
                }
                #[cfg(feature = "avx512")]
                {
                    if let Some(t) = archmage::X64V4Token::summon() {
                        run("v4", &mut |f| decode_pq_row_canon_vec_v4(t, f, dm));
                    }
                    if let Some(t) = archmage::X64V4xToken::summon() {
                        run("v4x", &mut |f| decode_pq_row_canon_vec_v4x(t, f, dm));
                    }
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                if let Some(t) = archmage::NeonToken::summon() {
                    run("neon", &mut |f| decode_pq_row_canon_vec_neon(t, f, dm));
                }
            }
            #[cfg(target_arch = "wasm32")]
            {
                if let Some(t) = archmage::Wasm128Token::summon() {
                    run("wasm128", &mut |f| {
                        decode_pq_row_canon_vec_wasm128(t, f, dm)
                    });
                }
            }
        }
    }

    /// HLG row canon: per-pixel select + OOTF chain, every compiled tier
    /// vs the scalar body. 10^7 random channels hits every special class
    /// incl. NaN/±0 through the clamp-blend; n ≡ 15 mod 16 for remainder.
    #[test]
    fn hlg_row_canon_vec_matches_scalar_body_on_every_compiled_tier() {
        use archmage::SimdToken as _;
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        let spec = special_channels();
        let mut row: Vec<[f32; 3]> = spec
            .iter()
            .enumerate()
            .map(|(i, &a)| [a, spec[(i + 7) % spec.len()], spec[(i + 13) % spec.len()]])
            .collect();
        let target = row.len() + 3_400_000;
        while row.len() < target {
            row.push([rng.f32_bits(), rng.f32_bits(), rng.f32_bits()]);
        }
        while row.len() % 16 != 15 {
            row.push([rng.f32_bits(), rng.f32_bits(), rng.f32_bits()]);
        }
        let n = row.len();

        // γ > 1 and γ = 1.2 paths; both luminance-basis variants.
        for (gamma, luma, tag) in [
            (1.2f32, BT2100_LUMA, "bt2020/γ1.2"),
            (1.5f32, [0.212_639, 0.715_168_7, 0.072_192_32], "srgb/γ1.5"),
            (0.999_999_94f32, BT2100_LUMA, "γ~1"),
        ] {
            let mut expect = row.clone();
            decode_hlg_row_canon_body(&mut expect, luma, gamma, 1000.0, 0.40288734);
            let run = |name: &str, f: &mut dyn FnMut(&mut [[f32; 3]])| {
                let mut got = row.clone();
                f(&mut got);
                for i in 0..n {
                    assert_eq!(
                        got[i].map(f32::to_bits),
                        expect[i].map(f32::to_bits),
                        "{tag}/{name}: differs at px {i} ({:?})",
                        row[i]
                    );
                }
            };
            run("scalar", &mut |r| {
                decode_hlg_row_canon_vec_scalar(
                    archmage::ScalarToken::summon().unwrap(),
                    r,
                    luma,
                    gamma,
                    1000.0,
                    0.40288734,
                )
            });
            #[cfg(target_arch = "x86_64")]
            {
                if let Some(t) = archmage::X64V3Token::summon() {
                    run("v3", &mut |r| {
                        decode_hlg_row_canon_vec_v3(t, r, luma, gamma, 1000.0, 0.40288734)
                    });
                }
                #[cfg(feature = "avx512")]
                {
                    if let Some(t) = archmage::X64V4Token::summon() {
                        run("v4", &mut |r| {
                            decode_hlg_row_canon_vec_v4(t, r, luma, gamma, 1000.0, 0.40288734)
                        });
                    }
                    if let Some(t) = archmage::X64V4xToken::summon() {
                        run("v4x", &mut |r| {
                            decode_hlg_row_canon_vec_v4x(t, r, luma, gamma, 1000.0, 0.40288734)
                        });
                    }
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                if let Some(t) = archmage::NeonToken::summon() {
                    run("neon", &mut |r| {
                        decode_hlg_row_canon_vec_neon(t, r, luma, gamma, 1000.0, 0.40288734)
                    });
                }
            }
            #[cfg(target_arch = "wasm32")]
            {
                if let Some(t) = archmage::Wasm128Token::summon() {
                    run("wasm128", &mut |r| {
                        decode_hlg_row_canon_vec_wasm128(t, r, luma, gamma, 1000.0, 0.40288734)
                    });
                }
            }
        }
    }

    // ── Display model — SDR pixel → nits, zensim `standard_4k` convention ──
    #[test]
    fn sdr_display_model_golden_nits() {
        let d = DisplayModel::STANDARD_4K;
        // mid-gray 128/255 → 43.73 cd/m² (matches cvvdp_features.rs path).
        let mid = d.sdr_linear_to_luminance(srgb_eotf(128.0 / 255.0));
        assert!(close(mid, 43.73, 0.05), "mid-gray nits = {mid}");
        // White 255 → peak + black + reflection = 200.40 cd/m².
        let white = d.sdr_linear_to_luminance(srgb_eotf(1.0));
        assert!(close(white, 200.398, 0.05), "white nits = {white}");
        // Black 0 → black + reflection floor.
        let black = d.sdr_linear_to_luminance(srgb_eotf(0.0));
        assert!(close(black, 0.597_887, 1e-4), "black nits = {black}");
    }

    #[test]
    fn pq_display_model_clamps_to_peak() {
        let d = DisplayModel::STANDARD_HDR_PQ_1000;
        // PQ peak (10000) clamps to the display's 1000 cd/m² ceiling.
        let hi = d.pq_to_luminance(1.0);
        assert!(
            close(hi, 1000.0 + d.y_black + d.y_refl, 1e-3),
            "pq peak = {hi}"
        );
        // A mid PQ value below the ceiling passes through (+ floor).
        let mid = d.pq_to_luminance(0.5);
        assert!(
            close(mid, 92.2466 + d.y_black + d.y_refl, 0.05),
            "pq mid = {mid}"
        );
    }

    #[test]
    fn decode_pq_row_reference_values() {
        // PQ 0.5 → 92.25 cd/m² (HDR_PLAN §1 golden) + the display lift.
        let lift =
            DisplayModel::STANDARD_HDR_PQ_1000.y_black + DisplayModel::STANDARD_HDR_PQ_1000.y_refl;
        let mut row = [[0.5f32; 3], [1.0; 3], [0.0; 3]];
        decode_pq_row(&mut row, 10_000.0);
        assert!(close(row[0][0], 92.25 + lift, 0.05), "{}", row[0][0]);
        assert!(close(row[1][1], 10_000.0 + lift, 1.0), "{}", row[1][1]);
        assert!(row[2][2] >= 0.0 && row[2][2] <= lift + 1e-3);
        // Display-limited decode clamps the highlight at peak.
        let mut row = [[1.0f32; 3]];
        decode_pq_row(&mut row, 1000.0);
        assert!(close(row[0][0], 1000.0 + lift, 0.5), "{}", row[0][0]);
    }

    #[test]
    fn decode_hlg_row_reference_values() {
        // Full-scale white (E' = 1 on all channels): E_s = 1, Y_s = 1,
        // F_D = peak · 1^(γ−1) · 1 = peak (+ lift).
        let lift =
            DisplayModel::STANDARD_HDR_PQ_1000.y_black + DisplayModel::STANDARD_HDR_PQ_1000.y_refl;
        let mut row = [[1.0f32; 3]];
        decode_hlg_row(&mut row, 1000.0, 5.0);
        assert!(close(row[0][0], 1000.0 + lift, 0.5), "{}", row[0][0]);
        // BT.2100 luma weights sum to 1 (spec identity).
        assert!(close(BT2100_LUMA.iter().sum::<f32>(), 1.0, 1e-4));
        // Monotone in signal value on a gray axis; zero stays at the lift.
        let mut prev = -1.0f32;
        for i in 0..=20 {
            let v = i as f32 / 20.0;
            let mut r = [[v; 3]];
            decode_hlg_row(&mut r, 1000.0, 5.0);
            assert!(r[0][0] >= prev, "not monotone at {v}");
            prev = r[0][0];
        }
        let mut r = [[0.0f32; 3]];
        decode_hlg_row(&mut r, 1000.0, 5.0);
        assert!(close(r[0][0], lift, 1e-4), "{}", r[0][0]);
    }
}
