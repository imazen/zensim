//! DVIFM-style block-visibility features — f956..f985 (Y channel at scale 0).
//!
//! Spec: `zenpapers/docs/iqa-methods/dvifm-zensim-feature-design.md`; numpy
//! reference `zenpapers/scripts/dvifm_block_visibility.py`. The family is a
//! five-level binomial pyramid (`[1 2 1]ᵀ[1 2 1]/16`, reflect borders,
//! zero-insertion + `4·B` expand onto the target shape) analysed in
//! non-overlapping 5×5 blocks anchored at (0,0) — partial border blocks are
//! dropped. Per level and block: the 25-pixel hard max `m_b`, corner 3×3
//! extrema per side, signed-power contrast `φ_g`, log-domain smooth
//! visibility `v(C) = (1 + (C/C₀)^{βς})^{-1/ς}` merged across sides with
//! `max`, then `F1 = mean_b v_b·m_b^P` and five triangular F2 bins on
//! `ln(min(C̃_ref, C̃_dist) + 1e-6)` weighted by `m_b^P` — 6 features per level,
//! 30 total. A level with no full block emits exact zeros.
//!
//! All kernel math runs in f64 so the scalar path is an exact reference for
//! the Python fixtures; the row kernels are generic over `F64x8Backend` and
//! run the identical IEEE-754 op sequence per lane (the `2·x` and `0.25·x`
//! factors are exact at any tier, and `mul_add` with a power-of-two factor
//! cannot change the rounding), so every SIMD tier is bit-identical to the
//! scalar path — asserted by `simd_tier_parity`. The streaming pump emits
//! each level's band rows in ascending order and pools blocks in one running
//! order, so every strip size is bit-identical to the whole-plane pass.
//!
//! The default constants (`g = 1, P = 1, C₀ = 0.01, β = 0.65, ς = 4`, the
//! reference self-check's F2 centres) are the design's SEED 1.0 placeholders
//! — the screen phase refits them from the TRAIN block-stats cache. The
//! placeholders are deliberately NOT a trained DVIFM score.

use std::collections::VecDeque;

use archmage::SimdToken;
use archmage::magetypes;
use magetypes::simd::backends::F64x8Backend;
use magetypes::simd::generic::f64x8 as GenericF64x8;

/// Number of pyramid levels (the talk's five: 1 HP + 3 BP + 1 LP).
pub(crate) const DVIFM_LEVELS: usize = 5;
/// Analysis block edge: odd, coprime with every power-of-two codec lattice.
pub(crate) const DVIFM_BLOCK: usize = 5;
/// Triangular F2 bins per level.
pub(crate) const DVIFM_BINS: usize = 5;
/// Features per level: one F1 + `DVIFM_BINS` F2.
pub(crate) const DVIFM_PER_LEVEL: usize = 1 + DVIFM_BINS;
/// Total emitted features (`f956..f985`).
pub(crate) const DVIFM_FEATURES: usize = DVIFM_LEVELS * DVIFM_PER_LEVEL;

/// Band construction: default Laplacian `G_l − E(G_{l+1})`; the local band
/// `G_l − B²G_l` is the design's first ablation. The G cascade is identical
/// either way — only the band plane differs.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum BandMode {
    Laplacian,
    Local,
}

/// Per-level shape constants.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DvifmLevelParams {
    /// Signed-power contrast exponent (φ_g).
    pub g: f64,
    /// Error power applied to the block hard max.
    pub p: f64,
    /// Visibility half-energy contrast.
    pub c0: f64,
    /// Visibility log-slope.
    pub beta: f64,
    /// Visibility knee sharpness (ς → ∞ is the hard knee).
    pub sharp: f64,
    /// Optional upper knee (ablation; `INFINITY` = the design's form).
    pub c_hi: f64,
    /// Triangular F2 bin centres on the `ln(min C̃ + 1e-6)` axis.
    pub f2_centers: [f64; DVIFM_BINS],
    /// Band construction for this level.
    pub band: BandMode,
    /// Corner-min edge discount (`edge=True` in the reference).
    pub edge: bool,
}

impl DvifmLevelParams {
    /// SEED 1.0 placeholders — the design's first-screen constants and the
    /// numpy reference self-check's bin centres `ln([1e-3,1e-2,5e-2,0.2,0.8])`.
    /// Retained as the fixture/test baseline; `Default` tracks the current
    /// screen-baked constants.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) const SEED: Self = Self {
        g: 1.0,
        p: 1.0,
        c0: 0.01,
        beta: 0.65,
        sharp: 4.0,
        c_hi: f64::INFINITY,
        f2_centers: [
            -6.907_755_278_982_137,    // ln(1e-3)
            -4.605_170_185_988_091,    // ln(1e-2)
            -2.995_732_273_553_991,    // ln(5e-2)
            -1.609_437_912_434_100_3,  // ln(0.2)
            -0.223_143_551_314_209_76, // ln(0.8)
        ],
        band: BandMode::Laplacian,
        edge: true,
    };
}

/// Which scale-0 plane feeds the DVIFM pump (crate-private; surfaced on
/// the training-gated `research::DvifmSpec` only).
///
/// `XybY` is the historical default: the producer's XYB channel 1,
/// normalised by [`DVIFM_NORM_SDR`]/[`DVIFM_NORM_PU`]. The `Ycbcr*`
/// variants convert the gamma-encoded sRGB input to full-range BT.709
/// Y′CbCr instead (the DVIFM talk's native colour space) and are SDR-only
/// — the HDR route keeps the PU-normalised `XybY` path.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
// The Ycbcr* variants are constructed only under `feature = "training"`
// (the spec surface); the default `XybY` is what served builds ever see.
#[cfg_attr(not(feature = "training"), allow(dead_code))]
pub(crate) enum DvifmInputPlane {
    /// XYB Y plane (existing behaviour; bit-identical to pre-field builds).
    #[default]
    XybY,
    /// BT.709 full-range Y′ from the gamma-encoded sRGB signal.
    YcbcrY,
    /// BT.709 full-range Cb (centred on 0; normalisation maps to [0, 1]).
    YcbcrCb,
    /// BT.709 full-range Cr (centred on 0; normalisation maps to [0, 1]).
    YcbcrCr,
}

/// The family parameters: one set per pyramid level plus the input-plane
/// selection the pump's rows come from.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DvifmParams {
    pub levels: [DvifmLevelParams; DVIFM_LEVELS],
    /// Scale-0 plane source (default [`DvifmInputPlane::XybY`]); read only
    /// on the training-gated walk — the non-training build always takes
    /// the default.
    #[cfg_attr(not(feature = "training"), allow(dead_code))]
    pub input_plane: DvifmInputPlane,
}

/// Screen-2 constants — derived from the Laplacian-band TRAIN block cache
/// (fit rows only, `row_index < 8000` of the 2026-09-19 minimal-top admitted
/// segment set; `dvifm-screen-2026-09-19/specs/dvifm-lap-derived.json`):
/// per level C₀ = p10 of `min(C̃_ref, C̃_dist)`, F2 centres = the {10,30,50,
/// 70,90}% quantiles of `ln(min C̃ + 1e-6)`; g = 1, P = 1, β = 0.65, ς = 4,
/// c_hi = ∞, edge discount on, Laplacian band — the design's first screen.
/// Kept as the round-2 record while `Default` tracks the current round.
#[allow(dead_code)]
#[allow(clippy::excessive_precision)] // baked quantile values, quoted in full
pub(crate) const DVIFM_SCREEN_LAP: [DvifmLevelParams; DVIFM_LEVELS] = [
    DvifmLevelParams {
        g: 1.0,
        p: 1.0,
        c0: 0.0017294263816438615,
        beta: 0.65,
        sharp: 4.0,
        c_hi: f64::INFINITY,
        f2_centers: [
            -6.3593874374616037,
            -5.035320449097286,
            -4.0195539725619414,
            -3.0716793079739837,
            -2.0480206073699097,
        ],
        band: BandMode::Laplacian,
        edge: true,
    },
    DvifmLevelParams {
        g: 1.0,
        p: 1.0,
        c0: 0.0017761460563633592,
        beta: 0.65,
        sharp: 4.0,
        c_hi: f64::INFINITY,
        f2_centers: [
            -6.3327465405235808,
            -4.870324628412102,
            -3.7595773363237082,
            -2.9265964629353478,
            -2.0872582738627732,
        ],
        band: BandMode::Laplacian,
        edge: true,
    },
    DvifmLevelParams {
        g: 1.0,
        p: 1.0,
        c0: 0.002016443555476144,
        beta: 0.65,
        sharp: 4.0,
        c_hi: f64::INFINITY,
        f2_centers: [
            -6.2059241356707071,
            -4.3622657471446074,
            -3.3855464852686756,
            -2.7253109124144967,
            -2.0380321799612053,
        ],
        band: BandMode::Laplacian,
        edge: true,
    },
    DvifmLevelParams {
        g: 1.0,
        p: 1.0,
        c0: 0.003740424606075978,
        beta: 0.65,
        sharp: 4.0,
        c_hi: f64::INFINITY,
        f2_centers: [
            -5.5882888295057835,
            -3.7874051334739729,
            -3.0092883951992846,
            -2.4612289364958166,
            -1.9052735467604098,
        ],
        band: BandMode::Laplacian,
        edge: true,
    },
    DvifmLevelParams {
        g: 1.0,
        p: 1.0,
        c0: 0.017739474773406982,
        beta: 0.65,
        sharp: 4.0,
        c_hi: f64::INFINITY,
        f2_centers: [
            -4.03190653958357,
            -2.9995830128137255,
            -2.4690981949129278,
            -1.970045059050014,
            -1.4573178491587075,
        ],
        band: BandMode::Laplacian,
        edge: true,
    },
];

/// Screen-3 constants — same derivation as `DVIFM_SCREEN_LAP` but on the
/// local-band cache (`G_l − B²·G_l`; `specs/dvifm-local-derived.json`).
/// Level 4 is the shared low-pass plane, so its constants are identical.
/// Kept as the round-3 record while `Default` tracks the current round.
#[allow(dead_code)]
#[allow(clippy::excessive_precision)] // baked quantile values, quoted in full
pub(crate) const DVIFM_SCREEN_LOCAL: [DvifmLevelParams; DVIFM_LEVELS] = [
    DvifmLevelParams {
        g: 1.0,
        p: 1.0,
        c0: 0.0017179555335587794,
        beta: 0.65,
        sharp: 4.0,
        c_hi: f64::INFINITY,
        f2_centers: [
            -6.3660384205374179,
            -5.0771838099160851,
            -4.0509926853489322,
            -3.0965231739704522,
            -2.0768170070202023,
        ],
        band: BandMode::Local,
        edge: true,
    },
    DvifmLevelParams {
        g: 1.0,
        p: 1.0,
        c0: 0.0017369159618283454,
        beta: 0.65,
        sharp: 4.0,
        c_hi: f64::INFINITY,
        f2_centers: [
            -6.3550686066866762,
            -4.9226504489722513,
            -3.8031681801002697,
            -2.962335986131186,
            -2.1237110965030142,
        ],
        band: BandMode::Local,
        edge: true,
    },
    DvifmLevelParams {
        g: 1.0,
        p: 1.0,
        c0: 0.0019748518825508654,
        beta: 0.65,
        sharp: 4.0,
        c_hi: f64::INFINITY,
        f2_centers: [
            -6.2267556406892153,
            -4.3993971486530379,
            -3.4262334766903426,
            -2.7637835577922258,
            -2.0757735898377474,
        ],
        band: BandMode::Local,
        edge: true,
    },
    DvifmLevelParams {
        g: 1.0,
        p: 1.0,
        c0: 0.0036813775077462196,
        beta: 0.65,
        sharp: 4.0,
        c_hi: f64::INFINITY,
        f2_centers: [
            -5.6041966735218747,
            -3.7882233755442565,
            -3.0261522625847137,
            -2.5011945261472182,
            -1.9427330173568897,
        ],
        band: BandMode::Local,
        edge: true,
    },
    DvifmLevelParams {
        g: 1.0,
        p: 1.0,
        c0: 0.017739474773406982,
        beta: 0.65,
        sharp: 4.0,
        c_hi: f64::INFINITY,
        f2_centers: [
            -4.03190653958357,
            -2.9995830128137255,
            -2.4690981949129278,
            -1.970045059050014,
            -1.4573178491587075,
        ],
        band: BandMode::Local,
        edge: true,
    },
];

/// Screen-4 constants — per-level (g, P, C₀, β, ς) fitted by 150 Adam epochs
/// on the local-band TRAIN block cache (fit rows only; head = linear 30→1
/// MSE, λ_β = 1e-3 prior toward 0.65; `specs/dvifm-local-fitted.json`), with
/// F2 centres then re-derived at the fitted g as the {10,30,50,70,90}%
/// quantiles of `ln(min C̃ + 1e-6)` on the same fit rows
/// (`specs/dvifm-local-fitted-final.json`). Local band, edge on, c_hi = ∞.
/// `DVIFM_SCREEN_LOCAL` is kept as the round-3 record.
#[allow(clippy::excessive_precision)] // baked quantile values, quoted in full
pub(crate) const DVIFM_SCREEN_FITTED: [DvifmLevelParams; DVIFM_LEVELS] = [
    DvifmLevelParams {
        g: 0.8229549277499173,
        p: 0.839164839021145,
        c0: 0.001492783539634792,
        beta: 0.604101903215656,
        sharp: 3.5450968244265835,
        c_hi: f64::INFINITY,
        f2_centers: [
            -5.197234115174323,
            -4.12354037297495,
            -3.258800367501043,
            -2.466767628390845,
            -1.615985826127243,
        ],
        band: BandMode::Local,
        edge: true,
    },
    DvifmLevelParams {
        g: 0.8740482654245578,
        p: 0.9267231301577087,
        c0: 0.0017371984815094774,
        beta: 0.6582619541066784,
        sharp: 4.096324180352423,
        c_hi: f64::INFINITY,
        f2_centers: [
            -5.5115575568537585,
            -4.270301029595456,
            -3.2847288406948953,
            -2.5392262555092624,
            -1.7942537854116223,
        ],
        band: BandMode::Local,
        edge: true,
    },
    DvifmLevelParams {
        g: 1.00567258718167,
        p: 1.1373112213843213,
        c0: 0.0019382839980098213,
        beta: 0.6210656868986264,
        sharp: 3.7036452355361607,
        c_hi: f64::INFINITY,
        f2_centers: [
            -6.263872914866205,
            -4.425306891513245,
            -3.447744257638887,
            -2.78159112990764,
            -2.0907573283398624,
        ],
        band: BandMode::Local,
        edge: true,
    },
    DvifmLevelParams {
        g: 0.9923898490930416,
        p: 1.0358388140325796,
        c0: 0.003823534343442909,
        beta: 0.6487053444624307,
        sharp: 3.835264422374912,
        c_hi: f64::INFINITY,
        f2_centers: [
            -5.558327229743577,
            -3.7586419475139277,
            -3.000786933353879,
            -2.479081379781561,
            -1.9242737207426723,
        ],
        band: BandMode::Local,
        edge: true,
    },
    DvifmLevelParams {
        g: 0.5240864965287544,
        p: 0.8345457349642225,
        c0: 0.022227412394756348,
        beta: 0.6173566191341664,
        sharp: 4.943042371601268,
        c_hi: f64::INFINITY,
        f2_centers: [
            -4.327125794285995,
            -3.274545098232736,
            -2.710117736533956,
            -2.2103097055026,
            -1.6975422772791406,
        ],
        band: BandMode::Local,
        edge: true,
    },
];

impl Default for DvifmParams {
    fn default() -> Self {
        Self {
            levels: DVIFM_SCREEN_FITTED,
            input_plane: DvifmInputPlane::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Input normalisation: map the scale-0 XYB Y plane to DVIFM's [0, 1] axis as
// (Y − min_Y)/s_Y where min_Y and s_Y are the channel's min and (max − min)
// over the encodable domain. Constants are DERIVED (below) and baked with a
// recompute test — the same convention as CSFW_PHI_*.
// ---------------------------------------------------------------------------

/// Derived by `derive_norm_sdr` over the full 256³ sRGB cube through
/// `srgb_to_positive_xyb_planar_into` (the real front-end, not a formula).
/// Pinned by `norm_constants_sdr_recompute`. Measured 2026-09-19: Y spans
/// ≈[0.010, 0.855] over the cube — black lands on the +0.01 Y bias and no
/// sRGB colour reaches Y = 1.
pub(crate) const DVIFM_Y_MIN_SDR: f64 = 0.010_000_014_677_643_776;
/// `max − min` of the encoded Y plane over the same cube.
pub(crate) const DVIFM_Y_SCALE_SDR: f64 = 0.845_308_577_641_844_7;

/// Derived by `derive_norm_pu` over the PU21 encodable grid: per channel
/// `{0.0} ∪ geomspace(PU21_L_MIN, PU21_L_MAX, 31)` through
/// `linear_to_pu_xyb_planar_into`. Y is monotone in each linear-light
/// channel so the extrema sit on cube corners, which the grid contains.
/// Pinned by `norm_constants_pu_recompute`.
pub(crate) const DVIFM_Y_MIN_PU: f64 = 0.009_999_999_776_482_582;
/// `max − min` of the encoded PU Y plane over the same grid.
pub(crate) const DVIFM_Y_SCALE_PU: f64 = 2.323_035_230_860_114;

/// Recompute the SDR Y normalisation over the full sRGB cube. Returns
/// `(min_Y, max_Y)`; the baked scale is `max − min`. Chunked cube enumeration
/// through the real front-end (a coarse grid can miss the extrema — the Y
/// channel mixes all three sRGB channels).
#[cfg_attr(not(test), allow(dead_code))] // recompute oracle; tests pin the baked constants
pub(crate) fn derive_norm_sdr() -> (f64, f64) {
    const CHUNK: usize = 65536;
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    let mut pixels: Vec<[u8; 3]> = Vec::with_capacity(CHUNK);
    let mut xb = vec![0.0f32; CHUNK];
    let mut yb = vec![0.0f32; CHUNK];
    let mut bb = vec![0.0f32; CHUNK];
    let flush = |pixels: &[[u8; 3]],
                 xb: &mut [f32],
                 yb: &mut [f32],
                 bb: &mut [f32],
                 lo: &mut f64,
                 hi: &mut f64| {
        let n = pixels.len();
        crate::color::srgb_to_positive_xyb_planar_into(
            pixels,
            &mut xb[..n],
            &mut yb[..n],
            &mut bb[..n],
        );
        for &v in &yb[..n] {
            let v = v as f64;
            if v < *lo {
                *lo = v;
            }
            if v > *hi {
                *hi = v;
            }
        }
    };
    for r in 0..=255u8 {
        for g in 0..=255u8 {
            for b in 0..=255u8 {
                pixels.push([r, g, b]);
                if pixels.len() == CHUNK {
                    flush(&pixels, &mut xb, &mut yb, &mut bb, &mut lo, &mut hi);
                    pixels.clear();
                }
            }
        }
    }
    if !pixels.is_empty() {
        flush(&pixels, &mut xb, &mut yb, &mut bb, &mut lo, &mut hi);
    }
    (lo, hi)
}

/// Recompute the PU-route Y normalisation over the PU21 encodable grid.
/// Returns `(min_Y, max_Y)`.
#[cfg_attr(not(test), allow(dead_code))] // recompute oracle; tests pin the baked constants
pub(crate) fn derive_norm_pu() -> (f64, f64) {
    const K: usize = 32; // {0} plus 31 log-spaced codes per channel
    let mut axis = Vec::with_capacity(K);
    axis.push(0.0f32);
    for i in 0..(K - 1) {
        let t = i as f64 / (K - 2) as f64;
        let v = crate::pu21::PU21_L_MIN as f64
            * (crate::pu21::PU21_L_MAX as f64 / crate::pu21::PU21_L_MIN as f64).powf(t);
        axis.push(v as f32);
    }
    let n = K * K * K;
    let mut pixels = Vec::with_capacity(n);
    for &r in &axis {
        for &g in &axis {
            for &b in &axis {
                pixels.push([r, g, b]);
            }
        }
    }
    let mut xb = vec![0.0f32; n];
    let mut yb = vec![0.0f32; n];
    let mut bb = vec![0.0f32; n];
    crate::color::linear_to_pu_xyb_planar_into(&pixels, &mut xb, &mut yb, &mut bb);
    let mut lo = f64::INFINITY;
    let mut hi = f64::NEG_INFINITY;
    for &v in &yb {
        let v = v as f64;
        if v < lo {
            lo = v;
        }
        if v > hi {
            hi = v;
        }
    }
    (lo, hi)
}

/// The per-route normalisation, chosen by the walk's front-end.
#[derive(Clone, Copy, Debug)]
pub(crate) struct DvifmNorm {
    pub min: f64,
    pub scale: f64,
}

/// SDR-route normalisation (positive-XYB front-end).
pub(crate) const DVIFM_NORM_SDR: DvifmNorm = DvifmNorm {
    min: DVIFM_Y_MIN_SDR,
    scale: DVIFM_Y_SCALE_SDR,
};
/// HDR-route normalisation (PU21 front-end).
pub(crate) const DVIFM_NORM_PU: DvifmNorm = DvifmNorm {
    min: DVIFM_Y_MIN_PU,
    scale: DVIFM_Y_SCALE_PU,
};

/// Y′CbCr-route normalisation for the BT.709 Y′ plane: `Y′` already spans
/// `[0, 1]` on the gamma-encoded display signal, so the map is identity.
pub(crate) const DVIFM_NORM_YCBCR_Y: DvifmNorm = DvifmNorm {
    min: 0.0,
    scale: 1.0,
};
/// Y′CbCr-route normalisation for the chroma planes: full-range BT.709
/// Cb/Cr are centred at 0 on `[-0.5, 0.5]`; `(c − min)/scale` lands them
/// on the same `[0, 1]` contrast axis the pyramid machinery expects.
pub(crate) const DVIFM_NORM_YCBCR_C: DvifmNorm = DvifmNorm {
    min: -0.5,
    scale: 1.0,
};

/// The route contract: which [`DvifmNorm`] a `(plane, front-end)` pair
/// uses, and which combinations are refused outright. `XybY` is the only
/// plane with an HDR route (PU normalisation); the Y′CbCr planes are
/// SDR-only because the gamma-encoded BT.709 signal they are defined on
/// does not exist for PQ input. A non-`XybY` plane is also incompatible
/// with a sampling contract — the producer's sampled rows are XYB planes
/// and cannot be reused as Y′CbCr rows.
pub(crate) fn dvifm_norm_for(plane: DvifmInputPlane, is_hdr: bool, sampled: bool) -> DvifmNorm {
    assert!(
        plane == DvifmInputPlane::XybY || !sampled,
        "dvifm input_plane != xyb_y cannot serve a sampling contract"
    );
    match (is_hdr, plane) {
        (false, DvifmInputPlane::XybY) => DVIFM_NORM_SDR,
        (true, DvifmInputPlane::XybY) => DVIFM_NORM_PU,
        (false, DvifmInputPlane::YcbcrY) => DVIFM_NORM_YCBCR_Y,
        (false, DvifmInputPlane::YcbcrCb | DvifmInputPlane::YcbcrCr) => DVIFM_NORM_YCBCR_C,
        (true, _) => {
            panic!("dvifm input_plane != xyb_y is SDR-only; HDR keeps the PU xyb_y route")
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar element math — shared by every path, bit-exact building blocks.
// ---------------------------------------------------------------------------

/// Signed power `φ_g(x) = sign(x)·|x|^g` (the reference's `phi`).
#[inline]
fn phi_g(x: f64, g: f64) -> f64 {
    x.signum() * x.abs().powf(g)
}

/// `log(1 + e^x)` without overflow (the reference's `np.logaddexp(0, x)`).
#[inline]
fn log1pexp(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

/// Smooth visibility `v(C) = (1 + (C/C₀)^{βς})^{-1/ς}` evaluated in the log
/// domain; the finite `c_hi` form subtracts the upper knee so v flattens at
/// `(c_hi/C₀)^{-β}` above it. `v(C) = 1` for `C ≤ 0`; finite on `(0, 1]`.
#[inline]
fn visibility(c: f64, lp: &DvifmLevelParams) -> f64 {
    // np.log(np.maximum(c, 0)) = −inf for c ≤ 0 → both softplus terms vanish.
    // `partial_cmp` (not `<=`) so a NaN contrast still takes the v=1 arm —
    // the same result `!(c > 0)` gave, kept explicit.
    if c.partial_cmp(&0.0) != Some(core::cmp::Ordering::Greater) {
        return 1.0;
    }
    let k = lp.beta * lp.sharp;
    let ell = c.ln();
    let lower = log1pexp(k * (ell - lp.c0.ln()));
    let upper = if lp.c_hi.is_finite() {
        log1pexp(k * (ell - lp.c_hi.ln()))
    } else {
        0.0
    };
    (-(lower - upper) / lp.sharp).exp()
}

/// Triangular memberships over the baked centres, ends clamped — mirrors
/// `hat_memberships` (`np.searchsorted` side="left").
#[inline]
fn hat_memberships(ell: f64, centers: &[f64; DVIFM_BINS]) -> [f64; DVIFM_BINS] {
    let mut h = [0.0f64; DVIFM_BINS];
    let ell = ell.clamp(centers[0], centers[DVIFM_BINS - 1]);
    // first index i with centers[i] >= ell (searchsorted side="left")
    let i = centers.partition_point(|&c| c < ell);
    let j = i.saturating_sub(1).min(DVIFM_BINS - 2);
    let t = (ell - centers[j]) / (centers[j + 1] - centers[j]);
    h[j] = 1.0 - t;
    h[j + 1] = t;
    h
}

/// Row index under numpy `reflect` padding (half-sample symmetric):
/// `−1 → 1`, `n → n−2`. A degenerate `n < 2` domain (deeper levels of tiny
/// images) collapses to index 0 — the numpy reference cannot express such
/// planes at all; zensim only needs them not to hang.
#[inline]
fn reflect_101(i: isize, n: usize) -> usize {
    if n < 2 {
        return 0;
    }
    let mut k = i;
    while k < 0 || k >= n as isize {
        k = if k < 0 { -k } else { 2 * n as isize - 2 - k };
    }
    k as usize
}

// ---------------------------------------------------------------------------
// Row kernels — generic over the SIMD token. `mul_add(x, 2, a)` computes
// `x·2 + a` in one FMA; because `2·x` is always exact the result is identical
// to the scalar `a + 2.0·x`. `·0.25` is likewise exact. Every tier therefore
// produces bit-identical rows.
// ---------------------------------------------------------------------------

/// Horizontal `[1 2 1]/4` of `row` (length `w`) into `out` (length `w`).
fn hblur_row<T: F64x8Backend>(t: T, row: &[f64], out: &mut [f64]) {
    let w = row.len();
    debug_assert_eq!(out.len(), w);
    if w < 2 {
        // Degenerate plane column (only reachable on tiny test images):
        // blur of a single-pixel reflected domain is the identity.
        out.copy_from_slice(row);
        return;
    }
    type V<T> = GenericF64x8<T>;
    let two = V::<T>::splat(t, 2.0);
    let q = V::<T>::splat(t, 0.25);
    out[0] = (row[1] + 2.0 * row[0] + row[1]) * 0.25;
    out[w - 1] = (row[w - 2] + 2.0 * row[w - 1] + row[w - 2]) * 0.25;
    let mut c = 1;
    while c + 8 < w {
        let a = V::<T>::from_array(t, row[c - 1..c + 7].try_into().unwrap());
        let b = V::<T>::from_array(t, row[c..c + 8].try_into().unwrap());
        let d = V::<T>::from_array(t, row[c + 1..c + 9].try_into().unwrap());
        let v = (b.mul_add(two, a) + d) * q;
        out[c..c + 8].copy_from_slice(&v.to_array());
        c += 8;
    }
    while c < w - 1 {
        out[c] = (row[c - 1] + 2.0 * row[c] + row[c + 1]) * 0.25;
        c += 1;
    }
}

/// Horizontal `[1 2 1]/4` sampled at even columns only: `out[c]` is the blur
/// at column `2c` of `row` (length `w`), `out` has length `w2 = ⌈w/2⌉`.
/// Strided taps stay scalar (identical values on every tier).
fn hblur_row_even<T: F64x8Backend>(_t: T, row: &[f64], out: &mut [f64]) {
    let w = row.len();
    debug_assert_eq!(out.len(), w.div_ceil(2));
    if w < 2 {
        out.copy_from_slice(row);
        return;
    }
    for (c, o) in out.iter_mut().enumerate() {
        let cc = 2 * c;
        *o = (row[reflect_101(cc as isize - 1, w)]
            + 2.0 * row[cc]
            + row[reflect_101(cc as isize + 1, w)])
            * 0.25;
    }
}

/// `out = (a + 2·b + c)/4` elementwise — the vertical `[1 2 1]` tap.
fn vblur3<T: F64x8Backend>(t: T, a: &[f64], b: &[f64], c: &[f64], out: &mut [f64]) {
    let n = out.len();
    type V<T> = GenericF64x8<T>;
    let two = V::<T>::splat(t, 2.0);
    let q = V::<T>::splat(t, 0.25);
    let mut i = 0;
    while i + 8 <= n {
        let av = V::<T>::from_array(t, a[i..i + 8].try_into().unwrap());
        let bv = V::<T>::from_array(t, b[i..i + 8].try_into().unwrap());
        let cv = V::<T>::from_array(t, c[i..i + 8].try_into().unwrap());
        let v = (bv.mul_add(two, av) + cv) * q;
        out[i..i + 8].copy_from_slice(&v.to_array());
        i += 8;
    }
    while i < n {
        out[i] = (a[i] + 2.0 * b[i] + c[i]) * 0.25;
        i += 1;
    }
}

/// `out = a + 2·b + c` elementwise — the expand taps; `blur121(z)·4` is
/// bitwise the un-normalised sum (`S/4` is exact, `·4` restores it).
fn esum3<T: F64x8Backend>(t: T, a: &[f64], b: &[f64], c: &[f64], out: &mut [f64]) {
    let n = out.len();
    type V<T> = GenericF64x8<T>;
    let two = V::<T>::splat(t, 2.0);
    let mut i = 0;
    while i + 8 <= n {
        let av = V::<T>::from_array(t, a[i..i + 8].try_into().unwrap());
        let bv = V::<T>::from_array(t, b[i..i + 8].try_into().unwrap());
        let cv = V::<T>::from_array(t, c[i..i + 8].try_into().unwrap());
        let v = bv.mul_add(two, av) + cv;
        out[i..i + 8].copy_from_slice(&v.to_array());
        i += 8;
    }
    while i < n {
        out[i] = a[i] + 2.0 * b[i] + c[i];
        i += 1;
    }
}

/// `out = a − b` elementwise.
fn sub_row<T: F64x8Backend>(t: T, a: &[f64], b: &[f64], out: &mut [f64]) {
    let n = out.len();
    type V<T> = GenericF64x8<T>;
    let mut i = 0;
    while i + 8 <= n {
        let v = V::<T>::from_array(t, a[i..i + 8].try_into().unwrap())
            - V::<T>::from_array(t, b[i..i + 8].try_into().unwrap());
        out[i..i + 8].copy_from_slice(&v.to_array());
        i += 8;
    }
    while i < n {
        out[i] = a[i] - b[i];
        i += 1;
    }
}

/// Zero-insertion row onto the target lattice: length `w`, `x[j]` at column
/// `2j`, `0.0` elsewhere (`_up`'s `z[::2, ::2] = x` row).
fn zrow(x: &[f64], w: usize, out: &mut Vec<f64>) {
    out.clear();
    out.resize(w, 0.0);
    for (j, &v) in x.iter().enumerate() {
        if 2 * j < w {
            out[2 * j] = v;
        }
    }
}

// ---------------------------------------------------------------------------
// Whole-plane reference path — the literal numpy dataflow, materialised.
// ---------------------------------------------------------------------------

/// A normalised f64 plane.
#[derive(Clone)]
#[cfg_attr(not(test), allow(dead_code))] // whole-plane reference path; test oracle
pub(crate) struct Plane {
    pub v: Vec<f64>,
    pub w: usize,
    pub h: usize,
}

#[cfg_attr(not(test), allow(dead_code))] // whole-plane reference path; test oracle
impl Plane {
    fn new(w: usize, h: usize) -> Self {
        Self {
            v: vec![0.0; w * h],
            w,
            h,
        }
    }
    fn row(&self, r: usize) -> &[f64] {
        &self.v[r * self.w..(r + 1) * self.w]
    }
    fn row_mut(&mut self, r: usize) -> &mut [f64] {
        let w = self.w;
        &mut self.v[r * w..(r + 1) * w]
    }
}

/// `blur121`: separable `[1 2 1]/16` with reflect borders — the literal
/// pad-then-convolve of the numpy reference.
#[cfg_attr(not(test), allow(dead_code))] // whole-plane reference path; test oracle
pub(crate) fn blur121_plane<T: F64x8Backend>(t: T, src: &Plane) -> Plane {
    let mut hb = Plane::new(src.w, src.h);
    for r in 0..src.h {
        hblur_row(t, src.row(r), hb.row_mut(r));
    }
    let mut out = Plane::new(src.w, src.h);
    for r in 0..src.h {
        let ra = reflect_101(r as isize - 1, src.h);
        let rc = reflect_101(r as isize + 1, src.h);
        vblur3(t, hb.row(ra), hb.row(r), hb.row(rc), out.row_mut(r));
    }
    out
}

/// `down`: `blur121` then `[::2, ::2]` (odd remainders keep the last even
/// index, matching numpy's slicing).
#[cfg_attr(not(test), allow(dead_code))] // whole-plane reference path; test oracle
fn downsample<T: F64x8Backend>(t: T, src: &Plane) -> Plane {
    let w2 = src.w.div_ceil(2);
    let h2 = src.h.div_ceil(2);
    let b = blur121_plane(t, src);
    let mut out = Plane::new(w2, h2);
    for j in 0..h2 {
        for c in 0..w2 {
            out.v[j * w2 + c] = b.row(2 * j)[2 * c];
        }
    }
    out
}

/// `expand`: zero insertion onto the target `(h, w)` lattice, `blur121`, `×4`
/// — mirrors the numpy `_up` literally (no separate crop needed: `z` already
/// has the target shape).
#[cfg_attr(not(test), allow(dead_code))] // whole-plane reference path; test oracle
fn expand<T: F64x8Backend>(t: T, down: &Plane, w: usize, h: usize) -> Plane {
    debug_assert_eq!(down.w, w.div_ceil(2));
    debug_assert_eq!(down.h, h.div_ceil(2));
    let mut z = Plane::new(w, h);
    for r in 0..down.h {
        if 2 * r >= h {
            break;
        }
        let zr = z.row_mut(2 * r);
        for (c, &v) in down.row(r).iter().enumerate() {
            if 2 * c < w {
                zr[2 * c] = v;
            }
        }
    }
    let mut zb = blur121_plane(t, &z);
    for v in zb.v.iter_mut() {
        *v *= 4.0;
    }
    zb
}

/// `laplacian_pyramid`: `L_l = G_l − E(G_{l+1})` for `l < n−1`, `L_{n−1} =
/// G_{n−1}`.
#[cfg_attr(not(test), allow(dead_code))] // whole-plane reference path; test oracle
pub(crate) fn laplacian_pyramid<T: F64x8Backend>(t: T, img: &Plane, n: usize) -> Vec<Plane> {
    let mut g = vec![img.clone()];
    for _ in 1..n {
        let next = downsample(t, g.last().unwrap());
        g.push(next);
    }
    let mut out = Vec::with_capacity(n);
    for l in 0..n - 1 {
        let e = expand(t, &g[l + 1], g[l].w, g[l].h);
        let mut band = Plane::new(g[l].w, g[l].h);
        for r in 0..g[l].h {
            sub_row(t, g[l].row(r), e.row(r), band.row_mut(r));
        }
        out.push(band);
    }
    out.push(g.pop().unwrap());
    out
}

/// `local_band_pyramid`: `L_l = G_l − B²G_l` for `l < n−1`, `L_{n−1} = G_{n−1}`.
#[cfg_attr(not(test), allow(dead_code))] // whole-plane reference path; test oracle
pub(crate) fn local_band_pyramid<T: F64x8Backend>(t: T, img: &Plane, n: usize) -> Vec<Plane> {
    let mut g = vec![img.clone()];
    for _ in 1..n {
        let next = downsample(t, g.last().unwrap());
        g.push(next);
    }
    let mut out = Vec::with_capacity(n);
    for gl in g.iter().take(n - 1) {
        let b2 = blur121_plane(t, &blur121_plane(t, gl));
        let mut band = Plane::new(gl.w, gl.h);
        for r in 0..gl.h {
            sub_row(t, gl.row(r), b2.row(r), band.row_mut(r));
        }
        out.push(band);
    }
    out.push(g.pop().unwrap());
    out
}

// ---------------------------------------------------------------------------
// Block statistics and feature pooling — one accumulation order shared by
// the whole-plane reference and the streaming pump.
// ---------------------------------------------------------------------------

/// One 5×5 block's raw statistics — the 18-float training record (m, peak,
/// 8 extrema per side) plus enough for every contrast mode.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BlockRec {
    /// Hard max |ρ_s − ρ_d| over the block.
    pub m: f64,
    /// Soft peak `Σ σ·δ / max(Σσ, 1e-30)`, σ = δ/(δ+0.01) (training record).
    #[allow(dead_code)] // training record field; the pooled features read `m`
    pub peak: f64,
    /// Per-corner 3×3 max/min of the src-side band.
    pub cmax_s: [f64; 4],
    pub cmin_s: [f64; 4],
    /// Per-corner 3×3 max/min of the dst-side band.
    pub cmax_d: [f64; 4],
    pub cmin_d: [f64; 4],
    /// Block mean of the src-side subtracted pedestal — the local DC the
    /// band was differenced from (expanded next Gaussian level for `lap`,
    /// B²G for `local`; for the low-pass level, the plane itself). The
    /// Weber-contrast denominator for the constants fit: `C̃/(mean+ε)`.
    pub mean_s: f64,
    /// Block mean of the dst-side subtracted pedestal.
    pub mean_d: f64,
}

/// The 20-f32 wire record one block serializes to on the training side
/// output (`research::Extraction::dvifm_block_stats`). Field order is the
/// struct's: `[m, peak, cmax_s×4, cmin_s×4, cmax_d×4, cmin_d×4, mean_s,
/// mean_d]` — the 18-f32 v1 prefix is unchanged, so v1 readers can still
/// parse a v2 blob given its record width.
#[cfg(any(feature = "training", test))]
pub(crate) const DVIFM_BLOCK_F32: usize = 20;

impl BlockRec {
    /// Narrow the f64 kernel record to the 20-f32 training record.
    ///
    /// The narrowing is the contract — the cache exists so per-level
    /// constants (g, P, C₀, β, ς, F2 centres) can be refit without
    /// re-extracting pixels, and f32 keeps the record at ~3.8 B per source
    /// pixel (design §"The block-stats cache").
    #[cfg(any(feature = "training", test))]
    pub(crate) fn to_f32(self) -> [f32; DVIFM_BLOCK_F32] {
        let mut out = [0.0f32; DVIFM_BLOCK_F32];
        out[0] = self.m as f32;
        out[1] = self.peak as f32;
        out[2..6].copy_from_slice(&self.cmax_s.map(|v| v as f32));
        out[6..10].copy_from_slice(&self.cmin_s.map(|v| v as f32));
        out[10..14].copy_from_slice(&self.cmax_d.map(|v| v as f32));
        out[14..18].copy_from_slice(&self.cmin_d.map(|v| v as f32));
        out[18] = self.mean_s as f32;
        out[19] = self.mean_d as f32;
        out
    }
}

const SOFT_PEAK_KAPPA: f64 = 0.01;

/// Scan one band block row (`DVIFM_BLOCK` rows of each side) and emit each
/// block's `BlockRec` in column order. Fixed row-major reduction inside the
/// block so every caller produces bit-identical records.
fn scan_block_row(
    rows_s: &[&[f64]; DVIFM_BLOCK],
    rows_d: &[&[f64]; DVIFM_BLOCK],
    w: usize,
    n: usize,
    mut emit: impl FnMut(BlockRec),
) {
    let nbx = w / n;
    let q = n.div_ceil(2); // corner sub-block edge: 3 for n=5, 2 for n=4
    for bx in 0..nbx {
        let c0 = bx * n;
        let mut m = 0.0f64;
        let mut den = 0.0f64;
        let mut num = 0.0f64;
        let mut cmax_s = [f64::NEG_INFINITY; 4];
        let mut cmin_s = [f64::INFINITY; 4];
        let mut cmax_d = [f64::NEG_INFINITY; 4];
        let mut cmin_d = [f64::INFINITY; 4];
        for r in 0..n {
            for c in c0..c0 + n {
                let d = (rows_s[r][c] - rows_d[r][c]).abs();
                if d > m {
                    m = d;
                }
                let s = d / (d + SOFT_PEAK_KAPPA);
                den += s;
                num += s * d;
            }
        }
        for (qi, &(r0, cq)) in [(0usize, 0usize), (0, n - q), (n - q, 0), (n - q, n - q)]
            .iter()
            .enumerate()
        {
            for r in r0..r0 + q {
                for c in c0 + cq..c0 + cq + q {
                    let sv = rows_s[r][c];
                    let dv = rows_d[r][c];
                    if sv > cmax_s[qi] {
                        cmax_s[qi] = sv;
                    }
                    if sv < cmin_s[qi] {
                        cmin_s[qi] = sv;
                    }
                    if dv > cmax_d[qi] {
                        cmax_d[qi] = dv;
                    }
                    if dv < cmin_d[qi] {
                        cmin_d[qi] = dv;
                    }
                }
            }
        }
        emit(BlockRec {
            m,
            peak: if den > 0.0 { num / den.max(1e-30) } else { 0.0 },
            cmax_s,
            cmin_s,
            cmax_d,
            cmin_d,
            // The batch oracle has no level-plane rows — the streaming
            // pump fills the Weber fields from `mean_q` after emit.
            mean_s: 0.0,
            mean_d: 0.0,
        });
    }
}

/// `block_stats` over one band plane pair — block-row-major `BlockRec`s.
#[cfg_attr(not(test), allow(dead_code))] // training-record producer; test oracle
pub(crate) fn block_stats_level(bs: &Plane, bd: &Plane) -> (usize, usize, Vec<BlockRec>) {
    debug_assert_eq!(bs.w, bd.w);
    debug_assert_eq!(bs.h, bd.h);
    let nby = bs.h / DVIFM_BLOCK;
    let nbx = bs.w / DVIFM_BLOCK;
    let mut recs = Vec::with_capacity(nby * nbx);
    for by in 0..nby {
        let rows_s: [&[f64]; DVIFM_BLOCK] = std::array::from_fn(|k| bs.row(by * DVIFM_BLOCK + k));
        let rows_d: [&[f64]; DVIFM_BLOCK] = std::array::from_fn(|k| bd.row(by * DVIFM_BLOCK + k));
        scan_block_row(&rows_s, &rows_d, bs.w, DVIFM_BLOCK, |rec| recs.push(rec));
    }
    (nby, nbx, recs)
}

/// `contrast_g`: per-block `C̃` for one side — `edge` selects the corner-min
/// (MAD-style edge discount) or the whole-block range.
fn contrast_g_rec(rec: &BlockRec, side: usize, g: f64, edge: bool) -> f64 {
    let (cmax, cmin) = if side == 0 {
        (&rec.cmax_s, &rec.cmin_s)
    } else {
        (&rec.cmax_d, &rec.cmin_d)
    };
    if edge {
        let mut c = f64::INFINITY;
        for qi in 0..4 {
            let cq = phi_g(cmax[qi], g) - phi_g(cmin[qi], g);
            if cq < c {
                c = cq;
            }
        }
        c
    } else {
        let mut mx = f64::NEG_INFINITY;
        let mut mn = f64::INFINITY;
        for qi in 0..4 {
            if cmax[qi] > mx {
                mx = cmax[qi];
            }
            if cmin[qi] < mn {
                mn = cmin[qi];
            }
        }
        phi_g(mx, g) - phi_g(mn, g)
    }
}

/// Two-state visibility: `1` for `C <= c0`, `0` above the knee (a NaN contrast
/// takes the `1` arm, like [`visibility`]'s `v = 1` arm). Non-increasing, so
/// the max-merge across sides is the gate of the smaller contrast.
#[inline]
fn gate_visibility(c: f64, lp: &DvifmLevelParams) -> f64 {
    if c > lp.c0 { 0.0 } else { 1.0 }
}

/// Per-level block sums — accumulated in one running order.
#[derive(Default)]
struct LevelSums {
    n: u64,
    f1: f64,
    /// Restored cut `dvifmgate`: the same F1 with the TWO-STATE visibility
    /// `v = 1 iff C <= c0` (merged across sides with `max`, as `f1` merges
    /// the smooth curve). F2 does not depend on `v`, so this one sum is the
    /// whole gate-form variant. Accumulated in the same order as `f1`; it
    /// cannot move any existing value.
    f1_gate: f64,
    f2: [f64; DVIFM_BINS],
}

/// One block's F1 terms under `lp`: the two side contrasts, the merged
/// visibility `v_b`, and the powered hard-max `m_b^P`. The block's F1
/// contribution is `ε_b = v_b · m_b^P` — [`pool_block`] adds exactly it,
/// and the steering field ([`block_field`]) reports the same values, so
/// "map mass" and "pooled score" cannot drift apart.
fn block_terms(rec: &BlockRec, lp: &DvifmLevelParams) -> (f64, f64, f64, f64) {
    let cs = contrast_g_rec(rec, 0, lp.g, lp.edge);
    let cd = contrast_g_rec(rec, 1, lp.g, lp.edge);
    let vb = visibility(cs, lp).max(visibility(cd, lp));
    let e = rec.m.powf(lp.p);
    (cs, cd, vb, e)
}

/// Pool one block's record into the running sums — `f1_parametric` +
/// `f2_binned` per block.
fn pool_block(sums: &mut LevelSums, lp: &DvifmLevelParams, rec: &BlockRec) {
    let (cs, cd, vb, e) = block_terms(rec, lp);
    sums.f1 += vb * e;
    sums.f1_gate += gate_visibility(cs, lp).max(gate_visibility(cd, lp)) * e;
    let ell = (cs.min(cd) + 1e-6).ln();
    let h = hat_memberships(ell, &lp.f2_centers);
    for (hj, f2) in h.iter().zip(sums.f2.iter_mut()) {
        *f2 += hj * e;
    }
    sums.n += 1;
}

/// Finalise one level's sums → `[F1, F2_0..F2_4]`; exact zeros when empty.
fn level_out(sums: &LevelSums) -> [f64; DVIFM_PER_LEVEL] {
    let mut out = [0.0; DVIFM_PER_LEVEL];
    if sums.n > 0 {
        let inv = 1.0 / sums.n as f64;
        out[0] = sums.f1 * inv;
        for j in 0..DVIFM_BINS {
            out[1 + j] = sums.f2[j] * inv;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// The DVIFM steering field — per-block ε_b = v_b·m_b^P over the level's
// full-block lattice, painted back over the level plane. The block vectors
// come from the SAME `block_terms` the pump pools, and `f1_sum` accumulates
// them in emit order, so the painted field's total mass is the pooled
// `sums.f1` bit-for-bit. `test` builds see it so the gate tests can run
// without the `training` feature.
// ---------------------------------------------------------------------------

/// One level's steering field: `ε_b`, `v_b`, `m_b` per block on the
/// full-block lattice (block-row-major; partial border blocks dropped).
#[cfg(any(test, all(feature = "training", feature = "custom-profiles")))]
#[derive(Clone, Debug)]
pub(crate) struct DvifmBlockField {
    /// Lattice shape `(nby, nbx)` at this level's plane dims.
    pub nby: usize,
    pub nbx: usize,
    /// `ε_b = v_b·m_b^P` — the block's F1 mass.
    pub eps: Vec<f64>,
    /// `v_b` — merged visibility.
    pub vis: Vec<f64>,
    /// `m_b` — block hard max |error|.
    pub err: Vec<f64>,
    /// `Σ_b ε_b` accumulated in record order — bitwise `LevelSums::f1`.
    pub f1_sum: f64,
}

/// Derive one level's field from its cached records — `block_terms` per
/// record in emit order, so `f1_sum` is bitwise-identical to the sum the
/// pump pooled while it emitted them.
#[cfg(any(test, all(feature = "training", feature = "custom-profiles")))]
pub(crate) fn block_field(
    recs: &[BlockRec],
    lp: &DvifmLevelParams,
    nby: usize,
    nbx: usize,
) -> DvifmBlockField {
    debug_assert_eq!(recs.len(), nby * nbx);
    let mut f = DvifmBlockField {
        nby,
        nbx,
        eps: Vec::with_capacity(recs.len()),
        vis: Vec::with_capacity(recs.len()),
        err: Vec::with_capacity(recs.len()),
        f1_sum: 0.0,
    };
    for rec in recs {
        let (_cs, _cd, vb, e) = block_terms(rec, lp);
        let eps_b = vb * e;
        f.f1_sum += eps_b;
        f.eps.push(eps_b);
        f.vis.push(vb);
        f.err.push(rec.m);
    }
    f
}

/// A block field painted over its level plane: `eps_density` carries
/// `ε_b/25` per block pixel (total mass = `f1_sum` exactly — the painted
/// cells are the whole blocks, partial border cells stay 0); `vis`/`err`
/// are the block-constant `v_b`/`m_b` planes.
#[cfg(any(test, all(feature = "training", feature = "custom-profiles")))]
#[derive(Clone, Debug)]
pub(crate) struct DvifmPainted {
    /// `ε_b/25` per pixel inside each full block; 0 on the dropped border.
    pub eps_density: Vec<f64>,
    /// `v_b` per pixel (f32 — a display/query plane, not a mass term).
    pub vis: Vec<f32>,
    /// `m_b` per pixel (f32).
    pub err: Vec<f32>,
}

/// Paint `f` onto a `w`×`h` canvas (the level's plane dims). The ε density
/// is uniform `ε_b/25` inside each block — the stated rule that makes a
/// fractional rectangle integral equal the sum of block terms cut by area.
#[cfg(any(test, all(feature = "training", feature = "custom-profiles")))]
pub(crate) fn paint_block_field(f: &DvifmBlockField, w: usize, h: usize) -> DvifmPainted {
    debug_assert!(f.nbx * DVIFM_BLOCK <= w && f.nby * DVIFM_BLOCK <= h);
    let mut out = DvifmPainted {
        eps_density: vec![0.0; w * h],
        vis: vec![0.0; w * h],
        err: vec![0.0; w * h],
    };
    let inv_nb = 1.0 / (DVIFM_BLOCK * DVIFM_BLOCK) as f64;
    for by in 0..f.nby {
        for bx in 0..f.nbx {
            let b = by * f.nbx + bx;
            let density = f.eps[b] * inv_nb;
            let (v, m) = (f.vis[b] as f32, f.err[b] as f32);
            for r in by * DVIFM_BLOCK..(by + 1) * DVIFM_BLOCK {
                for c in bx * DVIFM_BLOCK..(bx + 1) * DVIFM_BLOCK {
                    let i = r * w + c;
                    out.eps_density[i] = density;
                    out.vis[i] = v;
                    out.err[i] = m;
                }
            }
        }
    }
    out
}

/// Whole-plane feature extraction from normalised f64 planes — the parity
/// reference for the numpy fixtures and the streaming gate.
#[cfg_attr(not(test), allow(dead_code))] // whole-plane reference path; test oracle
pub(crate) fn dvifm_features_whole(
    ref_norm: &[f64],
    dst_norm: &[f64],
    w: usize,
    h: usize,
    params: &DvifmParams,
) -> [f64; DVIFM_FEATURES] {
    let img_s = Plane {
        v: ref_norm.to_vec(),
        w,
        h,
    };
    let img_d = Plane {
        v: dst_norm.to_vec(),
        w,
        h,
    };
    let t = archmage::ScalarToken::summon().expect("scalar token is infallible");
    let mut out = [0.0f64; DVIFM_FEATURES];
    // The two sides may use different band modes per level; build each
    // level's pyramid once per distinct mode used anywhere in the params.
    let mut pyr_s = [None, None];
    let mut pyr_d = [None, None];
    for l in 0..DVIFM_LEVELS {
        let lp = &params.levels[l];
        let mi = match lp.band {
            BandMode::Laplacian => 0,
            BandMode::Local => 1,
        };
        if pyr_s[mi].is_none() {
            let f: fn(archmage::ScalarToken, &Plane, usize) -> Vec<Plane> = match lp.band {
                BandMode::Laplacian => laplacian_pyramid,
                BandMode::Local => local_band_pyramid,
            };
            pyr_s[mi] = Some(f(t, &img_s, DVIFM_LEVELS));
            pyr_d[mi] = Some(f(t, &img_d, DVIFM_LEVELS));
        }
        let bs = &pyr_s[mi].as_ref().unwrap()[l];
        let bd = &pyr_d[mi].as_ref().unwrap()[l];
        let (_nby, _nbx, recs) = block_stats_level(bs, bd);
        let mut sums = LevelSums::default();
        for rec in &recs {
            pool_block(&mut sums, lp, rec);
        }
        out[l * DVIFM_PER_LEVEL..(l + 1) * DVIFM_PER_LEVEL].copy_from_slice(&level_out(&sums));
    }
    out
}

/// Normalise f32 rows onto the DVIFM contrast axis.
fn normalize_row(norm: DvifmNorm, src: &[f32], out: &mut Vec<f64>) {
    out.clear();
    out.reserve(src.len());
    for &v in src {
        out.push((v as f64 - norm.min) / norm.scale);
    }
}

// ---------------------------------------------------------------------------
// Streaming pump — band rows emitted in ascending order per level; block sums
// accumulate in the identical running order as the whole-plane path.
// ---------------------------------------------------------------------------

/// One level's streaming state for one side. Rings hold a bounded number of
/// rows — memory is `O(level width)`, not `O(plane)`.
#[derive(Default)]
struct LevelSide {
    /// Pending G rows awaiting their band emit; `gq[k]` = row `gq_first + k`.
    gq: VecDeque<Vec<f64>>,
    gq_first: usize,
    /// `hblur` at even columns (length w2) — the decimation taps.
    hb: VecDeque<Vec<f64>>,
    hb_first: usize,
    /// LAPLACIAN: `hblur` of z-row `2·(zb_first + k)` (length w).
    zb: VecDeque<Vec<f64>>,
    zb_first: usize,
    /// LOCAL: `hblur(G)` full-width rows (length w).
    hf: VecDeque<Vec<f64>>,
    hf_first: usize,
    /// LOCAL: `blur121(G)` rows (length w).
    b1: VecDeque<Vec<f64>>,
    b1_first: usize,
    /// LOCAL: `hblur(B1)` rows (length w).
    hb2: VecDeque<Vec<f64>>,
    hb2_first: usize,
}

/// What [`DvifmAccum::take_block_cache`] returns: each level's full-block
/// grid `(nby, nbx)`, each level's plane dims `(w, h)` (the canvas the
/// steering field paints over), and the records (level-major,
/// block-row-major).
#[cfg(any(feature = "training", test))]
pub(crate) type DvifmBlockTake = ([(u32, u32); 5], [(u32, u32); 5], Vec<Vec<BlockRec>>);

struct LevelPump {
    w: usize,
    h: usize,
    w2: usize,
    h2: usize,
    /// Level 4: the band is G itself — no decimation, no subtract.
    is_last: bool,
    mode: BandMode,
    arrived: usize,
    down_emitted: usize,
    band_emitted: usize,
    /// LOCAL: emitted B1 / hblur(B1) row counts.
    b1_emitted: usize,
    hb2_emitted: usize,
    side: [LevelSide; 2],
    /// Pending band rows (paired across sides) awaiting a full block row.
    band_q: [VecDeque<Vec<f64>>; 2],
    /// The level-plane rows (`g`) paired with `band_q` — same push/drain
    /// discipline; the block's local mean for the record's Weber fields.
    mean_q: [VecDeque<Vec<f64>>; 2],
    sums: LevelSums,
    lp: DvifmLevelParams,
}

impl LevelPump {
    fn new(w: usize, h: usize, is_last: bool, lp: DvifmLevelParams) -> Self {
        Self {
            w,
            h,
            w2: w.div_ceil(2),
            h2: h.div_ceil(2),
            is_last,
            mode: lp.band,
            arrived: 0,
            down_emitted: 0,
            band_emitted: 0,
            b1_emitted: 0,
            hb2_emitted: 0,
            side: [LevelSide::default(), LevelSide::default()],
            band_q: [VecDeque::new(), VecDeque::new()],
            mean_q: [VecDeque::new(), VecDeque::new()],
            sums: LevelSums::default(),
            lp,
        }
    }
}

fn ring_row(ring: &VecDeque<Vec<f64>>, first: usize, i: usize) -> &[f64] {
    debug_assert!(i >= first && i < first + ring.len());
    &ring[i - first]
}

/// The streaming DVIFM accumulator: feed scale-0 Y-plane row pairs in raster
/// order, then `finish` → 30 features. The pyramid cascade is implicit —
/// each level's decimation feeds the next level's pump as rows unlock.
pub(crate) struct DvifmAccum {
    w: usize,
    h: usize,
    norm: DvifmNorm,
    levels: Vec<LevelPump>,
    /// Optional per-level block-record cache (training side output; the
    /// served path leaves it `None`).
    block_cache: Option<Vec<Vec<BlockRec>>>,
}

impl DvifmAccum {
    /// Restored cut `dvifmgate`: the per-level gate-form F1 (`Σ v_gate·m^P / n`,
    /// exact zero for a level with no full block). Read after
    /// [`dvifm_finish_walk`]/[`dvifm_finish`] has flushed every level.
    pub(crate) fn gate_f1(&self) -> [f64; DVIFM_LEVELS] {
        let mut out = [0.0; DVIFM_LEVELS];
        for (o, pump) in out.iter_mut().zip(&self.levels) {
            if pump.sums.n > 0 {
                *o = pump.sums.f1_gate * (1.0 / pump.sums.n as f64);
            }
        }
        out
    }

    pub(crate) fn new(w: usize, h: usize, norm: DvifmNorm, params: &DvifmParams) -> Self {
        let mut levels = Vec::with_capacity(DVIFM_LEVELS);
        let (mut lw, mut lh) = (w, h);
        for l in 0..DVIFM_LEVELS {
            levels.push(LevelPump::new(
                lw,
                lh,
                l + 1 == DVIFM_LEVELS,
                params.levels[l],
            ));
            lw = lw.div_ceil(2);
            lh = lh.div_ceil(2);
        }
        Self {
            w,
            h,
            norm,
            levels,
            block_cache: None,
        }
    }

    /// Training side output: keep every block's 18-float record.
    ///
    /// Reachable only through `feature_v2::FoldWalkExtras::dvifm`, which
    /// exists under `feature = "training"` — the served path never enables
    /// it and never allocates the per-level vectors. `test` builds see it
    /// so the field gates can exercise the streaming path directly.
    #[cfg(any(feature = "training", test))]
    pub(crate) fn enable_block_cache(&mut self) {
        self.block_cache = Some(vec![Vec::new(); DVIFM_LEVELS]);
    }

    /// The collected block records, level-major — `None` unless enabled.
    #[cfg(any(feature = "training", test))]
    #[cfg_attr(not(test), allow(dead_code))] // training surface; tests inspect it
    pub(crate) fn block_cache(&self) -> Option<&Vec<Vec<BlockRec>>> {
        self.block_cache.as_ref()
    }

    /// Take the collected records after [`dvifm_finish`], plus each level's
    /// full-block grid `(nby, nbx)` and plane dims `(w, h)` — the record
    /// order is block-row-major over that lattice (partial border blocks
    /// are dropped per spec).
    ///
    /// MUST be called after the finish flush: pending band rows below the
    /// last full block row still contribute records, and taking earlier
    /// would truncate the tail.
    #[cfg(any(feature = "training", test))]
    pub(crate) fn take_block_cache(&mut self) -> Option<DvifmBlockTake> {
        let grid: [(u32, u32); 5] = std::array::from_fn(|l| {
            (
                (self.levels[l].h / DVIFM_BLOCK) as u32,
                (self.levels[l].w / DVIFM_BLOCK) as u32,
            )
        });
        let dims: [(u32, u32); 5] =
            std::array::from_fn(|l| (self.levels[l].w as u32, self.levels[l].h as u32));
        self.block_cache.take().map(|levels| (grid, dims, levels))
    }
}

/// Consume five pending band row pairs as one block row.
fn pump_consume_block_row<T: F64x8Backend>(
    pump: &mut LevelPump,
    cache: Option<&mut Vec<BlockRec>>,
    _t: T,
) {
    let rows_s: [&[f64]; DVIFM_BLOCK] = std::array::from_fn(|k| &pump.band_q[0][k][..]);
    let rows_d: [&[f64]; DVIFM_BLOCK] = std::array::from_fn(|k| &pump.band_q[1][k][..]);
    let rows_ms: [&[f64]; DVIFM_BLOCK] = std::array::from_fn(|k| &pump.mean_q[0][k][..]);
    let rows_md: [&[f64]; DVIFM_BLOCK] = std::array::from_fn(|k| &pump.mean_q[1][k][..]);
    let w = pump.w;
    let lp = pump.lp;
    let sums = &mut pump.sums;
    let mut cache = cache;
    let mut bx = 0usize;
    scan_block_row(&rows_s, &rows_d, w, DVIFM_BLOCK, |mut rec| {
        let c0 = bx * DVIFM_BLOCK;
        let mut ms = 0.0f64;
        let mut md = 0.0f64;
        for r in 0..DVIFM_BLOCK {
            for c in c0..c0 + DVIFM_BLOCK {
                ms += rows_ms[r][c];
                md += rows_md[r][c];
            }
        }
        rec.mean_s = ms / (DVIFM_BLOCK * DVIFM_BLOCK) as f64;
        rec.mean_d = md / (DVIFM_BLOCK * DVIFM_BLOCK) as f64;
        bx += 1;
        pool_block(sums, &lp, &rec);
        if let Some(c) = cache.as_mut() {
            c.push(rec);
        }
    });
    pump.band_q[0].drain(..DVIFM_BLOCK);
    pump.band_q[1].drain(..DVIFM_BLOCK);
    pump.mean_q[0].drain(..DVIFM_BLOCK);
    pump.mean_q[1].drain(..DVIFM_BLOCK);
}

/// Emit one Laplacian band row `L[r] = G[r] − E[r]` where
/// `E[r] = hz[r−1] + 2·hz[r] + hz[r+1]` and `hz[k]` is `hblur` of z-row `k`
/// on the `(h, w)` lattice — odd rows are literal zeros, matching
/// `blur121(z)·4` exactly.
fn pump_emit_band_row_lap<T: F64x8Backend>(
    pump: &mut LevelPump,
    cache: Option<&mut Vec<BlockRec>>,
    t: T,
    r: usize,
) {
    debug_assert_eq!(pump.band_emitted, r, "band rows emit in ascending order");
    pump.band_emitted += 1;
    let w = pump.w;
    let h = pump.h;
    let zeros = vec![0.0f64; w];
    for side_i in 0..2 {
        let (e, g) = {
            let sd = &pump.side[side_i];
            let tap = |i: isize| -> &[f64] {
                let k = reflect_101(i, h);
                if k % 2 == 1 {
                    &zeros
                } else {
                    ring_row(&sd.zb, sd.zb_first, k / 2)
                }
            };
            let a = tap(r as isize - 1).to_vec();
            let b = tap(r as isize).to_vec();
            let c = tap(r as isize + 1).to_vec();
            let mut e = vec![0.0f64; w];
            esum3(t, &a, &b, &c, &mut e);
            (e, ring_row(&sd.gq, sd.gq_first, r).to_vec())
        };
        let mut band = vec![0.0f64; w];
        sub_row(t, &g, &e, &mut band);
        pump.band_q[side_i].push_back(band);
        // Weber denominator: the subtracted pedestal E (expanded next
        // Gaussian level) — the local DC this band sits on.
        pump.mean_q[side_i].push_back(e);
        // Shrink rings: gq rows below the emitted index are dead; zb rows
        // below the lowest future z-tap are dead.
        let sd = &mut pump.side[side_i];
        while sd.gq_first < pump.band_emitted {
            sd.gq.pop_front();
            sd.gq_first += 1;
        }
        while sd.zb_first + 1 < pump.band_emitted / 2 {
            sd.zb.pop_front();
            sd.zb_first += 1;
        }
    }
    if pump.band_q[0].len() == DVIFM_BLOCK {
        pump_consume_block_row(pump, cache, t);
    }
}

/// Local-band progress: emit `B1` rows, `hblur(B1)` rows and band rows as
/// their taps unlock. `b1_emitted`/`hb2_emitted`/`band_emitted` are shared
/// per-level counters (the two sides advance in lockstep); `flush` allows
/// reflected taps at the trailing border.
fn pump_local_progress<T: F64x8Backend>(
    pump: &mut LevelPump,
    mut cache: Option<&mut Vec<BlockRec>>,
    t: T,
    flush: bool,
) {
    let h = pump.h;
    let w = pump.w;
    // B1[k] = vblur3(hf[k−1], hf[k], hf[k+1]) — needs hf[k+1]; mid-stream
    // only real taps (k+1 ≤ arrived−1) unlock, at flush the last row reflects.
    while pump.b1_emitted < h && (flush || pump.b1_emitted + 1 < pump.arrived) {
        let k = pump.b1_emitted;
        pump.b1_emitted += 1;
        for side_i in 0..2 {
            let sd = &mut pump.side[side_i];
            let a = ring_row(&sd.hf, sd.hf_first, reflect_101(k as isize - 1, h)).to_vec();
            let b = ring_row(&sd.hf, sd.hf_first, k).to_vec();
            let c = ring_row(&sd.hf, sd.hf_first, reflect_101(k as isize + 1, h)).to_vec();
            let mut b1 = vec![0.0f64; w];
            vblur3(t, &a, &b, &c, &mut b1);
            sd.b1.push_back(b1);
            // hf rows below the next emit's lowest tap are dead.
            let keep_from = pump.b1_emitted.saturating_sub(1);
            while sd.hf_first < keep_from {
                sd.hf.pop_front();
                sd.hf_first += 1;
            }
        }
    }
    // hb2[k] = hblur(b1[k]) for every emitted B1 row.
    while pump.hb2_emitted < pump.b1_emitted {
        let k = pump.hb2_emitted;
        pump.hb2_emitted += 1;
        for side_i in 0..2 {
            let sd = &mut pump.side[side_i];
            let src = ring_row(&sd.b1, sd.b1_first, k).to_vec();
            let mut row = vec![0.0f64; w];
            hblur_row(t, &src, &mut row);
            sd.hb2.push_back(row);
            // b1 rows below the next hb2 emit are dead.
            while sd.b1_first < pump.hb2_emitted {
                sd.b1.pop_front();
                sd.b1_first += 1;
            }
        }
    }
    // band[r] = G[r] − vblur3(hb2[r−1], hb2[r], hb2[r+1]) with reflected taps.
    while pump.band_emitted < h {
        let r = pump.band_emitted;
        let need = reflect_101(r as isize + 1, h);
        if need >= pump.hb2_emitted && !flush {
            break;
        }
        pump.band_emitted += 1;
        for side_i in 0..2 {
            let sd = &mut pump.side[side_i];
            let a = ring_row(&sd.hb2, sd.hb2_first, reflect_101(r as isize - 1, h)).to_vec();
            let b = ring_row(&sd.hb2, sd.hb2_first, r).to_vec();
            let c = ring_row(&sd.hb2, sd.hb2_first, reflect_101(r as isize + 1, h)).to_vec();
            let mut b2g = vec![0.0f64; w];
            vblur3(t, &a, &b, &c, &mut b2g);
            let g = ring_row(&sd.gq, sd.gq_first, r).to_vec();
            let mut band = vec![0.0f64; w];
            sub_row(t, &g, &b2g, &mut band);
            pump.band_q[side_i].push_back(band);
            // Weber denominator: the subtracted pedestal B²G — the local
            // DC this band sits on.
            pump.mean_q[side_i].push_back(b2g);
            // Ring housekeeping.
            while sd.gq_first < pump.band_emitted {
                sd.gq.pop_front();
                sd.gq_first += 1;
            }
            let keep_from = pump.band_emitted.saturating_sub(1);
            while sd.hb2_first < keep_from {
                sd.hb2.pop_front();
                sd.hb2_first += 1;
            }
        }
        if pump.band_q[0].len() == DVIFM_BLOCK {
            pump_consume_block_row(pump, cache.as_deref_mut(), t);
        }
    }
}

/// Push one normalised row pair into level `l`; returns the decimated row
/// pairs that unlocked (to cascade into level `l+1`).
fn level_push_row<T: F64x8Backend>(
    acc: &mut DvifmAccum,
    t: T,
    l: usize,
    row_s: Vec<f64>,
    row_d: Vec<f64>,
) {
    let mut outbox: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    {
        let pump = &mut acc.levels[l];
        let mut cache = acc.block_cache.as_mut().map(|c| &mut c[l]);
        if pump.is_last {
            // Low-pass level: the band is G itself — the mean rows are
            // the same plane.
            pump.mean_q[0].push_back(row_s.clone());
            pump.mean_q[1].push_back(row_d.clone());
            pump.band_q[0].push_back(row_s);
            pump.band_q[1].push_back(row_d);
            pump.arrived += 1;
            if pump.band_q[0].len() == DVIFM_BLOCK {
                pump_consume_block_row(pump, cache, t);
            }
        } else {
            let r = pump.arrived;
            pump.arrived += 1;
            let w2 = pump.w2;
            for (side_i, row) in [row_s, row_d].into_iter().enumerate() {
                let sd = &mut pump.side[side_i];
                let mut hb = vec![0.0f64; w2];
                hblur_row_even(t, &row, &mut hb);
                sd.hb.push_back(hb);
                if pump.mode == BandMode::Local {
                    let mut hf = vec![0.0f64; pump.w];
                    hblur_row(t, &row, &mut hf);
                    sd.hf.push_back(hf);
                }
                sd.gq.push_back(row);
            }
            // Emit decimated rows whose real taps (2j−1, 2j, 2j+1) arrived.
            while pump.down_emitted < pump.h2 && 2 * pump.down_emitted < r {
                let j = pump.down_emitted;
                pump.down_emitted += 1;
                let mut down_pair = [Vec::new(), Vec::new()];
                for (side_i, dp) in down_pair.iter_mut().enumerate() {
                    let sd = &mut pump.side[side_i];
                    let h = pump.h;
                    let tap =
                        |i: isize| -> &[f64] { ring_row(&sd.hb, sd.hb_first, reflect_101(i, h)) };
                    let (a, b, c) = (
                        tap(2 * j as isize - 1).to_vec(),
                        tap(2 * j as isize).to_vec(),
                        tap(2 * j as isize + 1).to_vec(),
                    );
                    let mut down = vec![0.0f64; w2];
                    vblur3(t, &a, &b, &c, &mut down);
                    *dp = down;
                }
                for side_i in 0..2 {
                    let sd = &mut pump.side[side_i];
                    while sd.hb_first < 2 * pump.down_emitted - 1 {
                        sd.hb.pop_front();
                        sd.hb_first += 1;
                    }
                }
                if pump.mode == BandMode::Laplacian {
                    for (side_i, dp) in down_pair.iter().enumerate() {
                        let mut z = Vec::new();
                        zrow(dp, pump.w, &mut z);
                        let mut zb = vec![0.0f64; pump.w];
                        hblur_row(t, &z, &mut zb);
                        pump.side[side_i].zb.push_back(zb);
                    }
                    for rr in [2 * j as isize - 1, 2 * j as isize] {
                        if rr >= 0 && (rr as usize) < pump.h {
                            pump_emit_band_row_lap(pump, cache.as_deref_mut(), t, rr as usize);
                        }
                    }
                }
                outbox.push((down_pair[0].clone(), down_pair[1].clone()));
            }
            if pump.mode == BandMode::Local {
                pump_local_progress(pump, cache, t, false);
            }
        }
    }
    for (s, d) in outbox {
        level_push_row(acc, t, l + 1, s, d);
    }
}

/// Flush level `l`: emit the tail decimated rows (reflected taps) and the
/// remaining band rows, then cascade and flush the next level.
fn level_flush<T: F64x8Backend>(acc: &mut DvifmAccum, t: T, l: usize) {
    let mut outbox: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
    {
        let pump = &mut acc.levels[l];
        let mut cache = acc.block_cache.as_mut().map(|c| &mut c[l]);
        if !pump.is_last {
            // Tail decimated rows (odd heights leave the last row's tap
            // reflected).
            while pump.down_emitted < pump.h2 {
                let j = pump.down_emitted;
                pump.down_emitted += 1;
                let mut down_pair = [Vec::new(), Vec::new()];
                for (side_i, dp) in down_pair.iter_mut().enumerate() {
                    let sd = &mut pump.side[side_i];
                    let h = pump.h;
                    let w2 = pump.w2;
                    let tap = |i: isize| -> Vec<f64> {
                        ring_row(&sd.hb, sd.hb_first, reflect_101(i, h)).to_vec()
                    };
                    let a = tap(2 * j as isize - 1);
                    let b = tap(2 * j as isize);
                    let c = tap(2 * j as isize + 1);
                    let mut down = vec![0.0f64; w2];
                    vblur3(t, &a, &b, &c, &mut down);
                    *dp = down;
                }
                if pump.mode == BandMode::Laplacian {
                    for (side_i, dp) in down_pair.iter().enumerate() {
                        let mut z = Vec::new();
                        zrow(dp, pump.w, &mut z);
                        let mut zb = vec![0.0f64; pump.w];
                        hblur_row(t, &z, &mut zb);
                        pump.side[side_i].zb.push_back(zb);
                    }
                    for rr in [2 * j as isize - 1, 2 * j as isize] {
                        if rr >= 0 && (rr as usize) < pump.h {
                            pump_emit_band_row_lap(pump, cache.as_deref_mut(), t, rr as usize);
                        }
                    }
                }
                outbox.push((down_pair[0].clone(), down_pair[1].clone()));
            }
            match pump.mode {
                BandMode::Laplacian => {
                    // Remaining band rows (the last row of even heights).
                    while pump.band_emitted < pump.h {
                        let r = pump.band_emitted;
                        pump_emit_band_row_lap(pump, cache.as_deref_mut(), t, r);
                    }
                }
                BandMode::Local => {
                    pump_local_progress(pump, cache, t, true);
                }
            }
        }
    }
    for (s, d) in outbox {
        if l + 1 < DVIFM_LEVELS {
            level_push_row(acc, t, l + 1, s, d);
        }
    }
    if l + 1 < DVIFM_LEVELS {
        level_flush(acc, t, l + 1);
    }
}

/// Push f32 strip rows (scale-0 Y planes) into the accumulator. `src`/`dst`
/// are `rows·w` f32 in raster order; normalised on the way in.
pub(crate) fn dvifm_push_rows<T: F64x8Backend>(
    t: T,
    acc: &mut DvifmAccum,
    src: &[f32],
    dst: &[f32],
) {
    debug_assert_eq!(src.len(), dst.len());
    debug_assert_eq!(src.len() % acc.w, 0);
    let mut s = Vec::new();
    let mut d = Vec::new();
    for r in 0..src.len() / acc.w {
        normalize_row(acc.norm, &src[r * acc.w..(r + 1) * acc.w], &mut s);
        normalize_row(acc.norm, &dst[r * acc.w..(r + 1) * acc.w], &mut d);
        level_push_row(acc, t, 0, std::mem::take(&mut s), std::mem::take(&mut d));
    }
}

/// Flush all levels and finalise the 30 features.
pub(crate) fn dvifm_finish<T: F64x8Backend>(t: T, acc: &mut DvifmAccum) -> [f64; DVIFM_FEATURES] {
    debug_assert_eq!(
        acc.levels[0].arrived, acc.h,
        "finish before every plane row was pushed"
    );
    level_flush(acc, t, 0);
    let mut out = [0.0f64; DVIFM_FEATURES];
    for (l, pump) in acc.levels.iter().enumerate() {
        // Partial trailing block rows are dropped per spec.
        debug_assert!(pump.band_q[0].len() < DVIFM_BLOCK);
        debug_assert!(pump.mean_q[0].len() == pump.band_q[0].len());
        out[l * DVIFM_PER_LEVEL..(l + 1) * DVIFM_PER_LEVEL].copy_from_slice(&level_out(&pump.sums));
    }
    out
}

/// Push a whole f32 plane pair at once (the `strip = h` case) — used by the
/// served path's whole-plane comparison tests.
#[cfg_attr(not(test), allow(dead_code))] // convenience wrapper; test oracle
pub(crate) fn dvifm_features_stream(
    src: &[f32],
    dst: &[f32],
    w: usize,
    h: usize,
    norm: DvifmNorm,
    params: &DvifmParams,
) -> [f64; DVIFM_FEATURES] {
    let t = archmage::ScalarToken::summon().expect("scalar token is infallible");
    let mut acc = DvifmAccum::new(w, h, norm, params);
    dvifm_push_rows(t, &mut acc, src, dst);
    dvifm_finish(t, &mut acc)
}

// ---------------------------------------------------------------------------
// Walk dispatch — the `#[magetypes]`/`incant!` shape `feature_v2` uses for
// every kernel. `incant!` resolves one tier per process, so every strip's
// push and the final flush run the SAME backend (a per-call pick would still
// be bit-safe here — the op sequence is identical per lane — but one pick is
// the cheaper and clearer contract).
// ---------------------------------------------------------------------------

#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]
fn dvifm_push_rows_entry(token: Token, acc: &mut DvifmAccum, src: &[f32], dst: &[f32]) {
    dvifm_push_rows(token, acc, src, dst)
}

#[magetypes(v4x, v4, v3, neon, wasm128, scalar)]
fn dvifm_finish_entry(token: Token, acc: &mut DvifmAccum) -> [f64; DVIFM_FEATURES] {
    dvifm_finish(token, acc)
}

/// Runtime dispatch for the strip-loop pump ([`crate::feature_v2`]'s
/// `foldapp_streaming_walk_impl` hook).
pub(crate) fn dvifm_push_rows_walk(acc: &mut DvifmAccum, src: &[f32], dst: &[f32]) {
    archmage::incant!(
        dvifm_push_rows_entry(acc, src, dst),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

/// Runtime dispatch for the walk's finalize.
pub(crate) fn dvifm_finish_walk(acc: &mut DvifmAccum) -> [f64; DVIFM_FEATURES] {
    archmage::incant!(
        dvifm_finish_entry(acc),
        [v4x, v4, v3, neon, wasm128, scalar]
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn rng_plane(seed: u64, w: usize, h: usize) -> Plane {
        // Deterministic LCG — identical values on every platform.
        let mut s = seed;
        let v = (0..w * h)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 33) & 0xffff_ffff) as f64 / u32::MAX as f64
            })
            .collect();
        Plane { v, w, h }
    }

    fn f32_plane(seed: u64, w: usize, h: usize) -> Vec<f32> {
        rng_plane(seed, w, h).v.iter().map(|&v| v as f32).collect()
    }

    fn scalar() -> archmage::ScalarToken {
        archmage::ScalarToken::summon().expect("scalar token is infallible")
    }

    #[test]
    fn norm_constants_sdr_recompute() {
        let (lo, hi) = derive_norm_sdr();
        assert!(
            (lo - DVIFM_Y_MIN_SDR).abs() <= 1e-12 && (hi - lo - DVIFM_Y_SCALE_SDR).abs() <= 1e-12,
            "recomputed ({lo}, {hi}) vs baked ({}, {})",
            DVIFM_Y_MIN_SDR,
            DVIFM_Y_MIN_SDR + DVIFM_Y_SCALE_SDR
        );
    }

    #[test]
    fn norm_constants_pu_recompute() {
        let (lo, hi) = derive_norm_pu();
        assert!(
            (lo - DVIFM_Y_MIN_PU).abs() <= 1e-9 && (hi - lo - DVIFM_Y_SCALE_PU).abs() <= 1e-9,
            "recomputed ({lo}, {hi}) vs baked ({}, {})",
            DVIFM_Y_MIN_PU,
            DVIFM_Y_MIN_PU + DVIFM_Y_SCALE_PU
        );
    }

    #[test]
    fn identity_is_zero() {
        let p = rng_plane(7, 97, 89);
        let f = dvifm_features_whole(&p.v, &p.v, p.w, p.h, &DvifmParams::default());
        for (i, v) in f.iter().enumerate() {
            assert_eq!(*v, 0.0, "feature {i} not exactly zero on an identical pair");
        }
    }

    #[test]
    fn monotone_in_noise() {
        // F1 must be non-decreasing as dst noise amplitude grows.
        let base = rng_plane(11, 125, 125);
        let noise = rng_plane(13, 125, 125);
        let params = DvifmParams::default();
        let mut prev = [-f64::INFINITY; DVIFM_LEVELS];
        for &a in &[0.0f64, 0.005, 0.02, 0.08, 0.3] {
            let d: Vec<f64> = base
                .v
                .iter()
                .zip(&noise.v)
                .map(|(&b, &n)| b + a * (n - 0.5))
                .collect();
            let f = dvifm_features_whole(&base.v, &d, 125, 125, &params);
            for l in 0..DVIFM_LEVELS {
                let f1 = f[l * DVIFM_PER_LEVEL];
                assert!(
                    f1 >= prev[l] - 1e-15,
                    "level {l} F1 not monotone: {prev:?} -> {f1}"
                );
                prev[l] = f1;
            }
        }
    }

    #[test]
    fn merged_visibility_dominates_sides() {
        let lp = DvifmLevelParams::SEED;
        for &c in &[1e-5f64, 1e-3, 1e-2, 0.05, 0.3, 1.0, 4.0] {
            let v = visibility(c, &lp);
            assert!((0.0..=1.0).contains(&v), "v({c}) = {v} out of (0,1]");
            for &d in &[1e-4f64, 1e-2, 0.1, 0.9] {
                let m = visibility(c, &lp).max(visibility(d, &lp));
                assert!(m >= visibility(c, &lp));
                assert!(m >= visibility(d, &lp));
            }
        }
        assert_eq!(visibility(0.0, &lp), 1.0);
        assert_eq!(visibility(-1.0, &lp), 1.0);
    }

    #[test]
    fn tied_curves_max_merge_is_v_of_min() {
        // With tied θ the max-merge equals v at the smaller contrast.
        let lp = DvifmLevelParams::SEED;
        for &a in &[1e-4f64, 1e-2, 0.2, 0.7] {
            for &b in &[1e-3f64, 0.05, 0.5] {
                let merged = visibility(a, &lp).max(visibility(b, &lp));
                let vmin = visibility(a.min(b), &lp);
                assert!((merged - vmin).abs() < 1e-12, "{a} {b}: {merged} vs {vmin}");
            }
        }
    }

    #[test]
    fn visibility_log_domain_matches_closed_form() {
        let lp = DvifmLevelParams::SEED;
        for &c in &[1e-12f64, 1e-8, 1e-4, 1e-2, 0.1, 1.0, 10.0, 1e6] {
            let direct = (1.0 + (c / lp.c0).powf(lp.beta * lp.sharp)).powf(-1.0 / lp.sharp);
            let v = visibility(c, &lp);
            assert!(
                (v - direct).abs() <= 1e-12 * direct.max(1.0),
                "c={c}: {v} vs {direct}"
            );
        }
        // Upper-knee ablation: v approaches (c_hi/c0)^(−β) from above as
        // C → ∞ (the residual logaddexp term is positive, never zero).
        let mut lp2 = lp;
        lp2.c_hi = 0.2;
        let v_hi = visibility(10.0, &lp2);
        let flat = (lp2.c_hi / lp2.c0).powf(-lp2.beta);
        assert!(
            v_hi > flat && v_hi < flat * 1.001,
            "upper knee {v_hi} vs asymptote {flat}"
        );
        // Still non-increasing through the knee (reference self-check pins
        // np.diff(v) <= 0 for finite c_hi), bounded below by the asymptote.
        assert!(visibility(0.5, &lp2) <= visibility(0.15, &lp2) + 1e-15);
        assert!(visibility(0.5, &lp2) > flat);
    }

    #[test]
    fn sharp_limit_visibility() {
        // ς→∞ limit: v(C) = max(1, C/C0)^(−β).
        for &c in &[1e-4f64, 1e-3, 1e-2, 0.1, 1.0] {
            let mut lp = DvifmLevelParams::SEED;
            lp.sharp = 1e6;
            let limit = (c / lp.c0).max(1.0).powf(-lp.beta);
            let v = visibility(c, &lp);
            assert!((v - limit).abs() < 1e-6, "c={c}: {v} vs {limit}");
        }
    }

    #[test]
    fn g1_plain_contrast_is_block_range() {
        // edge=False at g=1: C̃ = the whole-block pixel range.
        let rec = BlockRec {
            m: 0.0,
            peak: 0.0,
            cmax_s: [0.9, 0.8, 0.7, 0.6],
            cmin_s: [0.1, 0.2, 0.3, 0.4],
            cmax_d: [0.5; 4],
            cmin_d: [0.5; 4],
            mean_s: 0.0,
            mean_d: 0.0,
        };
        assert_eq!(contrast_g_rec(&rec, 0, 1.0, false), 0.8); // 0.9 − 0.1
        assert!((contrast_g_rec(&rec, 0, 1.0, true) - 0.2).abs() < 1e-15); // min corner
        assert_eq!(contrast_g_rec(&rec, 1, 1.0, true), 0.0);
    }

    #[test]
    fn signed_power_extrema() {
        // φ_g preserves sign; corner minima can be negative in a band.
        let rec = BlockRec {
            m: 0.0,
            peak: 0.0,
            cmax_s: [0.5; 4],
            cmin_s: [-0.5; 4],
            cmax_d: [0.0; 4],
            cmin_d: [0.0; 4],
            mean_s: 0.0,
            mean_d: 0.0,
        };
        assert_eq!(contrast_g_rec(&rec, 0, 1.0, true), 1.0); // 0.5 − (−0.5)
        // φ₂(0.5) − φ₂(−0.5) = 0.25 − (−0.25) = 0.5: the signed-power range
        // of a symmetric pair is NOT zero.
        assert_eq!(contrast_g_rec(&rec, 0, 2.0, true), 0.5);
        assert_eq!(contrast_g_rec(&rec, 1, 1.0, true), 0.0);
    }

    #[test]
    fn hard_max_is_full_block_support() {
        // A single 1-pixel delta in a 5×5 block must hit m_b exactly.
        let mut s = vec![0.0f64; 25];
        let mut d = vec![0.0f64; 25];
        d[12] = 0.7; // centre pixel
        let ps = Plane {
            v: s.clone(),
            w: 5,
            h: 5,
        };
        let pd = Plane {
            v: d.clone(),
            w: 5,
            h: 5,
        };
        let (_, _, recs) = block_stats_level(&ps, &pd);
        assert_eq!(recs[0].m, 0.7);
        s[0] = 1.0; // (0,0) is only in corner sub-block q0
        let ps = Plane { v: s, w: 5, h: 5 };
        let (_, _, recs) = block_stats_level(&ps, &pd);
        assert_eq!(recs[0].cmax_s, [1.0, 0.0, 0.0, 0.0]);
        let mut s = vec![0.0f64; 25];
        s[2 * 5 + 2] = 1.0; // block centre is in every corner sub-block
        let ps = Plane { v: s, w: 5, h: 5 };
        let (_, _, recs) = block_stats_level(&ps, &pd);
        assert_eq!(recs[0].cmax_s, [1.0; 4]);
    }

    #[test]
    fn pyramid_reconstructs_input() {
        let t = scalar();
        let img = rng_plane(17, 61, 47);
        let bands = laplacian_pyramid(t, &img, 5);
        let mut rec = bands[4].clone();
        for l in (0..4).rev() {
            let e = expand(t, &rec, bands[l].w, bands[l].h);
            let mut next = Plane::new(bands[l].w, bands[l].h);
            for r in 0..bands[l].h {
                let a = bands[l].row(r);
                let b = e.row(r);
                let o = next.row_mut(r);
                for c in 0..bands[l].w {
                    o[c] = a[c] + b[c];
                }
            }
            rec = next;
        }
        for (i, (&a, &b)) in rec.v.iter().zip(&img.v).enumerate() {
            assert!((a - b).abs() < 1e-12, "recon diverges at {i}: {a} vs {b}");
        }
    }

    #[test]
    fn laplacian_local_alias_identity() {
        // L_lap = L_local − B[(4s−1)·B·G], s = 1 on the even-even lattice.
        let t = scalar();
        let img = rng_plane(19, 40, 33);
        let lap = laplacian_pyramid(t, &img, 2)[0].clone();
        let loc = local_band_pyramid(t, &img, 2)[0].clone();
        let bg = blur121_plane(t, &img);
        let mut z = Plane::new(bg.w, bg.h);
        for r in 0..bg.h {
            for c in 0..bg.w {
                // (4s − 1): s = 1 on the even-even lattice, 0 elsewhere —
                // 3 at even-even, −1 at every odd coordinate.
                let s4m1 = if r % 2 == 0 && c % 2 == 0 { 3.0 } else { -1.0 };
                z.v[r * bg.w + c] = s4m1 * bg.v[r * bg.w + c];
            }
        }
        let alias = blur121_plane(t, &z);
        for i in 0..lap.v.len() {
            let lhs = lap.v[i];
            let rhs = loc.v[i] - alias.v[i];
            assert!((lhs - rhs).abs() < 1e-12, "i={i}: {lhs} vs {rhs}");
        }
    }

    /// Separable Gaussian blur with reflect borders — the reference's
    /// `_gauss_blur` (radius `⌊3σ + 0.5⌋`, kernel `exp(−(i/σ)²/2)`).
    fn gauss_blur(src: &Plane, sigma: f64) -> Plane {
        let rad = (3.0 * sigma + 0.5) as usize;
        let mut k = Vec::with_capacity(2 * rad + 1);
        let mut sum = 0.0;
        for i in 0..=2 * rad {
            let x = (i as f64 - rad as f64) / sigma;
            let v = (-0.5 * x * x).exp();
            k.push(v);
            sum += v;
        }
        for v in &mut k {
            *v /= sum;
        }
        let mut tmp = Plane::new(src.w, src.h);
        for r in 0..src.h {
            let row = src.row(r);
            let o = tmp.row_mut(r);
            for (c, oc) in o.iter_mut().enumerate().take(src.w) {
                let mut acc = 0.0;
                for (i, &kv) in k.iter().enumerate() {
                    let cc = reflect_101(c as isize + i as isize - rad as isize, src.w);
                    acc += kv * row[cc];
                }
                *oc = acc;
            }
        }
        let mut out = Plane::new(src.w, src.h);
        for r in 0..src.h {
            let o = out.row_mut(r);
            for (c, oc) in o.iter_mut().enumerate().take(src.w) {
                let mut acc = 0.0;
                for (i, &kv) in k.iter().enumerate() {
                    let rr = reflect_101(r as isize + i as isize - rad as isize, src.h);
                    acc += kv * tmp.row(rr)[c];
                }
                *oc = acc;
            }
        }
        out
    }

    /// n-generic block stats (scan_block_row is fixed at DVIFM_BLOCK).
    fn block_stats_n(bs: &Plane, bd: &Plane, n: usize) -> Vec<BlockRec> {
        let nbx = bs.w / n;
        let nby = bs.h / n;
        let q = n.div_ceil(2);
        let mut recs = Vec::with_capacity(nbx * nby);
        for by in 0..nby {
            for bx in 0..nbx {
                let (r0, c0) = (by * n, bx * n);
                let mut m = 0.0f64;
                let mut cmax = [f64::NEG_INFINITY; 4];
                let mut cmin = [f64::INFINITY; 4];
                let mut dmax = [f64::NEG_INFINITY; 4];
                let mut dmin = [f64::INFINITY; 4];
                for r in r0..r0 + n {
                    let rs = bs.row(r);
                    let rd = bd.row(r);
                    for c in c0..c0 + n {
                        let dd = (rs[c] - rd[c]).abs();
                        if dd > m {
                            m = dd;
                        }
                    }
                }
                for (qi, &(rq, cq)) in [(0usize, 0usize), (0, n - q), (n - q, 0), (n - q, n - q)]
                    .iter()
                    .enumerate()
                {
                    for r in r0 + rq..r0 + rq + q {
                        let rs = bs.row(r);
                        let rd = bd.row(r);
                        for c in c0 + cq..c0 + cq + q {
                            let sv = rs[c];
                            let dv = rd[c];
                            if sv > cmax[qi] {
                                cmax[qi] = sv;
                            }
                            if sv < cmin[qi] {
                                cmin[qi] = sv;
                            }
                            if dv > dmax[qi] {
                                dmax[qi] = dv;
                            }
                            if dv < dmin[qi] {
                                dmin[qi] = dv;
                            }
                        }
                    }
                }
                recs.push(BlockRec {
                    m,
                    peak: 0.0,
                    cmax_s: cmax,
                    cmin_s: cmin,
                    cmax_d: dmax,
                    cmin_d: dmin,
                    mean_s: 0.0,
                    mean_d: 0.0,
                });
            }
        }
        recs
    }

    #[test]
    fn five_level_grid_phase() {
        // Port of the reference `_phase_demo`: smooth content + per-8×8 DC
        // steps on the distorted side at codec-grid phases 0..7. F1's
        // max/min over phases must be smaller for n=5 (coprime with the
        // 8-px grid) than n=4. Reference numbers (binomial Laplacian, numpy
        // PCG64 noise): L0 1.070 vs 1.346, L1 1.531 vs 1.706, L2 1.668 vs
        // 1.709, L3 1.116 vs 1.157. The noise realisation here differs
        // (LCG), so only the clearest level-0 ordering is asserted.
        let t = scalar();
        let n_px = 240usize;
        let raw = rng_plane(3, n_px, n_px);
        let unit: Vec<f64> = raw.v.iter().map(|&u| (u - 0.5) * 12f64.sqrt()).collect();
        let noise = gauss_blur(
            &Plane {
                v: unit,
                w: n_px,
                h: n_px,
            },
            2.0,
        );
        let mut refp = Plane::new(n_px, n_px);
        for r in 0..n_px {
            for c in 0..n_px {
                refp.v[r * n_px + c] = 0.5
                    + 0.15 * (c as f64 / 23.0).sin() * (r as f64 / 31.0).cos()
                    + 0.05 * noise.v[r * n_px + c];
            }
        }
        // steps ~ uniform(−0.02, 0.02) on the (n_px/8 + 2)² lattice.
        let side = n_px / 8 + 2;
        let steps = rng_plane(5, side, side);
        let lp = DvifmLevelParams::SEED;
        let f1_at = |lvl: usize, n: usize, ph: usize| -> f64 {
            let mut dist = Plane::new(n_px, n_px);
            for r in 0..n_px {
                for c in 0..n_px {
                    let st = steps.v[((r + ph) / 8) * side + (c + ph) / 8];
                    dist.v[r * n_px + c] = refp.v[r * n_px + c] + 0.04 * st - 0.02;
                }
            }
            let bs = laplacian_pyramid(t, &refp, DVIFM_LEVELS);
            let bd = laplacian_pyramid(t, &dist, DVIFM_LEVELS);
            let recs = block_stats_n(&bs[lvl], &bd[lvl], n);
            let mut sums = LevelSums::default();
            for rec in &recs {
                pool_block(&mut sums, &lp, rec);
            }
            level_out(&sums)[0]
        };
        let mut spread5 = (f64::NEG_INFINITY, f64::INFINITY);
        let mut spread4 = (f64::NEG_INFINITY, f64::INFINITY);
        for ph in 0..8 {
            let f5 = f1_at(0, 5, ph);
            let f4 = f1_at(0, 4, ph);
            spread5.0 = spread5.0.max(f5);
            spread5.1 = spread5.1.min(f5);
            spread4.0 = spread4.0.max(f4);
            spread4.1 = spread4.1.min(f4);
        }
        let ratio5 = spread5.0 / spread5.1;
        let ratio4 = spread4.0 / spread4.1;
        assert!(
            ratio5 < ratio4,
            "n=5 should be less phase-sensitive than n=4 at L0: {ratio5} vs {ratio4}"
        );
    }

    #[test]
    fn f2_memberships_sum_to_one() {
        let lp = DvifmLevelParams::SEED;
        for &ell in &[
            -20.0f64,
            lp.f2_centers[0],
            -4.0,
            -1.0,
            lp.f2_centers[4],
            5.0,
        ] {
            let h = hat_memberships(ell, &lp.f2_centers);
            let s: f64 = h.iter().sum();
            assert!((s - 1.0).abs() < 1e-12, "ell={ell} sum={s}");
        }
    }

    /// The input-plane default is the historical XYB-Y tap — a spec that
    /// does not name a plane must reproduce pre-field bytes exactly.
    #[test]
    fn input_plane_defaults_to_xyb_y() {
        assert_eq!(DvifmInputPlane::default(), DvifmInputPlane::XybY);
        assert_eq!(DvifmParams::default().input_plane, DvifmInputPlane::XybY);
    }

    /// The Y′CbCr norms: Y′ is already `[0,1]` (identity map); Cb/Cr are
    /// centred at 0 on `[-0.5, 0.5]` and land on `[0,1]` under
    /// `(c − min)/scale` — the same contrast axis the pyramid expects.
    #[test]
    fn ycbcr_norms_place_the_planes() {
        let f = |v: f64, n: DvifmNorm| (v - n.min) / n.scale;
        assert_eq!(f(0.35, DVIFM_NORM_YCBCR_Y), 0.35);
        assert_eq!(f(1.0, DVIFM_NORM_YCBCR_Y), 1.0);
        assert_eq!(f(-0.5, DVIFM_NORM_YCBCR_C), 0.0);
        assert_eq!(f(0.0, DVIFM_NORM_YCBCR_C), 0.5);
        assert_eq!(f(0.5, DVIFM_NORM_YCBCR_C), 1.0);
    }

    /// The route contract: SDR XybY → SDR norm, HDR XybY → PU norm, SDR
    /// Y′CbCr → the plane's own norm; HDR + non-XYB and sampled + non-XYB
    /// are refused.
    #[test]
    fn norm_for_route_table() {
        for plane in [
            DvifmInputPlane::XybY,
            DvifmInputPlane::YcbcrY,
            DvifmInputPlane::YcbcrCb,
            DvifmInputPlane::YcbcrCr,
        ] {
            // Every plane serves un-sampled SDR.
            let _ = dvifm_norm_for(plane, false, false);
        }
        assert_eq!(
            dvifm_norm_for(DvifmInputPlane::XybY, false, false).min,
            DVIFM_NORM_SDR.min
        );
        assert_eq!(
            dvifm_norm_for(DvifmInputPlane::XybY, true, false).min,
            DVIFM_NORM_PU.min
        );
        assert_eq!(
            dvifm_norm_for(DvifmInputPlane::YcbcrY, false, false).min,
            DVIFM_NORM_YCBCR_Y.min
        );
        assert_eq!(
            dvifm_norm_for(DvifmInputPlane::YcbcrCb, false, false).min,
            DVIFM_NORM_YCBCR_C.min
        );
        assert_eq!(
            dvifm_norm_for(DvifmInputPlane::YcbcrCr, false, false).min,
            DVIFM_NORM_YCBCR_C.min
        );
        // XYB-Y stays servable under a sampling contract (the historical
        // route); only the new planes are refused there.
        let _ = dvifm_norm_for(DvifmInputPlane::XybY, true, true);
    }

    #[test]
    #[should_panic(expected = "SDR-only")]
    fn norm_for_rejects_hdr_ycbcr() {
        let _ = dvifm_norm_for(DvifmInputPlane::YcbcrY, true, false);
    }

    #[test]
    #[should_panic(expected = "sampling contract")]
    fn norm_for_rejects_sampled_ycbcr() {
        let _ = dvifm_norm_for(DvifmInputPlane::YcbcrCb, false, true);
    }

    #[test]
    fn whole_vs_pump_bit_identical() {
        for &(w, h) in &[(125usize, 125usize), (97, 89), (251, 129), (10, 7)] {
            let s32 = f32_plane(23, w, h);
            let d32 = f32_plane(29, w, h);
            let norm = DVIFM_NORM_SDR;
            let s64: Vec<f64> = s32
                .iter()
                .map(|&v| (v as f64 - norm.min) / norm.scale)
                .collect();
            let d64: Vec<f64> = d32
                .iter()
                .map(|&v| (v as f64 - norm.min) / norm.scale)
                .collect();
            let whole = dvifm_features_whole(&s64, &d64, w, h, &DvifmParams::default());
            let stream = dvifm_features_stream(&s32, &d32, w, h, norm, &DvifmParams::default());
            for (i, (&a, &b)) in whole.iter().zip(&stream).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "feature {i} at {w}x{h}: {a} vs {b}"
                );
            }
        }
    }

    /// Restored cut `dvifmgate`: the gate-form F1 is `Σ v_gate·m^P / n` over the
    /// same blocks in the same order as the curve F1. Wide-open knee
    /// (`c0 = +inf`) admits every block so it equals the plain mean of `m^P`
    /// recomputed from the cached block records; a shut knee (`-inf`) is an
    /// exact zero; the fitted knee lies between, level by level; and reading
    /// it moves none of the 30 curve/F2 features.
    #[test]
    fn gate_f1_semantics() {
        let (w, h) = (125usize, 130usize);
        let s32 = f32_plane(41, w, h);
        let d32 = f32_plane(43, w, h);
        let norm = DVIFM_NORM_SDR;
        let t = scalar();
        let run = |params: &DvifmParams| {
            let mut acc = DvifmAccum::new(w, h, norm, params);
            acc.enable_block_cache();
            dvifm_push_rows(t, &mut acc, &s32, &d32);
            let out = dvifm_finish(t, &mut acc);
            let gate = acc.gate_f1();
            (out, gate, acc)
        };
        let base = DvifmParams::default();
        let (out, gate, _) = run(&base);
        let mut open = base;
        let mut shut = base;
        for l in 0..DVIFM_LEVELS {
            open.levels[l].c0 = f64::INFINITY;
            shut.levels[l].c0 = f64::NEG_INFINITY;
        }
        let (_, gate_open, acc_open) = run(&open);
        let (_, gate_shut, _) = run(&shut);
        let cache = acc_open.block_cache().expect("cache enabled");
        for l in 0..DVIFM_LEVELS {
            let recs = &cache[l];
            assert!(!recs.is_empty(), "level {l} has full blocks at {w}x{h}");
            let mut sum = 0.0f64;
            for rec in recs {
                sum += rec.m.powf(open.levels[l].p);
            }
            assert_eq!(
                gate_open[l].to_bits(),
                (sum * (1.0 / recs.len() as f64)).to_bits(),
                "open gate == mean m^P, level {l}"
            );
            assert_eq!(gate_shut[l], 0.0, "shut gate level {l}");
            assert!(
                gate[l] >= 0.0 && gate[l] <= gate_open[l],
                "fitted gate level {l}: {} outside [0, {}]",
                gate[l],
                gate_open[l]
            );
        }
        // Reading the gate does not change any curve/F2 value: the same
        // params without the gate read reproduce them bit for bit.
        let plain = dvifm_features_stream(&s32, &d32, w, h, norm, &base);
        for (i, (&a, &b)) in plain.iter().zip(&out).enumerate() {
            assert_eq!(a.to_bits(), b.to_bits(), "feature {i}");
        }
        assert!(gate.iter().any(|&v| v > 0.0), "some level admits blocks");
    }

    #[test]
    fn strip_size_bit_identical() {
        for &(w, h) in &[(125usize, 130usize), (97, 101)] {
            let s32 = f32_plane(31, w, h);
            let d32 = f32_plane(37, w, h);
            let norm = DVIFM_NORM_SDR;
            let params = DvifmParams::default();
            let t = scalar();
            let want = dvifm_features_stream(&s32, &d32, w, h, norm, &params);
            for strip in [1usize, 2, 3, 5, 7, 11, 16, 33, 64, 97, 128] {
                let mut acc = DvifmAccum::new(w, h, norm, &params);
                let mut r = 0;
                while r < h {
                    let n = strip.min(h - r);
                    dvifm_push_rows(
                        t,
                        &mut acc,
                        &s32[r * w..(r + n) * w],
                        &d32[r * w..(r + n) * w],
                    );
                    r += n;
                }
                let got = dvifm_finish(t, &mut acc);
                for (i, (&a, &b)) in want.iter().zip(&got).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "feature {i} at {w}x{h} strip={strip}: {a} vs {b}"
                    );
                }
            }
        }
    }

    #[test]
    fn local_band_whole_vs_pump() {
        let mut params = DvifmParams::default();
        for lp in &mut params.levels {
            lp.band = BandMode::Local;
        }
        for &(w, h) in &[(125usize, 125usize), (101, 67)] {
            let s32 = f32_plane(41, w, h);
            let d32 = f32_plane(43, w, h);
            let norm = DVIFM_NORM_SDR;
            let s64: Vec<f64> = s32
                .iter()
                .map(|&v| (v as f64 - norm.min) / norm.scale)
                .collect();
            let d64: Vec<f64> = d32
                .iter()
                .map(|&v| (v as f64 - norm.min) / norm.scale)
                .collect();
            let whole = dvifm_features_whole(&s64, &d64, w, h, &params);
            let want = dvifm_features_stream(&s32, &d32, w, h, norm, &params);
            for (i, (&a, &b)) in whole.iter().zip(&want).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "whole feature {i} at {w}x{h}");
            }
            let t = scalar();
            for strip in [3usize, 7, 16] {
                let mut acc = DvifmAccum::new(w, h, norm, &params);
                let mut r = 0;
                while r < h {
                    let n = strip.min(h - r);
                    dvifm_push_rows(
                        t,
                        &mut acc,
                        &s32[r * w..(r + n) * w],
                        &d32[r * w..(r + n) * w],
                    );
                    r += n;
                }
                let got = dvifm_finish(t, &mut acc);
                for (i, (&a, &b)) in want.iter().zip(&got).enumerate() {
                    assert_eq!(
                        a.to_bits(),
                        b.to_bits(),
                        "local feature {i} at {w}x{h} strip={strip}"
                    );
                }
            }
        }
    }

    #[test]
    fn level_4_lowpass_carries_dc() {
        // Bands 0..3 pass zero under a constant offset; the low-pass carries
        // the DC difference.
        let w = 125;
        let h = 125;
        let s = vec![0.3f64; w * h];
        let d = vec![0.31f64; w * h];
        let f = dvifm_features_whole(&s, &d, w, h, &DvifmParams::default());
        for l in 0..4 {
            assert_eq!(f[l * DVIFM_PER_LEVEL], 0.0, "band level {l} saw DC");
        }
        assert!(f[4 * DVIFM_PER_LEVEL] > 0.0, "low-pass ignored DC");
    }

    #[test]
    fn empty_level_emits_zero() {
        let w = 4;
        let h = 4;
        let s = vec![0.5f64; w * h];
        let d = vec![0.4f64; w * h];
        let f = dvifm_features_whole(&s, &d, w, h, &DvifmParams::default());
        assert!(f.iter().all(|&v| v == 0.0));
        let s32: Vec<f32> = s.iter().map(|&v| v as f32).collect();
        let d32: Vec<f32> = d.iter().map(|&v| v as f32).collect();
        let f2 = dvifm_features_stream(&s32, &d32, w, h, DVIFM_NORM_SDR, &DvifmParams::default());
        assert!(f2.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn offset_free_bands() {
        // The band-pass planes and the error are offset-free: shifting both
        // sides by a constant leaves levels 0..3 unchanged to within f64
        // rounding ((x+c)−(y+c) is not bit-identical to x−y). The reference
        // pins the same property with np.allclose.
        let p = rng_plane(47, 100, 100);
        let q = rng_plane(53, 100, 100);
        let params = DvifmParams::default();
        let f1 = dvifm_features_whole(&p.v, &q.v, 100, 100, &params);
        let shifted_s: Vec<f64> = p.v.iter().map(|&v| v + 0.2).collect();
        let shifted_d: Vec<f64> = q.v.iter().map(|&v| v + 0.2).collect();
        let f2 = dvifm_features_whole(&shifted_s, &shifted_d, 100, 100, &params);
        for l in 0..4 {
            for j in 0..DVIFM_PER_LEVEL {
                let i = l * DVIFM_PER_LEVEL + j;
                let d = (f1[i] - f2[i]).abs();
                assert!(
                    d <= 1e-12 * f1[i].abs().max(1.0),
                    "band level {l} feature {j} moved under a common offset: {d}"
                );
            }
        }
    }

    // ---- numpy reference parity -----------------------------------------

    /// The four closed-form cases of `scripts/dvifm_parity_fixture.py` —
    /// the formulas must stay identical to that generator.
    fn parity_inputs(name: &str, h: usize, w: usize) -> (Plane, Plane) {
        let mut r = Plane::new(w, h);
        let mut d = Plane::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let (ry, cx) = (y as f64, x as f64);
                let (v, dv) = match name {
                    "a" => {
                        let v = 0.5
                            + 0.28 * (0.21 * ry + 0.13 * cx).sin() * (0.17 * ry - 0.11 * cx).cos()
                            + 0.07 * (0.53 * (ry + cx)).sin();
                        (
                            v,
                            v + 0.03 * (1.3 * ry - 0.7 * cx).sin() * (0.31 * ry + 0.9 * cx).cos(),
                        )
                    }
                    "b" => {
                        let v = 0.45
                            + 0.25 * (0.35 * ry).sin() * (0.28 * cx).cos()
                            + if cx >= 18.0 { 0.12 } else { 0.0 };
                        (v, v + 0.02 * (0.9 * ry - 0.4 * cx).cos())
                    }
                    "c" => {
                        let v = 0.5 + 0.2 * (0.9 * ry).sin() * (1.1 * cx).cos();
                        (v, v + 0.03 * (3.0 * ry + 2.0 * cx).sin())
                    }
                    "d" => {
                        let v = 0.5 + 0.3 * (0.8 * ry + 0.5 * cx).sin();
                        (v, v + 0.04 * (1.7 * ry - 0.6 * cx).cos())
                    }
                    _ => panic!("unknown parity case {name}"),
                };
                r.v[y * w + x] = v;
                d.v[y * w + x] = dv;
            }
        }
        (r, d)
    }

    #[test]
    fn numpy_parity_planes_and_features() {
        // Fixture written by scripts/dvifm_parity_fixture.py from the numpy
        // reference. Bound is 1e-6; the fixture itself is %.9e (~5e-10
        // quantisation) and inputs differ by ~1 ulp between libms.
        const TOL: f64 = 1e-6;
        let t = scalar();
        let text = include_str!("../tests/fixtures/dvifm_parity_2026-09-19.txt");
        let mut lines = text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'));
        while let Some(head) = lines.next() {
            let mut it = head.split_whitespace();
            assert_eq!(it.next(), Some("case"));
            let name = it.next().unwrap();
            let h: usize = it.next().unwrap().parse().unwrap();
            let w: usize = it.next().unwrap().parse().unwrap();
            let (refp, dist) = parity_inputs(name, h, w);
            let lap = laplacian_pyramid(t, &refp, DVIFM_LEVELS);
            loop {
                let line = lines.next().expect("case end");
                if line == "end" {
                    break;
                }
                let mut it = line.split_whitespace();
                match it.next().unwrap() {
                    "plane" => {
                        let l: usize = it.next().unwrap().parse().unwrap();
                        let ph: usize = it.next().unwrap().parse().unwrap();
                        let pw: usize = it.next().unwrap().parse().unwrap();
                        assert_eq!((lap[l].h, lap[l].w), (ph, pw), "{name} L{l} shape");
                        for r in 0..ph {
                            let row: Vec<f64> = lines
                                .next()
                                .unwrap()
                                .split_whitespace()
                                .map(|v| v.parse().unwrap())
                                .collect();
                            assert_eq!(row.len(), pw, "{name} L{l} row {r} width");
                            for (c, &e) in row.iter().enumerate() {
                                let g = lap[l].row(r)[c];
                                assert!((g - e).abs() <= TOL, "{name} L{l} ({r},{c}): {g} vs {e}");
                            }
                        }
                    }
                    "features" => {
                        let band = it.next().unwrap();
                        // The fixture was generated under the SEED constants —
                        // pin them; `DvifmParams::default()` tracks the current
                        // screen-baked constants.
                        let mut params = DvifmParams {
                            levels: [DvifmLevelParams::SEED; DVIFM_LEVELS],
                            input_plane: DvifmInputPlane::XybY,
                        };
                        if band == "local" {
                            for lp in &mut params.levels {
                                lp.band = BandMode::Local;
                            }
                        }
                        let got = dvifm_features_whole(&refp.v, &dist.v, w, h, &params);
                        for l in 0..DVIFM_LEVELS {
                            let row: Vec<f64> = lines
                                .next()
                                .unwrap()
                                .split_whitespace()
                                .map(|v| v.parse().unwrap())
                                .collect();
                            assert_eq!(row.len(), DVIFM_PER_LEVEL);
                            for (j, &e) in row.iter().enumerate() {
                                let g = got[l * DVIFM_PER_LEVEL + j];
                                assert!(
                                    (g - e).abs() <= TOL,
                                    "{name} {band} L{l} f{j}: {g} vs {e}"
                                );
                            }
                        }
                    }
                    other => panic!("bad fixture line {other}"),
                }
            }
        }
    }

    /// The u8 sRGB case generator the fixture's `ycbcr_rgb` mirrors.
    /// `dist` applies the clipped gain+lift of the Python `dist` side.
    fn ycbcr_parity_rgb(distorted: bool, w: usize, h: usize) -> Vec<[u8; 3]> {
        let mut img = Vec::with_capacity(w * h);
        for y in 0..h {
            for x in 0..w {
                img.push([
                    ((x * 37 + y * 11 + 3) % 256) as u8,
                    ((x * 91 + y * 7 + 13) % 256) as u8,
                    ((x * 53 + y * 29 + 7) % 256) as u8,
                ]);
            }
        }
        if distorted {
            for p in &mut img {
                for c in p.iter_mut() {
                    *c = ((*c as f64 * 0.82 + 21.0).round().clamp(0.0, 255.0)) as u8;
                }
            }
        }
        img
    }

    /// Y′CbCr companion fixture (`dvifm_ycbcr_parity_2026-09-19.txt`):
    /// pins the BT.709 conversion (plane_values rows) and the SEED-constant
    /// pump output on each plane after its route norm (features rows) —
    /// the colour space AND the kernel end-to-end.
    ///
    /// Feature bound is looser than plane bound: the fixture computes on
    /// the f32-rounded plane (emitted and analysed alike), but the Rust
    /// f32 mul_add chain still differs from numpy's f32-mul+f32-add by an
    /// ULP, and `ln(min C̃ + 1e-6)` amplifies that by ~1/C̃ near zero
    /// contrast — an ULP plane diff becomes ~1e-5 in a membership.
    #[test]
    fn numpy_parity_ycbcr_planes_and_features() {
        const TOL: f64 = 1e-6;
        const FEATURE_TOL: f64 = 5e-5;
        let text = include_str!("../tests/fixtures/dvifm_ycbcr_parity_2026-09-19.txt");
        let mut lines = text
            .lines()
            .map(str::trim)
            .filter(|l| !l.is_empty() && !l.starts_with('#'));
        let read_grid = |lines: &mut dyn Iterator<Item = &str>, h: usize, w: usize| -> Vec<f32> {
            let mut out = Vec::with_capacity(h * w);
            for _ in 0..h {
                for tok in lines.next().unwrap().split_whitespace() {
                    out.push(tok.parse().unwrap());
                }
            }
            out
        };
        while let Some(head) = lines.next() {
            let mut it = head.split_whitespace();
            assert_eq!(it.next(), Some("case"));
            let _name = it.next().unwrap();
            let h: usize = it.next().unwrap().parse().unwrap();
            let w: usize = it.next().unwrap().parse().unwrap();
            let srgb_s = ycbcr_parity_rgb(false, w, h);
            let srgb_d = ycbcr_parity_rgb(true, w, h);
            let rs = crate::RgbSlice::new(&srgb_s, w, h);
            let rd = crate::RgbSlice::new(&srgb_d, w, h);
            let planes = [
                ("y", crate::streaming::YcbcrPlane::Y, DVIFM_NORM_YCBCR_Y),
                ("cb", crate::streaming::YcbcrPlane::Cb, DVIFM_NORM_YCBCR_C),
                ("cr", crate::streaming::YcbcrPlane::Cr, DVIFM_NORM_YCBCR_C),
            ];
            // 1) converted plane values on the native range
            let mut converted: [Vec<f32>; 3] = [Vec::new(), Vec::new(), Vec::new()];
            for (pi, (pname, conv, _norm)) in planes.iter().enumerate() {
                let mut it = lines.next().unwrap().split_whitespace();
                assert_eq!(it.next(), Some("plane_values"));
                assert_eq!(it.next(), Some(*pname));
                let expect = read_grid(&mut lines, h, w);
                let mut got = vec![0.0f32; w * h];
                crate::streaming::convert_source_to_ycbcr_plane_into_slice(
                    &rs, &mut got, w, 0, *conv,
                );
                for (i, (&g, &e)) in got.iter().zip(expect.iter()).enumerate() {
                    assert!(
                        (g as f64 - e as f64).abs() <= TOL,
                        "{pname} px {i}: {g:e} vs {e:e}"
                    );
                }
                converted[pi] = got;
            }
            // 2) SEED-constant features per plane, after the route norm
            let params = DvifmParams {
                levels: [DvifmLevelParams::SEED; DVIFM_LEVELS],
                input_plane: DvifmInputPlane::XybY,
            };
            for (pi, (pname, conv, norm)) in planes.iter().enumerate() {
                let mut it = lines.next().unwrap().split_whitespace();
                assert_eq!(it.next(), Some("features"));
                assert_eq!(it.next().unwrap(), format!("ycbcr_{pname}").as_str());
                let mut d_plane = vec![0.0f32; w * h];
                crate::streaming::convert_source_to_ycbcr_plane_into_slice(
                    &rd,
                    &mut d_plane,
                    w,
                    0,
                    *conv,
                );
                let got = dvifm_features_stream(&converted[pi], &d_plane, w, h, *norm, &params);
                for l in 0..DVIFM_LEVELS {
                    let row: Vec<f64> = lines
                        .next()
                        .unwrap()
                        .split_whitespace()
                        .map(|v| v.parse().unwrap())
                        .collect();
                    assert_eq!(row.len(), DVIFM_PER_LEVEL);
                    for (j, &e) in row.iter().enumerate() {
                        let g = got[l * DVIFM_PER_LEVEL + j];
                        assert!(
                            (g - e).abs() <= FEATURE_TOL,
                            "{pname} L{l} f{j}: {g} vs {e}"
                        );
                    }
                }
            }
            assert_eq!(lines.next(), Some("end"));
        }
    }

    // Commit-5 gate: every SIMD tier is bit-identical to the scalar path.
    // The kernel's `mul_add`s all carry power-of-two factors (exact on any
    // tier — doubling/quadrupling cannot change the mantissa rounding), and
    // `GenericF64x8` is a fixed 8-lane type on every backend, so lane order
    // in every reduction is identical; each tier must reproduce scalar bits.
    #[test]
    fn simd_tier_parity() {
        use archmage::SimdToken as _;

        fn run<T: F64x8Backend + Copy>(
            t: T,
            w: usize,
            h: usize,
            strip: usize,
            mode: BandMode,
        ) -> [f64; DVIFM_FEATURES] {
            let s32 = f32_plane(53, w, h);
            let d32 = f32_plane(59, w, h);
            let mut params = DvifmParams::default();
            for lp in &mut params.levels {
                lp.band = mode;
            }
            let mut acc = DvifmAccum::new(w, h, DVIFM_NORM_SDR, &params);
            let mut r = 0;
            while r < h {
                let n = strip.min(h - r);
                dvifm_push_rows(
                    t,
                    &mut acc,
                    &s32[r * w..(r + n) * w],
                    &d32[r * w..(r + n) * w],
                );
                r += n;
            }
            dvifm_finish(t, &mut acc)
        }

        // Widths exercising every remainder path: full 8-lane + tail (125),
        // lane-adjacent (97), a width narrower than one lane (6), and the
        // block-friendly walk case (128). Strips are non-multiples of the
        // decimation stride and of 128.
        let cases: &[(usize, usize, usize)] =
            &[(125, 130, 7), (97, 101, 16), (6, 11, 3), (128, 128, 64)];
        for &mode in &[BandMode::Laplacian, BandMode::Local] {
            for &(w, h, strip) in cases {
                let want = run(scalar(), w, h, strip, mode);
                let check = |tier: &str, got: [f64; DVIFM_FEATURES]| {
                    for (i, (&a, &b)) in got.iter().zip(want.iter()).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "tier {tier} mode {mode:?} {w}x{h} strip={strip}: \
                             f{i} {a:e} != scalar {b:e}"
                        );
                    }
                };
                #[cfg(target_arch = "x86_64")]
                {
                    if let Some(t) = archmage::X64V3Token::summon() {
                        check("v3", run(t, w, h, strip, mode));
                    }
                    if let Some(t) = archmage::X64V4Token::summon() {
                        check("v4", run(t, w, h, strip, mode));
                    }
                    if let Some(t) = archmage::X64V4xToken::summon() {
                        check("v4x", run(t, w, h, strip, mode));
                    }
                }
                #[cfg(target_arch = "aarch64")]
                {
                    if let Some(t) = archmage::NeonToken::summon() {
                        check("neon", run(t, w, h, strip, mode));
                    }
                }
                #[cfg(target_arch = "wasm32")]
                {
                    if let Some(t) = archmage::Wasm128Token::summon() {
                        check("wasm128", run(t, w, h, strip, mode));
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------
    // Training side output: the block-record cache.
    // -------------------------------------------------------------------

    /// `to_f32` writes the struct's field order and nothing else — the
    /// 18-f32 v1 prefix is unchanged and the Weber means land at 18..20.
    #[cfg(feature = "training")]
    #[test]
    fn block_record_f32_field_order() {
        let rec = BlockRec {
            m: 0.5,
            peak: 0.25,
            cmax_s: [1.0, 2.0, 3.0, 4.0],
            cmin_s: [-1.0, -2.0, -3.0, -4.0],
            cmax_d: [10.0, 20.0, 30.0, 40.0],
            cmin_d: [-10.0, -20.0, -30.0, -40.0],
            mean_s: 0.6,
            mean_d: 0.7,
        };
        let v = rec.to_f32();
        assert_eq!(v[0], 0.5f32);
        assert_eq!(v[1], 0.25f32);
        assert_eq!(&v[2..6], &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(&v[6..10], &[-1.0, -2.0, -3.0, -4.0]);
        assert_eq!(&v[10..14], &[10.0, 20.0, 30.0, 40.0]);
        assert_eq!(&v[14..18], &[-10.0, -20.0, -30.0, -40.0]);
        assert_eq!(v[18], 0.6f32);
        assert_eq!(v[19], 0.7f32);
    }

    /// The served contract: no `enable_block_cache` call, no records.
    #[cfg(feature = "training")]
    #[test]
    fn block_cache_off_by_default() {
        let (w, h) = (97usize, 101usize);
        let s32 = f32_plane(31, w, h);
        let d32 = f32_plane(37, w, h);
        let mut acc = DvifmAccum::new(w, h, DVIFM_NORM_SDR, &DvifmParams::default());
        dvifm_push_rows(scalar(), &mut acc, &s32, &d32);
        let _ = dvifm_finish(scalar(), &mut acc);
        assert!(acc.block_cache().is_none());
    }

    /// The cache is the SAME block records the pooled features read: replay
    /// `pool_block` over them (same accumulation order — records are stored
    /// in emit order) reproduces the 30 outputs bit-for-bit, and each
    /// level's record count is its full-block grid.
    #[cfg(feature = "training")]
    #[test]
    fn block_cache_replays_to_pooled_features() {
        for &(w, h) in &[(125usize, 130usize), (97, 101)] {
            for &mode in &[BandMode::Laplacian, BandMode::Local] {
                let s32 = f32_plane(31, w, h);
                let d32 = f32_plane(37, w, h);
                let mut params = DvifmParams::default();
                for lp in &mut params.levels {
                    lp.band = mode;
                }
                let t = scalar();
                for strip in [h, 16] {
                    // Rebuild the accumulator per strip size — the cache is
                    // consumed by `take_block_cache`.
                    let mut acc = DvifmAccum::new(w, h, DVIFM_NORM_SDR, &params);
                    acc.enable_block_cache();
                    let mut r = 0;
                    while r < h {
                        let n = strip.min(h - r);
                        dvifm_push_rows(
                            t,
                            &mut acc,
                            &s32[r * w..(r + n) * w],
                            &d32[r * w..(r + n) * w],
                        );
                        r += n;
                    }
                    let want = dvifm_finish(t, &mut acc);
                    let (grid, _dims, levels) = acc.take_block_cache().expect("cache enabled");
                    assert_eq!(levels.len(), DVIFM_LEVELS);
                    let mut got = [0.0f64; DVIFM_FEATURES];
                    let (mut wl, mut hl) = (w, h);
                    for l in 0..DVIFM_LEVELS {
                        let (nby, nbx) = grid[l];
                        assert_eq!(nby as usize, hl / DVIFM_BLOCK);
                        assert_eq!(nbx as usize, wl / DVIFM_BLOCK);
                        wl = wl.div_ceil(2);
                        hl = hl.div_ceil(2);
                        assert_eq!(levels[l].len(), nby as usize * nbx as usize);
                        let mut sums = LevelSums::default();
                        for rec in &levels[l] {
                            pool_block(&mut sums, &params.levels[l], rec);
                        }
                        got[l * DVIFM_PER_LEVEL..(l + 1) * DVIFM_PER_LEVEL]
                            .copy_from_slice(&level_out(&sums));
                    }
                    for (i, (&a, &b)) in want.iter().zip(&got).enumerate() {
                        assert_eq!(
                            a.to_bits(),
                            b.to_bits(),
                            "mode {mode:?} {w}x{h} strip={strip}: cached f{i} {a:e} != {b:e}"
                        );
                    }
                    // The 20-f32 wire form keeps the record's information:
                    // the narrowed replay agrees to f32 rounding, not less.
                    let mut got32 = [0.0f64; DVIFM_FEATURES];
                    for l in 0..DVIFM_LEVELS {
                        let mut sums = LevelSums::default();
                        for rec in &levels[l] {
                            let v = rec.to_f32();
                            let rec32 = BlockRec {
                                m: v[0] as f64,
                                peak: v[1] as f64,
                                cmax_s: [v[2] as f64, v[3] as f64, v[4] as f64, v[5] as f64],
                                cmin_s: [v[6] as f64, v[7] as f64, v[8] as f64, v[9] as f64],
                                cmax_d: [v[10] as f64, v[11] as f64, v[12] as f64, v[13] as f64],
                                cmin_d: [v[14] as f64, v[15] as f64, v[16] as f64, v[17] as f64],
                                mean_s: v[18] as f64,
                                mean_d: v[19] as f64,
                            };
                            pool_block(&mut sums, &params.levels[l], &rec32);
                        }
                        got32[l * DVIFM_PER_LEVEL..(l + 1) * DVIFM_PER_LEVEL]
                            .copy_from_slice(&level_out(&sums));
                    }
                    for (i, (&a, &b)) in want.iter().zip(&got32).enumerate() {
                        assert!(
                            (a - b).abs() <= a.abs() * 1e-5 + 1e-9,
                            "mode {mode:?} {w}x{h} strip={strip}: f32-record f{i} {a:e} vs {b:e}"
                        );
                    }
                }
            }
        }
    }

    // -------------------------------------------------------------------
    // The steering field (S1): the painted ε map IS the pooled F1 mass.
    // -------------------------------------------------------------------

    fn rec_bits_eq(a: &BlockRec, b: &BlockRec) -> bool {
        a.m.to_bits() == b.m.to_bits()
            && a.peak.to_bits() == b.peak.to_bits()
            && a.cmax_s
                .iter()
                .zip(&b.cmax_s)
                .all(|(x, y)| x.to_bits() == y.to_bits())
            && a.cmin_s
                .iter()
                .zip(&b.cmin_s)
                .all(|(x, y)| x.to_bits() == y.to_bits())
            && a.cmax_d
                .iter()
                .zip(&b.cmax_d)
                .all(|(x, y)| x.to_bits() == y.to_bits())
            && a.cmin_d
                .iter()
                .zip(&b.cmin_d)
                .all(|(x, y)| x.to_bits() == y.to_bits())
    }

    /// The field's `f1_sum` is bitwise the pooled numerator: `block_field`
    /// runs the same `vb*e` adds in the same emit order `pool_block` did.
    /// `dims` reports each level's own plane dims (halved, ceil).
    #[test]
    fn block_field_f1_sum_is_the_pooled_numerator() {
        for &(w, h) in &[(125usize, 130usize), (97, 101)] {
            for &mode in &[BandMode::Laplacian, BandMode::Local] {
                let s32 = f32_plane(31, w, h);
                let d32 = f32_plane(37, w, h);
                let mut params = DvifmParams::default();
                for lp in &mut params.levels {
                    lp.band = mode;
                }
                let t = scalar();
                let mut acc = DvifmAccum::new(w, h, DVIFM_NORM_SDR, &params);
                acc.enable_block_cache();
                dvifm_push_rows(t, &mut acc, &s32, &d32);
                let feats = dvifm_finish(t, &mut acc);
                let (grid, dims, levels) = acc.take_block_cache().expect("cache enabled");
                let (mut wl, mut hl) = (w, h);
                for l in 0..DVIFM_LEVELS {
                    assert_eq!(
                        dims[l],
                        (wl as u32, hl as u32),
                        "{mode:?} {w}x{h} level {l} dims"
                    );
                    let (nby, nbx) = (grid[l].0 as usize, grid[l].1 as usize);
                    let f = block_field(&levels[l], &params.levels[l], nby, nbx);
                    let mut sums = LevelSums::default();
                    for rec in &levels[l] {
                        pool_block(&mut sums, &params.levels[l], rec);
                    }
                    assert_eq!(
                        f.f1_sum.to_bits(),
                        sums.f1.to_bits(),
                        "{mode:?} {w}x{h} level {l}: field mass != pooled f1"
                    );
                    let n = (nby * nbx) as f64;
                    if n > 0.0 {
                        let f1 = feats[l * DVIFM_PER_LEVEL];
                        assert!(
                            (f.f1_sum / n - f1).abs() <= f1.abs() * 1e-12 + 1e-18,
                            "{mode:?} {w}x{h} level {l}: f1_sum/n {} vs F1 {f1}",
                            f.f1_sum / n
                        );
                    }
                    wl = wl.div_ceil(2);
                    hl = hl.div_ceil(2);
                }
            }
        }
    }

    /// The painted ε map integrates back to the block terms: whole-canvas
    /// mass ≈ `f1_sum` (the SAT's different summation order stays well
    /// inside the 1e-6 f64 gate); block-aligned rects return their block's
    /// ε_b; a rect cutting a block gets the area-weighted share (the
    /// stated cut rule).
    #[cfg(feature = "custom-profiles")] // the map owner lives behind custom-profiles
    #[test]
    fn painted_eps_map_rect_queries_match_block_terms() {
        let (w, h) = (125usize, 130usize);
        let s32 = f32_plane(31, w, h);
        let d32 = f32_plane(37, w, h);
        let params = DvifmParams::default();
        let t = scalar();
        let mut acc = DvifmAccum::new(w, h, DVIFM_NORM_SDR, &params);
        acc.enable_block_cache();
        dvifm_push_rows(t, &mut acc, &s32, &d32);
        let _ = dvifm_finish(t, &mut acc);
        let (grid, dims, levels) = acc.take_block_cache().unwrap();
        for l in 0..DVIFM_LEVELS {
            let (nby, nbx) = (grid[l].0 as usize, grid[l].1 as usize);
            let (lw, lh) = (dims[l].0 as usize, dims[l].1 as usize);
            let f = block_field(&levels[l], &params.levels[l], nby, nbx);
            let painted = paint_block_field(&f, lw, lh);
            // vis/err planes are block-constant (f32-painted); the dropped
            // border is 0.
            assert_eq!(painted.vis[0], f.vis[0] as f32);
            assert_eq!(painted.err[0], f.err[0] as f32);
            if lw % DVIFM_BLOCK != 0 {
                assert_eq!(painted.eps_density[(lh / 2) * lw + lw - 1], 0.0);
            }
            let map =
                crate::attribution::AttributionResult::from_f64_canvas(painted.eps_density, lw, lh);
            let total = map.query_rect(0, 0, lw, lh);
            assert!(
                (total - f.f1_sum).abs() <= f.f1_sum.abs() * 1e-12 + 1e-15,
                "level {l}: map total {total} vs f1_sum {}",
                f.f1_sum
            );
            for &(by, bx) in &[
                (0usize, 0usize),
                (nby - 1, nbx - 1),
                (nby / 2, nbx / 2),
                (0, nbx - 1),
            ] {
                let b = by * nbx + bx;
                let q = map.query_rect(
                    bx * DVIFM_BLOCK,
                    by * DVIFM_BLOCK,
                    bx * DVIFM_BLOCK + DVIFM_BLOCK,
                    by * DVIFM_BLOCK + DVIFM_BLOCK,
                );
                assert!(
                    (q - f.eps[b]).abs() <= f.eps[b].abs() * 1e-9 + 1e-15,
                    "level {l} block ({bx},{by}): rect {q} vs eps {}",
                    f.eps[b]
                );
                // Cut rule: a rect covering the left half of the block by
                // area receives exactly ε_b/2 (uniform painted density).
                let qh = map.query_rect_frac(
                    bx as f64 * DVIFM_BLOCK as f64,
                    by as f64 * DVIFM_BLOCK as f64,
                    bx as f64 * DVIFM_BLOCK as f64 + DVIFM_BLOCK as f64 / 2.0,
                    by as f64 * DVIFM_BLOCK as f64 + DVIFM_BLOCK as f64,
                );
                assert!(
                    (qh - f.eps[b] * 0.5).abs() <= f.eps[b].abs() * 1e-6 + 1e-12,
                    "level {l} block ({bx},{by}): half-rect {qh} vs eps/2 {}",
                    f.eps[b] * 0.5
                );
            }
        }
    }

    /// Identity pair: `m_b` and `ε_b` are exactly 0 everywhere (the map is
    /// all-zero), `f1_sum` is +0. `v_b` is NOT pinned to 1 — it is the
    /// block's content visibility `v(C̃)` (both sides identical, so the
    /// merge is `v` at the shared contrast).
    #[test]
    fn field_identity_is_zero() {
        let (w, h) = (97usize, 89usize);
        let p = f32_plane(7, w, h);
        let params = DvifmParams::default();
        let t = scalar();
        let mut acc = DvifmAccum::new(w, h, DVIFM_NORM_SDR, &params);
        acc.enable_block_cache();
        dvifm_push_rows(t, &mut acc, &p, &p);
        let _ = dvifm_finish(t, &mut acc);
        let (grid, dims, levels) = acc.take_block_cache().unwrap();
        for l in 0..DVIFM_LEVELS {
            let lp = &params.levels[l];
            let (nby, nbx) = (grid[l].0 as usize, grid[l].1 as usize);
            let f = block_field(&levels[l], lp, nby, nbx);
            for (i, &e) in f.eps.iter().enumerate() {
                let want_v = visibility(contrast_g_rec(&levels[l][i], 0, lp.g, lp.edge), lp);
                assert_eq!(e, 0.0, "level {l} block {i} eps on identity");
                assert_eq!(f.err[i], 0.0, "level {l} block {i} err on identity");
                assert_eq!(f.vis[i], want_v, "level {l} block {i} vis on identity");
            }
            assert_eq!(f.f1_sum, 0.0);
            let painted = paint_block_field(&f, dims[l].0 as usize, dims[l].1 as usize);
            assert!(painted.eps_density.iter().all(|&v| v == 0.0));
            assert!(painted.err.iter().all(|&v| v == 0.0));
        }
    }

    /// The cached records — and therefore the field derived from them —
    /// are bitwise identical across strip sizes: the pump's emit order and
    /// record arithmetic do not depend on chunking.
    #[test]
    fn field_strip_size_bit_identical() {
        let (w, h) = (125usize, 130usize);
        let s32 = f32_plane(31, w, h);
        let d32 = f32_plane(37, w, h);
        let params = DvifmParams::default();
        let t = scalar();
        let run = |strip: usize| -> DvifmBlockTake {
            let mut acc = DvifmAccum::new(w, h, DVIFM_NORM_SDR, &params);
            acc.enable_block_cache();
            let mut r = 0;
            while r < h {
                let n = strip.min(h - r);
                dvifm_push_rows(
                    t,
                    &mut acc,
                    &s32[r * w..(r + n) * w],
                    &d32[r * w..(r + n) * w],
                );
                r += n;
            }
            let _ = dvifm_finish(t, &mut acc);
            acc.take_block_cache().unwrap()
        };
        let want = run(h);
        for strip in [1usize, 2, 3, 5, 7, 11, 16, 33, 64, 97] {
            let got = run(strip);
            assert_eq!(got.0, want.0, "grid differs at strip {strip}");
            assert_eq!(got.1, want.1, "dims differ at strip {strip}");
            for l in 0..DVIFM_LEVELS {
                assert_eq!(got.2[l].len(), want.2[l].len());
                for (i, (a, b)) in got.2[l].iter().zip(&want.2[l]).enumerate() {
                    assert!(
                        rec_bits_eq(a, b),
                        "level {l} block {i} differs at strip {strip}"
                    );
                }
            }
        }
    }
}
