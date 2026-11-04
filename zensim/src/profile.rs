//! Named metric profiles.
//!
//! Each [`ZensimProfile`] variant selects a named model. [`ZensimProfile::latest`]
//! returns the general-purpose default ([`ZensimProfile::A`]) and
//! [`ZensimProfile::codec_target`] selects the same adopted model for codec
//! quality targets. Pin the crate version when reproducing exact
//! scores: a profile name alone does not freeze its coefficients or calibration.

/// Named metric profile. Scores for a given profile should be approximately
/// stable across crate versions. Profiles may be removed in future versions.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ZensimProfile {
    /// Preview v0.1 — linear-weights profile (no MLP). 344k synthetic
    /// pairs, 5-fold CV SROCC = 0.9936. Uses the [`WEIGHTS_PREVIEW_V0_1`]
    /// coefficients with the classic `100 − 18·d^0.7` score mapping.
    ///
    /// Kept for compatibility with zensim 0.1.x / 0.2.x, which shipped the same
    /// weights and mapping. Scores match those releases closely but not
    /// bit-exactly: kernel-fusion and summation-order changes since 0.2.7 shift
    /// scores of images 64 px and larger by about 1e-3 to 1.5e-2, and inputs below
    /// 64 px now reflect-pad (different, better-behaved scores at those sizes).
    #[cfg(feature = "legacy-models")]
    PreviewV0_1,
    /// Preview v0.2. Concordance-filtered 218k pairs, Nelder-Mead SROCC=0.9960.
    /// Linear-weights profile, no MLP forward pass. `latest()` returned this
    /// through zensim 0.2.x; the 0.2.7 compatibility note on
    /// [`Self::PreviewV0_1`] applies here too.
    #[cfg(feature = "legacy-models")]
    PreviewV0_2,
    /// General-purpose profile `A` (`zensim-a`), returned by [`Self::latest`]
    /// since 0.3.0. Adopted on 2026-11-04 (research name `mixF3`, `F` in the
    /// research crate).
    ///
    /// A monotone MLP with full-resolution chroma signals over 492
    /// revision-7 damage features and 492 fragility products, every damage
    /// input entering with non-negative weight. Trained on synthetic codec
    /// ladders, coverage pairs and pairwise legs with the human-rating legs
    /// narrowed to TID2013 and MDID, then continued on the JPEG AIC-2026 public
    /// training set with the base legs rehearsed. See `docs/PROFILES.md` for
    /// held-out evaluation and the recipe reference.
    A,
    /// **INTERNAL — legacy 228-weight linear scoring, no MLP head.**
    /// Tags results from the training entry points `compute_zensim_with_config`
    /// and `compute_zensim_with_ref_and_config`, which score via a plain
    /// `dot(features, WEIGHTS_PREVIEW_V0_2)` distance (the caller-supplied
    /// [`crate::ZensimConfig`] drives blur/scale knobs directly) and
    /// never run an MLP forward pass. The tag exists so `ZensimResult::profile()`
    /// never claims an MLP-scored result for a linear-only computation.
    #[cfg(all(feature = "legacy-models", any(feature = "training", test)))]
    #[doc(hidden)]
    LegacyLinearV0_2,
}

impl ZensimProfile {
    /// Latest recommended general-purpose profile: [`Self::A`].
    pub fn latest() -> Self {
        Self::A
    }

    /// The adopted model for codec-quality targets: [`Self::A`]. The public
    /// crate ships no separate codec profile; this currently matches [`Self::latest`].
    pub const fn codec_target() -> Self {
        Self::A
    }

    /// Canonical name string, e.g. `"zensim-a"`.
    pub fn name(&self) -> &'static str {
        match self {
            #[cfg(feature = "legacy-models")]
            Self::PreviewV0_1 => "zensim-preview-v0.1",
            #[cfg(feature = "legacy-models")]
            Self::PreviewV0_2 => "zensim-preview-v0.2",
            Self::A => "zensim-a",
            #[cfg(all(feature = "legacy-models", any(feature = "training", test)))]
            Self::LegacyLinearV0_2 => "zensim-legacy-linear-v0.2",
        }
    }

    /// Internal parameters for this profile.
    pub(crate) fn params(&self) -> &'static ProfileParams {
        match self {
            #[cfg(feature = "legacy-models")]
            Self::PreviewV0_1 => &PROFILE_PREVIEW_V0_1,
            #[cfg(feature = "legacy-models")]
            Self::PreviewV0_2 => &PROFILE_PREVIEW_V0_2,
            Self::A => &PROFILE_A,
            #[cfg(all(feature = "legacy-models", any(feature = "training", test)))]
            Self::LegacyLinearV0_2 => &PROFILE_LEGACY_LINEAR_V0_2,
        }
    }
}

impl ProfileParams {
    /// Every bake whose forward pass consumes the v1-layout feature vector:
    /// the primary, the optional D2 secondary, and the optional ensemble
    /// classifier. Used by the fold engine's per-profile weight-skipping to
    /// take the UNION of what the profile actually reads — missing one would
    /// silently skip a family a live consumer still looks at.
    pub(crate) fn scoring_bake_bytes(&self) -> impl Iterator<Item = &'static [u8]> + '_ {
        self.mlp_bytes.into_iter().map(|f| f())
    }
}

impl core::str::FromStr for ZensimProfile {
    type Err = &'static str;

    fn from_str(name: &str) -> Result<Self, Self::Err> {
        match name {
            #[cfg(feature = "legacy-models")]
            "zensim-preview-v0.1" => Ok(Self::PreviewV0_1),
            #[cfg(feature = "legacy-models")]
            "zensim-preview-v0.2" => Ok(Self::PreviewV0_2),
            "zensim-a" => Ok(Self::A),
            #[cfg(all(feature = "legacy-models", any(feature = "training", test)))]
            "zensim-legacy-linear-v0.2" => Ok(Self::LegacyLinearV0_2),
            _ => Err("unknown zensim profile"),
        }
    }
}

impl core::fmt::Display for ZensimProfile {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

/// Internal struct holding everything needed to compute scores for a profile.
///
/// Each parameter's effect on computation path and performance is documented
/// on the corresponding field of `ZensimConfig` in `metric.rs`.
#[derive(Debug)]
#[non_exhaustive]
pub struct ProfileParams {
    /// Linear preview weights, or A's historical projection used by trained
    /// steering maps. A's score comes entirely from its MLP bake.
    pub weights: &'static [f64],
    /// Box blur radius at scale 0 (kernel width = `2 * radius + 1`).
    pub blur_radius: usize,
    /// Number of iterated box blur passes (1 = rectangular, 3 ≈ Gaussian).
    pub blur_passes: u8,
    /// Number of pyramid scales (typically 4).
    pub num_scales: usize,
    /// Linear score mapping coefficient A in `100 - A × d^B`.
    /// A uses its baked calibration instead.
    pub score_mapping_a: f64,
    /// Linear score mapping exponent B in `100 - A × d^B`.
    /// A uses its baked calibration instead.
    pub score_mapping_b: f64,
    /// When `true`, the MLP bake's raw output is returned **directly**
    /// as the final score (no `100 − A·d^B` transform). This is correct
    /// for V0_5+ bakes, which are affine-calibrated to the
    /// MCOS-aligned 0–100 scale during baking. Set `false` for V0_1 /
    /// V0_2 linear profiles whose raw output is a distance.
    pub(crate) skip_score_mapping: bool,
    /// MLP scorer for non-linear profiles. When `Some`, the scoring
    /// path replaces the linear `dot(features, weights)` with a forward
    /// pass through the MLP loaded from these bytes (`ZNPR` v3 format).
    /// `None` for V0_1 / V0_2 — they keep the classic linear path.
    ///
    /// A loader keeps the embedded bake out of the linear preview path.
    pub(crate) mlp_bytes: Option<fn() -> &'static [u8]>,

    /// EXP-CROSS-CODEC-V10 (2026-05-20): when `true`, the final score
    /// is returned **without clamping or soft-clamping** — the
    /// PCHIP spline's linear extrapolation past the knot endpoints
    /// flows through to the user. V10 profiles set this so the
    /// score-space dial can dip below 0 for "pathological /
    /// unreasonable" codec output (worst codec at q=0, butter > 12)
    /// rather than collapsing into a tie block at 0.
    ///
    /// Default `false` preserves legacy [0, 100] semantics for V9 and
    /// earlier profiles whose splines/anchors only span [0, 100].
    pub(crate) extrapolate_score: bool,
}

#[cfg(all(feature = "legacy-models", any(feature = "training", test)))]
impl ProfileParams {
    /// Create custom params for weight training/exploration.
    ///
    /// Custom parameters use linear scoring (`mlp_bytes = None`).
    pub fn custom(
        weights: &'static [f64],
        blur_radius: usize,
        blur_passes: u8,
        num_scales: usize,
        score_mapping_a: f64,
        score_mapping_b: f64,
    ) -> Self {
        Self {
            weights,
            blur_radius,
            blur_passes,
            num_scales,
            score_mapping_a,
            score_mapping_b,
            // Custom training params get the legacy distance-to-score mapping
            // (V0_2 semantics). V0_8+ pre-calibrated bakes set the flag in
            // their static `ProfileParams` definitions.
            skip_score_mapping: false,
            mlp_bytes: None,

            extrapolate_score: false,
        }
    }
}

// --- Profile definitions ---

// `PreviewV0_1` is the linear V0.1 profile published through zensim
// 0.1.x / 0.2.x. Identical to `PROFILE_PREVIEW_V0_2` except for the
// weight vector, so scores match those releases bit-for-bit.
#[cfg(feature = "legacy-models")]
static PROFILE_PREVIEW_V0_1: ProfileParams = ProfileParams {
    weights: &WEIGHTS_PREVIEW_V0_1,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    skip_score_mapping: false,
    mlp_bytes: None,

    extrapolate_score: false,
};

/// Parameters for the opt-in 228-weight v0.2 linear scorer.
#[cfg(feature = "legacy-models")]
static PROFILE_PREVIEW_V0_2: ProfileParams = ProfileParams {
    weights: &WEIGHTS_PREVIEW_V0_2,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    skip_score_mapping: false,
    mlp_bytes: None,

    extrapolate_score: false,
};

/// Legacy linear weights and score mapping used by the custom-configuration
/// training entry points in `metric.rs`.
#[cfg(all(feature = "legacy-models", any(feature = "training", test)))]
static PROFILE_LEGACY_LINEAR_V0_2: ProfileParams = ProfileParams {
    weights: &WEIGHTS_PREVIEW_V0_2,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    skip_score_mapping: false,
    mlp_bytes: None,

    extrapolate_score: false,
};

/// `ZensimProfile::A` bake: the adopted general model (H128 monotone MLP,
/// 984 inputs, derived-input declaration `fx2`), stripped of training-provenance
/// metadata. The mapping to the research identity is in
/// `docs/PROFILES.md`.
pub(crate) fn mlp_bake_a_general() -> &'static [u8] {
    include_bytes!("../weights/rev7_mix_h128_2026-11-03.bin")
}

/// Shared runtime shape of the revision-7 profiles. Feature IDs, formula revision,
/// derived inputs and output calibration all come from the bake's metadata.
const fn rev7_params(bake: fn() -> &'static [u8]) -> ProfileParams {
    ProfileParams {
        weights: &A_MAP_WEIGHTS,
        blur_radius: 5,
        blur_passes: 1,
        num_scales: 4,
        score_mapping_a: 18.0,
        score_mapping_b: 0.7,
        skip_score_mapping: true,
        mlp_bytes: Some(bake),

        extrapolate_score: true,
    }
}

static PROFILE_A: ProfileParams = rev7_params(mlp_bake_a_general);

// A's steering map uses this historical 156-coefficient projection. It is
// independent of the opt-in 228-input linear scoring path.
pub(crate) static A_MAP_WEIGHTS: [f64; 156] = [
    0.0000000000,
    0.0374713114,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0002534408,
    0.0022940503,
    64.5319059554,
    0.0025568180,
    0.0024655971,
    0.0001182877,
    5.0947987726,
    0.6253588192,
    0.0036665038,
    0.0012089836,
    15.7983738285,
    0.0005742272,
    0.0000000000,
    0.5175522882,
    0.0017844759,
    0.0000000000,
    1.3480049939,
    1.4246254310,
    0.0302900947,
    0.0000000000,
    0.0030975336,
    0.0021003750,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0025534968,
    69.2987507117,
    0.0000000000,
    0.0000000000,
    0.0020508776,
    0.0006647708,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0001735594,
    0.0001311405,
    0.0022817212,
    0.0000000000,
    0.0024169014,
    16.1367160769,
    0.0028778096,
    0.0000000000,
    0.0001602903,
    94.6093105684,
    14.2170395553,
    35.3539050513,
    1.3630743451,
    68.1526923123,
    0.0008809755,
    0.0000000000,
    0.2013576637,
    0.0000000000,
    0.0000000000,
    0.1249634554,
    0.0035217432,
    0.0498144992,
    0.2131147170,
    1.7331839707,
    0.0000000000,
    61.9252606811,
    0.0217888369,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0011283888,
    69.6715917151,
    0.0000000000,
    0.0000000000,
    0.0030681356,
    0.0508113272,
    0.0000000000,
    0.0022822823,
    0.0000000000,
    1.0470138384,
    0.0030747304,
    0.0031241737,
    0.0026097417,
    0.0035396334,
    64.5861323041,
    0.0025540825,
    0.0000000000,
    0.0000861748,
    21.9866716620,
    1.6042001813,
    1.6650862341,
    177.6428823130,
    8.8094144980,
    74.3890350730,
    0.0000000000,
    2.4416214848,
    0.0000000000,
    0.0000000000,
    7.6251827816,
    2.3284049327,
    0.0254137606,
    0.1682704477,
    0.0000000000,
    0.0000000000,
    2.1893370574,
    1.3307404051,
    0.3266457358,
    1.9962771956,
    2.5501952064,
    0.0000000000,
    61.8997311172,
    0.0000000000,
    0.0013826336,
    0.0000029596,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    31.5515340494,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0006549325,
    0.0000000000,
    64.6916010441,
    0.0023106921,
    0.0407007643,
    0.0000418198,
    117.9545428313,
    8.0998531896,
    0.0015372472,
    771.7280930717,
    6.7617998299,
    277.1059304530,
    396.4838353042,
    0.0000000000,
    0.0000000000,
    13.9788488769,
    4.8576545185,
    8.6129380883,
    0.0182069151,
    27.1648870672,
    3.0652861197,
    0.8003146330,
    315.2451919583,
    43.9295462389,
    0.0023695407,
    0.0000000000,
    0.0022416240,
    0.0000000000,
    72.4855222582,
    0.0024129895,
    0.1189369864,
    0.0014196010,
];

// --- Weight arrays ---

/// Preview v0.1 linear weights (344k synthetic pairs, 5-fold CV SROCC=0.9936).
/// 228 entries = 4 scales × 3 channels × 19 features. The canonical V0.1
/// coefficient vector backing [`ZensimProfile::PreviewV0_1`].
#[cfg(feature = "legacy-models")]
pub static WEIGHTS_PREVIEW_V0_1: [f64; 228] = [
    // --- Basic features (13/ch × 3ch × 4 scales = 156) ---
    0.0000000000,
    0.1391674808,
    0.0000000000,
    0.0055172171,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0010650645,
    0.0071194723,
    69.6110793540,
    0.0106660235,
    0.0076379521,
    0.0051069220, // Scale 0 Channel X
    17.8445125125,
    1.9157888513,
    0.0109886875,
    0.0048996910,
    0.0000000000,
    0.0018418193,
    0.0000000000,
    1.5940983560,
    0.0072914879,
    0.0000000000,
    0.2695940535,
    0.5232582347,
    0.1101639205, // Scale 0 Channel Y
    0.0000000000,
    0.0097680540,
    0.0075408094,
    4.2314204599,
    0.0082993863,
    0.0060063585,
    0.0000000000,
    0.0000000000,
    0.0076442067,
    0.4127212154,
    0.0000000000,
    0.0000000000,
    0.0061137647, // Scale 0 Channel B
    0.0027028659,
    0.1421516497,
    0.0000000000,
    0.0000000000,
    0.0006394302,
    0.0004174259,
    0.0084670378,
    0.0000000000,
    0.0102579245,
    0.0000000000,
    0.0097535151,
    0.0000000000,
    0.0000000091, // Scale 1 Channel X
    22.0713261440,
    52.8548074123,
    87.4350424152,
    5.5343470971,
    8.5458130239,
    0.0026243365,
    0.0000000000,
    0.6444438326,
    0.0000000000,
    0.0000000000,
    0.4690274655,
    0.0111775837,
    0.0000000000, // Scale 1 Channel Y
    0.7853068895,
    0.5804301701,
    0.0000000000,
    241.7223774962,
    0.0852474584,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0046043128,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0092126667, // Scale 1 Channel B
    0.1907664071,
    1.1388072940,
    0.0069950673,
    0.0000000000,
    3.2949756637,
    0.0097480604,
    0.0114461871,
    0.0101092121,
    0.0120198795,
    0.0000000000,
    0.0102984460,
    0.0000000000,
    0.0003411392, // Scale 2 Channel X
    77.8638757528,
    4.9774136371,
    5.7998312546,
    0.0000000000,
    32.6107435348,
    0.0000000000,
    0.0000000000,
    7.3147158634,
    0.0000000000,
    112.3320506295,
    6.5803001760,
    0.9144713387,
    0.0800661074, // Scale 2 Channel Y
    0.6380873029,
    3.4344996615,
    0.0000000000,
    7.9969790535,
    4.0547889928,
    1.2673476404,
    7.9809497222,
    8.8252344733,
    0.0000000000,
    190.1707930678,
    0.0000000000,
    0.0042434316,
    0.0000117426, // Scale 2 Channel B
    42.4928921475,
    1.8499402382,
    18.0908263404,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0022710707,
    0.0000000000,
    0.0000000000,
    0.0068807271,
    0.1494089476,
    0.0001752242, // Scale 3 Channel X
    396.2394144642,
    33.6112684912,
    0.0053195470,
    331.9368790619,
    437.6418006190,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    15.5115983050,
    0.0052803584,
    0.0703659816, // Scale 3 Channel Y
    112.4036508580,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0073096632,
    0.0000000000,
    0.0091600012,
    0.0000000000,
    0.0000000000,
    0.0072861510,
    0.0493312705,
    0.0049937361, // Scale 3 Channel B
    // --- Peak features (6/ch × 3ch × 4 scales = 72) ---
    1.6405231709,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000, // Scale 0 Channel X
    1.8173590152,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    28.5681479205, // Scale 0 Channel Y
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000, // Scale 0 Channel B
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000, // Scale 1 Channel X
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    1.7833707251,
    0.0000000000, // Scale 1 Channel Y
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    17.5252532711,
    0.0000000000, // Scale 1 Channel B
    0.0000000000,
    31.1123311855,
    0.0000000000,
    0.0000000000,
    3.4969161675,
    0.0000000000, // Scale 2 Channel X
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000, // Scale 2 Channel Y
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    3.4593661665,
    0.0000000000, // Scale 2 Channel B
    56.7768222287,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    5.3758924006,
    0.0000000000, // Scale 3 Channel X
    0.0000000000,
    1.6125342576,
    47.2133536610,
    0.0000000000,
    0.0000000000,
    0.0000000000, // Scale 3 Channel Y
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000, // Scale 3 Channel B
];

/// Preview v0.2 weights (concordance-filtered 218k synthetic pairs, Nelder-Mead).
/// SROCC = 0.9942 on full 344k synthetic dataset.
/// Raw distance SROCC: TID2013=0.8427, KADIK10k=0.8192, CID22=0.8676.
/// Layout: 4 scales × 3 channels × (13 basic + 6 peak) features = 228.
#[cfg(feature = "legacy-models")]
pub static WEIGHTS_PREVIEW_V0_2: [f64; 228] = {
    let mut out = [0.0; 228];
    let mut i = 0;
    while i < 156 {
        out[i] = A_MAP_WEIGHTS[i];
        i += 1;
    }
    let peaks = [
        0.4487279165,
        0.0000000000,
        0.0000000000,
        0.0000000000,
        0.0000000000,
        0.0000000000,
        0.4482423487,
        1.7175745862,
        0.0000000000,
        0.0000000000,
        0.0000000000,
        8.3046937974,
        0.0000000000,
        0.0000000000,
        0.0000000000,
        0.0000000000,
        0.0000000000,
        0.0000000000,
        0.0000000000,
        0.0000000000,
        0.0000000000,
        0.0000000000,
        0.0000000000,
        0.0000000000,
        0.0000000000,
        0.0000000000,
        0.0000000000,
        0.0000000000,
        0.5506673335,
        0.0000000000,
        0.0000000000,
        0.0000000000,
        0.0000000000,
        0.0000000000,
        5.5320293663,
        0.0000000000,
        0.0000000000,
        8.0037734436,
        0.0000000000,
        0.0000000000,
        1.0761444814,
        0.0000000000,
        1.5371198008,
        0.0000000000,
        0.0000000000,
        2.8277784618,
        0.0000000000,
        0.0000000000,
        1.6209056242,
        0.0000000000,
        0.0000000000,
        0.0000000000,
        0.9606478120,
        0.0000000000,
        13.6786977377,
        0.0000000000,
        0.0000000000,
        0.0000000000,
        1.7799580769,
        0.0000000000,
        1.6034688535,
        0.4040027990,
        13.5194607333,
        2.6215292388,
        0.0000000000,
        0.0000000000,
        1.6611285265,
        6.9307627972,
        0.0000000000,
        0.7474376328,
        12.8104312758,
        0.0000000000,
    ];
    let mut j = 0;
    while j < 72 {
        out[156 + j] = peaks[j];
        j += 1;
    }
    out
};

// --- Canonical aliases ---
//
