//! Named metric profiles.
//!
//! Each [`ZensimProfile`] variant bundles weights and parameters that affect
//! score output. A given profile should produce approximately the same scores
//! across versions, but profiles may be removed in future major versions as
//! the algorithm evolves.
//!
//! # Profile-version policy
//!
//! Versions that have publicly shipped (`PreviewV0_1`, `PreviewV0_2`) stay
//! enabled in the default feature set — they're stable, default-on, part of
//! the published API surface.
//!
//! **Any new profile version is opt-in.** Newer variants (`PreviewV0_4`,
//! and any future `PreviewV0_5+` we add) MUST be gated behind the
//! `__experimental_versions` cargo feature. They stay opt-in until
//! validated against external anchors (CID22 paper Table 4, KonJND-1k
//! PJND), proven not to regress on the human-rated holdout sets
//! (KADID10k_val, TID2013_val), and explicitly promoted in a release
//! notes entry. Promotion = removing the `#[cfg(feature = "...")]` gate;
//! never silently flip a `latest()` to a still-gated variant.
//!
//! This matters because the experimental feature also activates
//! `zenpredict` (AGPL-3.0-only OR LicenseRef-Imazen-Commercial) and the
//! bundled trained-weight `.bin` files. Default builds remain
//! MIT/Apache-2.0 with no AGPL transitive obligations.

/// Named metric profile. Scores for a given profile should be approximately
/// stable across crate versions. Profiles may be removed in future versions.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ZensimProfile {
    /// Preview v0.1. Trained on 344k synthetic pairs, 5-fold CV SROCC=0.9936.
    PreviewV0_1,
    /// Preview v0.2. Concordance-filtered 218k pairs, Nelder-Mead SROCC=0.9960.
    PreviewV0_2,
    /// Preview v0.4. MLP-scored profile (228 → 64 LeakyReLU → 1) trained
    /// 2026-04-30 with synthetic + KADID_train + TID_train mixed
    /// supervision. **Experimental** — gated behind the
    /// `__experimental_versions` cargo feature; not part of the
    /// crates.io-published surface.
    #[cfg(feature = "__experimental_versions")]
    PreviewV0_4,
}

impl ZensimProfile {
    /// Latest recommended general-purpose profile.
    pub fn latest() -> Self {
        Self::PreviewV0_2
    }

    /// Canonical name string, e.g. `"zensim-preview-v0.1"`.
    pub fn name(&self) -> &'static str {
        match self {
            Self::PreviewV0_1 => "zensim-preview-v0.1",
            Self::PreviewV0_2 => "zensim-preview-v0.2",
            #[cfg(feature = "__experimental_versions")]
            Self::PreviewV0_4 => "zensim-preview-v0.4",
        }
    }

    /// Internal parameters for this profile.
    pub(crate) fn params(&self) -> &'static ProfileParams {
        match self {
            Self::PreviewV0_1 => &PROFILE_PREVIEW_V0_1,
            Self::PreviewV0_2 => &PROFILE_PREVIEW_V0_2,
            #[cfg(feature = "__experimental_versions")]
            Self::PreviewV0_4 => &PROFILE_PREVIEW_V0_4,
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
#[cfg_attr(not(feature = "training"), allow(dead_code))]
#[non_exhaustive]
pub struct ProfileParams {
    /// Scoring weights (one per feature, length = `FEATURES_PER_SCALE * num_scales`).
    /// Empty `&[]` for MLP-scored profiles — see [`mlp_bytes`](Self::mlp_bytes).
    pub weights: &'static [f64],
    /// Box blur radius at scale 0 (kernel width = `2 * radius + 1`).
    pub blur_radius: usize,
    /// Number of iterated box blur passes (1 = rectangular, 3 ≈ Gaussian).
    pub blur_passes: u8,
    /// Number of pyramid scales (typically 4).
    pub num_scales: usize,
    /// Score mapping coefficient A in `100 - A × d^B`.
    pub score_mapping_a: f64,
    /// Score mapping exponent B in `100 - A × d^B`.
    pub score_mapping_b: f64,
    /// MLP scorer for non-linear profiles. When `Some`, the scoring
    /// path replaces the linear `dot(features, weights)` with a forward
    /// pass through the MLP loaded from these bytes (`ZNPK` v1 format).
    /// `None` for V0_1 / V0_2 — they keep the classic linear path.
    ///
    /// Stored as a function pointer rather than `&'static [u8]` so the
    /// bytes can be lazily baked at first use (V0_4 placeholder is built
    /// at runtime from `WEIGHTS_PREVIEW_V0_2`; trained bakes will move
    /// to a `static` byte array once the training pipeline lands).
    pub(crate) mlp_bytes: Option<fn() -> &'static [u8]>,
}

#[cfg(feature = "training")]
impl ProfileParams {
    /// Create custom params for weight training/exploration.
    ///
    /// MLP scoring is disabled (`mlp_bytes = None`) for custom params;
    /// research-side MLP exploration goes through `zensim-validate`'s
    /// `--algorithm mlp` arm instead.
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
            mlp_bytes: None,
        }
    }
}

// --- Profile definitions ---

static PROFILE_PREVIEW_V0_1: ProfileParams = ProfileParams {
    weights: &WEIGHTS_PREVIEW_V0_1,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    mlp_bytes: None,
};

static PROFILE_PREVIEW_V0_2: ProfileParams = ProfileParams {
    weights: &WEIGHTS_PREVIEW_V0_2,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    mlp_bytes: None,
};

/// V0_4 trained MLP weights — 228 → 64 LeakyReLU → 1 final linear.
///
/// **Bake provenance (2026-05-09 swap)**: this slot now ships the
/// **V0_5 SSIM2-proxy MLP** trained 2026-05-01 (file
/// `runs/v04_mlp_ssim2_holdout_20260501T045510.bin`). Source corpus:
/// safe-synthetic 218k source-disjoint 80/20, target = ssim2.
/// Recovery register's "current CID22 leader" — the swap was
/// recommended at `docs/RECOVERY_REGISTER_2026-05-08.md` line 24.
///
/// On the full benchmark datasets (`zensim-bench`'s
/// `dataset_metric_baseline`):
/// **CID22 0.8934, KADID 0.8505, TID 0.8492** — improves the prior
/// 2026-04-30 mixed-supervision bake (CID22 0.8893 / KADID 0.8432 /
/// TID 0.8401) by +0.004 / +0.007 / +0.009 respectively. Same byte
/// format (60,932 bytes ZNPR v2). Outputs raw distance directly
/// (0..90 range, mean 2.8 over the training distribution) —
/// compatible with the classic `100 - 18·d^0.7` score mapping.
///
/// Predecessor (the 2026-04-30 mixed-supervision bake) is preserved
/// at `/mnt/v/output/zensim/synthetic-v2/runs/v04_mlp_v5znpr2_20260430T044620.bin`
/// (byte-identical to the prior shipped state).
///
/// File: `zensim/weights/v0_7_2026-05-11.bin` (md5 `0ad0dace`).
/// Trained on **leak-free** safe-synthetic CSV (1,015 perceptual
/// duplicates of CID22 holdout removed; -28% rows). h=128, TV=10,
/// **seed=1** (strict upgrade from initial seed=0 ship — see below),
/// KonJND-aligned. Affine-calibrated (α=31.2540, β=-4.0305, R²=0.76)
/// against ssim2 on the synth corpus.
///
/// **First honest clean-corpus bake that exceeds fast-ssim2 on CID22
/// aggregate by > 0.003**: V0_7 CID22 SROCC = 0.8933 vs fast-ssim2
/// 0.8895 (**+0.0038**). Per-band CID22:
/// - B0 (<50): 0.4370 vs ssim2 0.4418 (-0.005, near-parity)
/// - B1 [50,65): 0.4424 vs ssim2 0.4694 (-0.027, loses)
/// - B2 [65,90): **0.7893** vs ssim2 0.7722 (**+0.017 BEATS**)
/// - B3 (≥90, n=43): **0.1944** vs ssim2 0.1121 (**+0.082 BEATS**)
/// - Near-PJND [58,68]: 0.3741 vs ssim2 0.3908 (-0.017, near-parity)
///
/// Wins B2 and B3, near-parity on B0 and Near-PJND, only meaningful
/// loss is B1. **Non-mono q-step rate = 5.46 % (within 5.5 % target)** —
/// tighter smoothness than the initial seed=0 ship (5.67 %).
///
/// **Holdout-overlap remediation**: V0_5's 0.8900 was inflated by
/// 11.77 % training-pair leak from 22 of 49 CID22 holdout refs (via
/// hex-hashed crops circumventing the filename blocklist). The
/// dHash-64 audit + stage-2 sliding-window detector
/// (`zensim-validate/src/bin/check_holdout_overlap{,_stage2}.rs`)
/// drove the cleanup; cleaned features file at
/// `/tmp/zensim_loop/safe_synth_clean_features.csv`.
///
/// Function and slot names preserved (`mlp_bake_preview_v0_4`,
/// `PROFILE_PREVIEW_V0_4`) for source-compat with consumer pinning.
/// Predecessors archived at `zensim/weights/archive/`:
/// - `v0_5_2026-05-11.bin` (md5 `0133d165`, training leak 11.77 %)
/// - `v0_7_seed0_2026-05-11.bin` (md5 `b31741e3`, initial V0_7
///   ship before seed=1 swap; CID22 0.8912, non-mono 5.67 %)
/// Gated behind `__experimental_versions` because the `weights/`
/// directory is excluded from the published crate.
#[cfg(feature = "__experimental_versions")]
pub(crate) fn mlp_bake_preview_v0_4() -> &'static [u8] {
    include_bytes!("../weights/v0_7_2026-05-11.bin")
}

#[cfg(feature = "__experimental_versions")]
static PROFILE_PREVIEW_V0_4: ProfileParams = ProfileParams {
    // Linear weights are unused on the MLP path but kept non-empty so
    // any caller that introspects `params.weights` length without
    // checking `mlp_bytes.is_some()` sees a sensible (V0_2-equivalent)
    // value.
    weights: &WEIGHTS_PREVIEW_V0_2,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    // V0_4 outputs raw distance, same semantics as V0_1/V0_2 — use the
    // classic mapping so V0_4 scores stay drop-in comparable.
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    mlp_bytes: Some(mlp_bake_preview_v0_4),
};

// --- Weight arrays ---

/// Preview v0.1 weights (344k synthetic pairs, 5-fold CV SROCC=0.9936).
/// SROCC = 0.9941 on full training set.
/// Layout: 4 scales × 3 channels × (13 basic + 6 peak) features = 228.
#[allow(clippy::excessive_precision)]
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
#[allow(clippy::excessive_precision)]
pub static WEIGHTS_PREVIEW_V0_2: [f64; 228] = [
    // --- Basic features (13/ch × 3ch × 4 scales = 156) ---
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
    0.0001182877, // Scale 0 Channel X
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
    0.0302900947, // Scale 0 Channel Y
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
    0.0020508776, // Scale 0 Channel B
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
    0.0001602903, // Scale 1 Channel X
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
    0.0498144992, // Scale 1 Channel Y
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
    0.0030681356, // Scale 1 Channel B
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
    0.0000861748, // Scale 2 Channel X
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
    0.0254137606, // Scale 2 Channel Y
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
    0.0000029596, // Scale 2 Channel B
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
    0.0000418198, // Scale 3 Channel X
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
    0.0182069151, // Scale 3 Channel Y
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
    0.0014196010, // Scale 3 Channel B
    // --- Peak features (6/ch × 3ch × 4 scales = 72) ---
    0.4487279165,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000, // Scale 0 Channel X
    0.4482423487,
    1.7175745862,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    8.3046937974, // Scale 0 Channel Y
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
    0.5506673335,
    0.0000000000, // Scale 1 Channel Y
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    5.5320293663,
    0.0000000000, // Scale 1 Channel B
    0.0000000000,
    8.0037734436,
    0.0000000000,
    0.0000000000,
    1.0761444814,
    0.0000000000, // Scale 2 Channel X
    1.5371198008,
    0.0000000000,
    0.0000000000,
    2.8277784618,
    0.0000000000,
    0.0000000000, // Scale 2 Channel Y
    1.6209056242,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.9606478120,
    0.0000000000, // Scale 2 Channel B
    13.6786977377,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    1.7799580769,
    0.0000000000, // Scale 3 Channel X
    1.6034688535,
    0.4040027990,
    13.5194607333,
    2.6215292388,
    0.0000000000,
    0.0000000000, // Scale 3 Channel Y
    1.6611285265,
    6.9307627972,
    0.0000000000,
    0.7474376328,
    12.8104312758,
    0.0000000000, // Scale 3 Channel B
];

// --- Canonical aliases ---
//
// Preferred name going forward — the explicit `LINEAR_` prefix
// disambiguates from V0_4's MLP weights, which ship as packed ZNPR v2
// bytes rather than a flat coefficient array. The unprefixed
// `WEIGHTS_PREVIEW_V0_X` names are kept indefinitely for source
// compatibility with code written against zensim 0.2.x and earlier.

/// Alias for [`WEIGHTS_PREVIEW_V0_1`]. Linear scoring weights for the
/// V0_1 profile, 228 entries (4 scales × 3 channels × 19 features).
/// Use this name in new code; the unprefixed alias is kept forever.
pub use self::WEIGHTS_PREVIEW_V0_1 as LINEAR_WEIGHTS_PREVIEW_V0_1;

/// Alias for [`WEIGHTS_PREVIEW_V0_2`]. Linear scoring weights for the
/// V0_2 profile. See [`LINEAR_WEIGHTS_PREVIEW_V0_1`] for naming
/// rationale.
pub use self::WEIGHTS_PREVIEW_V0_2 as LINEAR_WEIGHTS_PREVIEW_V0_2;
