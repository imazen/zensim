//! Named metric profiles with permanently stable scoring.
//!
//! Each [`ZensimProfile`] variant bundles all parameters that affect score output:
//! weights, structural params, and score mapping. Same profile name = same scores
//! forever. The crate accumulates profiles; old ones never change or get removed.

/// Named metric profile. Each variant is permanently stable — same name produces
/// identical scores across all crate versions. The list only grows.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ZensimProfile {
    /// General-purpose v0.1. Trained on 12k synthetic pairs.
    GeneralV0_1,
    /// General-purpose v0.2. Trained on 163k synthetic pairs, SROCC=0.9857.
    GeneralV0_2,
}

impl ZensimProfile {
    /// Latest recommended general-purpose profile.
    pub fn latest() -> Self {
        Self::GeneralV0_2
    }

    /// Canonical name string, e.g. `"zensim-general-v0.2"`.
    pub fn name(&self) -> &'static str {
        match self {
            Self::GeneralV0_1 => "zensim-general-v0.1",
            Self::GeneralV0_2 => "zensim-general-v0.2",
        }
    }

    /// Internal parameters for this profile.
    pub(crate) fn params(&self) -> &'static ProfileParams {
        match self {
            Self::GeneralV0_1 => &PROFILE_GENERAL_V0_1,
            Self::GeneralV0_2 => &PROFILE_GENERAL_V0_2,
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
/// See [`ZensimConfig`](crate::metric::ZensimConfig) for detailed documentation
/// of each parameter's effect on computation path and performance.
#[cfg_attr(not(feature = "training"), allow(dead_code))]
pub struct ProfileParams {
    pub weights: &'static [f64; 156],
    pub blur_radius: usize,
    pub blur_passes: u8,
    pub num_scales: usize,
    /// When > 0.0, forces the full-image path and adds per-pixel activity masking.
    /// See [`ZensimConfig::masking_strength`](crate::metric::ZensimConfig::masking_strength).
    pub masking_strength: f32,
    pub score_mapping_a: f64,
    pub score_mapping_b: f64,
}

#[cfg(feature = "training")]
impl ProfileParams {
    /// Create custom params for weight training/exploration.
    pub fn custom(
        weights: &'static [f64; 156],
        blur_radius: usize,
        blur_passes: u8,
        num_scales: usize,
        masking_strength: f32,
        score_mapping_a: f64,
        score_mapping_b: f64,
    ) -> Self {
        Self {
            weights,
            blur_radius,
            blur_passes,
            num_scales,
            masking_strength,
            score_mapping_a,
            score_mapping_b,
        }
    }
}

// --- Profile definitions ---

static PROFILE_GENERAL_V0_1: ProfileParams = ProfileParams {
    weights: &WEIGHTS_GENERAL_V0_1,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    masking_strength: 0.0,
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
};

static PROFILE_GENERAL_V0_2: ProfileParams = ProfileParams {
    weights: &WEIGHTS_GENERAL_V0_2,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    masking_strength: 0.0,
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
};

// --- Weight arrays ---

/// Weights from gpu_ssim2_v1_12k.txt (12k synthetic pairs).
#[allow(clippy::excessive_precision)]
pub static WEIGHTS_GENERAL_V0_1: [f64; 156] = [
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
    0.0000000000, // Scale 0 Channel X
    0.0000000000,
    5.5184017295,
    0.0000000000,
    0.0000000000,
    11.9765353949,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.5570284539,
    0.0000000000,
    0.0664559172, // Scale 0 Channel Y
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
    0.0000000000, // Scale 0 Channel B
    0.0000000000,
    0.6808485268,
    0.0000000000,
    0.0000000000,
    6.9767245232,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000, // Scale 1 Channel X
    0.0000000000,
    7.9013437585,
    0.0000000000,
    0.0000000000,
    17.4878517746,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    1.0055655288,
    0.0000000000,
    3.8923240907, // Scale 1 Channel Y
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    10.0338032542,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    483.4483668293,
    0.0000000000,
    0.0000000000,
    0.0000000000, // Scale 1 Channel B
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    10.7281206112,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000, // Scale 2 Channel X
    109.4432649067,
    0.0000000000,
    31.8753801022,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.6075500208,
    0.0000000000,
    0.0000000000,
    0.0000000000, // Scale 2 Channel Y
    0.0000000000,
    0.3038513640,
    0.0000000000,
    0.0000000000,
    5.1950960267,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000, // Scale 2 Channel B
    2.4490060438,
    0.5716384101,
    0.0000000000,
    22.0447588384,
    36.8994405754,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0597352886, // Scale 3 Channel X
    23.9181465497,
    20.1430790262,
    1.8721810331,
    116.4610403364,
    0.0000000000,
    0.0000000000,
    153.8515182409,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.1634755027,
    0.0000000000,
    0.0000000000, // Scale 3 Channel Y
    12.8371881627,
    5.6454725963,
    2.4126482900,
    17.3653275481,
    23.9178928431,
    7.2115384615,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0000000000, // Scale 3 Channel B
];

/// Weights from gpu_ssim2_v2_163k.txt (163k synthetic pairs, 149.5k valid).
/// SROCC = 0.9857 on training set.
#[allow(clippy::excessive_precision)]
pub static WEIGHTS_GENERAL_V0_2: [f64; 156] = [
    0.0000000000,
    0.3054918030,
    0.0000000000,
    0.0120060829,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0025082274,
    0.0158347215,
    1.0232620956,
    0.0232095092,
    0.0157660229,
    0.0112574353, // Scale 0 Channel X
    2.6230901445,
    5.0412606075,
    0.0223917844,
    0.0098804550,
    0.0000000000,
    0.0038950338,
    0.0156205026,
    4.1215632793,
    0.0165163863,
    0.0000000000,
    0.6505034669,
    1.0661232328,
    0.2024220909, // Scale 0 Channel Y
    0.0000000000,
    0.0224667817,
    0.0214887709,
    8.6744486523,
    0.0212844299,
    0.0139801711,
    0.0000000000,
    0.0000000000,
    0.0167307326,
    1.0988427328,
    0.0000000000,
    0.0000000000,
    0.0147283435, // Scale 0 Channel B
    0.0056672813,
    0.3216560111,
    0.0000000000,
    0.0000000000,
    0.0013174343,
    0.0009618480,
    0.0235250311,
    0.0000000000,
    0.0233925065,
    51.1749644772,
    0.0240114888,
    0.0000000000,
    0.0000002541, // Scale 1 Channel X
    56.0466844974,
    13.7089560010,
    0.9639517783,
    11.6058052403,
    21.6136075666,
    0.0066979598,
    0.0000000000,
    1.8412885819,
    0.0000000000,
    0.0000000000,
    1.1198246434,
    0.0233930872,
    0.0010262427, // Scale 1 Channel Y
    1.7751266787,
    1.3741364948,
    0.0000000000,
    51.6748592180,
    1.9402960207,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0105201745,
    276.1886791157,
    0.0000000000,
    0.0000000000,
    0.0230782705, // Scale 1 Channel B
    0.5138116129,
    2.9679983985,
    0.0168215442,
    0.0000000000,
    9.4726552791,
    0.0227755446,
    0.0251548240,
    0.0231763042,
    0.0250751373,
    256.0304873419,
    0.0251168295,
    0.0000000000,
    0.0075593685, // Scale 2 Channel X
    17.1521567314,
    13.6143902459,
    15.2230380318,
    281.6839646804,
    67.7830673948,
    0.0224871201,
    0.0005901792,
    19.7594335859,
    0.0000000000,
    0.0000000000,
    2.4182112031,
    2.1322464656,
    0.0007368699, // Scale 2 Channel Y
    1.4831017806,
    0.0000000000,
    0.0000000000,
    38.8427093881,
    8.5955250320,
    2.7105025533,
    0.0000000000,
    0.0000000000,
    0.0000000000,
    0.0223146216,
    0.0000000000,
    0.0098977948,
    0.0000317606, // Scale 2 Channel B
    4.1642190683,
    13.5967841625,
    0.0000000000,
    0.0075158970,
    14.7858065617,
    0.0102946185,
    0.0047139742,
    0.0057291302,
    0.0000000000,
    1.0257943538,
    0.0146241140,
    0.3198369698,
    0.0000000000, // Scale 3 Channel X
    0.0582460099,
    0.0195498754,
    0.0114503725,
    869.4011573597,
    0.0001269371,
    0.3471799558,
    415.8779165902,
    82.2604764730,
    28.2928124905,
    0.2216564412,
    8.4103544255,
    0.0109363789,
    0.0151401628, // Scale 3 Channel Y
    8.2615006567,
    0.0069805260,
    0.0507613523,
    374.9053338534,
    69.6581630071,
    0.0150326812,
    0.0201076132,
    0.0246660472,
    0.0083943755,
    1.1493739991,
    0.0154112867,
    0.1279347879,
    0.0124929133, // Scale 3 Channel B
];
