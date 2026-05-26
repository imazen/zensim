/// Display profile for PPD-dependent quality scoring.
///
/// Per CODEC_TARGET_GOALS.md G11: "score 60 = visually lossless" should
/// mean different byte budgets for different displays. An image encoded
/// for Retina mobile needs fewer bytes to look identical than the same
/// image for a 1080p desktop — because the desktop viewer literally
/// cannot see the fine-grained artifacts that the Retina viewer can.
///
/// The Tier 1 implementation (post-network affine shift) stores per-PPD
/// calibration coefficients in the bake metadata. The runtime applies
/// `score_display = α(ppd) + β(ppd) · score_agnostic` when a display
/// profile is provided.
#[derive(Clone, Copy, Debug)]
pub struct DisplayProfile {
    /// Pixels per degree of visual angle.
    /// PPI × (π/180) × viewing_distance_cm / 2.54
    pub ppd: f32,
    /// Peak display luminance in nits (cd/m²).
    pub peak_nits: f32,
    /// Typical ambient light in lux.
    pub ambient_lux: f32,
}

impl DisplayProfile {
    /// Standard 1080p desktop monitor at ~60 cm viewing distance.
    pub const DESKTOP_1080P: Self = Self {
        ppd: 53.0,
        peak_nits: 350.0,
        ambient_lux: 200.0,
    };
    /// 4K 27" desktop at ~60 cm.
    pub const DESKTOP_4K_27: Self = Self {
        ppd: 93.0,
        peak_nits: 600.0,
        ambient_lux: 200.0,
    };
    /// MacBook Pro Retina 14" at ~45 cm.
    pub const MACBOOK_RETINA: Self = Self {
        ppd: 99.0,
        peak_nits: 500.0,
        ambient_lux: 300.0,
    };
    /// iPhone 14 Pro at ~25 cm.
    pub const IPHONE_14_PRO: Self = Self {
        ppd: 67.0,
        peak_nits: 2000.0,
        ambient_lux: 500.0,
    };
    /// Modern OLED phone, everyday indoor SDR viewing — the
    /// `modern_oled_phone_indoor` zenmetrics display the
    /// `zensim-b-phone` bake was trained on (6.1" 2532×1170 at 0.35 m
    /// → 109.97 ppd; 400 nit indoor SDR auto-brightness setpoint, not
    /// the panel's 1000–2000 nit HDR/sunlight peak; ~1000:1 effective
    /// contrast once the OLED's sub-milli-nit black is washed out by
    /// 250 lux ambient reflection).
    pub const PHONE_OLED_INDOOR: Self = Self {
        ppd: 109.97,
        peak_nits: 400.0,
        ambient_lux: 250.0,
    };
    /// iPhone 16 Pro at ~25 cm.
    pub const IPHONE_16_PRO: Self = Self {
        ppd: 69.0,
        peak_nits: 2000.0,
        ambient_lux: 500.0,
    };
    /// iPad Pro M4 at ~35 cm.
    pub const IPAD_PRO_M4: Self = Self {
        ppd: 80.0,
        peak_nits: 1600.0,
        ambient_lux: 400.0,
    };
    /// 55" 4K TV at 3 meters.
    pub const TV_4K_55_3M: Self = Self {
        ppd: 56.0,
        peak_nits: 1000.0,
        ambient_lux: 50.0,
    };
    /// 300 DPI print at 30 cm.
    pub const PRINT_300DPI_30CM: Self = Self {
        ppd: 115.0,
        peak_nits: 100.0,
        ambient_lux: 500.0,
    };
    /// Generic web delivery — the default when no display is specified.
    /// Matches the display-agnostic behavior of all previous bakes.
    pub const WEB_GENERIC: Self = Self {
        ppd: 60.0,
        peak_nits: 350.0,
        ambient_lux: 200.0,
    };
    /// Mohammadi 2025 AIC-3 evaluation display: "ccfl lcd, 64.27 ppd".
    pub const MOHAMMADI_AIC3: Self = Self {
        ppd: 64.27,
        peak_nits: 250.0,
        ambient_lux: 200.0,
    };
}

/// Display target for CVVDP-calibrated bakes.
///
/// Each variant corresponds to a bake trained on CVVDP scores computed
/// at that display's viewing conditions. The naming convention is
/// `zensim-{gen}-{display}` where `gen` is a generation letter (b =
/// current) and `display` is the viewing condition.
///
/// # Examples
/// ```
/// use zensim::display::DisplayTarget;
/// let target = DisplayTarget::Phone; // zensim-b-phone (modern OLED phone, ~110 PPD)
/// let profile = target.display_profile();
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DisplayTarget {
    /// Desktop monitor at typical desk distance (PPD ≈ 53, ~1080p at 60cm).
    /// This is the default when no display is specified.
    /// Bake name: `zensim-b-desktop` (or just `zensim-b`).
    Desktop,
    /// Smartphone, modern OLED at hand-held distance (PPD ≈ 110,
    /// `modern_oled_phone_indoor`: 6.1" 2532×1170 at 0.35 m, 400 nit
    /// indoor SDR setpoint, ~1000:1 effective contrast under 250 lux).
    /// Bake name: `zensim-b-phone`.
    Phone,
    /// Living-room TV at couch distance (PPD ≈ 56, 55" 4K at 3m).
    /// Bake name: `zensim-b-tv`.
    Tv,
}

impl DisplayTarget {
    /// The display profile (PPD, luminance, ambient) for this target.
    pub fn display_profile(self) -> DisplayProfile {
        match self {
            Self::Desktop => DisplayProfile::DESKTOP_1080P,
            Self::Phone => DisplayProfile::PHONE_OLED_INDOOR,
            Self::Tv => DisplayProfile::TV_4K_55_3M,
        }
    }

    /// Canonical bake name string: `zensim-b-{suffix}`.
    pub fn bake_name(self) -> &'static str {
        match self {
            Self::Desktop => "zensim-b-desktop",
            Self::Phone => "zensim-b-phone",
            Self::Tv => "zensim-b-tv",
        }
    }

    /// Short suffix for filenames and CLI.
    pub fn suffix(self) -> &'static str {
        match self {
            Self::Desktop => "desktop",
            Self::Phone => "phone",
            Self::Tv => "tv",
        }
    }

    /// CVVDP-trained bake bytes for this display target.
    /// Returns `None` for targets whose CVVDP backfill hasn't landed yet.
    pub fn bake_bytes(self) -> Option<&'static [u8]> {
        match self {
            Self::Desktop => Some(crate::profile::mlp_bake_cvvdp_desktop()),
            Self::Phone => Some(crate::profile::mlp_bake_cvvdp_phone_interim()),
            Self::Tv => None, // awaiting CVVDP TV backfill
        }
    }
}

impl core::fmt::Display for DisplayTarget {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.bake_name())
    }
}

impl core::str::FromStr for DisplayTarget {
    type Err = &'static str;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "desktop" | "zensim-b-desktop" | "zensim-b" => Ok(Self::Desktop),
            "phone" | "zensim-b-phone" => Ok(Self::Phone),
            "tv" | "zensim-b-tv" => Ok(Self::Tv),
            _ => Err("expected one of: desktop, phone, tv"),
        }
    }
}

/// Per-display affine calibration coefficients.
///
/// `score_display = alpha + beta * score_agnostic`
///
/// Higher PPD → stricter scores (same distortion is MORE visible on
/// high-PPI displays). The affine shift encodes this: at PPD=67
/// (iPhone), β < 1.0 compresses the score range (making "score 60"
/// require a higher-quality encode).
#[derive(Clone, Copy, Debug)]
pub struct DisplayCalibration {
    /// PPD bracket this calibration applies to.
    pub ppd: f32,
    /// Additive offset.
    pub alpha: f32,
    /// Multiplicative scale.
    pub beta: f32,
}
