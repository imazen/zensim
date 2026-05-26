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
