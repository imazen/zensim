//! Named metric profiles.
//!
//! Each [`ZensimProfile`] variant selects a named model. [`ZensimProfile::latest`]
//! returns the general-purpose default ([`ZensimProfile::E`]) and
//! [`ZensimProfile::codec_target`] the model specialised for codec artifacts
//! ([`ZensimProfile::ECodec`]). Pin the crate version when reproducing exact
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
    PreviewV0_1,
    /// Preview v0.2. Concordance-filtered 218k pairs, Nelder-Mead SROCC=0.9960.
    /// Linear-weights profile, no MLP forward pass. `latest()` returned this
    /// through zensim 0.2.x; the 0.2.7 compatibility note on
    /// [`Self::PreviewV0_1`] applies here too.
    PreviewV0_2,
    /// Legacy MLP profile `A` (`zensim-a`): 372 features at the first formula
    /// revision, a monotone network and a calibrated output spline. It was the
    /// codec-quality target of the 0.3.0 development line before [`Self::E`] and
    /// is kept for callers whose thresholds are calibrated to its scores. New code
    /// should use [`Self::latest`] or [`Self::codec_target`].
    A,
    /// General-purpose profile `E` (`zensim-e`), the default since 0.3.0.
    ///
    /// A monotone MLP over 410 revision-7 damage features and 410 fragility
    /// products (each damage feature scaled by the reference's local flatness),
    /// with every damage input entering with non-negative weight, so more damage
    /// never raises the score. Trained on synthetic codec ladders scored by
    /// SSIMULACRA2 together with human-rated datasets covering codecs, noise, blur,
    /// colour and contrast changes. Scores run from 100 (identical) downwards and
    /// can go negative for severe damage.
    E,
    /// Codec-specialised profile `ECodec` (`zensim-e-codec`), returned by
    /// [`Self::codec_target`].
    ///
    /// The same architecture and features as [`Self::E`], fitted to human
    /// judgments of compression artifacts. It ranks codec output more accurately
    /// than `E` and is less reliable on other distortion types (noise, colour
    /// shifts, contrast changes), so use it where the input is codec output.
    ECodec,
    /// **INTERNAL — legacy 228-weight linear scoring, no MLP head.**
    /// Tags results from the training entry points `compute_zensim_with_config`
    /// and `compute_zensim_with_ref_and_config`, which score via a plain
    /// `dot(features, WEIGHTS_PREVIEW_V0_2)` distance (the caller-supplied
    /// [`crate::ZensimConfig`] drives blur/scale/masking knobs directly) and
    /// never run an MLP forward pass. The tag exists so `ZensimResult::profile()`
    /// never claims an MLP-scored result for a linear-only computation.
    #[cfg(any(feature = "training", test))]
    #[doc(hidden)]
    LegacyLinearV0_2,
}

impl ZensimProfile {
    /// Latest recommended general-purpose profile: [`Self::E`].
    pub fn latest() -> Self {
        Self::E
    }

    /// The profile specialised for codec artifacts: [`Self::ECodec`]. Encoders
    /// that target a score use this so later model rotations reach them without
    /// changing their profile selection.
    pub const fn codec_target() -> Self {
        Self::ECodec
    }

    /// Canonical name string, e.g. `"zensim-e"`.
    pub fn name(&self) -> &'static str {
        match self {
            Self::PreviewV0_1 => "zensim-preview-v0.1",
            Self::PreviewV0_2 => "zensim-preview-v0.2",
            Self::A => "zensim-a",
            Self::E => "zensim-e",
            Self::ECodec => "zensim-e-codec",
            #[cfg(any(feature = "training", test))]
            Self::LegacyLinearV0_2 => "zensim-legacy-linear-v0.2",
        }
    }

    /// Internal parameters for this profile.
    pub(crate) fn params(&self) -> &'static ProfileParams {
        match self {
            Self::PreviewV0_1 => &PROFILE_PREVIEW_V0_1,
            Self::PreviewV0_2 => &PROFILE_PREVIEW_V0_2,
            Self::A => &PROFILE_A,
            Self::E => &PROFILE_E,
            Self::ECodec => &PROFILE_E_CODEC,
            #[cfg(any(feature = "training", test))]
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
        [
            self.mlp_bytes,
            self.mlp_bytes_b3,
            self.ensemble_classifier_bytes,
        ]
        .into_iter()
        .flatten()
        .map(|f| f())
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
#[derive(Debug)]
#[non_exhaustive]
pub struct ProfileParams {
    /// Scoring weights (one per feature, length = `FEATURES_PER_SCALE * num_scales`).
    /// Empty `&[]` for MLP-scored profiles (the bake bytes live in a
    /// crate-private `mlp_bytes` field accessed via the profile API).
    pub weights: &'static [f64],
    /// Box blur radius at scale 0 (kernel width = `2 * radius + 1`).
    pub blur_radius: usize,
    /// Number of iterated box blur passes (1 = rectangular, 3 ≈ Gaussian).
    pub blur_passes: u8,
    /// Number of pyramid scales (typically 4).
    pub num_scales: usize,
    /// Score mapping coefficient A in `100 - A × d^B`. **Ignored when
    /// [`Self::skip_score_mapping`] is `true` (the bake is already
    /// MCOS-calibrated).**
    pub score_mapping_a: f64,
    /// Score mapping exponent B in `100 - A × d^B`. **Ignored when
    /// [`Self::skip_score_mapping`] is `true`.**
    pub score_mapping_b: f64,
    /// When `true`, the MLP bake's raw output is returned **directly**
    /// as the final score (no `100 − A·d^B` transform). This is correct
    /// for V0_5+ bakes, which are affine-calibrated to the
    /// MCOS-aligned 0–100 scale during baking. Set `false` for V0_1 /
    /// V0_2 linear profiles whose raw output is a distance.
    pub skip_score_mapping: bool,
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

    /// Optional **secondary** MLP bake for D2-style multi-output ensembles
    /// (V_20 input-shaping B3 specialist). When `Some`, the runtime forwards
    /// both `mlp_bytes` and `mlp_bytes_b3` and mixes their RAW outputs at
    /// `mlp_primary_mix` weight before the score-mapping step. The secondary
    /// bake MUST be affine-calibrated to the primary's output scale (use
    /// `zensim-validate/bin/affine_calibrate`). Both bakes MUST share
    /// `n_inputs` — their forward path runs over the same 228-feature
    /// vector; `feature_transforms` metadata is per-bake so the primary
    /// can be untransformed while the secondary applies V_20-style
    /// per-feature shaping.
    ///
    /// `None` (default for V_18 ship and earlier) keeps the single-bake
    /// path with zero overhead.
    pub(crate) mlp_bytes_b3: Option<fn() -> &'static [u8]>,

    /// Mix weight on the **primary** bake (`mlp_bytes`). The secondary
    /// gets `1.0 - mlp_primary_mix`. Only used when `mlp_bytes_b3` is
    /// `Some`. Per the D2 design doc, α = 0.7 gives CID22 B3 +0.080
    /// at aggregate −0.004 vs V_18 ship alone. α = 0.8 gives B3 +0.052
    /// at aggregate match.
    pub(crate) mlp_primary_mix: f32,

    /// When `true`, the runtime computes the **extended-features**
    /// block (228 → 300 features — adds 72 masked features) per pair
    /// via `Zensim::compute_extended_features`. Required when the
    /// profile's MLP bake has `n_inputs > 228`. Default `false` keeps
    /// the V_18 / V_20 IS 228-feature fast path.
    ///
    /// The masking pass reuses the already-computed flatness map, so
    /// extended-features overhead is moderate (~10–30 % per-pair compute
    /// vs standard). Distinct from `compute_iw_features` which adds
    /// a separate weighted pool (more expensive).
    ///
    /// Added 2026-05-15 to enable V_20 extended-shaping profile
    /// (PreviewV0_5_Extended at 300-feat input) after the V_20a IW
    /// (372-feat) path was falsified for CID22 transfer.
    pub extended_features: bool,

    /// When `true`, the runtime also computes the **IW pool** block
    /// (300 → 372 features — adds 72 IW-weighted features per Wang &
    /// Li 2011). Implies `extended_features = true` for the standard
    /// IW layout. Default `false`.
    ///
    /// **Note**: setting this for a shipping profile is currently
    /// discouraged — V_20a IW bakes catastrophically failed CID22
    /// transfer (Z-RMSE 0.869 vs V_18's 0.455 at k=1; near-random at
    /// k=8). Reserved for research bakes that need full 372-input
    /// scoring through the standard runtime.
    pub compute_iw_features: bool,

    /// When `true`, the final score is wrapped through a logistic
    /// soft-clamp `100 / (1 + exp(-(raw - 50) / 20))` instead of the
    /// hard `raw.clamp(0, 100)`. Preserves rank ordering at the
    /// extremes (no ties → SROCC stays defined when V_20+ multi-bake
    /// mixes extrapolate below 0 or above 100).
    ///
    /// The soft-clamp is a no-op in the `[5, 95]` range (output differs
    /// from input by less than 1.5 units at the band centers); it only
    /// reshapes the tails. Cost: one `exp` per score (~1 ns).
    ///
    /// Use on profiles that mix bakes of different shapes or that
    /// extrapolate outside training distribution. PreviewV0_4 (V_18 +
    /// V_20 IS multi-bake) ships with this `true` — the V_20 IS B3
    /// specialist's raw output extends past 100 on some pairs after
    /// the linear mix, and the hard clamp was creating tie blocks
    /// that collapsed SROCC to 0 on TID B0/B1 pairs.
    ///
    /// Default `false` keeps the hard-clamp legacy semantics for
    /// the V_18 ship lineage and earlier profiles.
    ///
    /// Added 2026-05-16 (T3.2). See zensim/CLAUDE.md `V_20 input-shaping
    /// learnings > Soft-clamp the multi-bake output` for the design rationale.
    pub soft_clamp_score: bool,

    /// EXP-CROSS-CODEC-V10 (2026-05-20): when `true`, the final score
    /// is returned **without clamping or soft-clamping** — the
    /// PCHIP spline's linear extrapolation past the knot endpoints
    /// flows through to the user. V10 profiles set this so the
    /// score-space dial can dip below 0 for "pathological /
    /// unreasonable" codec output (worst codec at q=0, butter > 12)
    /// rather than collapsing into a tie block at 0.
    ///
    /// When set, `soft_clamp_score` is ignored. Default `false`
    /// preserves legacy [0, 100] semantics for V9 and earlier
    /// profiles whose splines/anchors only span [0, 100].
    pub extrapolate_score: bool,

    /// **Correct-by-construction bounded squash** for *linear*
    /// (`mlp_bytes == None`) profiles. When `true`, the final score is
    /// `100 · exp(−(a/100) · d^b)` of the non-negative-weight feature
    /// distance `d = Σ wᵢ fᵢ` (with `a = score_mapping_a`, `b =
    /// score_mapping_b`), instead of the legacy `100 − a·d^b`.
    ///
    /// Because every feature is a non-negative dissimilarity
    /// (`fᵢ ≥ 0`, `= 0` iff locally identical) and the V0_2 weights are
    /// all non-negative, `d ≥ 0` with `d = 0` iff identical. The squash
    /// `S(d) = exp(−k·d^b)` is strictly decreasing on `[0, ∞)`,
    /// `S(0) = 1`, `S → 0`. Composing them gives a score that is
    /// **bounded `[0, 100]`, equal to 100 iff identical (its unique
    /// global maximum), and monotone non-increasing in every error
    /// feature — all by construction, on the entire input domain**
    /// (including content far off the training manifold). It is a
    /// strictly-monotone transform of `d`, so SROCC is identical to the
    /// legacy `100 − a·d^b` mapping; only the unbounded-below tail and
    /// mid-range scale differ. See
    /// `docs/METRIC_INVARIANTS_MECHANISM_AND_REDESIGN_2026-05-26.md` §4.
    ///
    /// Ignored on MLP profiles (`mlp_bytes.is_some()`), whose
    /// correct-by-construction form requires a monotone architecture
    /// (the larger follow-on in §5 of that doc). Default `false`.
    pub bounded_squash: bool,

    /// **Ensemble routing classifier** — when `Some`, the runtime
    /// loads the classifier bake, forwards the feature vector through
    /// it, and routes to either `mlp_bytes` (the balanced/primary
    /// bake) or `mlp_bytes_compression` (the alternative bake) based
    /// on the classifier's sign.
    ///
    /// The classifier is a small (372 → 64 → 1 or 300 → 64 → 1) MLP
    /// trained to predict `is_compression_corpus` on the canonical
    /// 5-corpus held-out set. Its output is a pre-sigmoid logit;
    /// `logit > 0` (i.e. `sigmoid(logit) > 0.5`) routes the pair to
    /// the compression bake.
    ///
    /// `None` (default) keeps the single-bake (`mlp_bytes`) forward
    /// path; `mlp_bytes_compression` is ignored.
    ///
    /// Added 2026-05-18 for PreviewV0_5Ensemble. The ensemble bake +
    /// classifier together produce per-corpus SROCC at or near
    /// `max(balanced, compression)` on every canonical corpus while
    /// staying within the compression-trail § A.10 gate's −0.10
    /// synthetic tolerance vs the balanced ship. See
    /// `benchmarks/exp_ensemble_v05_eval_2026-05-18.md`.
    pub(crate) ensemble_classifier_bytes: Option<fn() -> &'static [u8]>,

    /// Alternative MLP bake routed to when the classifier (see
    /// `ensemble_classifier_bytes`) outputs a positive logit. The
    /// "compression-trail" bake in the production setup; called
    /// secondary here to avoid name conflicts with the existing
    /// `mlp_bytes_b3` D2-style ensemble slot (which mixes RAW
    /// outputs linearly rather than routing).
    ///
    /// Ignored when `ensemble_classifier_bytes` is `None`. Both
    /// bakes MUST accept the same input-feature shape (typically
    /// 300 features in the V0_5 generation).
    ///
    /// Added 2026-05-18.
    pub(crate) mlp_bytes_compression: Option<fn() -> &'static [u8]>,
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
            // Custom training params get the legacy distance-to-score mapping
            // (V0_2 semantics). V0_8+ pre-calibrated bakes set the flag in
            // their static `ProfileParams` definitions.
            skip_score_mapping: false,
            mlp_bytes: None,
            mlp_bytes_b3: None,
            mlp_primary_mix: 1.0,
            extended_features: false,
            compute_iw_features: false,
            soft_clamp_score: false,
            extrapolate_score: false,
            bounded_squash: false,
            ensemble_classifier_bytes: None,
            mlp_bytes_compression: None,
        }
    }
}

// --- Profile definitions ---

// `PreviewV0_1` is the linear V0.1 profile published through zensim
// 0.1.x / 0.2.x. Identical to `PROFILE_PREVIEW_V0_2` except for the
// weight vector, so scores match those releases bit-for-bit.
static PROFILE_PREVIEW_V0_1: ProfileParams = ProfileParams {
    weights: &WEIGHTS_PREVIEW_V0_1,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    bounded_squash: false,
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    skip_score_mapping: false,
    mlp_bytes: None,
    mlp_bytes_b3: None,
    mlp_primary_mix: 1.0,
    extended_features: false,
    compute_iw_features: false,
    soft_clamp_score: false,
    extrapolate_score: false,
    ensemble_classifier_bytes: None,
    mlp_bytes_compression: None,
};

/// `ZensimProfile::PreviewV0_2` params — the plain 228-weight linear V0_2
/// scoring path (no MLP head), published through zensim 0.1.x / 0.2.x.
/// Byte-identical to `PROFILE_LEGACY_LINEAR_V0_2` and to
/// [`ProfileParams::builder`]'s defaults; kept as its own ungated static so
/// the always-available `PreviewV0_2` variant can dispatch to it in every
/// build.
static PROFILE_PREVIEW_V0_2: ProfileParams = ProfileParams {
    weights: &WEIGHTS_PREVIEW_V0_2,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    bounded_squash: false,
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    skip_score_mapping: false,
    mlp_bytes: None,
    mlp_bytes_b3: None,
    mlp_primary_mix: 1.0,
    extended_features: false,
    compute_iw_features: false,
    soft_clamp_score: false,
    extrapolate_score: false,
    ensemble_classifier_bytes: None,
    mlp_bytes_compression: None,
};

/// `ZensimProfile::A` bake bytes — **rotated 2026-05-27 to
/// `v47-strict-QAT-native`** (`v47_strict_qat_native_2026-05-27.bin`, 27 KB,
/// sha256 `d0ef7a30…`). This is a bake rotation of the `Profile::A` slot, not
/// an API change — the `A` name and the metric architecture are
/// unchanged.
///
/// **Why rotated:** the prior V39 bake (`v39_v32plus_spline…`, still backing
/// `PreviewV0_4`) is *not a correct similarity metric* — it scores
/// `identity = 0` on every reference and its codec dial is non-invertible
/// (q-sweep 67.7 % monotone / 53.6 % tied; high-quality encodes collapse to
/// score 0). v47-strict is **masked-monotone-by-construction** (W1 ≥ 0 on the
/// 300 sign-safe features, rank_w ≤ 0, α ≡ 1): 0 inversions, 0 above-identity,
/// `identity = 97.69` (the dial max). Best codec dial measured — q-sweep
/// **94.33 % monotone / 0.33 % tied**, clean monotone median q5→q95
/// (1.40→88.50). Global ordering verified: identity 97.69 > honest-q20
/// 40.36 > channel-invert 12.21 > block-zero 0.00 (the regression-test
/// ordering V39 inverts).
///
/// 372-input MLP (372 → 128 → 64 + per-sample-α + tanh pin), f16+zerobias
/// encoder + f32 identity passthrough, with a monotone PCHIP dial spline
/// (`zentrain.output_calibration_spline`, fit in-pass on the projected+
/// quantized net). Produced by ONE `zensim_mlp_train --manifest
/// zensim/weights/manifests/v47_strict_qat.toml` pass (QAT-native, no Python
/// post-step).
///
/// **Held-out panel** (vs the prior V39, which it replaces): CID22 0.8657,
/// KADID 0.7933, TID 0.7927, KonJND 0.4185, AIC-3 0.7680, AIC-4 0.8854.
/// Costs vs V39 are rank-SROCC on the non-compression analytic-distortion
/// corpora (KADID/TID −0.13, integrity guards) + KonJND (f16 drops PJND
/// precision) — V39's higher rank there is moot because its dial is
/// non-invertible. Methodology (all 8 sub-points):
/// `benchmarks/v0_qat_native_methodology_2026-05-27.md`. q-sweep:
/// `benchmarks/qsweep_qat_native_vs_v39_2026-05-27.md`. The prior V39 bytes
/// were relocated to
/// `zensim-experimental/weights/v39_v32plus_spline_seed17_2026-05-25.bin`
/// (backing `zensim_experimental::preview_v0_4()`) for reproducibility.
pub(crate) fn mlp_bake_a_v47_qat() -> &'static [u8] {
    include_bytes!("../weights/v47_strict_qat_native_byid_2026-09-06.bin")
}

/// Profile `A` params. Its bake identity is recorded in `docs/PROFILES.md`.
static PROFILE_A: ProfileParams = ProfileParams {
    // Linear weights are unused on the MLP path but kept non-empty so
    // any caller that introspects `params.weights` length without
    // checking `mlp_bytes.is_some()` sees a sensible (V0_2-equivalent)
    // value.
    weights: &WEIGHTS_PREVIEW_V0_2,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    bounded_squash: false,
    // v47-strict-QAT bake carries its own monotone PCHIP spline calibration
    // via the `zentrain.output_calibration_spline` metadata — the raw MLP
    // output is dial-honest after the spline applies. No legacy
    // `100 - 18·d^0.7` transform needed.
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    skip_score_mapping: true,
    mlp_bytes: Some(mlp_bake_a_v47_qat),
    mlp_bytes_b3: None,
    mlp_primary_mix: 1.0,
    // v47-strict is a 372-feature bake (extended + IW-pool features).
    extended_features: true,
    compute_iw_features: true,
    // Hard clamp at [0, 100] post-spline (spline output is already
    // calibrated into the dial range; clamp catches numerical drift).
    soft_clamp_score: false,
    extrapolate_score: true,
    ensemble_classifier_bytes: None,
    mlp_bytes_compression: None,
};

/// `ZensimProfile::LegacyLinearV0_2` params — the plain 228-weight linear
/// V0_2 scoring path (no MLP head), matching [`ProfileParams::builder`]'s
/// defaults byte-for-byte (which in turn reconstruct the removed
/// `PreviewV0_2` built-in profile; see commit `493c91cd`). Backs the
/// doc-hidden `compute_zensim_with_config` /
/// `compute_zensim_with_ref_and_config` research entry points in
/// `metric.rs` — nothing else constructs a `Zensim` with this profile.
#[cfg(any(feature = "training", test))]
static PROFILE_LEGACY_LINEAR_V0_2: ProfileParams = ProfileParams {
    weights: &WEIGHTS_PREVIEW_V0_2,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    bounded_squash: false,
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    skip_score_mapping: false,
    mlp_bytes: None,
    mlp_bytes_b3: None,
    mlp_primary_mix: 1.0,
    extended_features: false,
    compute_iw_features: false,
    soft_clamp_score: false,
    extrapolate_score: false,
    ensemble_classifier_bytes: None,
    mlp_bytes_compression: None,
};

/// `ZensimProfile::E` bake: the general revision-7 model (H128 monotone MLP,
/// 820 inputs), stripped of its training-provenance metadata; its weights and
/// serving metadata equal the evaluated artifact byte for byte. The mapping to
/// the research identity is in `docs/PROFILES.md`.
pub(crate) fn mlp_bake_e_general() -> &'static [u8] {
    include_bytes!("../weights/rev7_general_h128_2026-10-27.bin")
}

/// `ZensimProfile::ECodec` bake: the codec revision-7 model, prepared the same
/// way as [`mlp_bake_e_general`].
pub(crate) fn mlp_bake_e_codec() -> &'static [u8] {
    include_bytes!("../weights/rev7_codec_h128_2026-10-27.bin")
}

/// Shared runtime shape of the revision-7 profiles. Feature IDs, formula revision,
/// derived inputs and output calibration all come from the bake's metadata.
const fn rev7_params(bake: fn() -> &'static [u8]) -> ProfileParams {
    ProfileParams {
        weights: &WEIGHTS_PREVIEW_V0_2,
        blur_radius: 5,
        blur_passes: 1,
        num_scales: 4,
        bounded_squash: false,
        score_mapping_a: 18.0,
        score_mapping_b: 0.7,
        skip_score_mapping: true,
        mlp_bytes: Some(bake),
        mlp_bytes_b3: None,
        mlp_primary_mix: 1.0,
        extended_features: true,
        compute_iw_features: true,
        soft_clamp_score: false,
        extrapolate_score: true,
        ensemble_classifier_bytes: None,
        mlp_bytes_compression: None,
    }
}

static PROFILE_E: ProfileParams = rev7_params(mlp_bake_e_general);
static PROFILE_E_CODEC: ProfileParams = rev7_params(mlp_bake_e_codec);

// --- Weight arrays ---

/// Preview v0.1 linear weights (344k synthetic pairs, 5-fold CV SROCC=0.9936).
/// 228 entries = 4 scales × 3 channels × 19 features. The canonical V0.1
/// coefficient vector backing [`ZensimProfile::PreviewV0_1`].
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
