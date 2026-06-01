//! Named metric profiles.
//!
//! Each [`ZensimProfile`] variant bundles weights and parameters that affect
//! score output. A given profile produces approximately the same scores
//! across crate patch versions. Variant names track the crate's minor
//! version that introduced them — `PreviewV0_3` is the variant shipped
//! by zensim 0.3.x; the underlying bake bytes inside the variant may
//! be swapped during 0.3.x patches as long as scores stay approximately
//! stable (per the [`zensim::CLAUDE.md`] shipping policy).
//!
//! Profile names and crate-version names are **two different things**:
//! - **Profile name** (`PreviewV0_3`) is the API surface — what
//!   downstream code matches on or constructs.
//! - **Bake-bytes version** (V0_18, V0_18-zerobiased, ...) is the
//!   implementation detail recorded in the methodology doc paired
//!   with each crate release.
//!
//! [`zensim::CLAUDE.md`]: https://github.com/imazen/zensim/blob/main/CLAUDE.md

/// Named metric profile. Scores for a given profile stay approximately
/// stable across crate versions. Variants may be removed in future
/// major (or minor-for-0.x) version bumps.
// `PartialEq`/`Eq`/`Hash` are hand-written below (not derived): the
// `Custom` variant holds `&'static ProfileParams`, and `ProfileParams`
// contains `f64` fields (not `Eq`) plus fn-pointer bake slots, so a
// derive would not compile. The manual impls compare named variants by
// discriminant and `Custom` profiles by `(params-pointer, name)` identity.
#[non_exhaustive]
#[derive(Clone, Copy, Debug)]
pub enum ZensimProfile {
    /// **`A` — the canonical generation-A profile (external name
    /// `zensim-a`).** The default general-purpose perceptual metric.
    ///
    /// Naming convention (see `docs/NAMING_CONVENTION.md`): EXTERNAL
    /// variant names are `A`, `A_Phone`, … (generation letter +
    /// optional display suffix) — a STABLE behavioral contract. The
    /// INTERNAL bake backing this variant rotates freely; its identity
    /// (currently the `v47-strict-QAT` bake) is recorded in the
    /// mapping table in `docs/CODEC_TARGET_METRIC.md`, NOT inlined here
    /// (so this doc can't go stale on rotation).
    A,
    /// Preview v0.2. Concordance-filtered 218k pairs, Nelder-Mead SROCC=0.9960.
    /// Linear-weights profile, no MLP forward pass. The historical
    /// stable default — `latest()` returned this through zensim 0.2.x.
    PreviewV0_2,
    /// Preview v0.3 — **canonical shipping profile (recovery phase 4,
    /// 2026-05-24 PM)**. Multi-dataset Tuner v5 bake (file
    /// `v_tuner_v11_2026-05-24.bin`, 54 KB packed i8 + zerobias + lz4,
    /// md5 `cac9416124a5e5f8ff577bc78e15ea1f`).
    ///
    /// 372-input MLP (372 → 128 → 128 identity passthrough) with
    /// per_sample_alpha_head + tanh_output_head + 7-knot PCHIP spline
    /// calibration. Trained on 5 groups (safesyn 196k + cid22_train
    /// 17.6k + kadid 10.1k + tid 3k + konjnd_dense 20.2k) with
    /// `tanh_output_head_scale = 30.0` and a konjnd-aggregation aux
    /// loss for per-source PJND calibration.
    ///
    /// Cross-corpus SROCC (held-out, 5-seed CI median):
    /// CID22 0.860, KonJND 0.285, AIC-3 0.776, AIC-4 0.929,
    /// monotonicity (JPEG q-sweep) 0.948.
    ///
    /// **Full 0-100 dial coverage** — score p5 = 28 (was 48 in the
    /// V_18 lineage), JND lands at score 60 bit-exact (was 79).
    /// Mean score at butter=3.5 (low-q) = 37 (was 55 flat floor).
    /// Per-unit cross-codec consistency 2.36 % of dial span
    /// (proportionally tighter than the V_18 ship's 2.63 %).
    ///
    /// **66/72 adjacent q-pairs discretely targetable** at ±1 score
    /// unit across {zenjpeg, zenwebp, zenavif, zenjxl}. AVIF and JXL
    /// reach all 18/18 pairs. Byte-identical short-circuit returns
    /// score = 100 exactly for any codec's lossless mode.
    ///
    /// Methodology: `benchmarks/v_tuner_v11_methodology_2026-05-24.md`.
    /// Per-codec q-range: `benchmarks/v_tuner_v5_per_codec_q_range_2026-05-24.md`.
    /// Integration guide: `docs/CODEC_TARGET_METRIC.md`.
    ///
    /// **Deprecated alias of [`Self::A`]** — behaves identically (same
    /// bake, same scores). Kept until internal call sites migrate off
    /// the `PreviewV0_3` name. Prefer `ZensimProfile::A` /
    /// `ZensimProfile::codec_target()`.
    #[deprecated(note = "use ZensimProfile::A (PreviewV0_3 is a deprecated alias)")]
    PreviewV0_3,
    /// **Externally-defined profile** — an escape hatch for profiles
    /// constructed outside this crate (for example the unpublished
    /// `zensim-experimental` crate, which preserves the historical
    /// research bakes off the published download). Build the parameters
    /// with [`ProfileParams::builder`] from a bake's bytes, promote them
    /// to `'static` (e.g. via a `OnceLock`), then wrap them here.
    ///
    /// `params` drives the full scoring runtime exactly as a built-in
    /// variant's does — every head / spline / per-codec-calibration
    /// behaviour lives in the bake bytes, so a `Custom` profile is
    /// bit-identical to the equivalent built-in. `name` is the display
    /// string returned by [`Self::name`] / [`Display`](core::fmt::Display).
    Custom {
        /// Profile parameters: weights, bake bytes, and dispositions.
        params: &'static ProfileParams,
        /// Display name, e.g. `"zensim-experimental-tuner-v4"`.
        name: &'static str,
    },
}

impl ZensimProfile {
    /// Current preview-stable profile. Returns [`Self::A`].
    ///
    /// Use this only when you explicitly want "whatever the current
    /// preview is" — the returned variant will rotate as new previews
    /// ship. For pinned reproducibility, name the variant directly
    /// (`ZensimProfile::A`). For the stable codec-target
    /// contract that codec crates should target, use [`Self::codec_target`].
    pub const fn latest_preview() -> Self {
        Self::A
    }

    /// Deprecated. Use [`Self::latest_preview`] for the rotating-preview
    /// alias, [`Self::codec_target`] for the stable codec contract, or
    /// name a `PreviewV0_X` variant directly for pinned reproducibility.
    #[deprecated(
        since = "0.3.0",
        note = "use latest_preview() or codec_target() or name a Preview variant directly"
    )]
    pub fn latest() -> Self {
        Self::A
    }

    /// **Canonical codec-target metric.** The stable, version-independent
    /// alias for "the bake all zen codecs train and target to."
    /// Returns [`Self::A`] — the current production codec-target bake.
    ///
    /// Codec crates (`zenjpeg`, `zenwebp`, `zenjxl`, `zenavif`, ...)
    /// should construct their `Zensim` instance via
    /// `Zensim::new(ZensimProfile::codec_target())` so that bake rotations
    /// (Tuner v5, v6, ...) flow through automatically without per-codec
    /// version edits.
    ///
    /// **Use cases this is purpose-built for:**
    /// - **Quality-target dial** — `Zensim::compute(source, distorted)`
    ///   inside an iterative encode-rescore-adjust outer loop
    ///   (see `zenwebp::EncodeConfig::target_zensim` for the reference
    ///   pattern; pattern is ~100 LOC per codec).
    /// - **Picker training** — train cross-codec pickers against this
    ///   bake's per-codec score parquets at
    ///   `/mnt/v/zen/picker-training/2026-05-19/butter/*.parquet`.
    ///
    /// **What this is NOT for:**
    /// - General-purpose ranking across heterogeneous distortion
    ///   families (KADID/TID/KonJND SROCC is poor by design — those
    ///   constraints were relaxed to gain monotonic dial behavior).
    ///   For general ranking, use [`Self::balanced_v3`].
    /// - In-encoder per-block RDO distortion term. zensim is per-image
    ///   (~14 ms at 1024² × 5–20k RDO calls = 70–280 s/image, infeasible).
    ///   See `docs/RDO_LOSS_FEASIBILITY_2026-05-24.md`.
    ///
    /// **Measured cross-codec consistency** (`benchmarks/tuner_v10_cross_codec_baseline_2026-05-24.md`):
    /// at matched-perceptual-quality anchors, median |Δ| = 1.18 score
    /// units, p90 = 3.58 across {JPEG, WebP, AVIF, JXL}.
    /// In the score 60–90 band (where codec consumers operate), median
    /// |Δ| drops to 0.6–1.5 — tight enough for production dial use.
    /// **Known gap**: scores below 55 are clamped flat (low-q dial
    /// dead zone); production code targeting `score < 55` should
    /// expect non-monotonic codec output until task #6 (Tuner v11)
    /// ships.
    ///
    /// **Why not Balanced or Compression?** Measured on the same
    /// 68,788 matched-anchor pairs (2026-05-24): Balanced v3 yields
    /// median |Δ| = 3.06 (p90 20.71), Compression v3 yields median
    /// |Δ| = 2.64 (p90 15.32). Tuner is **3-6× tighter cross-codec**,
    /// because Tuner trains with explicit cross-codec equivalence
    /// pair-loss while the other trails optimize within-codec rank
    /// fidelity at the cost of cross-codec spread. Picking either
    /// of those as the codec-target dial would give users ±20-50
    /// score-unit precision swings across codecs.
    ///
    /// See [`docs/CODEC_TARGET_METRIC.md`] in the zensim repo for the
    /// integration guide.
    ///
    /// Returns [`Self::A`]. The backing bake (and any future rotation)
    /// is recorded in the variant→bake mapping table in
    /// `docs/CODEC_TARGET_METRIC.md` — not inlined here, so this doc
    /// can't go stale when the bake rotates (see
    /// `docs/NAMING_CONVENTION.md`).
    pub const fn codec_target() -> Self {
        Self::A
    }

    /// Canonical name string, e.g. `"zensim-preview-v0.1"`.
    pub fn name(&self) -> &'static str {
        match self {
            Self::A => "zensim-a",
            Self::PreviewV0_2 => "zensim-preview-v0.2",
            #[allow(deprecated)]
            Self::PreviewV0_3 => "zensim-preview-v0.3",
            // Experimental / historical profiles live in `zensim-experimental`
            // and surface here as `Custom` with their original name string.
            Self::Custom { name, .. } => name,
        }
    }

    /// Internal parameters for this profile.
    pub(crate) fn params(&self) -> &'static ProfileParams {
        match self {
            // `A` is canonical; `PreviewV0_3` is its deprecated alias —
            // identical params, identical scores.
            #[allow(deprecated)]
            Self::A | Self::PreviewV0_3 => &PROFILE_A,
            Self::PreviewV0_2 => &PROFILE_PREVIEW_V0_2,
            // Experimental / historical profiles carry their own
            // `&'static ProfileParams` built in `zensim-experimental`.
            Self::Custom { params, .. } => params,
        }
    }
}

impl core::fmt::Display for ZensimProfile {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.name())
    }
}

// Hand-written equality/hashing (see the note on the enum). Named
// (fieldless) variants compare by discriminant; a `Custom` profile is
// equal only to another `Custom` with the same `params` pointer AND the
// same `name`, and is never equal to a named variant.
impl PartialEq for ZensimProfile {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (
                ZensimProfile::Custom { params: a, name: na },
                ZensimProfile::Custom { params: b, name: nb },
            ) => core::ptr::eq(*a, *b) && na == nb,
            (ZensimProfile::Custom { .. }, _) | (_, ZensimProfile::Custom { .. }) => false,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

impl Eq for ZensimProfile {}

impl core::hash::Hash for ZensimProfile {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        if let ZensimProfile::Custom { params, name } = self {
            (*params as *const ProfileParams as usize).hash(state);
            name.hash(state);
        }
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
    /// V_18 ship (PreviewV0_3) and earlier profiles.
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
    /// When set, [`soft_clamp_score`] is ignored. Default `false`
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

impl ProfileParams {
    /// Start building a custom [`ProfileParams`] for use with
    /// [`ZensimProfile::Custom`].
    ///
    /// This is the supported extension point for profiles defined outside
    /// this crate (e.g. the unpublished `zensim-experimental` crate, which
    /// preserves the historical research bakes). Defaults match the linear
    /// `PreviewV0_2` baseline (`blur_radius = 5`, `blur_passes = 1`,
    /// `num_scales = 4`, score mapping `100 − 18·d^0.7`, all dispositions
    /// off, no MLP). Override only what differs:
    ///
    /// ```
    /// use zensim::profile::ProfileParams;
    /// fn my_bake() -> &'static [u8] { include_bytes!("../weights/v47_strict_qat_native_2026-05-27.bin") }
    /// let params = ProfileParams::builder()
    ///     .mlp(my_bake)
    ///     .extended_features(true)
    ///     .compute_iw_features(true)
    ///     .skip_score_mapping(true)
    ///     .extrapolate_score(true)
    ///     .build();
    /// # let _ = params;
    /// ```
    pub fn builder() -> ProfileParamsBuilder {
        ProfileParamsBuilder {
            inner: ProfileParams {
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
            },
        }
    }
}

/// Builder for a custom [`ProfileParams`]. See [`ProfileParams::builder`].
///
/// Every method takes and returns `self` for chaining; finish with
/// [`build`](Self::build). The bake slots take a `fn() -> &'static [u8]`
/// (typically a tiny wrapper around `include_bytes!`) so the bytes can be
/// embedded by the defining crate and resolved lazily.
#[derive(Debug)]
pub struct ProfileParamsBuilder {
    inner: ProfileParams,
}

impl ProfileParamsBuilder {
    /// Linear scoring weights (one per feature). Defaults to the
    /// `PreviewV0_2` weight vector; unused on the MLP path but kept
    /// sensible for callers that introspect `params.weights`.
    pub fn weights(mut self, weights: &'static [f64]) -> Self {
        self.inner.weights = weights;
        self
    }

    /// Box-blur radius at scale 0 and the number of iterated passes
    /// (1 = rectangular, 3 ≈ Gaussian). Defaults to `(5, 1)`.
    pub fn blur(mut self, radius: usize, passes: u8) -> Self {
        self.inner.blur_radius = radius;
        self.inner.blur_passes = passes;
        self
    }

    /// Number of pyramid scales. Defaults to 4.
    pub fn num_scales(mut self, num_scales: usize) -> Self {
        self.inner.num_scales = num_scales;
        self
    }

    /// Score-mapping coefficients `A`, `B` in `100 − A·d^B`. Ignored when
    /// [`skip_score_mapping`](Self::skip_score_mapping) is set. Defaults to
    /// `(18.0, 0.7)`.
    pub fn score_mapping(mut self, a: f64, b: f64) -> Self {
        self.inner.score_mapping_a = a;
        self.inner.score_mapping_b = b;
        self
    }

    /// Return the bake's raw (already-calibrated) output directly instead
    /// of applying `100 − A·d^B`. Correct for MCOS/spline-calibrated bakes.
    pub fn skip_score_mapping(mut self, v: bool) -> Self {
        self.inner.skip_score_mapping = v;
        self
    }

    /// Use the bounded saturating squash `100·exp(−(A/100)·d^B)` for a
    /// linear (non-MLP) profile. Ignored when an MLP bake is set.
    pub fn bounded_squash(mut self, v: bool) -> Self {
        self.inner.bounded_squash = v;
        self
    }

    /// Primary MLP bake. Switches scoring from the linear dot-product to a
    /// forward pass through the loaded network.
    pub fn mlp(mut self, bytes: fn() -> &'static [u8]) -> Self {
        self.inner.mlp_bytes = Some(bytes);
        self
    }

    /// Secondary (D2-style) MLP bake mixed with the primary at
    /// `primary_mix` weight in raw-output space before score mapping.
    pub fn mlp_secondary(mut self, bytes: fn() -> &'static [u8], primary_mix: f32) -> Self {
        self.inner.mlp_bytes_b3 = Some(bytes);
        self.inner.mlp_primary_mix = primary_mix;
        self
    }

    /// Ensemble routing: a classifier bake whose sign selects between the
    /// primary [`mlp`](Self::mlp) bake (negative logit) and the
    /// `compression` bake (positive logit).
    pub fn ensemble(
        mut self,
        classifier: fn() -> &'static [u8],
        compression: fn() -> &'static [u8],
    ) -> Self {
        self.inner.ensemble_classifier_bytes = Some(classifier);
        self.inner.mlp_bytes_compression = Some(compression);
        self
    }

    /// Compute the extended (228 → 300) masked-feature block. Required for
    /// any bake whose input width exceeds 228.
    pub fn extended_features(mut self, v: bool) -> Self {
        self.inner.extended_features = v;
        self
    }

    /// Compute the IW-pool (300 → 372) feature block (implies extended).
    pub fn compute_iw_features(mut self, v: bool) -> Self {
        self.inner.compute_iw_features = v;
        self
    }

    /// Wrap the final score in a logistic soft-clamp instead of a hard
    /// `clamp(0, 100)` (preserves rank order at the extremes).
    pub fn soft_clamp_score(mut self, v: bool) -> Self {
        self.inner.soft_clamp_score = v;
        self
    }

    /// Return the (spline-extrapolated) score without any clamp, allowing
    /// values below 0 / above 100. Overrides `soft_clamp_score`.
    pub fn extrapolate_score(mut self, v: bool) -> Self {
        self.inner.extrapolate_score = v;
        self
    }

    /// Finish building.
    pub fn build(self) -> ProfileParams {
        self.inner
    }
}

// --- Profile definitions ---

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

/// `ZensimProfile::A` (PreviewV0_3) bake bytes — **rotated 2026-05-27 to
/// `v47-strict-QAT-native`** (`v47_strict_qat_native_2026-05-27.bin`, 27 KB,
/// sha256 `d0ef7a30…`). This is a bake rotation of the `Profile::A` slot, not
/// an API change — the `PreviewV0_3` name and the metric architecture are
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
/// (1.40→88.50). Global ordering verified: identity 97.69 > honest-q20 40.36
/// > channel-invert 12.21 > block-zero 0.00 (the regression-test ordering V39
/// inverts).
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
/// remain on disk at `zensim/weights/v39_v32plus_spline_seed17_2026-05-25.bin`
/// (still backing `PreviewV0_4`) for reproducibility.
pub(crate) fn mlp_bake_a_v47_qat() -> &'static [u8] {
    include_bytes!("../weights/v47_strict_qat_native_2026-05-27.bin")
}

/// Generation-A profile params (external `ZensimProfile::A`, and the
/// deprecated `PreviewV0_3` alias). Backing bake recorded in the
/// mapping table in `docs/CODEC_TARGET_METRIC.md`.
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

// --- Weight arrays ---

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

/// Alias for [`WEIGHTS_PREVIEW_V0_2`]. Linear scoring weights for the
/// V0_2 profile. See [`LINEAR_WEIGHTS_PREVIEW_V0_1`] for naming
/// rationale.
pub use self::WEIGHTS_PREVIEW_V0_2 as LINEAR_WEIGHTS_PREVIEW_V0_2;
