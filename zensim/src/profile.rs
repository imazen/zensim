//! Named metric profiles.
//!
//! Each [`ZensimProfile`] variant bundles weights and parameters that affect
//! score output. A given profile produces approximately the same scores
//! across crate patch versions. The canonical shipping profile is
//! [`ZensimProfile::A`]; the underlying bake bytes inside a variant may
//! be swapped during patch releases as long as scores stay approximately
//! stable (per the [`zensim::CLAUDE.md`] shipping policy).
//!
//! Profile names and crate-version names are **two different things**:
//! - **Profile name** (`A`, `PreviewV0_2`) is the API surface — what
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
    /// **`A` — the DEPRECATED generation-A profile (external name
    /// `zensim-a`), the v47-strict-QAT MLP bake.** Superseded by the
    /// deterministic generation-B linear profile [`Self::B`], which
    /// [`Self::codec_target`] / [`Self::latest_preview`] now return.
    ///
    /// **Deprecated since 0.3.0.** Gated behind the `deprecated-profiles`
    /// feature (ON by default so existing code keeps working). To drop `A`
    /// and its ~27 KB MLP bake, build with `--no-default-features` and
    /// re-add the other defaults you need (`avx512`, `imgref`, `threads`).
    /// Migrate to [`Self::codec_target`] (the codec-quality dial),
    /// [`Self::latest_preview`], or name [`Self::B`] directly. A future
    /// minor release may move `deprecated-profiles` out of `default`.
    ///
    /// Naming convention (see `docs/NAMING_CONVENTION.md`): EXTERNAL
    /// variant names are `A`, `A_Phone`, … (generation letter +
    /// optional display suffix). The INTERNAL bake backing this variant
    /// is the `v47-strict-QAT` bake, recorded in the mapping table in
    /// `docs/CODEC_TARGET_METRIC.md`.
    #[cfg(feature = "deprecated-profiles")]
    #[deprecated(
        since = "0.3.0",
        note = "generation-A (v47 MLP) is superseded by the deterministic linear `B`; \
                use ZensimProfile::codec_target() / latest_preview() (now `B`) or name \
                ZensimProfile::B directly. `A` remains behind the default-on \
                `deprecated-profiles` feature; disable it to drop `A`."
    )]
    A,
    /// Preview v0.1 — linear-weights profile (no MLP). 344k synthetic
    /// pairs, 5-fold CV SROCC = 0.9936. Uses the [`WEIGHTS_PREVIEW_V0_1`]
    /// coefficients with the classic `100 − 18·d^0.7` score mapping.
    ///
    /// Retained for backwards-compatibility: this variant shipped in the
    /// published zensim 0.1.x / 0.2.x line with the same weights and
    /// mapping. Scores match those releases closely but not bit-exactly:
    /// kernel-fusion and summation-order changes since 0.2.7 shift
    /// ≥ 64px scores by ~1e-3..1.5e-2 units (measured, synthetic probes),
    /// and sub-64px inputs now reflect-pad (substantially different — and
    /// better-behaved — scores below 64px). For new code prefer
    /// [`Self::A`] (codec target) or [`Self::PreviewV0_2`] (linear
    /// general-ranking).
    PreviewV0_1,
    /// Preview v0.2. Concordance-filtered 218k pairs, Nelder-Mead SROCC=0.9960.
    /// Linear-weights profile, no MLP forward pass. The historical
    /// stable default — `latest()` returned this through zensim 0.2.x.
    PreviewV0_2,
    /// **`B` — generation-B SDR profile (external name `zensim-b`).**
    /// Deterministic LINEAR core: a 35-weight lasso ensemble over the 372
    /// features (`ens-Pline-cid80`, 823 B), dial-calibrated against the
    /// shared multiband anchor scale. Ships the 2026-07-04 campaign's SDR
    /// pick: beats `A` on 7/9 held-out panel axes (CID22 0.8733 vs 0.8657,
    /// KonJND 0.5439 vs 0.4185) with the dial panel green (monotonicity
    /// 0.9711, 0 dead zones). Fit is closed-form and byte-reproducible —
    /// no training seed exists, so the MLP family's collapse mode
    /// structurally cannot occur. Provenance + repro:
    /// `benchmarks/provenance_best_results_2026-07-04.md`.
    ///
    /// **SDR content only** — for HDR (absolute-luminance) content use
    /// [`ZensimProfile::BHdr`]. The two share a *partial* mid-range dial
    /// anchor (cross-model MAE ≈5pt, seam ≤0.92pt inside the overlap) but
    /// NOT a full range: `B`'s dial is calibrated `[0,100]`, `BHdr`'s only
    /// over its HDR data range — outside that it extrapolates. See
    /// `benchmarks/profile_b_methodology_2026-07-12.md` §3b.
    B,
    /// **`BHdr` — generation-B HDR profile (external name `zensim-b-hdr`).**
    /// Deterministic LINEAR core on SHAPED features (11.8 KB,
    /// `hdrmix-lasso0.0003-shaped`, cvvdp-mix target `0.5·ssim2 +
    /// 0.5·(JOD−6)/4`). UPIQ-HDR |SROCC| **0.7536 point estimate** (vs `A`
    /// 0.6933, prior anchored2 bake 0.7313; cvvdp 0.758 is the cross-metric
    /// reference) — the point is the max of a 7-candidate λ grid selected on
    /// UPIQ itself; selection-adjusted (maxT) p = 0.22, so an in-domain
    /// improvement over the prior bake is **not established** (family median
    /// ≈ tie; non-inferiority is). vs the prior bake it wins CID22/TID/
    /// KonJND/AIC-3 and loses KADID (−0.015) / AIC-4 (−0.014). Feature
    /// regime = PU-linear integrated
    /// (`Zensim::compute_pu_linear_extended_features` — absolute nits, no u8
    /// shell, no display-peak anchor; extraction via `zenmetrics score-pairs
    /// --hdr --hdr-features-pu-linear`). Dial spline (18 knots, monotone,
    /// `[0, 95.77]`) is fit on the **SDR** multiband anchor — unlike the
    /// prior bake's HDR-anchored `anchored2` dial; median dial on real HDR
    /// content shifts +20.7 vs the prior bake (rank-invariant, semantics
    /// unvalidated). Full audit:
    /// `benchmarks/bhdr_improvement_split_lineage_2026-07-12.md` §7.
    ///
    /// **HDR content only** — measured INVALID on SDR codec content
    /// (rank agreement 0.72, unbounded extrapolation); route SDR to
    /// [`ZensimProfile::B`] / [`ZensimProfile::A`].
    BHdr,
    /// **`C` — generation-C SDR profile (external name `zensim-c`), the
    /// SOTA-944 campaign's wave-11 ship candidate.** A 944-input MLP
    /// (folded-720+append+append2 feature regime) from the k=8-confirmed
    /// corrected-mix recipe — the first shipped bake trained after the
    /// KADID orientation fix and the first shipped **dead-column-pruned**
    /// bake (caller width 944, internal layer-0 width 667 via
    /// `FeatureTransform::Drop`; the runtime sizes feature vectors by
    /// `Model::caller_input_width()`, never `n_inputs()`).
    ///
    /// Headline panel (bake_verdict `--regime 944`, committed verdict):
    /// CID22 0.8867, KonJND |0.4988|, LIVE 0.9604, CSIQ 0.9331, nonphoto
    /// 0.9251, imazen26 0.9215, HF-NL per-ref 0.7334, dial monotonicity
    /// 99.32% / 0.0% tied **in dial units** (the first packaged 944 cell
    /// to hold the ≥93% dial bar), M3a coherence 0.862 (GOLD ≥0.85),
    /// corruption via the *companion head* `corrhead944_s13` (pass_q20
    /// 0.793 — the 944 dial's own corruption ordering is broken by
    /// design; the head is the owner). Balanced-profile floors 7/8 (the
    /// miss: CID22 B9 band tail 0.139 vs ≥0.15). vs [`Self::B`]: `C`
    /// takes CID22/nonphoto/LIVE/dial-mono/M3a and carries the
    /// corruption head; `B` keeps KonJND (0.519 vs 0.499) and HF-NL
    /// per-ref (0.825 vs 0.733). **`B` remains the default /
    /// [`Self::codec_target`].** Known honest caveat: G-RANGE FAILs on
    /// the CID22-val anchor domain (4.50% above-knot raw mass; near-top
    /// saturation — anchor-densification lever registered, untested).
    ///
    /// **944-regime scoring contract.** `C` consumes the folded-944
    /// feature layout, which the standard 372-feature
    /// [`Zensim::compute`](crate::Zensim::compute) pipeline does not
    /// produce — `compute` on a non-identical pair returns
    /// [`ModelForwardFailed`](crate::ZensimError::ModelForwardFailed)
    /// (byte-identical pairs still short-circuit to 100). Score `C` by
    /// extracting folded-944 features and forwarding them, exactly as
    /// the eval/loop tooling does:
    /// `Zensim::compute_folded720_append2_features` (or the fused
    /// `compute_folded944_score_and_attribution` session entry), both
    /// behind the `feature-regime-v2` feature, then
    /// [`crate::score_features_with_profile`]`(ZensimProfile::C, …)`.
    /// **SDR content only** — `C` is structurally SDR (its HDR-gated
    /// append2 slots are pruned); route HDR content to
    /// [`Self::BHdr`] explicitly.
    ///
    /// Full provenance, training data shas, exact reproduction chain
    /// and verification gates: `docs/PROFILE_C_REPRODUCTION_2026-08-05.md`
    /// and the mapping table in `docs/CODEC_TARGET_METRIC.md`.
    C,
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
    ///
    /// Gated behind the non-default `custom-profiles` feature — the
    /// `zensim-experimental` crate enables it; default builds don't carry
    /// this variant (or the [`ProfileParams::builder`] machinery).
    #[cfg(feature = "custom-profiles")]
    Custom {
        /// Profile parameters: weights, bake bytes, and dispositions.
        params: &'static ProfileParams,
        /// Display name, e.g. `"zensim-experimental-tuner-v4"`.
        name: &'static str,
    },
    /// **INTERNAL — legacy 228-weight linear scoring, no MLP head.**
    /// Tags results from the doc-hidden research/feature-extraction entry
    /// points `compute_zensim_with_config` and
    /// `compute_zensim_with_ref_and_config`, which score via a plain
    /// `dot(features, WEIGHTS_PREVIEW_V0_2)` distance (the caller-supplied
    /// [`crate::ZensimConfig`] drives blur/scale/masking knobs directly) and
    /// never run an MLP forward pass. This is **not** the same pipeline as
    /// [`Self::A`] (MLP-scored) — the tag exists purely so
    /// `ZensimResult::profile()` never claims an MLP-scored result for a
    /// linear-only computation. Gated behind the same `training`/`test` cfg
    /// as the two functions above; default builds never construct or
    /// observe this variant.
    ///
    /// Declared LAST (after `Custom`) so enabling this variant's cfg
    /// (`training`/`test`) never shifts `Custom`'s discriminant when
    /// `custom-profiles` is also enabled — see
    /// `enum_no_repr_variant_discriminant_changed` in
    /// `cargo semver-checks`.
    #[cfg(any(feature = "training", test))]
    #[doc(hidden)]
    LegacyLinearV0_2,
}

impl ZensimProfile {
    /// Current preview-stable profile. Returns [`Self::B`] (the
    /// deterministic generation-B linear profile; generation-A is
    /// deprecated).
    ///
    /// Use this only when you explicitly want "whatever the current
    /// preview is" — the returned variant will rotate as new previews
    /// ship. For pinned reproducibility, name the variant directly
    /// (`ZensimProfile::B`). For the stable codec-target contract that
    /// codec crates should target, use [`Self::codec_target`] (also `B`).
    pub const fn latest_preview() -> Self {
        Self::B
    }

    /// Deprecated. Use [`Self::latest_preview`] for the rotating-preview
    /// alias, [`Self::codec_target`] for the stable codec contract, or
    /// name [`Self::B`] directly for pinned reproducibility.
    #[deprecated(
        since = "0.3.0",
        note = "use latest_preview() or codec_target() (now `B`) or name ZensimProfile::B directly"
    )]
    pub fn latest() -> Self {
        Self::B
    }

    /// **Canonical codec-target metric.** The stable, version-independent
    /// alias for "the bake all zen codecs train and target to."
    /// Returns [`Self::B`] — the deterministic generation-B linear profile
    /// (generation-A, the prior return value, is deprecated).
    ///
    /// Codec crates (`zenjpeg`, `zenwebp`, `zenjxl`, `zenavif`, ...)
    /// should construct their `Zensim` instance via
    /// `Zensim::new(ZensimProfile::codec_target())` so that profile
    /// rotations flow through automatically without per-codec version edits.
    ///
    /// **Use cases this is purpose-built for:**
    /// - **Quality-target dial** — `Zensim::compute(source, distorted)`
    ///   inside an iterative encode-rescore-adjust outer loop
    ///   (see `zenwebp::EncodeConfig::target_zensim` for the reference
    ///   pattern; pattern is ~100 LOC per codec). The user names a target
    ///   score, the codec binary-searches the quality knob that hits it.
    /// - **Picker training** — train cross-codec pickers against this
    ///   profile's per-codec score parquets.
    ///
    /// **Why `B` for the dial (measured 2026-07-11 with real encoders,
    /// `benchmarks/b_knob_validation_real_encoders_2026-07-11.md`):**
    /// - **Mechanics:** |ρ| = 0.953 vs codec quality, 79.9 % strict-monotone
    ///   q-steps — a search converges.
    /// - **Consistency at a target:** at a fixed dial value the true
    ///   quality (MOS) scatter is resid-SD ±6.33 score units — on par with
    ///   ssim2 (±6.04) and tighter than deprecated `A` (±6.60).
    /// - **Reach:** normalized reachable-dial span 0.68 (vs `A` 0.64,
    ///   ssim2 0.40) — more of the 0–100 dial is actually addressable.
    /// - **Independent-reference consistency at scale:** on ~1 M codec
    ///   cells, η²(butteraugli | dial-decile) = 0.582 (vs `A` 0.344) — a
    ///   fixed `B` value pins an independent perceptual reference far
    ///   tighter than `A` does.
    /// - **Deterministic + byte-reproducible** (closed-form linear fit, no
    ///   training seed) — the MLP family's collapse mode structurally
    ///   cannot occur. Repro: `benchmarks/profile_b_methodology_2026-07-12.md`.
    ///
    /// **What this is NOT for:**
    /// - In-encoder per-block RDO distortion term. zensim is per-image
    ///   (~14 ms at 1024² × 5–20k RDO calls = 70–280 s/image, infeasible).
    ///   See `docs/RDO_LOSS_FEASIBILITY_2026-05-24.md`.
    ///
    /// **Known dial limits (honest gaps):**
    /// - On jxl/webp the 85–92 top-shoulder band is compressed (a raw-output
    ///   ceiling; recalibratable only by trading away calibration). Per-codec
    ///   reachable zones + the JXL butteraugli-distance→dial table are in
    ///   `benchmarks/jxl_nearlossless_dial_2026-07-05.md` (Part 11).
    /// - `B` ↔ `A` is a **trade**, not strict dominance: `B` wins the human-MOS
    ///   holdouts (CID22 0.876, AIC-3/AIC-4) while `A` wins raw ssim2-agreement
    ///   on the ~1 M-cell codec sweeps. `B` is the better *dial*; both are
    ///   documented in the methodology doc.
    ///
    /// **SDR path uses `B`; HDR (absolute-nits) content routes to the
    /// `BHdr` weights automatically** via `params_pu_linear` — one dial,
    /// two domains. See `docs/CODEC_TARGET_METRIC.md` for the integration
    /// guide.
    pub const fn codec_target() -> Self {
        Self::B
    }

    /// Canonical name string, e.g. `"zensim-b"`.
    #[allow(deprecated)] // matches the deprecated `A` arm when the feature is on
    pub fn name(&self) -> &'static str {
        match self {
            #[cfg(feature = "deprecated-profiles")]
            Self::A => "zensim-a",
            Self::PreviewV0_1 => "zensim-preview-v0.1",
            Self::PreviewV0_2 => "zensim-preview-v0.2",
            Self::B => "zensim-b",
            Self::BHdr => "zensim-b-hdr",
            Self::C => "zensim-c",
            #[cfg(any(feature = "training", test))]
            Self::LegacyLinearV0_2 => "zensim-legacy-linear-v0.2",
            // Experimental / historical profiles live in `zensim-experimental`
            // and surface here as `Custom` with their original name string.
            #[cfg(feature = "custom-profiles")]
            Self::Custom { name, .. } => name,
        }
    }

    /// Internal parameters for the ABSOLUTE-LUMINANCE (PU-linear) entry
    /// points. Generation-B routes here: [`ZensimProfile::B`] dispatches to
    /// the `BHdr` weights when fed nits, so one profile serves both domains
    /// with each bake on its own feature distribution — the invalid pairing
    /// (SDR weights on PU features) becomes unrepresentable through `B`.
    ///
    /// Routing keys on the ENTRY PATH (the caller already declared the
    /// domain by decoding to nits), never on pixel values: value-sniffing
    /// (e.g. peak-nits thresholds) would flip models across a threshold
    /// with ~5-10pt cross-model per-pair scatter — a visible seam
    /// (measured 2026-07-04, benchmarks/provenance_best_results doc).
    /// `A`, `BHdr` and `C` are unrouted (A has one bake; BHdr is the
    /// explicit HDR handle for callers who want no routing; `C` is
    /// structurally SDR-only — on the PU path its 944-wide bake cannot
    /// consume the 372-wide PU features, so scoring fails loud instead
    /// of silently borrowing another generation's HDR weights).
    pub(crate) fn params_pu_linear(&self) -> &'static ProfileParams {
        match self {
            Self::B => &PROFILE_B_HDR,
            _ => self.params(),
        }
    }

    /// Internal parameters for this profile.
    #[allow(deprecated)] // matches the deprecated `A` arm when the feature is on
    pub(crate) fn params(&self) -> &'static ProfileParams {
        match self {
            #[cfg(feature = "deprecated-profiles")]
            Self::A => &PROFILE_A,
            Self::PreviewV0_1 => &PROFILE_PREVIEW_V0_1,
            Self::PreviewV0_2 => &PROFILE_PREVIEW_V0_2,
            Self::B => &PROFILE_B,
            Self::BHdr => &PROFILE_B_HDR,
            Self::C => &PROFILE_C,
            #[cfg(any(feature = "training", test))]
            Self::LegacyLinearV0_2 => &PROFILE_LEGACY_LINEAR_V0_2,
            // Experimental / historical profiles carry their own
            // `&'static ProfileParams` built in `zensim-experimental`.
            #[cfg(feature = "custom-profiles")]
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
            #[cfg(feature = "custom-profiles")]
            (
                ZensimProfile::Custom {
                    params: a,
                    name: na,
                },
                ZensimProfile::Custom {
                    params: b,
                    name: nb,
                },
            ) => core::ptr::eq(*a, *b) && na == nb,
            #[cfg(feature = "custom-profiles")]
            (ZensimProfile::Custom { .. }, _) | (_, ZensimProfile::Custom { .. }) => false,
            _ => core::mem::discriminant(self) == core::mem::discriminant(other),
        }
    }
}

impl Eq for ZensimProfile {}

impl core::hash::Hash for ZensimProfile {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        #[cfg(feature = "custom-profiles")]
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

#[cfg(feature = "custom-profiles")]
impl ProfileParams {
    /// Start building a custom [`ProfileParams`] for use with
    /// [`ZensimProfile::Custom`].
    ///
    /// This is the supported extension point for profiles defined outside
    /// this crate (e.g. the unpublished `zensim-experimental` crate, which
    /// preserves the historical research bakes). Defaults match the linear
    /// baseline backed by [`WEIGHTS_PREVIEW_V0_2`] (`blur_radius = 5`,
    /// `blur_passes = 1`, `num_scales = 4`, score mapping `100 − 18·d^0.7`,
    /// all dispositions off, no MLP). Override only what differs:
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
#[cfg(feature = "custom-profiles")]
#[derive(Debug)]
pub struct ProfileParamsBuilder {
    inner: ProfileParams,
}

#[cfg(feature = "custom-profiles")]
impl ProfileParamsBuilder {
    /// Linear scoring weights (one per feature). Defaults to the
    /// [`WEIGHTS_PREVIEW_V0_2`] vector; unused on the MLP path but kept
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
///
/// Gated behind the default-on `deprecated-profiles` feature (with the
/// deprecated [`ZensimProfile::A`] variant it backs).
#[cfg(feature = "deprecated-profiles")]
pub(crate) fn mlp_bake_a_v47_qat() -> &'static [u8] {
    include_bytes!("../weights/v47_strict_qat_native_2026-05-27.bin")
}

/// Generation-A profile params (external `ZensimProfile::A`). Backing
/// bake recorded in the mapping table in `docs/CODEC_TARGET_METRIC.md`.
/// Gated behind the default-on `deprecated-profiles` feature.
#[cfg(feature = "deprecated-profiles")]
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

/// `ZensimProfile::B` bake bytes — `ens-Pline-cid80`, INCLUSIVE-WINSOR + DENSE-DIAL
/// (7.3 KB, sha256 `b6fe5233…`, rotated 2026-07-07 from the 2026-07-05
/// `b78adb15` bake — same raw weights/spline, only the winsor fit corpus
/// changed; see the near-lossless fix in guard 1). Two rank-invariant
/// guards on the raw linear ensemble, both via the standard metadata:
///
/// 1. **Winsor tail guard** — 372 `winsor_p99` `zentrain.feature_transforms`
///    (fit-corpus [p0.1, p99.9] per feature), applied by
///    `Predictor::predict_transformed` (the SAME mechanism `BHdr` uses).
///    Clips the tail, bounding raw output inside the spline domain and
///    eliminating the f155 tiny-screen pathology (f155 p99.9 = 0.776 still
///    catches the 14,532 offender). CID22 0.8732 -> 0.8764, KonJND 0.5466.
///    **Fit corpus is near-lossless-INCLUSIVE (2026-07-07):** the predecessor's
///    hdr_v3mix-only bounds clamped 245/372 features CONSTANT across SDR
///    near-lossless, pinning that dial band at ~91.5; adding the zenjxl
///    near-lossless SDR sweep to the corpus frees 340/372 lower bounds so the
///    features pass through (dial climbs to ~96.1, matching ssim2 ~96;
///    near-lossless per-image rank-vs-ssim2 0.657 -> 0.771), f155's upper
///    guard unmoved. Recipe: scripts/v_next/build_inclusive_winsor_corpus.py.
///    See benchmarks/jxl_nearlossless_dial_2026-07-05.md §7 + §8.
/// 2. **Dense-dial spline** — ONLY the output spline's TOP is extended, by the
///    training-fitted concave saturation (`dense_dial_refit_b.py`, decay
///    k=3.31 fit on 600 near-lossless multiband-anchor rows), so near-lossless
///    codec-knob configs (raw up to ~2.8, above any training/real-content raw
///    ~1.12) resolve toward 100 instead of piling at the top knot. The winsor
///    bake's bottom + in-distribution knots are kept VERBATIM, so both raw
///    tails stay inside the knot domain [-1.97, 3.92] (outlier gate G-RANGE
///    PASS, 0 extrapolating). Clears the dial dead-zone (0.0563 FAIL -> 0.0005
///    PASS, all G3 dial gates green) with rank IDENTICAL to the winsor-only
///    bake (spline is monotone => rank-invariant: CID22 0.8764 / KonJND 0.5466
///    / AIC-3 0.7774; inversions ~0.026 ~= the winsor bake's).
///    Identity=100 is guaranteed by the `is_identical` short-circuit, not the
///    spline top. See benchmarks/provenance "best-of-both dial probe".
pub(crate) fn linear_bake_b_cid80() -> &'static [u8] {
    include_bytes!("../weights/b_sdr_linear_cid80_inclwinsor_dense_dial_2026-07-07.bin")
}

/// `ZensimProfile::BHdr` bake bytes — `hdrmix-lasso0.0003-shaped`
/// (11.8 KB, sha256 `7d7f2123…`): the shaped PU-linear HDR pick, trained on
/// the **cvvdp-mix** target (`0.5·ssim2 + 0.5·(JOD−6)/4`). Carries the mixed
/// `winsor_p99` / `quantile_bins` / `signed_cbrt` / `yeo_johnson` shape
/// recipe, so it was never exposed to the raw-feature tail that afflicted B
/// SDR. Promoted 2026-07-12 from the prior pure-ssim2 `anchored2` bake
/// (`373eac56…`, preserved at `weights/bhdr_linear_shaped_anchored2_2026-07-04.bin`).
/// ⚠ Same-day audit: the UPIQ lift 0.7313→0.7536 is a scoreboard-selected
/// point estimate (maxT-adjusted p=0.22 — NOT established); vs the prior
/// bake it wins CID22/TID/KonJND/AIC-3, loses KADID/AIC-4, and its dial is
/// SDR-anchored (the prior bake's was HDR-anchored). Full audit + numbers:
/// `benchmarks/bhdr_improvement_split_lineage_2026-07-12.md` §7.
pub(crate) fn linear_bake_bhdr_shaped() -> &'static [u8] {
    include_bytes!("../weights/bhdr_linear_shaped_cvvdpmix_2026-07-12.bin")
}

/// Generation-B SDR profile params. Same 372-feature front end as
/// `PROFILE_A`; the bake is a single identity-activation linear layer with
/// its own anchored dial spline, so the MLP forward path drives it
/// unchanged.
static PROFILE_B: ProfileParams = ProfileParams {
    weights: &WEIGHTS_PREVIEW_V0_2,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    bounded_squash: false,
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    skip_score_mapping: true,
    mlp_bytes: Some(linear_bake_b_cid80),
    mlp_bytes_b3: None,
    mlp_primary_mix: 1.0,
    extended_features: true,
    compute_iw_features: true,
    soft_clamp_score: false,
    extrapolate_score: true,
    ensemble_classifier_bytes: None,
    mlp_bytes_compression: None,
};

/// Generation-B HDR profile params. Identical runtime shape to `PROFILE_B`;
/// the shaped-feature dispatch happens via the bake's transform metadata.
/// Callers MUST feed absolute-luminance content through the PU-linear
/// feature path (see the `BHdr` variant docs) — SDR sRGB input is
/// out-of-domain for this bake.
static PROFILE_B_HDR: ProfileParams = ProfileParams {
    weights: &WEIGHTS_PREVIEW_V0_2,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    bounded_squash: false,
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    skip_score_mapping: true,
    mlp_bytes: Some(linear_bake_bhdr_shaped),
    mlp_bytes_b3: None,
    mlp_primary_mix: 1.0,
    extended_features: true,
    compute_iw_features: true,
    soft_clamp_score: false,
    extrapolate_score: true,
    ensemble_classifier_bytes: None,
    mlp_bytes_compression: None,
};

/// `ZensimProfile::C` bake bytes — internal `W10L9_s4003_packed`
/// (`c_sdr_mlp944_corrmix_2026-08-05.bin`, 165,696 B, sha256
/// `1a2c8d522fed8034b279ff018aa052f19d0b9f419f12cf22cca303a0b4abb7f4`).
/// The SOTA-944 wave-11 battery-selected cell: 944 → 128 → 1 MLP
/// (f16 + zerobias-0.005, LeakyReLU), trained seed 4003 on the 10-group
/// corrected mix (corrected `ext_kadid`, KonJND-BPG leg, teacher tables,
/// `tkadis` dropped), dial-packaged (`bake_dial_refit add-spline` on
/// `anchor944_dial.parquet`) then packed with dead-column pruning
/// (944 caller lines → 667 internal; 277 all-class-1 drops, identity
/// gate BIT-identical on 2,035 anchor rows). Carries its full
/// `zentrain.repro` (input shas + seed + argv) embedded in the bake
/// metadata. Selection: `freeze_check --select` over the k=8 family
/// (7/8 floors, sel_comp 0.9579, M3a 0.8626 GOLD tie-break).
/// Records: campaign appendix K (`benchmarks/sota944_campaign_2026-08-03.md`)
/// and `docs/PROFILE_C_REPRODUCTION_2026-08-05.md`. The pinning test
/// `profile_c_tests::weight_sha256_pinned` fails loud on any silent
/// byte swap of this file.
pub(crate) fn mlp_bake_c_corrmix_944() -> &'static [u8] {
    include_bytes!("../weights/c_sdr_mlp944_corrmix_2026-08-05.bin")
}

/// Generation-C SDR profile params. The bake is a 944-input (pruned to
/// 667 internal) MLP over the folded-720+append+append2 feature regime —
/// NOT the 372-feature v1 pipeline — so the `extended_features` /
/// `compute_iw_features` flags below only shape the legacy `compute()`
/// path's extraction (whose 372-wide vector the 944 bake refuses,
/// failing loud). Full-fidelity scoring goes through the folded-944
/// extraction + [`crate::score_features_with_profile`]; see the
/// [`ZensimProfile::C`] docs for the contract.
static PROFILE_C: ProfileParams = ProfileParams {
    weights: &WEIGHTS_PREVIEW_V0_2,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    bounded_squash: false,
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    // The bake carries its own monotone PCHIP dial spline
    // (`zentrain.output_calibration_spline`, refit on the packed net);
    // raw output is dial-honest after the spline applies.
    skip_score_mapping: true,
    mlp_bytes: Some(mlp_bake_c_corrmix_944),
    mlp_bytes_b3: None,
    mlp_primary_mix: 1.0,
    extended_features: true,
    compute_iw_features: true,
    soft_clamp_score: false,
    // Negative dial tail is part of the contract (pack --neg-tail):
    // worse-than-worst-codec inputs score below 0 instead of tying at 0.
    extrapolate_score: true,
    ensemble_classifier_bytes: None,
    mlp_bytes_compression: None,
};

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
// Preferred name going forward — the explicit `LINEAR_` prefix
// disambiguates from V0_4's MLP weights, which ship as packed ZNPR v2
// bytes rather than a flat coefficient array. The unprefixed
// `WEIGHTS_PREVIEW_V0_X` names are kept indefinitely for source
// compatibility with code written against zensim 0.2.x and earlier.

/// Alias for [`WEIGHTS_PREVIEW_V0_1`]. Linear V0.1 coefficients (228).
/// The `LINEAR_` prefix disambiguates these linear coefficients from
/// MLP bake bytes.
pub use self::WEIGHTS_PREVIEW_V0_1 as LINEAR_WEIGHTS_PREVIEW_V0_1;

/// Alias for [`WEIGHTS_PREVIEW_V0_2`]. Linear scoring weights (228). The
/// `LINEAR_` prefix disambiguates these linear coefficients from MLP bake
/// bytes. Kept indefinitely for source compatibility with code written
/// against zensim 0.2.x and earlier.
pub use self::WEIGHTS_PREVIEW_V0_2 as LINEAR_WEIGHTS_PREVIEW_V0_2;

#[cfg(test)]
mod profile_b_tests {
    use super::*;
    use crate::{PixelFormat, StridedBytes, Zensim};

    fn synth(w: usize, h: usize, seed: u32) -> Vec<u8> {
        let mut px = vec![0u8; w * h * 3];
        for (i, v) in px.iter_mut().enumerate() {
            *v = (((i as u32).wrapping_mul(2654435761).wrapping_add(seed)) >> 24) as u8;
        }
        px
    }

    #[test]
    fn profile_b_loads_scores_and_holds_identity() {
        let (w, h) = (64usize, 64usize);
        let refe = synth(w, h, 7);
        let mut dist = refe.clone();
        for v in dist.iter_mut().step_by(5) {
            *v = v.saturating_sub(20);
        }
        for profile in [ZensimProfile::B, ZensimProfile::BHdr] {
            let z = Zensim::new(profile);
            let src = StridedBytes::try_new(&refe, w, h, w * 3, PixelFormat::Srgb8Rgb).unwrap();
            let dst = StridedBytes::try_new(&dist, w, h, w * 3, PixelFormat::Srgb8Rgb).unwrap();
            let ident = z.compute(&src, &src).unwrap().score();
            assert!(
                (ident - 100.0).abs() < 1e-9,
                "{profile}: identity must be 100, got {ident}"
            );
            let s = z.compute(&src, &dst).unwrap().score();
            assert!(
                s.is_finite() && s < ident && s > -50.0,
                "{profile}: distorted score sane+below identity, got {s}"
            );
        }
        assert_eq!(ZensimProfile::B.name(), "zensim-b");
        assert_eq!(ZensimProfile::BHdr.name(), "zensim-b-hdr");
    }

    /// Both shipped B bakes MUST carry `feature_transforms` so the runtime
    /// dispatches `predict_transformed` and winsorizes the input tail. B
    /// SDR carries 372 `winsor_p99`; BHdr carries the mixed shape recipe.
    /// This guards against a future re-bake silently reverting to the raw
    /// `predict` path — which is exactly what exposed the f155
    /// tiny-screen-content tail (raw min -1131) before 2026-07-05. See
    /// benchmarks/provenance_best_results_2026-07-04.md §f155.
    #[test]
    fn both_b_bakes_carry_winsorizing_transforms() {
        for profile in [ZensimProfile::B, ZensimProfile::BHdr] {
            let loader = profile.params().mlp_bytes.expect("B/BHdr ship an mlp bake");
            let model = crate::mlp::Model::from_bytes(loader()).expect("bake parses");
            assert!(
                model.has_nontrivial_feature_transforms(),
                "{profile}: bake must carry feature_transforms (winsor guard); \
                 a raw bake reintroduces the f155 tail"
            );
        }
    }
}

#[cfg(test)]
mod profile_b_routing_tests {
    use super::*;
    use crate::Zensim;

    #[test]
    fn b_routes_to_hdr_weights_on_the_nits_path() {
        let (w, h) = (64usize, 64usize);
        let mut refe = vec![0.0f32; w * h * 3];
        for (i, v) in refe.iter_mut().enumerate() {
            *v = 1.0 + 180.0 * ((i % 97) as f32 / 97.0);
        }
        let mut dist = refe.clone();
        for v in dist.iter_mut().skip(1).step_by(4) {
            *v *= 0.7;
        }
        let sb = Zensim::new(ZensimProfile::B)
            .compute_pu_linear(&refe, &dist, w, h, w * 3, w * 3)
            .unwrap()
            .score();
        let sh = Zensim::new(ZensimProfile::BHdr)
            .compute_pu_linear(&refe, &dist, w, h, w * 3, w * 3)
            .unwrap()
            .score();
        assert!(
            (sb - sh).abs() < 1e-12,
            "B on nits must equal BHdr (routed): {sb} vs {sh}"
        );
        // Identity via the routed path still honors the contract.
        let ident = Zensim::new(ZensimProfile::B)
            .compute_pu_linear(&refe, &refe, w, h, w * 3, w * 3)
            .unwrap()
            .score();
        assert!((ident - 100.0).abs() < 1e-9);
    }
}

#[cfg(test)]
mod descriptor_hdr_routing_tests {
    use super::*;
    use crate::Zensim;
    use crate::source::{AlphaMode, ImageSource, PixelFormat};

    struct NitsSource {
        rgba: Vec<f32>,
        w: usize,
        h: usize,
    }
    impl ImageSource for NitsSource {
        fn width(&self) -> usize {
            self.w
        }
        fn height(&self) -> usize {
            self.h
        }
        fn pixel_format(&self) -> PixelFormat {
            PixelFormat::LinearF32Rgba
        }
        fn alpha_mode(&self) -> AlphaMode {
            AlphaMode::Opaque
        }
        fn is_hdr(&self) -> bool {
            true
        }
        fn row_bytes(&self, y: usize) -> &[u8] {
            bytemuck::cast_slice(&self.rgba[y * self.w * 4..(y + 1) * self.w * 4])
        }
    }

    fn nits_pair(w: usize, h: usize) -> (NitsSource, NitsSource, Vec<f32>, Vec<f32>) {
        let mut r = vec![0.0f32; w * h * 4];
        for (i, v) in r.iter_mut().enumerate() {
            *v = if i % 4 == 3 {
                1.0
            } else {
                0.5 + 400.0 * ((i % 89) as f32 / 89.0)
            };
        }
        let mut d = r.clone();
        for v in d.iter_mut().step_by(9) {
            *v *= 0.8;
        }
        let rgb =
            |px: &[f32]| -> Vec<f32> { px.chunks_exact(4).flat_map(|c| c[..3].to_vec()).collect() };
        let (rr, dd) = (rgb(&r), rgb(&d));
        (
            NitsSource { rgba: r, w, h },
            NitsSource { rgba: d, w, h },
            rr,
            dd,
        )
    }

    #[test]
    fn compute_routes_descriptor_flagged_hdr_to_pu_linear() {
        let (w, h) = (64usize, 64usize);
        let (src, dst, r_rgb, d_rgb) = nits_pair(w, h);
        // B always; the deprecated A only when its feature is on.
        #[allow(deprecated)]
        let profiles: &[ZensimProfile] = &[
            ZensimProfile::B,
            #[cfg(feature = "deprecated-profiles")]
            ZensimProfile::A,
        ];
        for &profile in profiles {
            let z = Zensim::new(profile);
            let via_compute = z.compute(&src, &dst).unwrap().score();
            let via_pu = z
                .compute_pu_linear(&r_rgb, &d_rgb, w, h, w * 3, w * 3)
                .unwrap()
                .score();
            assert!(
                (via_compute - via_pu).abs() < 1e-12,
                "{profile}: compute() must route to pu_linear: {via_compute} vs {via_pu}"
            );
        }
        // B on descriptor-HDR == BHdr (the routed weights).
        let b = Zensim::new(ZensimProfile::B)
            .compute(&src, &dst)
            .unwrap()
            .score();
        let bh = Zensim::new(ZensimProfile::BHdr)
            .compute(&src, &dst)
            .unwrap()
            .score();
        assert!((b - bh).abs() < 1e-12);
    }
}

#[cfg(test)]
mod linearf32_sdr_not_hdr_tests {
    use super::*;
    use crate::Zensim;
    use crate::source::{AlphaMode, ImageSource, PixelFormat};

    /// LinearF32Rgba is a CONTAINER, not an HDR signal — SDR content ships
    /// in it too (display-relative [0,1] linear). Only `is_hdr()` routes.
    struct F32Source {
        rgba: Vec<f32>,
        w: usize,
        h: usize,
        hdr: bool,
    }
    impl ImageSource for F32Source {
        fn width(&self) -> usize {
            self.w
        }
        fn height(&self) -> usize {
            self.h
        }
        fn pixel_format(&self) -> PixelFormat {
            PixelFormat::LinearF32Rgba
        }
        fn alpha_mode(&self) -> AlphaMode {
            AlphaMode::Opaque
        }
        fn is_hdr(&self) -> bool {
            self.hdr
        }
        fn row_bytes(&self, y: usize) -> &[u8] {
            bytemuck::cast_slice(&self.rgba[y * self.w * 4..(y + 1) * self.w * 4])
        }
    }

    #[test]
    fn flag_not_format_decides_the_pipeline() {
        let (w, h) = (64usize, 64usize);
        let mut r = vec![0.0f32; w * h * 4];
        for (i, v) in r.iter_mut().enumerate() {
            *v = if i % 4 == 3 {
                1.0
            } else {
                0.05 + 0.9 * ((i % 83) as f32 / 83.0)
            };
        }
        let mut d = r.clone();
        for v in d.iter_mut().step_by(7) {
            *v *= 0.75;
        }
        let z = Zensim::new(ZensimProfile::B);
        let mk = |rgba: &Vec<f32>, hdr: bool| F32Source {
            rgba: rgba.clone(),
            w,
            h,
            hdr,
        };
        // SDR-flagged linear f32: flows the SDR pipeline (no refusal, no PU).
        let sdr = z.compute(&mk(&r, false), &mk(&d, false)).unwrap().score();
        // Same bytes HDR-flagged: routed to the PU-linear pipeline.
        let hdr = z.compute(&mk(&r, true), &mk(&d, true)).unwrap().score();
        assert!(sdr.is_finite() && hdr.is_finite());
        assert!(
            (sdr - hdr).abs() > 1e-6,
            "identical bytes must take DIFFERENT pipelines when only the \
             descriptor flag differs (sdr={sdr}, hdr={hdr}); equality would \
             mean the format, not the flag, was routing"
        );
        // Identity contract holds through both pipelines.
        for hdrflag in [false, true] {
            let s = z
                .compute(&mk(&r, hdrflag), &mk(&r, hdrflag))
                .unwrap()
                .score();
            assert!((s - 100.0).abs() < 1e-9);
        }
    }
}

#[cfg(test)]
mod profile_c_tests {
    use super::*;
    use crate::source::RgbSlice;
    use crate::{Zensim, ZensimError};

    /// Deterministic SDR test pair: a structured gradient+texture image
    /// and a distorted copy (every 7th byte attenuated). Same shape as
    /// the sibling profile test fixtures.
    fn sdr_pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
        let mut src = vec![[0u8; 3]; w * h];
        for (i, px) in src.iter_mut().enumerate() {
            let x = i % w;
            let y = i / w;
            px[0] = ((x * 255) / w.max(1)) as u8;
            px[1] = ((y * 255) / h.max(1)) as u8;
            px[2] = (((x ^ y) * 7) % 256) as u8;
        }
        let mut dst = src.clone();
        for (i, px) in dst.iter_mut().enumerate() {
            if i % 7 == 0 {
                px[0] = (px[0] as u16 * 3 / 4) as u8;
                px[1] = px[1].saturating_add(9);
            }
        }
        (src, dst)
    }

    /// The shipped `C` weight file is pinned by sha256 — a silent byte
    /// swap of `c_sdr_mlp944_corrmix_2026-08-05.bin` fails this test
    /// loudly. Expected digest = the wave-11 battery's committed
    /// `W10L9_s4003_packed.bin` (campaign appendix K.R, verdict
    /// `bake_sha256`, Tower-mirror spot-check — all the same bytes).
    #[test]
    fn weight_sha256_pinned() {
        use sha2::{Digest, Sha256};
        let bytes = mlp_bake_c_corrmix_944();
        assert_eq!(bytes.len(), 165_696, "C weight byte length changed");
        let mut hasher = Sha256::new();
        hasher.update(bytes);
        let digest = hasher.finalize();
        let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(
            hex, "1a2c8d522fed8034b279ff018aa052f19d0b9f419f12cf22cca303a0b4abb7f4",
            "C weight bytes do not match the pinned wave-11 W10L9_s4003_packed sha256"
        );
    }

    /// `C` is the first PRUNED shipped bake: the caller-facing width is
    /// 944 (the folded-944 feature layout) while the internal layer-0
    /// width is 667 (`FeatureTransform::Drop` on the dead raw lines).
    /// The runtime MUST size feature vectors by `caller_input_width()`,
    /// never `n_inputs()` (metric.rs fix `ae852b1b`).
    #[test]
    fn bake_loads_caller_width_944_internal_667() {
        let model = crate::mlp::Model::from_bytes(mlp_bake_c_corrmix_944())
            .expect("shipped C bake must parse");
        assert_eq!(model.caller_input_width(), 944, "caller-facing width");
        assert_eq!(model.n_inputs(), 667, "internal (pruned) layer-0 width");
        assert_eq!(model.n_outputs(), 1);
        assert_eq!(model.n_layers(), 2);
    }

    /// Standard identity fixture: byte-identical inputs score exactly
    /// 100 through the `is_identical` short-circuit (profile-independent
    /// contract; mirrors the B/BHdr identity tests).
    #[test]
    fn identity_fixture_scores_100() {
        let (src, _) = sdr_pair(64, 64);
        let z = Zensim::new(ZensimProfile::C);
        let score = z
            .compute(&RgbSlice::new(&src, 64, 64), &RgbSlice::new(&src, 64, 64))
            .expect("identity compute")
            .score();
        assert!(
            (score - 100.0).abs() < 1e-9,
            "identity must score 100, got {score}"
        );
    }

    /// Pin the documented limitation: the standard 372-feature
    /// `compute()` pipeline cannot feed the 944-wide `C` bake, so a
    /// non-identical pair fails LOUD (`ModelForwardFailed`) instead of
    /// silently scoring a prefix. Full-fidelity scoring goes through the
    /// folded-944 extraction + `score_features_with_profile` (see the
    /// `ZensimProfile::C` docs).
    #[test]
    fn compute_on_non_identical_pair_fails_loud() {
        let (src, dst) = sdr_pair(64, 64);
        let z = Zensim::new(ZensimProfile::C);
        let err = z
            .compute(&RgbSlice::new(&src, 64, 64), &RgbSlice::new(&dst, 64, 64))
            .expect_err("372-wide extraction must not silently feed a 944 bake");
        assert!(
            matches!(err, ZensimError::ModelForwardFailed { .. }),
            "expected ModelForwardFailed, got {err:?}"
        );
    }

    /// Default-build forward smoke over the real shipped bytes: a
    /// caller-width (944) vector loads, transforms (winsor / signed-cbrt
    /// / 277 Drop lines), standardizes, forwards, and lands on the dial
    /// spline — finite output, capped at ≤100 by the product runtime.
    #[test]
    fn forward_zero_vector_is_finite() {
        let features = vec![0.0f64; 944];
        let score = crate::score_features_with_profile(ZensimProfile::C, &features, 64, 64)
            .expect("caller-width vector must forward");
        assert!(score.is_finite(), "got {score}");
        assert!(score <= 100.0 + 1e-9, "runtime caps at 100, got {score}");
    }

    /// End-to-end 944-regime sanity on the real extraction: identical
    /// pair scores high (>= 90) and strictly above a visibly distorted
    /// pair. This is the profile's actual scoring contract (folded-944
    /// features -> score_features_with_profile), the same call shape
    /// bake_verdict / the jxl loop use.
    #[cfg(feature = "feature-regime-v2")]
    #[test]
    fn folded944_end_to_end_scores_sanely() {
        let (src, dst) = sdr_pair(128, 128);
        let z = Zensim::new(ZensimProfile::C);
        let same = RgbSlice::new(&src, 128, 128);
        let iden = z
            .compute_folded720_append2_features(&same, &same)
            .expect("folded-944 extraction (identical)");
        assert_eq!(iden.features().len(), 944);
        let s_iden =
            crate::score_features_with_profile(ZensimProfile::C, iden.features(), 128, 128)
                .expect("forward identical");
        let distorted = RgbSlice::new(&dst, 128, 128);
        let dist = z
            .compute_folded720_append2_features(&same, &distorted)
            .expect("folded-944 extraction (distorted)");
        let s_dist =
            crate::score_features_with_profile(ZensimProfile::C, dist.features(), 128, 128)
                .expect("forward distorted");
        assert!(s_iden.is_finite() && s_dist.is_finite());
        // The folded-944 regime carries REFERENCE-ONLY conditioner slots
        // (content descriptors), so an identical pair is NOT the zero
        // vector and the MLP reads content-conditioned quality: on this
        // deliberately off-manifold synthetic xor-texture the identical
        // pair measures 87.87 (real-content identity sits near the dial
        // top; dial p95 = 96.7 on the 944 dial grid). The floor below is
        // a sanity bound against sign flips / garbage forwards, not a
        // calibration claim.
        assert!(
            (80.0..=100.0 + 1e-9).contains(&s_iden),
            "identical-pair 944 features must land in the sane high band, got {s_iden}"
        );
        assert!(
            s_dist < s_iden,
            "distorted must score below identical ({s_dist} !< {s_iden})"
        );
    }
}
