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
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ZensimProfile {
    /// Preview v0.1. Trained on 344k synthetic pairs, 5-fold CV SROCC=0.9936.
    /// Linear-weights profile, no MLP forward pass.
    PreviewV0_1,
    /// Preview v0.2. Concordance-filtered 218k pairs, Nelder-Mead SROCC=0.9960.
    /// Linear-weights profile, no MLP forward pass. The historical
    /// stable default — `latest()` returned this through zensim 0.2.x.
    PreviewV0_2,
    /// Preview v0.3. MLP-scored profile, shipped 2026-05-13 with
    /// zensim 0.3.0. Underlying bake at crate version 0.3.0 is **V0_18**:
    /// 228→384→1 LeakyReLU MLP, I8 quantized with per-output f32 scales,
    /// 93 KB bake (md5 `2cc537470e68f7379e759811ddd22900`). Built by
    /// 3-way concat construction over V0_16 + cycle-14-s1 + cycle-14-s42
    /// (mathematically equivalent to averaging three 228→128→1 MLPs).
    ///
    /// Cross-corpus SROCC (held-out): CID22 0.8934, KADID 0.9427, TID 0.9525,
    /// AIC-3 0.7998, AIC-4 0.9153.
    ///
    /// Methodology: `benchmarks/v0_18_methodology_2026-05-13.md`.
    ///
    /// The bake bytes inside this variant may be swapped during 0.3.x
    /// patches (e.g. to a zero-biased + LZ4-compressed V0_18 once that
    /// path lands per the `zenpredict` roadmap). Cross-patch score
    /// stability is the contract; bit-identity is not.
    PreviewV0_3,
    /// Preview v0.4. **Multi-bake D2 α=0.7 ensemble (2026-05-15)**:
    /// V_18 ship as the primary (228 → 384 → 1, no feature transforms,
    /// affine-calibrated to MCOS 0..100) + V_20 input-shaping seed=1
    /// (228 → 128 → 1, 98 per-feature transforms, calibrated to same
    /// scale) as the secondary, raw-output linear mix at α=0.7.
    ///
    /// Per `benchmarks/v0_20_three_directions_summary_2026-05-15.md`,
    /// the α=0.7 mix gives the Pareto-optimal CID22 trade vs V_18 alone:
    ///   - CID22 aggregate: 0.8890 (V_18 ship 0.8934, −0.004)
    ///   - CID22 B3 [30, 40) priority band: 0.1047 (V_18 0.0246, **+0.080**)
    ///   - Closes the V_18 weakness vs fast-ssim2 (which lands at 0.1335
    ///     on B3 — V_20_4 lands closer to ssim2 there).
    ///
    /// Runtime cost: ~2× forward-pass time vs PreviewV0_3 (both bakes
    /// run on the same 228-feature vector). Bakes embed via
    /// `include_bytes!`; no extra heap allocations beyond the second
    /// Predictor's scratch buffer.
    PreviewV0_4,
}

impl ZensimProfile {
    /// Latest recommended general-purpose profile.
    /// Returns [`Self::PreviewV0_3`] in zensim 0.3.x.
    pub fn latest() -> Self {
        Self::PreviewV0_3
    }

    /// Canonical name string, e.g. `"zensim-preview-v0.1"`.
    pub fn name(&self) -> &'static str {
        match self {
            Self::PreviewV0_1 => "zensim-preview-v0.1",
            Self::PreviewV0_2 => "zensim-preview-v0.2",
            Self::PreviewV0_3 => "zensim-preview-v0.3",
            Self::PreviewV0_4 => "zensim-preview-v0.4",
        }
    }

    /// Internal parameters for this profile.
    pub(crate) fn params(&self) -> &'static ProfileParams {
        match self {
            Self::PreviewV0_1 => &PROFILE_PREVIEW_V0_1,
            Self::PreviewV0_2 => &PROFILE_PREVIEW_V0_2,
            Self::PreviewV0_3 => &PROFILE_PREVIEW_V0_3,
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
    /// Score mapping coefficient A in `100 - A × d^B`. **Ignored when
    /// [`skip_score_mapping`] is `true` (the bake is already
    /// MCOS-calibrated).**
    pub score_mapping_a: f64,
    /// Score mapping exponent B in `100 - A × d^B`. **Ignored when
    /// [`skip_score_mapping`] is `true`.**
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
    skip_score_mapping: false,
    mlp_bytes: None,
    mlp_bytes_b3: None,
    mlp_primary_mix: 1.0,
    extended_features: false,
    compute_iw_features: false,
};

static PROFILE_PREVIEW_V0_2: ProfileParams = ProfileParams {
    weights: &WEIGHTS_PREVIEW_V0_2,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    skip_score_mapping: false,
    mlp_bytes: None,
    mlp_bytes_b3: None,
    mlp_primary_mix: 1.0,
    extended_features: false,
    compute_iw_features: false,
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
/// File: `zensim/weights/v0_16_2026-05-12.bin` (md5 `baf3fdcb`,
/// affine-calibrated α=28.0366, β=-5.0738, R²=0.7423 against ssim2
/// truth on unified JPEG parquet; raw bake md5 was `b3f5fc59`).
/// Trained on **fully-purged** safe-synthetic CSV (144,791 rows
/// after the 2026-05-12 d≤16 purge; manifest at
/// `benchmarks/contaminated_sources_purged_2026-05-12.txt`).
/// h=128, **TV=20** (raised from V0_15's TV=15 to recover B1 closure
/// honestly), seed=1, KonJND-aligned.
///
/// **V0_16 honest results** vs fast-ssim2:
/// - CID22 SROCC = **0.8919** (+0.0024 above ssim2's 0.8895)
/// - AIC-3 CTC = **0.7990** (+0.0025 above ssim2's 0.7965)
/// - Non-mono q-step rate = **2.30 %** (1/2.5 of V0_8's 5.87 %;
///   best of any bake)
/// - val_mean = 0.9403 (V0_15 had 0.9427; lower training fit, better
///   generalization to CID22)
///
/// **V0_16 per-band CID22 (closes B0/B1 honestly)**:
/// - B0 (<50): 0.4214 vs ssim2 0.4418 (-0.020, was V0_15 -0.049)
/// - B1 [50,65): **0.4559** vs ssim2 0.4694 (-0.014, MATCHES V0_8
///   tainted -0.014 HONESTLY)
/// - B2 [65,90): 0.7802 vs ssim2 0.7722 (+0.008)
/// - B3 (≥90, n=43): 0.1723 vs ssim2 0.1121 (+0.060)
/// - Near-PJND: 0.3547 vs ssim2 0.3908 (-0.036, was V0_15 -0.046)
///
/// **Key finding**: V0_16 with stronger TV=20 on clean data
/// recovers the B1 closure that V0_8 had via training-set leakage.
/// The B1 floor isn't a fundamental ceiling — it's a regularization
/// hyperparameter selection. V0_15 (TV=15) was undersmoothed for B1.
///
/// V0_15 (TV=15) wins B2/B3 and AIC-3; V0_16 (TV=20) wins B0/B1/
/// Near-PJND/CID22-agg/non-mono. Decision: ship V0_16 for better
/// band coverage per Goal #1 (match-or-exceed ssim2 across all
/// quality bands).
///
/// Predecessors archived at `zensim/weights/archive/`:
/// - `v0_4_2026-04-30.bin` (original V0_4 placeholder MLP)
/// - `v0_5_2026-05-11.bin` (original 11.77% leak)
/// - `v0_7_seed0_2026-05-11.bin`
/// - `v0_7_seed1_tv10_2026-05-11.bin`
/// - `v0_8_tainted_2026-05-11.bin` (+0.0034 inflation 2026-05-12)
/// - `v0_15_2026-05-12.bin` (md5 `73d5e418`, first honest ship,
///   superseded same day by V0_16's better B0/B1 coverage)
/// - `v0_16_2026-05-12.bin` (the base V0_18 was constructed from)
/// - `v0_17_2026-05-13.bin` (F32 of V0_18; V0_18 is the I8 re-bake)
/// - `v0_19_2026-05-14.bin` (briefly shipped 2026-05-14 then reverted
///   same day; commit f8a3280 → revert. The "contamination cleanup"
///   that motivated V0_19 used dHash-64 at d≤16 which is the LOOSE
///   screening threshold; user review of the side-by-side montages
///   confirmed those matches were vastly different images. Re-audit
///   at d≤10 (the strict "very likely same image" threshold) found
///   zero cross-corpus CID22 ↔ KADID matches. V0_18's CID22 SROCC
///   0.8934 is therefore NOT inflated. V0_19 archived at
///   `zensim/weights/archive/v0_19_overcleaned_2026-05-14.bin` for
///   reference.)
///
/// **2026-05-14 ship-form swap (this commit)**: the shipped bytes are
/// the zerobiased + LZ4-compressed variant of V_18 at 17,940 bytes
/// (down from the raw 93,064 bytes — a 5.2× reduction). Per-pair
/// outputs are score-equivalent to the raw bake (CID22 SROCC 0.8933
/// reproduces exactly across 4292 pairs). zenpredict 0.2.0+ ships
/// LZ4 decompression unconditionally so no consumer-facing feature
/// flag is needed. The raw 93 KB variant lives at
/// `zensim/weights/v0_18_2026-05-13.bin` for reproduction reference.
pub(crate) fn mlp_bake_preview_v0_3() -> &'static [u8] {
    include_bytes!("../weights/v0_18_zerobiased_lz4_2026-05-13.bin")
}

static PROFILE_PREVIEW_V0_3: ProfileParams = ProfileParams {
    // Linear weights are unused on the MLP path but kept non-empty so
    // any caller that introspects `params.weights` length without
    // checking `mlp_bytes.is_some()` sees a sensible (V0_2-equivalent)
    // value.
    weights: &WEIGHTS_PREVIEW_V0_2,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    // V0_18 bake is already MCOS-calibrated (affine α=28.04 β=-5.07
    // inherited from V0_16). The runtime returns the bake's raw output
    // directly as the score — applying the V0_2 `100 - 18·d^0.7` on
    // top of an MCOS-aligned value would produce garbage. Set
    // `skip_score_mapping = true` to bypass the transform.
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    skip_score_mapping: true,
    mlp_bytes: Some(mlp_bake_preview_v0_3),
    mlp_bytes_b3: None,
    mlp_primary_mix: 1.0,
    extended_features: false,
    compute_iw_features: false,
};

/// V_20 input-shaping seed=1 bake, **affine-calibrated** to V_18's
/// output scale (α=28.0366, β=-5.0738 — inherited from V_18 / V_16
/// lineage). Used as the B3-specialist secondary in
/// `PreviewV0_4` (D2 α=0.7 multi-bake ensemble).
///
/// Architecture: 228 → 128 (LeakyReLU) → 1 (Identity). Carries 98
/// per-feature `feature_transforms` metadata (and the corresponding
/// `feature_transform_params` blob); the runtime's
/// `apply_mlp_scoring` dispatches to `predict_transformed` when this
/// metadata is present.
///
/// Methodology: `benchmarks/v0_20_input_shaping_methodology_2026-05-15.md`.
/// Source eval: `benchmarks/v0_20_input_shaping_eval_2026-05-15.log`.
pub(crate) fn mlp_bake_v0_20_is_calibrated() -> &'static [u8] {
    include_bytes!("../weights/v0_20_is_calibrated_2026-05-15.bin")
}

static PROFILE_PREVIEW_V0_4: ProfileParams = ProfileParams {
    weights: &WEIGHTS_PREVIEW_V0_2,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    // Both bakes are affine-calibrated to MCOS 0..100 already.
    // skip_score_mapping returns the raw mix directly as the score.
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    skip_score_mapping: true,
    mlp_bytes: Some(mlp_bake_preview_v0_3),
    mlp_bytes_b3: Some(mlp_bake_v0_20_is_calibrated),
    // **α = 0.4 in RAW-output space matches the D2 design doc's
    // Z-norm α = 0.7 prediction**: CID22 B3 [30, 40) +0.080 lift
    // (same as the Z-norm prediction), with CID22 aggregate −0.008
    // (slightly higher than the Z-norm prediction's −0.004). The
    // runtime mix happens BEFORE the score-mapping step, so we mix
    // in raw-prediction space — different from the offline
    // `ensemble_mix` tool which Z-normalizes per-bake before mixing.
    //
    // Per-α trade (raw space, see `benchmarks/v0_20_4_runtime_mix_sweep_2026-05-15.md`):
    //   α=0.4: B3 = 0.1042 (+0.080),  agg = 0.8855 (-0.008) ← default ship
    //   α=0.5: B3 = 0.0816 (+0.057),  agg = 0.8870 (-0.006)
    //   α=0.6: B3 = 0.0572 (+0.033),  agg = 0.8886 (-0.005)
    //   α=0.7: B3 = 0.0417 (+0.017),  agg = 0.8900 (-0.003) ← conservative
    //
    // Pick α=0.4 to maximize the priority-band lift (CLAUDE.md
    // "B0..B5 lift is the dominant priority"). For a smaller B3 lift
    // at lower aggregate cost, swap to a fresh ProfileParams with
    // mlp_primary_mix tuned higher.
    mlp_primary_mix: 0.4,
    extended_features: false,
    compute_iw_features: false,
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
