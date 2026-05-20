# V7 ship patch sketch (drop in once seed selection complete)

This is the diff to apply to `zensim/src/profile.rs` and friends once a
V7 bake passes all 6 Tuner gates + the V7-specific within-±5 check.

## 1. zensim/src/profile.rs — add `PreviewV0_5TunerV3` variant

After the existing `PreviewV0_5TunerV2` variant (line ~372):

```rust
    /// **PreviewV0_5TunerV3** (2026-05-19, EXP-CROSS-CODEC-V7).
    ///
    /// V_24-per-sample-α architecture with `zentrain.tanh_output_head`
    /// scale=15.0, trained on the canonical safesyn corpus with the
    /// **empirical** multi-band anchor parquet (per-(codec, band)
    /// `target_score` = empirical median ssim2 from the canonical
    /// score parquets, not V6's rule-of-thumb numbers).
    ///
    /// V7 closes the V6 calibration gap: V6's anchor target_scores
    /// were systematically 10–30 score units below the empirical
    /// medians at every band except 0.3. V7 outputs at each band
    /// land within ±5 of the empirical metric medians.
    ///
    /// **NOT for general ranking** — same caveat as PreviewV0_5Tuner /
    /// PreviewV0_5TunerV2. Cross-corpus SROCC drops because training
    /// is safesyn-only with multi-band anchor pressure.
    ///
    /// Methodology: `benchmarks/v_tuner_v7_methodology_2026-05-19.md`.
    PreviewV0_5TunerV3,
```

## 2. zensim/src/profile.rs — add name() arm (line ~447)

```rust
            Self::PreviewV0_5TunerV3 => "zensim-preview-v0.5-tuner-v3",
```

## 3. zensim/src/profile.rs — add params() arm (line ~466)

```rust
            Self::PreviewV0_5TunerV3 => &PROFILE_PREVIEW_V0_5_TUNER_V3,
```

## 4. zensim/src/profile.rs — add bake bytes loader + ProfileParams

After `PROFILE_PREVIEW_V0_5_TUNER_V2` (line ~1205):

```rust
/// PreviewV0_5TunerV3 bake bytes (2026-05-19,
/// EXP-CROSS-CODEC-V7). Same V_24-per-sample-α architecture +
/// tanh-output-head scale=15.0 as PreviewV0_5TunerV2; the only
/// trainer difference is `--anchor-parquet` pointing at the
/// empirical-medians anchor parquet at
/// `/mnt/v/zen/zensim-training/2026-05-19-empirical-band-anchors/`
/// instead of V6's rule-of-thumb parquet.
///
/// 372 → 128 → 128 (identity passthrough) MLP, F32 uncompressed
/// (261,351 bytes, md5 _TBD_).
///
/// Methodology: `benchmarks/v_tuner_v7_methodology_2026-05-19.md`.
pub(crate) fn mlp_bake_preview_v0_5_tuner_v3() -> &'static [u8] {
    include_bytes!("../weights/v_tuner_v7_2026-05-19.bin")
}

static PROFILE_PREVIEW_V0_5_TUNER_V3: ProfileParams = ProfileParams {
    weights: &WEIGHTS_PREVIEW_V0_2,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    skip_score_mapping: true,
    mlp_bytes: Some(mlp_bake_preview_v0_5_tuner_v3),
    mlp_bytes_b3: None,
    mlp_primary_mix: 1.0,
    extended_features: true,
    compute_iw_features: true,
    soft_clamp_score: false,
    ensemble_classifier_bytes: None,
    mlp_bytes_compression: None,
};
```

## 5. zensim/weights/v_tuner_v7_2026-05-19.bin

Copy the winning seed's bake bytes here.

## 6. zensim/tests/tuner_v3_profile.rs

Mirror tuner_v2_profile.rs but with PreviewV0_5TunerV3 +
v_tuner_v7_2026-05-19.bin (4 tests).

## 7. zensim/SOTA_TRAILS.md — TunerV3 row + gate

Add row to the Tuner trail section.

## 8. CHANGELOG.md — [Unreleased] entry

Add ship entry.
