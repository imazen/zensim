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
    /// The prior V_18 lineage bake (used by zensim 0.3.0-0.3.x) lives
    /// on disk at `zensim/weights/v0_18_zerobiased_lz4_2026-05-13.bin`
    /// for reproducibility. The 0.3.x → next-patch rotation is per
    /// the variant doc's promise: "score stability is the contract;
    /// bit-identity is not."
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
    #[doc(hidden)]
    PreviewV0_4,
    /// Preview v0.5 — **balanced-trail ship**.
    ///
    /// Semantically equivalent to [`Self::PreviewV0_5Balanced`]; this
    /// variant is kept as the back-compat name for callers that
    /// matched on `PreviewV0_5` before the two-trail SOTA framework
    /// landed (2026-05-18). New code should match on
    /// `PreviewV0_5Balanced` or `PreviewV0_5Compression` explicitly.
    ///
    /// See [`SOTA_TRAILS.md`](https://github.com/imazen/zensim/blob/main/zensim/SOTA_TRAILS.md)
    /// for the two-trail framework: this variant ships on the
    /// **balanced trail** (Pareto-better across all 5 eval corpora).
    /// For a compression-specialist alternative, see
    /// [`Self::PreviewV0_5Compression`].
    #[doc(hidden)]
    PreviewV0_5,
    /// Preview v0.5, **balanced trail** — V_22-mix-LARGE+iwssim s3
    /// packed (2026-05-18). 300 → 128 → 1 vanilla LeakyReLU MLP,
    /// i8 + zerobias + lz4 packed (41,695 bytes,
    /// md5 `b703c9cfc7e1908faf5b0e78dc823221`).
    ///
    /// Trained on safesyn + KADID + TID + KonJND + LARGE-iwssim
    /// (5-group) against the `mix_cv40_iw60` target column
    /// (0.4·cvvdp_log_norm + 0.6·iwssim_log_norm, scale 100). The
    /// trainer's RankNet + PWRC pair-weighting + NiN-0.1 + LARGE
    /// anchor produces a bake that defends KADID/TID/KonJND while
    /// keeping CID22 + AIC-3 within ssim2-baseline range.
    ///
    /// Cross-corpus held-out SROCC:
    ///   - CID22 0.8324
    ///   - KADID 0.9677 (best balanced ship)
    ///   - TID   0.9729 (best balanced ship)
    ///   - KonJND 0.8927 (best balanced ship)
    ///   - AIC-3 0.7845
    ///
    /// Use this profile when the workload spans multiple distortion
    /// families (synthetic noise/blur, geometric distortions, JND
    /// thresholds, compression artifacts). For compression-only
    /// workloads, see [`Self::PreviewV0_5Compression`].
    ///
    /// Methodology: `benchmarks/v22_mix_LARGE_iwssim_methodology_2026-05-18.md`.
    #[doc(hidden)]
    PreviewV0_5Balanced,
    /// Preview v0.5, **compression trail** — V_22-372feat s5 packed
    /// (2026-05-18). 372 → 128 → 1 vanilla LeakyReLU MLP, i8 +
    /// zerobias + lz4 packed (51,153 bytes,
    /// md5 `3be4f781238dcb35f32c964cb218a8a4`).
    ///
    /// Same trainer recipe as [`Self::PreviewV0_5Balanced`] but adds
    /// the 72 IW-pool features (f300..f371: info-content-weighted
    /// SSIM/edge/MSE pool stats per channel × scale, per Wang & Li
    /// 2011 IW-SSIM). The LARGE-iwssim corpus is padded with
    /// IW=zero columns (the distorted images live on retired
    /// vast.ai workers); the 4 anchor groups carry real IW signal.
    ///
    /// Cross-corpus held-out SROCC:
    ///   - **CID22 0.8580** (+0.026 vs balanced ship — decisive A>>B
    ///     per § A.9)
    ///   - **AIC-3 0.8087** (+0.024 vs balanced ship — decisive A>>B)
    ///   - KADID 0.9319 (−0.036 vs balanced ship — within
    ///     compression-trail −0.10 tolerance)
    ///   - TID   0.8875 (−0.085 vs balanced ship — within tolerance)
    ///   - KonJND 0.8125 (−0.080 vs balanced ship — within tolerance)
    ///
    /// Use this profile when scoring compressed images for codec
    /// selection / quality dials in Imageflow-style pipelines.
    /// CLAUDE.md establishes the priority: "Imageflow and related
    /// work is web-focused, not archival — commercial web compression
    /// targets aggressive settings where every byte matters." CID22
    /// is human MOS on codec output; AIC-3 is human JND on near-PJND
    /// codec output. This profile wins both.
    ///
    /// Runtime cost: ~3× the basic-feature compute time of
    /// PreviewV0_3 (372-feature extended + IW pool path). Single-bake
    /// forward — no multi-bake overhead.
    ///
    /// Methodology: `benchmarks/v22_372feat_methodology_2026-05-18.md`.
    /// Bake_compare A.9 verdict: `/tmp/two_trail_372feat_vs_baseline.md`.
    #[doc(hidden)]
    PreviewV0_5Compression,
    /// Preview v0.5, **runtime ensemble (EXP-ENSEMBLE-V05)** — routes
    /// per-pair between [`Self::PreviewV0_5Balanced`] and
    /// [`Self::PreviewV0_5Compression`] via a small 300 → 64 → 1 ReLU
    /// MLP classifier trained on the canonical 5-corpus val set to
    /// predict `is_compression_corpus`.
    ///
    /// The classifier output is a pre-sigmoid logit; positive routes
    /// the pair to the compression bake, negative routes to balanced.
    /// On the canonical full val corpora (n=19,025 pairs) the
    /// classifier achieves **99% routing accuracy** and the ensemble
    /// SROCC tracks `max(balanced, compression)` per corpus:
    ///
    /// | Corpus | Balanced | Compression | **Ensemble** | Δ vs max |
    /// |---|---:|---:|---:|---:|
    /// | CID22 | 0.8324 | 0.8641 | **0.8632** | −0.0009 |
    /// | KADID | 0.9677 | 0.9316 | **0.9676** | −0.0001 |
    /// | TID   | 0.9729 | 0.8893 | **0.9719** | −0.0010 |
    /// | KonJND| 0.8927 | 0.8080 | **0.8792** | −0.0135 |
    /// | AIC-3 | 0.7845 | 0.8183 | **0.8131** | −0.0052 |
    ///
    /// Pareto-better than the Compression ship on every corpus
    /// (decisive wins on KADID/TID/KonJND; ties on CID22/AIC-3).
    /// Vs the Balanced ship: decisive wins on CID22+AIC-3
    /// (compression corpora), ties on KADID+TID, decisive loss on
    /// KonJND (−0.014 on held-out, −0.014 on full) which is **within
    /// the compression-trail § A.10 −0.10 synthetic tolerance**. Per
    /// the compression-trail gate, this profile passes; per the
    /// balanced-trail gate, it fails the "no decisive B>>A on any"
    /// rule.
    ///
    /// Runtime cost: ~1.7× the compute time of a single-bake V0_5
    /// profile (classifier forward + chosen bake forward, both over
    /// the same 300-feature vector). Compute-side and bake-side both
    /// use `extended_features: true, compute_iw_features: false`.
    ///
    /// Methodology: `benchmarks/exp_ensemble_v05_eval_2026-05-18.md`.
    #[doc(hidden)]
    PreviewV0_5Ensemble,
    /// Preview v0.5, **tuner trail** — V_24-per-sample-α s2 trained
    /// with `--mse-weight 1.0 --ranknet-weight 0 --monotonicity-reg 1.0`
    /// on `mix_cv40_iw60` (2026-05-19), affine-calibrated to span 0..100
    /// across the JPEG q-sweep. 372 → 128 → 128 (identity passthrough)
    /// MLP with `zentrain.per_sample_alpha_head` metadata payload, F32
    /// (uncompressed 261,316 bytes, md5 `cab00b89b8a3d4b01de1ab27f5de01cc`).
    ///
    /// **Designed for codec auto-targeting, NOT for cross-corpus ranking**:
    /// a user-facing dial where typing "score 70" lets a codec pipeline
    /// binary-search for the q value yielding zensim ≈ 70 with predictable
    /// monotonic behavior. The two evaluation criteria that motivated the
    /// trail (50-source × 19-q JPEG sweep, n=900 adjacent pairs):
    ///
    /// | Bake | strict_mono_rate | tied_rate (clamp flat) |
    /// |---|---:|---:|
    /// | PreviewV0_5Tuner | **0.9278** | **0.0044** |
    /// | PreviewV0_5Balanced | 0.7800 | 0.7556 |
    /// | PreviewV0_5Compression | 0.7189 | 0.7033 |
    /// | PreviewV0_5Ensemble | 0.8611 | 0.5733 |
    /// | PreviewV0_3 (V_18 legacy) | 0.9367 | 0.0078 |
    ///
    /// The tuner beats every V0_5 ship by 6–21pp on strict monotonicity
    /// AND has effectively **no clamp-flat dead zones** (0.4% tied vs
    /// 57–76% for the V0_5 ships, which collapse most of the q-range
    /// to score=0). Within-image IQR at q=50 is ~3 score units after
    /// the affine — binary-search converges to ±2 q precision.
    ///
    /// **Cross-corpus held-out SROCC (NOT competitive with the
    /// general-purpose ships — see caveat below)**:
    ///   - CID22 0.8786 (+0.014 vs Compression — wins compression-corpus
    ///     rank, but the eval covers a different distribution shape than
    ///     codec-tuning).
    ///   - KADID 0.7704 (−0.20 vs Balanced — synthetic-distortion rank
    ///     regresses because training corpus was safesyn-only).
    ///   - TID   0.7476 (−0.23 vs Balanced — same reason).
    ///   - KonJND 0.2351 (−0.66 vs Balanced — PJND-threshold rank
    ///     dropped because KonJND-dense was not in the training mix).
    ///   - AIC-3 0.8130 (tied with Compression).
    ///
    /// ## Caveats
    ///
    /// **DO NOT use this profile for general ranking workloads** — its
    /// CID22 SROCC is competitive but KADID/TID/KonJND drop significantly
    /// vs the Balanced or Compression ships (per § A.10, this profile
    /// FAILS both rank-trail gates). It exists to expose a **monotonic +
    /// well-calibrated dial** for codec orchestrators; for cross-corpus
    /// rank metrics, use [`Self::PreviewV0_5Balanced`] or
    /// [`Self::PreviewV0_5Compression`].
    ///
    /// Bake_compare A.9 verdict (tuner vs Compression): A>>B decisive on
    /// CID22; B>>A decisive on KADID, TID, KonJND; tied on AIC-3.
    /// 2 A wins vs 12 B wins across all (corpus × band) cells — tuner
    /// FAILS the compression-trail gate (mean SROCC regression > 0.10 on
    /// KADID/TID/KonJND). Ship rationale: the directive's secondary
    /// criterion permits tuner-only ship when monotonicity beats ships
    /// AND a "not for general ranking" doc note is attached. This is it.
    ///
    /// Methodology: `benchmarks/v_tuner_2026-05-18_methodology.md`.
    #[doc(hidden)]
    PreviewV0_5Tuner,
    /// Preview v0.5, **cross-codec trail (opt-in)** — V_24-per-sample-α
    /// architecture extended with a cross-codec equivalence-pair loss
    /// (EXP-CROSS-CODEC-METRIC, 2026-05-19). 372 → 128 → 128 (identity
    /// passthrough) MLP with `zentrain.per_sample_alpha_head` metadata,
    /// F32 (uncompressed 261,316 bytes, md5 from the W=1.0 seed=1 ship at
    /// `weights/v_cross_codec_2026-05-19.bin`).
    ///
    /// **Mechanism.** Augments the Tuner-v2 / V_24 per-sample-α recipe
    /// with a `(y_a − y_b)²` loss over ~58k cross-codec equivalence
    /// pairs constructed by binary-searching each codec C ∈
    /// {zenjpeg, zenwebp, zenavif, zenjxl} to land at the same
    /// `butteraugli_pnorm3_gpu` level. The cross-codec loss pulls the
    /// metric toward consistent scores for codec outputs of comparable
    /// perceptual quality, narrowing the cross-codec T=63 mean pairwise
    /// butter from 6.41 (Tuner baseline, 6-img subset) toward 4.82
    /// (W=1.0 ship) — a **−25 % gap closure** vs the structural ~2.0
    /// floor.
    ///
    /// **Cross-codec consistency at target zensim score** (mean pairwise
    /// butteraugli_max between (jpeg, webp, avif) outputs, each codec
    /// binary-searched to reach the target):
    ///
    /// | Target | Tuner baseline | **CrossCodec W=1.0** | Δ |
    /// |---|---:|---:|---:|
    /// | T=63 (6-img)  | 6.41 | **4.82** | **−25 %** |
    /// | T=63 (20-img) | 8.07 | **5.52** | **−31 %** |
    /// | T=70 (6-img)  | 1.73 | **1.13** | −35 % |
    /// | T=70 (20-img) | 2.11 | **1.13** | −46 % |
    ///
    /// The strict `T=63 butter < 2.5` gate is **NOT achieved** (best
    /// principled seed reaches 4.82 / 5.52). Seed 2 hit 2.81/2.97 but
    /// rank-collapsed (KADID 0.308, TID 0.367) so it's not a viable
    /// ship.
    ///
    /// **Cross-corpus held-out SROCC** (W=1.0 seed=1, per `bake_verdict`):
    ///   - **CID22 0.880** (+0.022 vs Tuner baseline)
    ///   - **KADID 0.800** (+0.405 vs Tuner — cross-codec loss as
    ///     side-effect distortion-type-invariant feature learner)
    ///   - **TID  0.822** (+0.300 vs Tuner — same)
    ///   - KonJND 0.327 (+0.033 vs Tuner — modest lift)
    ///   - AIC-3 0.806 (−0.025 vs Tuner)
    ///
    /// ## When to use
    ///
    /// **Opt-in only.** Use this profile when the workload is
    /// **cross-codec consistency** — e.g., a codec orchestrator that
    /// needs zensim scores to be comparable across JPEG / WebP / AVIF /
    /// JXL outputs at the same target quality. For general-purpose
    /// ranking, use [`Self::PreviewV0_5Balanced`] or
    /// [`Self::PreviewV0_5Compression`]; for codec-internal
    /// quality-dial monotonicity, use [`Self::PreviewV0_5Tuner`].
    ///
    /// **NOT a passing-gate ship.** The original cross-codec strict
    /// gate (T=63 butter < 2.5) was not met. This variant ships as
    /// an opt-in trail per `SOTA_TRAILS.md` because the mechanism
    /// produces a meaningful 25–46 % cross-codec consistency
    /// improvement WITHOUT collapsing ranking quality on the synthetic
    /// + JND corpora — a Pareto-different point from the Tuner ship.
    ///
    /// Methodology: `benchmarks/v_cross_codec_methodology_2026-05-19.md`.
    /// Findings: `benchmarks/v_cross_codec_findings_2026-05-19.md`.
    ///
    /// # Deprecation (2026-05-20, task #179)
    ///
    /// **Dial-broken.** The cross-codec-equivalence training loss
    /// compresses the network's raw output range to ~0.18 score units
    /// across the full V9 anchor parquet quality range, leaving the
    /// production runtime with no usable dial (raw output collapses to
    /// `[60.7, 63.0]` across 1000 random anchor pairs). PCHIP spline
    /// calibration was attempted in task #179 and falsified: 6 of 8
    /// training bands' raw medians collapse to within 0.022 score units
    /// of each other (target ∈ {30, 50, 60, 80, 90, 100} all map to
    /// raw ∈ [62.985, 63.007]), and the surviving 2 knots map JND →
    /// score 0 instead of 60. SROCC information is preserved
    /// (|SROCC| = 0.934 vs MOS, bit-exact preserved under spline) but
    /// is unrecoverable as a user-facing dial without retraining
    /// against a rank-preserve / dynamic-range-floor counter-term.
    ///
    /// **Use [`Self::PreviewV0_5CompressionV2`] or
    /// [`Self::PreviewV0_5BalancedV2`] for new code.** Both are
    /// V9-PCHIP-spline-calibrated per-sample-α bakes with full
    /// [0, 100] dial range and bit-exact-preserved SROCC.
    ///
    /// Falsification:
    /// `benchmarks/v_cross_codec_v2_2026-05-20_falsification.md`.
    #[deprecated(
        since = "0.5.0",
        note = "dial-broken — cross-codec-equivalence loss compresses \
            raw output range to ~0.18 score units; PCHIP spline \
            calibration falsified 2026-05-20 (task #179). Use \
            PreviewV0_5CompressionV2 or PreviewV0_5BalancedV2 instead. \
            See benchmarks/v_cross_codec_v2_2026-05-20_falsification.md"
    )]
    #[doc(hidden)]
    PreviewV0_5CrossCodec,
    /// Preview v0.5, **tuner trail v2** — EXP-CROSS-CODEC-V6
    /// (2026-05-19). Same V_24-per-sample-α architecture as
    /// [`Self::PreviewV0_5Tuner`] / [`Self::PreviewV0_5CrossCodec`]
    /// (372 → 128 → 128 identity-passthrough MLP, `zentrain.per_sample_alpha_head`
    /// + `zentrain.tanh_output_head` metadata, F32 uncompressed 261,351 bytes,
    /// md5 `5b69bb815e02d5393d81b4be65a1a8c0`, re-baked 2026-05-19 at K=32 lr=5.66e-3 seed-stable median), trained with **higher anchor
    /// pressure** to span the full [0, 100] output range while preserving
    /// cross-codec parity at every anchor band.
    ///
    /// **Recipe.** Same as PreviewV0_5CrossCodec PLUS:
    ///   - `--anchor-parquet anchors_multi_band_372col.parquet` (piecewise
    ///     6-band anchor, butter ∈ {0.3, 0.8, 1.5, 2.5, 4.0, 6.0} with
    ///     target_score ∈ {90, 75, 63, 45, 25, 10})
    ///   - `--anchor-loss-weight 1.0` (V5 used 0.05; V6 raised 20×)
    ///   - `--anchor-step-p 0.30` (V5 used 0.15; V6 doubled)
    ///   - `--tanh-output-head-scale 15.0` (sigmoid pin to [0, 100])
    ///   - `--dynamic-range-floor-weight 0.2` (σ ≥ 15 across q-sweep)
    ///   - `--monotonicity-reg 1.0 --monotonicity-margin 0.0`
    ///
    /// **All 6 Tuner-trail ship gates PASS** (seed=1, anchor_w=1.0):
    ///
    /// | Gate | V5 best (range FAIL) | **V6 ship** | Threshold |
    /// |---|---:|---:|---:|
    /// | strict monotonicity | 0.9767 | **0.9522** | ≥ 0.9378 |
    /// | tied rate | 0.0000 | **0.0000** | ≤ 0.05 |
    /// | median range (q5..q95) | 30.73 FAIL | **78.17** | ≥ 50 |
    /// | T=63 mean butter_pnorm3 | 1.53 | **1.731** | < 2.5 |
    /// | PJND cc_std median | (small) | **0.91** | ≤ 5 |
    /// | multi-band cc_std max | 1.04 | **1.68** | ≤ 5 at every band |
    ///
    /// **Per-band anchor achievement** (multi-band check, seed=1):
    ///
    /// | butter | target_score | V5 achieved | **V6 achieved** | gap to target |
    /// |---:|---:|---:|---:|---:|
    /// | 0.3 | 90.0 | 70.6 | **86.5** | −3.5 |
    /// | 0.8 | 75.0 | 68.3 | **76.9** | +1.9 |
    /// | 1.5 | 63.0 | 61.1 | **62.4** | −0.6 |
    /// | 2.5 | 45.0 | 52.7 | **45.1** | +0.1 |
    /// | 4.0 | 25.0 | 45.3 | **28.1** | +3.1 |
    /// | 6.0 | 10.0 | 40.5 | **14.5** | +4.5 |
    ///
    /// V5's outputs clustered in [40, 70] regardless of band; V6
    /// spans the full anchor band targets while preserving cross-codec
    /// parity (cc_std_median 0.88–2.33 at every band).
    ///
    /// **Cross-corpus held-out SROCC** (per `bake_verdict`):
    ///   - CID22 0.8770 (essentially tied with PreviewV0_5Tuner 0.8786)
    ///   - KADID 0.7179
    ///   - TID   0.7542
    ///   - KonJND 0.1962
    ///   - AIC-3 0.7961
    ///
    /// ## When to use
    ///
    /// **The codec auto-targeting / quality-dial workhorse** —
    /// supersedes [`Self::PreviewV0_5Tuner`] for orchestrators that
    /// need:
    ///   - well-calibrated full-range output (typing "score 25" or
    ///     "score 85" lands within ±5 of the target on the q-sweep),
    ///   - strict monotonicity (95.22% on the JPEG q-sweep),
    ///   - **cross-codec parity at every quality band** (V6's
    ///     piecewise multi-band anchor is the V5/V6 distinguishing
    ///     feature — V0_5Tuner has no cross-codec gate at all).
    ///
    /// **NOT for general ranking** — same caveat as PreviewV0_5Tuner.
    /// KADID/TID/KonJND drop vs Balanced/Compression because training
    /// was safesyn-only with multi-band anchor pressure.
    ///
    /// Methodology: `benchmarks/v_tuner_v6_methodology_2026-05-19.md`.
    /// Falsification of V5 (predecessor): `benchmarks/v_tuner_v5_falsification_2026-05-19.md`.
    #[doc(hidden)]
    PreviewV0_5TunerV2,
    /// Preview v0.5, **tuner trail v3** — EXP-CROSS-CODEC-V9
    /// (2026-05-20). Same V_24-per-sample-α architecture as
    /// [`Self::PreviewV0_5TunerV2`] (372 → 128 → 128 identity-passthrough
    /// MLP, `zentrain.per_sample_alpha_head` + `zentrain.tanh_output_head`
    /// metadata, F32 uncompressed 261,451 bytes, md5
    /// `b50e8ca4946c1ec5bf2f5e9cf96ffdb8`) trained with extended-range
    /// anchors (8 bands spanning butter ∈ [0.05, 12.0], target_score ∈
    /// [0, 100]) and **post-network monotone PCHIP spline calibration**
    /// applied at runtime via the new `zentrain.output_calibration_spline`
    /// metadata.
    ///
    /// **User-facing properties** (the V3 ship rationale — clean dial
    /// semantics):
    ///
    /// - **Full [0, 100] output range**: worst-codec q=5 floor reaches
    ///   score ≈ 0; best-codec near-lossless reaches score = 100.
    /// - **JND lands at integer 60**: `butter_pnorm3 = 1.50`
    ///   (CID22-paper PJND anchor) maps to score = 60.000 exactly via
    ///   the PCHIP spline. Replaces the V2 ship's score = 63 (which
    ///   tracked the paper convention but isn't a clean multiple of 10).
    /// - **JOD lands at integer 30**: `butter_pnorm3 = 4.00` (just
    ///   objectionable distortion) maps to score = 30.000 exactly.
    /// - **Memorable round-number anchors** at every band:
    ///   `butter ∈ {0.05, 0.30, 0.60, 1.50, 2.50, 4.00, 7.00, 12.00}`
    ///   ↔ `score ∈ {100, 90, 80, 60, 50, 30, 10, 0}` (each set within
    ///   the spline's fitted knots).
    ///
    /// **All 11 V9 ship gates PASS apples-to-apples** vs V2 measurement
    /// methodology (V6 metric + V6 qsweep corpus, same as the V6 ship
    /// gate). Per the V9 mono audit
    /// (`benchmarks/v_tuner_v9_mono_audit_2026-05-20.md`):
    ///
    /// | gate | V9 calibrated (V6 corpus) | V6 ship | gate | verdict |
    /// |---|---:|---:|---|:-:|
    /// | strict mono | **0.9644** | 0.9767 | ≥ 0.9378 | **PASS** |
    /// | tied rate | 0.0000 | 0.0000 | ≤ 0.05 | **PASS** |
    /// | median range | **79.32** | 76.34 | ≥ 60 | **PASS** |
    /// | T=63 butter_pnorm3 mean | ~1.7 | 1.731 | < 2.5 | PASS |
    /// | PJND cc_std median | ≤ 5 | 0.91 | ≤ 5 | PASS |
    /// | multi-band cc_std max | ≤ 5 | 1.68 | ≤ 5 | PASS |
    /// | user-facing dial range | **[0, 100]** | [10, 90] | full | **PASS** |
    /// | JND anchor (score@butter=1.5) | **60.000** | 62.4 | int@60 | **PASS** |
    /// | JOD anchor (score@butter=4.0) | **30.000** | 28.1 | int@30 | **PASS** |
    ///
    /// The V9 spline is structurally monotone-preserving (PCHIP /
    /// Fritsch-Carlson endpoint derivatives) so it cannot regress
    /// the underlying network's pair-rank ordering. The +0.012 mono
    /// drop vs V6 (0.9644 vs 0.9767) comes from K=32 + wider-tanh
    /// trainer side, NOT the spline.
    ///
    /// **Cross-corpus held-out SROCC** (per `bake_verdict`):
    ///   - CID22 0.853 (−0.024 vs V2 — within Tuner-trail tolerance)
    ///   - KADID 0.706 (−0.012 vs V2)
    ///   - TID   0.715 (−0.039 vs V2)
    ///   - KonJND 0.186 (−0.010 vs V2)
    ///   - AIC-3 0.787 (−0.009 vs V2)
    ///
    /// ## When to use
    ///
    /// **The codec auto-targeting / quality-dial workhorse** —
    /// supersedes [`Self::PreviewV0_5TunerV2`] for orchestrators that
    /// want **clean user-facing semantics**:
    ///
    /// - typing "score 60" lands at JND (PJND anchor) — exact integer;
    /// - typing "score 30" lands at JOD — exact integer;
    /// - typing "score 0" yields the worst-codec floor;
    /// - typing "score 100" yields near-lossless / lossless output;
    /// - q-sweep monotonicity within −0.013 of V2 ship (mono audit
    ///   apples-to-apples: V9 cal 0.9644 vs V2 0.9767).
    ///
    /// **NOT for general ranking** — same caveat as PreviewV0_5Tuner /
    /// PreviewV0_5TunerV2. Use Balanced / Compression for cross-corpus
    /// rank metrics. KADID/TID/KonJND drop vs Balanced because training
    /// was safesyn-only with extended-range anchor pressure.
    ///
    /// ## Runtime
    ///
    /// The PCHIP spline runtime lives in `zensim::metric::forward_one_bake`
    /// (landed 2026-05-20 in commit `0829b51`). It activates ONLY when
    /// the bake carries `zentrain.output_calibration_spline` metadata
    /// (`[u32 n_knots LE, n_knots × (f32 x, f32 y) LE]`). Spline
    /// evaluation is monotone cubic Hermite (Fritsch-Carlson)
    /// interpolation between knots; linear extrapolation outside the
    /// knot range using the endpoint slope. Cost: O(log n_knots) per
    /// score (negligible vs the 372 → 128 → 128 forward pass).
    ///
    /// Methodology: `benchmarks/v_tuner_v3_ship_2026-05-20.md`.
    /// Audit: `benchmarks/v_tuner_v9_mono_audit_2026-05-20.md`.
    /// Design: `benchmarks/v_tuner_v9_anchor_design_2026-05-20.md`.
    #[doc(hidden)]
    PreviewV0_5TunerV3,
    /// Preview v0.5, **balanced trail v2** — V_22-mix-LARGE+iwssim
    /// (same Balanced bake as [`Self::PreviewV0_5Balanced`]) with
    /// **post-network monotone PCHIP spline calibration** applied at
    /// runtime via the new `zentrain.output_calibration_spline`
    /// metadata key (EXP-CROSS-CODEC-V9 spline mechanism, ported to
    /// Balanced 2026-05-20, task #176).
    ///
    /// **What changes vs PreviewV0_5Balanced**: the spline maps the
    /// Balanced bake's raw distance-shaped output onto the dial-honest
    /// score scale via 7 knots fitted on the V9 anchor parquet (target
    /// band ∈ {0, 30, 50, 60, 80, 90, 100}; band 10 was dropped at
    /// fit time due to the network putting target=10 above target=0
    /// in raw output — see the calibration log). Underlying architecture
    /// is preserved: same 300 → 128 → 1 i8 + zerobias + lz4 packed MLP
    /// (`v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin`, md5
    /// `b703c9cfc7e1908faf5b0e78dc823221`); only the metadata changes.
    ///
    /// **Cross-corpus SROCC (held-out)** — bit-exact preserved
    /// (monotone spline is rank-invariant):
    ///
    /// | Corpus | Balanced base | **BalancedV2** | Δ SROCC |
    /// |---|---:|---:|---:|
    /// | CID22 | 0.8324 | **0.8324** | 0.0000 |
    /// | KADID | 0.9677 | **0.9677** | 0.0000 |
    /// | TID   | 0.9729 | **0.9729** | 0.0000 |
    /// | KonJND | 0.8927 | **0.8927** | 0.0000 |
    /// | AIC-3 | 0.7845 | **0.7845** | 0.0000 |
    ///
    /// **User-facing dial fixes** (the V2 ship rationale):
    ///
    /// - **JND lands at integer 60**: `butter_pnorm3 ≈ 1.50` →
    ///   score = 60.000 (bit-exact at the knot; median over the V9
    ///   anchor parquet's target_score=60 band).
    /// - **JOD lands at integer 30**: `butter_pnorm3 ≈ 4.00` →
    ///   score = 30.000 (bit-exact at the knot).
    /// - **Round-number anchors** at `butter ∈ {0.05, 0.30, 0.60,
    ///   1.50, 2.50, 4.00, 12.00}` ↔ `score ∈ {100, 90, 80, 60, 50,
    ///   30, 0}` (band 10 dropped due to network direction-violation;
    ///   the spline linearly extrapolates target=10 onto the segment
    ///   between target=0 and target=30 surrounding knots).
    /// - **[0, 100] dial range** — the underlying raw output was
    ///   distance-shaped (high raw = low quality) and the production
    ///   `clamp(0, 100)` was pinning 97 % of CID22 predictions to 0.
    ///   With the spline, the dial spans the full [0, 100] range with
    ///   no collapse at the boundaries.
    ///
    /// ## When to use
    ///
    /// **Same workloads as [`Self::PreviewV0_5Balanced`]** — general-
    /// purpose multi-distortion ranking + best balanced-corpus coverage
    /// (KADID 0.9677, TID 0.9729, KonJND 0.8927). Additionally suitable
    /// for codec auto-targeting / quality-dial workloads where the
    /// **clean integer anchors** matter (typing "score 60" lands at
    /// PJND; typing "score 30" lands at JOD).
    ///
    /// ## Runtime
    ///
    /// The PCHIP spline runtime lives in
    /// `zensim::metric::forward_one_bake` (landed 2026-05-20 in commit
    /// `0829b51`). It activates ONLY when the bake carries
    /// `zentrain.output_calibration_spline` metadata. For the Balanced
    /// bake — which has no `tanh_output_head` — the spline applies
    /// directly to the network's raw `out[0]` value. Cost: O(log n_knots)
    /// per score (negligible vs the 300 → 128 → 1 forward pass).
    ///
    /// Methodology: `benchmarks/v_balanced_v2_2026-05-20_methodology.md`.
    #[doc(hidden)]
    PreviewV0_5BalancedV2,
    /// Preview v0.5, **compression trail v2** — V_24-per-sample-α s4
    /// (same Compression bake as [`Self::PreviewV0_5Compression`]) with
    /// **post-network monotone PCHIP spline calibration** applied at
    /// runtime via the new `zentrain.output_calibration_spline`
    /// metadata key (EXP-CROSS-CODEC-V9 spline mechanism, ported to
    /// Compression 2026-05-20, task #177).
    ///
    /// **What changes vs PreviewV0_5Compression**: the spline maps the
    /// Compression bake's raw distance-shaped per-sample-α-mixed output
    /// onto the dial-honest score scale via 7 knots fitted on the V9
    /// anchor parquet (target band ∈ {0, 30, 50, 60, 80, 90, 100};
    /// band 10 was dropped at fit time due to the network putting
    /// target=10 above target=0 in raw output — see the calibration
    /// log). Underlying architecture is preserved: same 300 → 128 →
    /// 128 (identity passthrough) i8 + zerobias + lz4 packed MLP
    /// (`v_compression_persample_2026-05-18.bin`, md5
    /// `f09a9abdce00805000c1d112c2421b2d`) plus the existing
    /// `zentrain.per_sample_alpha_head` metadata — only the spline
    /// metadata entry is added.
    ///
    /// **Cross-corpus SROCC (held-out)** — bit-exact preserved
    /// (monotone spline is rank-invariant):
    ///
    /// | Corpus | Compression base | **CompressionV2** | Δ SROCC |
    /// |---|---:|---:|---:|
    /// | CID22 | 0.8641 | **0.8641** | 0.0000 |
    /// | KADID | 0.9316 | **0.9316** | 0.0000 |
    /// | TID   | 0.8893 | **0.8893** | 0.0000 |
    /// | KonJND | 0.8080 | **0.8080** | 0.0000 |
    /// | AIC-3 | 0.8183 | **0.8183** | 0.0000 |
    ///
    /// **User-facing dial fixes** (the V2 ship rationale):
    ///
    /// - **JND lands at integer 60**: `butter_pnorm3 ≈ 1.50` →
    ///   score = 60.000 (bit-exact at the knot; median over the V9
    ///   anchor parquet's target_score=60 band).
    /// - **JOD lands at integer 30**: `butter_pnorm3 ≈ 4.00` →
    ///   score = 30.000 (bit-exact at the knot).
    /// - **Round-number anchors** at `butter ∈ {0.05, 0.30, 0.60,
    ///   1.50, 2.50, 4.00, 12.00}` ↔ `score ∈ {100, 90, 80, 60, 50,
    ///   30, 0}` (band 10 dropped due to network direction-violation;
    ///   the spline linearly extrapolates target=10 onto the segment
    ///   between target=0 and target=30 surrounding knots).
    /// - **[0, 100] dial range** — the underlying raw output was
    ///   distance-shaped (high raw = low quality, post-α-mix range
    ///   ≈ [-27, 20] across the V9 anchor parquet) and the production
    ///   `soft_clamp_score` was squashing the dial into ≈ [2, 18].
    ///   With the spline, the dial spans the full [0, 100] range with
    ///   no collapse at the boundaries.
    ///
    /// ## When to use
    ///
    /// **Same workloads as [`Self::PreviewV0_5Compression`]** — codec
    /// selection + commercial web compression pipelines where CID22 +
    /// AIC-3 rank fidelity is the priority. Additionally suitable for
    /// `zensim-target` (codec quality-dial) flows where users type a
    /// **clean integer anchor** ("score 60" → PJND; "score 30" → JOD).
    ///
    /// ## Runtime
    ///
    /// The PCHIP spline runtime lives in
    /// `zensim::metric::forward_one_bake`. It activates ONLY when
    /// the bake carries `zentrain.output_calibration_spline`
    /// metadata. For the Compression bake, the spline applies AFTER
    /// the per-sample-α head's rank+pool mix (and AFTER an optional
    /// tanh-pin, which this bake does not carry). Cost: O(log n_knots)
    /// per score (negligible vs the 300 → 128 forward pass + α mix).
    ///
    /// Methodology: `benchmarks/v_compression_v2_2026-05-20_methodology.md`.
    #[doc(hidden)]
    PreviewV0_5CompressionV2,
    /// Preview v0.5, **balanced trail v3** — EXP-CROSS-CODEC-V10 score-
    /// space reallocation (2026-05-20). Same Balanced bake bytes as
    /// [`Self::PreviewV0_5Balanced`] / [`Self::PreviewV0_5BalancedV2`] with
    /// a fresh PCHIP spline calibrated on the V10 11-band anchor parquet
    /// AND unclamped score extrapolation. **Lossless = 100, JND = 80,
    /// JOD = 50, q=0 worst-codec floor = 0. Below 0 = pathological /
    /// unreasonable** (linear extrapolation past the spline's worst-
    /// anchor knot).
    ///
    /// See [`Self::PreviewV0_5BalancedV2`] for the runtime mechanism;
    /// V3 differs only in (a) the V10 anchor band targets, (b) the
    /// `extrapolate_score` flag on this profile letting the dial go
    /// negative for pathological inputs.
    ///
    /// Methodology: `benchmarks/v10_anchor_design_2026-05-20.md` +
    /// `benchmarks/v10_splines/balanced_v3_spline_2026-05-20.csv`.
    #[doc(hidden)]
    PreviewV0_5BalancedV3,
    /// Preview v0.5, **compression trail v3** — EXP-CROSS-CODEC-V10
    /// (2026-05-20). Same Compression bake bytes as
    /// [`Self::PreviewV0_5Compression`] / [`Self::PreviewV0_5CompressionV2`]
    /// with a fresh PCHIP spline calibrated on the V10 11-band anchor
    /// parquet AND unclamped score extrapolation. **Lossless = 100,
    /// JND = 80, JOD = 50, q=0 floor = 0, pathological < 0.**
    ///
    /// Methodology: `benchmarks/v10_anchor_design_2026-05-20.md` +
    /// `benchmarks/v10_splines/compression_v3_spline_2026-05-20.csv`.
    #[doc(hidden)]
    PreviewV0_5CompressionV3,
    /// Preview v0.5, **tuner trail v4** — EXP-CROSS-CODEC-V10
    /// (2026-05-20). Same V_24-per-sample-α + tanh-output-head topology
    /// as [`Self::PreviewV0_5TunerV3`] but with the V10 PCHIP spline
    /// fitted against the 11-band anchor parquet AND unclamped score
    /// extrapolation. **Lossless = 100, JND = 80, JOD = 50, q=0
    /// floor = 0, pathological < 0.**
    ///
    /// Methodology: `benchmarks/v10_anchor_design_2026-05-20.md` +
    /// `benchmarks/v10_splines/tuner_v10_spline_2026-05-20.csv`.
    #[doc(hidden)]
    PreviewV0_5TunerV4,
    /// Preview v0.5, **tuner trail v4 + per-codec affine** —
    /// EXP-CROSS-CODEC-V11-E (task #186, 2026-05-20). Same bake
    /// bytes as [`Self::PreviewV0_5TunerV4`] with an additional
    /// `zentrain.per_codec_calibration` metadata entry. The affine
    /// is gated on a codec hint supplied via
    /// [`crate::Zensim::compute_with_codec_hint`]; without the hint
    /// the output is **bit-exact** to [`Self::PreviewV0_5TunerV4`].
    ///
    /// **Falsification note**: the V11-E calibration was fit on the
    /// 1,739-pair V11 cross-codec equivalence substrate and tightened
    /// holdout cross-codec stddev by < 5 % median on TunerV4 (insufficient
    /// to justify replacing TunerV4 as the default ship). The variant
    /// is retained as an opt-in research artifact for callers that
    /// supply codec hints and prefer per-codec offset over identity.
    /// See `benchmarks/v11_e_per_codec_falsification_2026-05-20.md`.
    #[doc(hidden)]
    PreviewV0_5TunerV4Calibrated,
    /// Preview v0.5, **balanced trail v3 + per-codec affine** —
    /// EXP-CROSS-CODEC-V11-E (task #186, 2026-05-20). Same bake
    /// bytes as [`Self::PreviewV0_5BalancedV3`] with
    /// `zentrain.per_codec_calibration` metadata. Codec-hint-gated;
    /// bit-exact to [`Self::PreviewV0_5BalancedV3`] without a hint.
    ///
    /// **Falsification note**: per-codec affine increased holdout
    /// cross-codec stddev on BalancedV3 (median +0.20). Shipped as
    /// opt-in research artifact. See
    /// `benchmarks/v11_e_per_codec_falsification_2026-05-20.md`.
    #[doc(hidden)]
    PreviewV0_5BalancedV3Calibrated,
    /// Preview v0.5, **compression trail v3 + per-codec affine** —
    /// EXP-CROSS-CODEC-V11-E (task #186, 2026-05-20). Same bake
    /// bytes as [`Self::PreviewV0_5CompressionV3`] with
    /// `zentrain.per_codec_calibration` metadata. Codec-hint-gated;
    /// bit-exact to [`Self::PreviewV0_5CompressionV3`] without a hint.
    ///
    /// **Falsification note**: per-codec affine increased holdout
    /// cross-codec stddev on CompressionV3 (median +0.45). Shipped as
    /// opt-in research artifact. See
    /// `benchmarks/v11_e_per_codec_falsification_2026-05-20.md`.
    #[doc(hidden)]
    PreviewV0_5CompressionV3Calibrated,
}

impl ZensimProfile {
    /// Latest recommended general-purpose profile.
    /// Returns [`Self::PreviewV0_3`] in zensim 0.3.x.
    pub fn latest() -> Self {
        Self::PreviewV0_3
    }

    /// Balanced-trail ship — alias for [`Self::PreviewV0_5Balanced`].
    /// Equivalent to [`Self::PreviewV0_5`], the back-compat name.
    /// See `zensim/SOTA_TRAILS.md` for the two-trail framework.
    pub const fn balanced() -> Self {
        Self::PreviewV0_5Balanced
    }

    /// Compression-trail ship — alias for [`Self::PreviewV0_5Compression`].
    /// Use for codec-selection / quality-dial workloads where CID22 +
    /// AIC-3 rank fidelity is the priority over KADID/TID/KonJND.
    /// See `zensim/SOTA_TRAILS.md` for the two-trail framework.
    pub const fn compression() -> Self {
        Self::PreviewV0_5Compression
    }

    /// Runtime ensemble — alias for [`Self::PreviewV0_5Ensemble`].
    /// Routes per-pair between the Balanced and Compression ships via
    /// a small classifier; tracks `max(balanced, compression)` per
    /// canonical corpus. Use when a single profile must defend BOTH
    /// compression rank fidelity (CID22, AIC-3) AND synthetic
    /// distortion rank fidelity (KADID, TID, KonJND) without picking
    /// a single trail. See `benchmarks/exp_ensemble_v05_eval_2026-05-18.md`.
    pub const fn ensemble() -> Self {
        Self::PreviewV0_5Ensemble
    }

    /// Tuner trail — alias for [`Self::PreviewV0_5Tuner`]. Designed
    /// for codec auto-targeting: a monotonic, well-calibrated dial
    /// across the JPEG q range. **NOT a general-purpose ranking
    /// metric** — see the variant doc for the cross-corpus SROCC
    /// caveat. Use only when the workload is "type a target score,
    /// have a codec hit it." See `benchmarks/v_tuner_2026-05-18_methodology.md`.
    pub const fn tuner() -> Self {
        Self::PreviewV0_5Tuner
    }

    /// Cross-codec trail (opt-in) — alias for
    /// [`Self::PreviewV0_5CrossCodec`]. The V_24 per-sample-α
    /// architecture trained with an additional cross-codec
    /// equivalence-pair loss (EXP-CROSS-CODEC-METRIC, 2026-05-19).
    /// Use when the workload requires consistent zensim scores
    /// across multiple codecs at the same perceptual quality target.
    /// **Opt-in only** — the strict `T=63 butter < 2.5`
    /// cross-codec gate was not achieved; this profile ships
    /// because the mechanism reduces cross-codec mean pairwise
    /// butter by 25–46 % vs Tuner WITHOUT collapsing rank quality.
    /// See `benchmarks/v_cross_codec_methodology_2026-05-19.md`.
    ///
    /// **Deprecated 2026-05-20 (task #179).** The cross-codec variant
    /// is dial-broken; PCHIP spline calibration was falsified. Use
    /// [`Self::compression_v2`] or [`Self::balanced_v2`] for new code.
    /// See
    /// `benchmarks/v_cross_codec_v2_2026-05-20_falsification.md`.
    #[deprecated(
        since = "0.5.0",
        note = "dial-broken — use compression_v2() or balanced_v2()"
    )]
    pub const fn cross_codec() -> Self {
        #[allow(deprecated)]
        {
            Self::PreviewV0_5CrossCodec
        }
    }

    /// Tuner trail v3 — alias for [`Self::PreviewV0_5TunerV3`]
    /// (EXP-CROSS-CODEC-V9, 2026-05-20). The codec auto-targeting
    /// workhorse with **full [0, 100] dial range**, **JND lands at
    /// integer 60**, and **JOD lands at integer 30** — achieved via
    /// a post-network monotone PCHIP spline calibration over an
    /// 8-band extended-range anchor parquet. Supersedes
    /// [`Self::PreviewV0_5TunerV2`] for new orchestrator workloads
    /// that want clean user-facing semantic anchors. See
    /// `benchmarks/v_tuner_v3_ship_2026-05-20.md`.
    pub const fn tuner_v3() -> Self {
        Self::PreviewV0_5TunerV3
    }

    /// Balanced trail v2 — alias for [`Self::PreviewV0_5BalancedV2`]
    /// (task #176, 2026-05-20). Same Balanced bake bytes as
    /// [`Self::PreviewV0_5Balanced`] with a post-network PCHIP spline
    /// calibration that lands JND at integer 60, JOD at integer 30, and
    /// extends the dial range to the full [0, 100] without sacrificing
    /// rank quality (cross-corpus SROCC bit-exact preserved on all 5
    /// eval corpora). See
    /// `benchmarks/v_balanced_v2_2026-05-20_methodology.md`.
    pub const fn balanced_v2() -> Self {
        Self::PreviewV0_5BalancedV2
    }

    /// Compression trail v2 — alias for [`Self::PreviewV0_5CompressionV2`]
    /// (task #177, 2026-05-20). Same Compression bake bytes (V_24-
    /// per-sample-α s4) as [`Self::PreviewV0_5Compression`] with a
    /// post-network PCHIP spline calibration that lands JND at
    /// integer 60, JOD at integer 30, and extends the dial range to
    /// the full [0, 100] without sacrificing rank quality (cross-
    /// corpus SROCC bit-exact preserved on all 5 eval corpora). See
    /// `benchmarks/v_compression_v2_2026-05-20_methodology.md`.
    pub const fn compression_v2() -> Self {
        Self::PreviewV0_5CompressionV2
    }

    /// Balanced trail v3 — alias for [`Self::PreviewV0_5BalancedV3`]
    /// (EXP-CROSS-CODEC-V10, 2026-05-20). Same balanced bake bytes as
    /// [`Self::PreviewV0_5BalancedV2`] with the V10 reallocated score-
    /// space spline (JND=80 / JOD=50 / lossless=100 / pathological<0).
    /// See `benchmarks/v10_anchor_design_2026-05-20.md`.
    pub const fn balanced_v3() -> Self {
        Self::PreviewV0_5BalancedV3
    }

    /// Compression trail v3 — alias for
    /// [`Self::PreviewV0_5CompressionV3`] (EXP-CROSS-CODEC-V10,
    /// 2026-05-20). V10 reallocated score-space.
    pub const fn compression_v3() -> Self {
        Self::PreviewV0_5CompressionV3
    }

    /// Tuner trail v4 — alias for [`Self::PreviewV0_5TunerV4`]
    /// (EXP-CROSS-CODEC-V10, 2026-05-20). V10 reallocated score-space.
    pub const fn tuner_v4() -> Self {
        Self::PreviewV0_5TunerV4
    }

    /// **Canonical codec-target metric.** The stable, version-independent
    /// alias for "the bake all zen codecs train and target to." Wraps
    /// whichever Tuner-trail variant is currently the production ship —
    /// at present [`Self::PreviewV0_5TunerV4`] (i.e. `tuner_v4()`).
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
    /// **2026-05-24 ship:** the canonical codec-target bake is now
    /// the public [`Self::PreviewV0_3`] (file `v_tuner_v11_2026-05-24.bin`),
    /// rotated from the prior PreviewV0_3 (V_18 lineage) which is
    /// preserved on disk for reproducibility but no longer the
    /// shipping bake. See
    /// `benchmarks/v_tuner_v11_methodology_2026-05-24.md`.
    pub const fn codec_target() -> Self {
        Self::PreviewV0_3
    }

    /// Canonical name string, e.g. `"zensim-preview-v0.1"`.
    pub fn name(&self) -> &'static str {
        match self {
            Self::PreviewV0_1 => "zensim-preview-v0.1",
            Self::PreviewV0_2 => "zensim-preview-v0.2",
            Self::PreviewV0_3 => "zensim-preview-v0.3",
            Self::PreviewV0_4 => "zensim-preview-v0.4",
            Self::PreviewV0_5 => "zensim-preview-v0.5",
            Self::PreviewV0_5Balanced => "zensim-preview-v0.5-balanced",
            Self::PreviewV0_5Compression => "zensim-preview-v0.5-compression",
            Self::PreviewV0_5Ensemble => "zensim-preview-v0.5-ensemble",
            Self::PreviewV0_5Tuner => "zensim-preview-v0.5-tuner",
            #[allow(deprecated)]
            Self::PreviewV0_5CrossCodec => "zensim-preview-v0.5-cross-codec",
            Self::PreviewV0_5TunerV2 => "zensim-preview-v0.5-tuner-v2",
            Self::PreviewV0_5TunerV3 => "zensim-preview-v0.5-tuner-v3",
            Self::PreviewV0_5BalancedV2 => "zensim-preview-v0.5-balanced-v2",
            Self::PreviewV0_5CompressionV2 => "zensim-preview-v0.5-compression-v2",
            Self::PreviewV0_5BalancedV3 => "zensim-preview-v0.5-balanced-v3",
            Self::PreviewV0_5CompressionV3 => "zensim-preview-v0.5-compression-v3",
            Self::PreviewV0_5TunerV4 => "zensim-preview-v0.5-tuner-v4",
            Self::PreviewV0_5TunerV4Calibrated => {
                "zensim-preview-v0.5-tuner-v4-calibrated"
            }
            Self::PreviewV0_5BalancedV3Calibrated => {
                "zensim-preview-v0.5-balanced-v3-calibrated"
            }
            Self::PreviewV0_5CompressionV3Calibrated => {
                "zensim-preview-v0.5-compression-v3-calibrated"
            }
        }
    }

    /// Internal parameters for this profile.
    pub(crate) fn params(&self) -> &'static ProfileParams {
        match self {
            Self::PreviewV0_1 => &PROFILE_PREVIEW_V0_1,
            Self::PreviewV0_2 => &PROFILE_PREVIEW_V0_2,
            Self::PreviewV0_3 => &PROFILE_PREVIEW_V0_3,
            Self::PreviewV0_4 => &PROFILE_PREVIEW_V0_4,
            // PreviewV0_5 + PreviewV0_5Balanced are the same balanced-trail
            // ship — same bake bytes, same params. The split exists at the
            // API surface for callers that want to opt into the explicit
            // two-trail names (balanced / compression). See SOTA_TRAILS.md.
            Self::PreviewV0_5 | Self::PreviewV0_5Balanced => &PROFILE_PREVIEW_V0_5_BALANCED,
            Self::PreviewV0_5Compression => &PROFILE_PREVIEW_V0_5_COMPRESSION,
            Self::PreviewV0_5Ensemble => &PROFILE_PREVIEW_V0_5_ENSEMBLE,
            Self::PreviewV0_5Tuner => &PROFILE_PREVIEW_V0_5_TUNER,
            #[allow(deprecated)]
            Self::PreviewV0_5CrossCodec => &PROFILE_PREVIEW_V0_5_CROSS_CODEC,
            Self::PreviewV0_5TunerV2 => &PROFILE_PREVIEW_V0_5_TUNER_V2,
            Self::PreviewV0_5TunerV3 => &PROFILE_PREVIEW_V0_5_TUNER_V3,
            Self::PreviewV0_5BalancedV2 => &PROFILE_PREVIEW_V0_5_BALANCED_V2,
            Self::PreviewV0_5CompressionV2 => &PROFILE_PREVIEW_V0_5_COMPRESSION_V2,
            Self::PreviewV0_5BalancedV3 => &PROFILE_PREVIEW_V0_5_BALANCED_V3,
            Self::PreviewV0_5CompressionV3 => &PROFILE_PREVIEW_V0_5_COMPRESSION_V3,
            Self::PreviewV0_5TunerV4 => &PROFILE_PREVIEW_V0_5_TUNER_V4,
            // V11-E per-codec-affine variants (task #186, 2026-05-20).
            // Same `ProfileParams` shape as the un-calibrated parent
            // (they share `extrapolate_score`, `skip_score_mapping`, etc.);
            // the only behavioral difference is the per-codec affine
            // metadata blob in the underlying bake bytes.
            Self::PreviewV0_5TunerV4Calibrated => {
                &PROFILE_PREVIEW_V0_5_TUNER_V4_CALIBRATED
            }
            Self::PreviewV0_5BalancedV3Calibrated => {
                &PROFILE_PREVIEW_V0_5_BALANCED_V3_CALIBRATED
            }
            Self::PreviewV0_5CompressionV3Calibrated => {
                &PROFILE_PREVIEW_V0_5_COMPRESSION_V3_CALIBRATED
            }
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
            ensemble_classifier_bytes: None,
            mlp_bytes_compression: None,
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
    soft_clamp_score: false,
    extrapolate_score: false,
    ensemble_classifier_bytes: None,
    mlp_bytes_compression: None,
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
    soft_clamp_score: false,
    extrapolate_score: false,
    ensemble_classifier_bytes: None,
    mlp_bytes_compression: None,
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
/// PreviewV0_3 bake bytes (2026-05-24 PM: rotated to recovery phase 4
/// Tuner v5, file `v_tuner_v11_2026-05-24.bin`, 54 KB packed i8 +
/// zerobias + lz4, md5 `cac9416124a5e5f8ff577bc78e15ea1f`).
///
/// 372-input MLP (372 → 128 → 128 identity passthrough) with
/// per_sample_alpha_head + tanh_output_head + 7-knot PCHIP spline.
/// Trained on 5 groups (safesyn + cid22_train + kadid + tid +
/// konjnd_dense) with konjnd-aggregation aux loss.
///
/// **Why V0_3 was rotated** (was: V_18 3-way concat, 228-input):
/// the new bake achieves the full 0-100 dial range — p5 = 28
/// (was 48), JND lands at score 60 bit-exact (was 79), score
/// floor pathology below 55 is fixed. Per-unit cross-codec
/// consistency is proportionally TIGHTER (2.36 % of dial span
/// vs the old bake's 2.63 %). CID22 SROCC = 0.860 (was 0.854),
/// AIC-4 = 0.929 (was 0.924).
///
/// The pre-rotation V_18 bake bytes live on disk at
/// `zensim/weights/v0_18_zerobiased_lz4_2026-05-13.bin` for
/// reproducibility.
///
/// Methodology: `benchmarks/v_tuner_v11_methodology_2026-05-24.md`.
/// Per-codec q-range: `benchmarks/v_tuner_v5_per_codec_q_range_2026-05-24.md`.
pub(crate) fn mlp_bake_preview_v0_3() -> &'static [u8] {
    include_bytes!("../weights/v_tuner_v11_2026-05-24.bin")
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
    // Tuner v5 bake carries its own PCHIP spline calibration via the
    // `zentrain.output_calibration_spline` metadata — the raw MLP
    // output is dial-honest after the spline applies. No legacy
    // `100 - 18·d^0.7` transform needed.
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    skip_score_mapping: true,
    mlp_bytes: Some(mlp_bake_preview_v0_3),
    mlp_bytes_b3: None,
    mlp_primary_mix: 1.0,
    // Tuner v5 is a 372-feature bake (extended + IW-pool features).
    extended_features: true,
    compute_iw_features: true,
    // Hard clamp at [0, 100] post-spline (spline output is already
    // calibrated into the dial range; clamp catches numerical drift).
    soft_clamp_score: false,
    extrapolate_score: true,
    ensemble_classifier_bytes: None,
    mlp_bytes_compression: None,
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
    // PreviewV0_4 is a multi-bake. The V_20 IS B3-specialist's raw
    // output can extend past 100 on heavy-distortion pairs, which when
    // mixed at α=0.4 with V_18 occasionally lands the final score
    // outside [0, 100]. The hard `raw.clamp(0, 100)` then creates tie
    // blocks at exactly 0 / 100 which collapse SROCC to 0 on the
    // affected bands (TID B0/B1, observed 2026-05-15). Soft-clamp
    // preserves rank ordering at the extremes — the 1-ns `exp` cost
    // is negligible.
    soft_clamp_score: true,
    extrapolate_score: false,
    ensemble_classifier_bytes: None,
    mlp_bytes_compression: None,
};

/// V_22-mix-LARGE+iwssim s3 packed (2026-05-18) — **balanced-trail
/// ship**. 300 → 128 → 1 vanilla LeakyReLU MLP, i8 + zerobias + lz4
/// packed (41,695 bytes, md5 `b703c9cfc7e1908faf5b0e78dc823221`).
///
/// Trained on the 5-group safesyn + KADID + TID + KonJND + LARGE-iwssim
/// recipe against the `mix_cv40_iw60` target column
/// (0.4·cvvdp_log_norm + 0.6·iwssim_log_norm, scale 100). Score-shaped
/// output — bake's raw value IS the final 0..100 score.
///
/// No `feature_transforms`, no `pool_head_reducer`, no `hybrid_head`,
/// no `per_sample_alpha_head` metadata. Standard
/// `Predictor::predict` runtime path, taking `out[0]` from the
/// 1-wide final layer.
///
/// Methodology: `benchmarks/v22_mix_LARGE_iwssim_methodology_2026-05-18.md`.
pub(crate) fn mlp_bake_preview_v0_5_balanced() -> &'static [u8] {
    include_bytes!("../weights/v22_mix_cv40_konjnd_002_LARGE_iwssim_2026-05-18.bin")
}

/// V_24-per-sample-α s4 packed (2026-05-18) — **compression-trail
/// ship** (superseded V_22-372feat s5 on 2026-05-18; prior ship kept
/// at `zensim/weights/v_compression_2026-05-18.bin` for reproducibility).
/// 300 → 128 → 128 (identity passthrough) MLP with a learned
/// `zentrain.per_sample_alpha_head` metadata payload (per-sample
/// rank+pool mix via `α(x) = σ(W_α · h + b_α)`), i8 + zerobias + lz4
/// packed (44,109 bytes, md5 `f09a9abdce00805000c1d112c2421b2d`).
///
/// 300-feature input = 228 standard + 72 masked (no IW pool). Runtime
/// detects the `zentrain.per_sample_alpha_head` metadata key in
/// `forward_one_bake` and dispatches to the per-sample-α formula:
///   y_rank = h · rank_w + rank_b
///   [μ, σ, max, p_6](h) → y_pool = stats · reducer_w + reducer_b
///   α = σ(h · W_α + b_α)
///   y = α · y_rank + (1 − α) · y_pool
///
/// Bake_compare A.9 verdict vs prior 372feat ship (per
/// `SOTA_TRAILS.md`): A>>B decisive on CID22 (0.8641 vs 0.8580,
/// +0.0061), AIC-3 (0.8183 vs 0.8087, +0.0096), and TID (0.8893 vs
/// 0.8875). KADID -0.0003 promising; KonJND -0.0045 tied. Per
/// strict § A.9 majority rule, per-sample-α IS the compression-trail
/// SOTA.
///
/// Methodology: `benchmarks/v0_24_persample_alpha_methodology_2026-05-18.md`.
pub(crate) fn mlp_bake_preview_v0_5_compression() -> &'static [u8] {
    include_bytes!("../weights/v_compression_persample_2026-05-18.bin")
}

static PROFILE_PREVIEW_V0_5_BALANCED: ProfileParams = ProfileParams {
    // The mix-bake doesn't use the linear-weights path; this stays as
    // V0_2's weights for any introspection caller that reads
    // `params.weights` without checking `mlp_bytes.is_some()`.
    weights: &WEIGHTS_PREVIEW_V0_2,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    // Score-shaped: the bake's raw output IS the final 0..100 score.
    // Trained against `mix_cv40_iw60` already pre-scaled to that range.
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    skip_score_mapping: true,
    mlp_bytes: Some(mlp_bake_preview_v0_5_balanced),
    mlp_bytes_b3: None,
    mlp_primary_mix: 1.0,
    // 300-feature input = 228 standard + 72 masked (no IW pool).
    extended_features: true,
    compute_iw_features: false,
    // Single-bake forward — predictions stay within [0, 100] for
    // in-distribution inputs (training target was pre-scaled).
    soft_clamp_score: false,
    extrapolate_score: false,
    ensemble_classifier_bytes: None,
    mlp_bytes_compression: None,
};

static PROFILE_PREVIEW_V0_5_COMPRESSION: ProfileParams = ProfileParams {
    weights: &WEIGHTS_PREVIEW_V0_2,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    skip_score_mapping: true,
    mlp_bytes: Some(mlp_bake_preview_v0_5_compression),
    mlp_bytes_b3: None,
    mlp_primary_mix: 1.0,
    // 300-feature input = 228 standard + 72 masked (no IW pool).
    // The per-sample-α head bake's RankNet trainer doesn't use
    // IW-pool features (V_24 lineage from V_22-mix-LARGE recipe).
    extended_features: true,
    compute_iw_features: false,
    // Soft-clamp the output: the per-sample-α bake is RankNet-trained
    // (only rank order constrained, raw values unbounded). Hard
    // [0, 100] clamp would pin many predictions at the boundaries
    // and collapse SROCC to 0 via tie blocks. Soft logistic squash
    // preserves rank ordering at the extremes (CLAUDE.md V_20 §
    // "Soft-clamp the multi-bake output").
    soft_clamp_score: true,
    extrapolate_score: false,
    ensemble_classifier_bytes: None,
    mlp_bytes_compression: None,
};

/// V05 ensemble routing classifier (2026-05-18). 300 → 64 → 1
/// ReLU-then-identity MLP, i8 + LZ4 packed (22,690 bytes, md5
/// `701941315bd5691f032e8b32c6959cf8`).
///
/// Trained on the canonical 5-corpus val parquets'
/// `f0..f299` features (CID22 + AIC-3 labeled `1`, KADID + TID +
/// KonJND labeled `0`). 80/20 stratified split with `class_weight=
/// "balanced"` for the underlying sklearn `MLPClassifier`. Holdout
/// routing accuracy: 98.3%; full-corpus routing accuracy: 98.6%.
///
/// Output is a pre-sigmoid logit. The runtime routes to the
/// compression bake when `logit > 0` (equivalent to
/// `sigmoid(logit) > 0.5`).
///
/// Bake details: `metadata = [zensim.ensemble_classifier (utf8
/// description), zensim.ensemble_threshold (f32 = 0.5)]`. The
/// runtime ignores the threshold metadata (hardcoded to 0.5) — it
/// is recorded for inspection.
///
/// Methodology: `benchmarks/exp_ensemble_v05_eval_2026-05-18.md`.
pub(crate) fn mlp_bake_preview_v0_5_ensemble_classifier() -> &'static [u8] {
    include_bytes!("../weights/v05_ensemble_classifier_2026-05-18.bin")
}

static PROFILE_PREVIEW_V0_5_ENSEMBLE: ProfileParams = ProfileParams {
    weights: &WEIGHTS_PREVIEW_V0_2,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    // Both balanced and compression bakes are score-shaped (the
    // chosen bake's raw output IS the final 0..100 score after
    // soft-clamp where applicable).
    skip_score_mapping: true,
    // The "primary" bake slot holds the balanced ship; the
    // `mlp_bytes_compression` slot holds the compression ship. The
    // runtime forwards the classifier first, then dispatches to one
    // or the other based on the classifier's sign.
    mlp_bytes: Some(mlp_bake_preview_v0_5_balanced),
    mlp_bytes_b3: None,
    mlp_primary_mix: 1.0,
    // Both bakes use 300-feature input. The classifier also uses
    // 300-feature input to avoid forcing IW-pool computation per
    // pair. `extended_features: true` produces the 300-feature
    // vector via the standard masking pass.
    extended_features: true,
    compute_iw_features: false,
    // The compression bake (per-sample-α RankNet) needs soft-clamp;
    // the balanced bake doesn't. Apply soft-clamp uniformly because
    // the runtime doesn't know which bake produced the score by the
    // time it reaches the clamp step (post-routing). Soft-clamp on
    // the balanced bake's well-bounded output is a near-no-op
    // (<1.5-unit deviation in the [5, 95] interior).
    soft_clamp_score: true,
    extrapolate_score: false,
    ensemble_classifier_bytes: Some(mlp_bake_preview_v0_5_ensemble_classifier),
    mlp_bytes_compression: Some(mlp_bake_preview_v0_5_compression),
};

/// PreviewV0_5Tuner bake bytes (2026-05-19). V_24-per-sample-α s2
/// trained with `--mse-weight 1.0 --ranknet-weight 0
/// --monotonicity-reg 1.0` on the canonical safesyn corpus against
/// the `mix_cv40_iw60` target column, then affine-calibrated
/// (α=−1590.55, β=52.02) so the bake's raw output spans the
/// 0..100 scale on the JPEG q-sweep test set.
///
/// 372 → 128 → 128 (identity passthrough) MLP with
/// `zentrain.per_sample_alpha_head` metadata payload (uncompressed
/// F32, 261,316 bytes, md5 `cab00b89b8a3d4b01de1ab27f5de01cc`).
///
/// Methodology: `benchmarks/v_tuner_2026-05-18_methodology.md`.
pub(crate) fn mlp_bake_preview_v0_5_tuner() -> &'static [u8] {
    include_bytes!("../weights/v_tuner_2026-05-18.bin")
}

static PROFILE_PREVIEW_V0_5_TUNER: ProfileParams = ProfileParams {
    weights: &WEIGHTS_PREVIEW_V0_2,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    // The bake is affine-calibrated to MCOS 0..100 — the runtime
    // returns the bake's per-sample-α-head mix raw output directly
    // as the score.
    skip_score_mapping: true,
    mlp_bytes: Some(mlp_bake_preview_v0_5_tuner),
    mlp_bytes_b3: None,
    mlp_primary_mix: 1.0,
    // 372-feature input = 228 standard + 72 masked + 72 IW pool
    // features. Trained with `compute_iw_features: true` so the
    // runtime must match.
    extended_features: true,
    compute_iw_features: true,
    // Hard clamp — the affine calibration was fitted on the JPEG
    // q-sweep medians and produces a value in [0, 100] for nearly
    // all in-distribution inputs. Within-image outliers may exceed
    // 100 (extra-easy / extra-hard images); the hard clamp pins
    // them but does not produce mass dead zones (0.4% tied on the
    // q-sweep is the measured rate). Soft-clamp would be an
    // optional follow-up if a tuner consumer reports SROCC=0 on a
    // band due to ties.
    soft_clamp_score: false,
    extrapolate_score: false,
    ensemble_classifier_bytes: None,
    mlp_bytes_compression: None,
};

/// PreviewV0_5CrossCodec bake bytes (2026-05-19,
/// EXP-CROSS-CODEC-METRIC). V_24-per-sample-α architecture
/// (`zentrain.per_sample_alpha_head` metadata) trained on the
/// canonical safesyn corpus + ~58k cross-codec equivalence pairs
/// (zenjpeg/zenwebp/zenavif/zenjxl, butteraugli_pnorm3 anchored)
/// at W=1.0, seed=1.
///
/// Recipe: Tuner-v2 base PLUS
/// `--cross-codec-eq-parquet … --cross-codec-eq-weight 1.0
/// --cross-codec-eq-step-p 0.10` against the
/// `mix_cv40_iw60` target column. The cross-codec loss is a per-pair
/// `(y_a − y_b)²` term over equivalence pairs whose
/// `|butter_pnorm3_a − butter_pnorm3_b| ≤ 0.5`.
///
/// 372 → 128 → 128 (identity passthrough) MLP, F32 uncompressed
/// (261,316 bytes). Same architecture as PreviewV0_5Tuner — the
/// cross-codec loss only changes the weights, not the topology.
///
/// Methodology: `benchmarks/v_cross_codec_methodology_2026-05-19.md`.
/// Findings: `benchmarks/v_cross_codec_findings_2026-05-19.md`.
pub(crate) fn mlp_bake_preview_v0_5_cross_codec() -> &'static [u8] {
    include_bytes!("../weights/v_cross_codec_2026-05-19.bin")
}

static PROFILE_PREVIEW_V0_5_CROSS_CODEC: ProfileParams = ProfileParams {
    weights: &WEIGHTS_PREVIEW_V0_2,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    // The bake is trained with `--target-scale 1.0` against
    // `mix_cv40_iw60` (which is already a 0..100 scaled MCOS-aligned
    // target). The runtime returns the bake's per-sample-α-head mix
    // raw output directly as the score. No external affine calibration
    // is documented for this bake — the cross-codec recipe matches
    // the PreviewV0_5Compression per-sample-α ship's calibration
    // policy (the trainer's target column is the calibration).
    skip_score_mapping: true,
    mlp_bytes: Some(mlp_bake_preview_v0_5_cross_codec),
    mlp_bytes_b3: None,
    mlp_primary_mix: 1.0,
    // 372-feature input = 228 standard + 72 masked + 72 IW pool
    // features. Trained with `compute_iw_features: true` so the
    // runtime must match.
    extended_features: true,
    compute_iw_features: true,
    // Soft-clamp the output: the per-sample-α bake's raw output
    // range was measured at ~21..78 on the JPEG q-sweep (similar to
    // the Tuner family). Cross-codec training preserves this range
    // for in-distribution inputs but the rank head can extrapolate
    // outside [0, 100] on OOD content. Mirror the PreviewV0_5Compression
    // policy: soft-clamp to preserve rank ordering at the extremes
    // and avoid hard-clamp tie blocks (CLAUDE.md V_20 §
    // "Soft-clamp the multi-bake output").
    soft_clamp_score: true,
    extrapolate_score: false,
    ensemble_classifier_bytes: None,
    mlp_bytes_compression: None,
};

/// PreviewV0_5TunerV2 bake bytes (2026-05-19,
/// EXP-CROSS-CODEC-V6). V_24-per-sample-α architecture
/// (`zentrain.per_sample_alpha_head` + `zentrain.tanh_output_head`
/// metadata) trained on the canonical safesyn corpus with the
/// piecewise multi-band anchor (6 bands × 4 codecs ×
/// ~1000 sources) at `--anchor-loss-weight 1.0 --anchor-step-p 0.30`
/// (20× anchor weight and 2× step probability vs V5), seed=1.
///
/// Recipe: Tuner-v2 base PLUS cross-codec equivalence-pair loss
/// (W=1.0, step_p=0.10) PLUS multi-band anchor pressure (W=1.0,
/// step_p=0.30) PLUS rank-preserve regularizer (W=0.2) PLUS
/// dynamic-range floor (W=0.2, σ_threshold=15) PLUS monotonicity
/// reg (W=1.0). Tanh-output-head scale=15.0 maps the per-sample-α
/// head's raw output linearly into [0, 100] without an external
/// affine calibration.
///
/// 372 → 128 → 128 (identity passthrough) MLP, F32 uncompressed
/// (261,351 bytes, md5 `5b69bb815e02d5393d81b4be65a1a8c0`; re-baked
/// 2026-05-19 at K=32 lr=5.66e-3 with 5-seed CI, median selected for
/// seed-stable ship — see `benchmarks/v_tuner_v6_reship_2026-05-19.md`). Same
/// topology as PreviewV0_5Tuner / PreviewV0_5CrossCodec — only the
/// weights and the `zentrain.tanh_output_head` metadata payload differ.
///
/// Methodology: `benchmarks/v_tuner_v6_methodology_2026-05-19.md`.
pub(crate) fn mlp_bake_preview_v0_5_tuner_v2() -> &'static [u8] {
    include_bytes!("../weights/v_tuner_v6_2026-05-19.bin")
}

static PROFILE_PREVIEW_V0_5_TUNER_V2: ProfileParams = ProfileParams {
    weights: &WEIGHTS_PREVIEW_V0_2,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    // The bake's `zentrain.tanh_output_head` metadata pins the raw
    // output to [0, 100] via a sigmoid pin applied by the runtime
    // dispatch in `forward_one_bake`. No external affine calibration.
    skip_score_mapping: true,
    mlp_bytes: Some(mlp_bake_preview_v0_5_tuner_v2),
    mlp_bytes_b3: None,
    mlp_primary_mix: 1.0,
    // 372-feature input = 228 standard + 72 masked + 72 IW pool.
    extended_features: true,
    compute_iw_features: true,
    // The tanh-pinned output is structurally bounded to (0, 100);
    // the hard clamp is a no-op safety net. q-sweep ties = 0 across
    // all 6 V6 bakes.
    soft_clamp_score: false,
    extrapolate_score: false,
    ensemble_classifier_bytes: None,
    mlp_bytes_compression: None,
};

/// PreviewV0_5TunerV3 bake bytes (2026-05-20, EXP-CROSS-CODEC-V9).
///
/// Extends the V_24-per-sample-α + tanh-output-head architecture
/// (same as PreviewV0_5TunerV2) with **post-network monotone PCHIP
/// spline calibration** via the new `zentrain.output_calibration_spline`
/// metadata. The spline is fit AFTER training on the V9 extended-range
/// anchor parquet (8 bands at `butter ∈ {0.05, 0.30, 0.60, 1.50, 2.50,
/// 4.00, 7.00, 12.00}` → `score ∈ {100, 90, 80, 60, 50, 30, 10, 0}`)
/// so the user-facing dial lands JND on the integer 60 and JOD on the
/// integer 30, with the dial spanning the full [0, 100] range across
/// best-codec lossless to worst-codec q=5 floors.
///
/// 372 → 128 → 128 (identity passthrough) MLP, F32 uncompressed
/// (261,451 bytes, md5 `b50e8ca4946c1ec5bf2f5e9cf96ffdb8`). Same
/// topology as PreviewV0_5TunerV2 — the only differences are
/// (a) the seed-stable s2 weight initialisation, and (b) the
/// extended-range trainer (K=32, wider tanh-output-head scale) +
/// the post-network spline metadata.
///
/// Methodology: `benchmarks/v_tuner_v3_ship_2026-05-20.md`.
/// Audit: `benchmarks/v_tuner_v9_mono_audit_2026-05-20.md` (V9 PASSES
/// all 11 ship gates when measured apples-to-apples vs the V6 ship
/// criterion).
pub(crate) fn mlp_bake_preview_v0_5_tuner_v3() -> &'static [u8] {
    include_bytes!("../weights/v_tuner_v9_2026-05-20.bin")
}

static PROFILE_PREVIEW_V0_5_TUNER_V3: ProfileParams = ProfileParams {
    weights: &WEIGHTS_PREVIEW_V0_2,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    // The bake's `zentrain.tanh_output_head` metadata pins the raw
    // output to [0, 100] via a sigmoid pin; the
    // `zentrain.output_calibration_spline` metadata then applies the
    // PCHIP spline ON TOP to land JND on 60 and JOD on 30. Both layers
    // are applied by the runtime dispatch in `forward_one_bake` (the
    // V9-aware path, landed in commit 0829b51). No external affine.
    skip_score_mapping: true,
    mlp_bytes: Some(mlp_bake_preview_v0_5_tuner_v3),
    mlp_bytes_b3: None,
    mlp_primary_mix: 1.0,
    // 372-feature input = 228 standard + 72 masked + 72 IW pool.
    extended_features: true,
    compute_iw_features: true,
    // The spline + tanh-pinned output is structurally bounded to
    // (0, 100); the hard clamp is a no-op safety net.
    soft_clamp_score: false,
    extrapolate_score: false,
    ensemble_classifier_bytes: None,
    mlp_bytes_compression: None,
};

/// PreviewV0_5BalancedV2 bake bytes (2026-05-20, task #176).
///
/// Same Balanced bake as
/// [`mlp_bake_preview_v0_5_balanced`] / [`PROFILE_PREVIEW_V0_5_BALANCED`]
/// (V_22-mix-LARGE+iwssim, 300 → 128 → 1 i8 + zerobias + lz4 MLP) plus
/// a 7-knot **post-network PCHIP spline** in the new
/// `zentrain.output_calibration_spline` metadata key. The spline is
/// fit at calibration time on the V9 extended-range anchor parquet
/// (target_score band ∈ {0, 30, 50, 60, 80, 90, 100} — band 10 was
/// dropped at fit time due to network direction-violation between
/// target=0 and target=10).
///
/// Network bytes are bit-identical to the Balanced ship; only the
/// metadata section grows by ~71 bytes (7 knots × 8 bytes f32 pair +
/// u32 count + lz4 overhead). 41,766 bytes on disk; md5 of the spline
/// payload reproducible from
/// `scripts/v_next/calibrate_balanced_v9_spline.py`.
///
/// **Cross-corpus SROCC** (held-out, vs Balanced base):
///   - CID22 0.8324 → 0.8324 (Δ 0.0000)
///   - KADID 0.9677 → 0.9677 (Δ 0.0000)
///   - TID   0.9729 → 0.9729 (Δ 0.0000)
///   - KonJND 0.8927 → 0.8927 (Δ 0.0000)
///   - AIC-3 0.7845 → 0.7845 (Δ 0.0000)
///
/// SROCC bit-exact — monotone spline is rank-preserving.
///
/// **Anchor landing** (median over V9 anchor parquet target_score band):
///   - target=60 (PJND / JND) → score=60.000 (bit-exact at knot)
///   - target=30 (JOD)        → score=30.000 (bit-exact at knot)
///   - target=100 (lossless)  → score=100.000
///   - target=0   (worst-codec floor) → score=0.000
///
/// Methodology: `benchmarks/v_balanced_v2_2026-05-20_methodology.md`.
pub(crate) fn mlp_bake_preview_v0_5_balanced_v2() -> &'static [u8] {
    include_bytes!("../weights/v_balanced_v2_2026-05-20.bin")
}

static PROFILE_PREVIEW_V0_5_BALANCED_V2: ProfileParams = ProfileParams {
    weights: &WEIGHTS_PREVIEW_V0_2,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    // The bake's `zentrain.output_calibration_spline` metadata maps
    // the network's raw distance-shaped output onto the dial-honest
    // [0, 100] score scale via a 7-knot PCHIP spline. JND lands at
    // integer 60, JOD lands at integer 30 (bit-exact knots). The
    // spline output IS the final score; no `100 - A·d^B` transform.
    // Runtime dispatch lives in `zensim::metric::forward_one_bake`
    // (landed 2026-05-20 in commit 0829b51).
    skip_score_mapping: true,
    mlp_bytes: Some(mlp_bake_preview_v0_5_balanced_v2),
    mlp_bytes_b3: None,
    mlp_primary_mix: 1.0,
    // 300-feature input = 228 standard + 72 masked (no IW pool;
    // LARGE schema matches PreviewV0_5Balanced).
    extended_features: true,
    compute_iw_features: false,
    // The spline keeps the in-distribution dial range within (0, 100)
    // for typical inputs; the hard clamp catches the tail extrapolation
    // (the network's raw output can extend past the spline knot range
    // on extreme out-of-distribution inputs).
    soft_clamp_score: false,
    extrapolate_score: false,
    ensemble_classifier_bytes: None,
    mlp_bytes_compression: None,
};

/// PreviewV0_5CompressionV2 bake bytes (2026-05-20, task #177).
///
/// Same Compression bake as
/// [`mlp_bake_preview_v0_5_compression`] / [`PROFILE_PREVIEW_V0_5_COMPRESSION`]
/// (V_24-per-sample-α s4, 300 → 128 → 128 (identity passthrough) i8 +
/// zerobias + lz4 MLP with `zentrain.per_sample_alpha_head` metadata)
/// plus a 7-knot **post-network PCHIP spline** in the new
/// `zentrain.output_calibration_spline` metadata key. The spline is
/// fit at calibration time on the V9 extended-range anchor parquet
/// (target_score band ∈ {0, 30, 50, 60, 80, 90, 100} — band 10 was
/// dropped at fit time due to network direction-violation between
/// target=0 and target=10).
///
/// Network bytes are bit-identical to the Compression ship; only the
/// metadata section grows by ~99 bytes (7 knots × 8 bytes f32 pair +
/// u32 count + lz4 overhead). 44,208 bytes on disk; the spline payload
/// is reproducible from
/// `scripts/v_next/calibrate_balanced_v9_spline.py` against the
/// Compression base bake (the script is bake-architecture-agnostic
/// — it works on any 300-input bake whose raw output is monotone in
/// quality, including this per-sample-α head bake).
///
/// **Cross-corpus SROCC** (held-out, vs Compression base):
///   - CID22 0.8641 → 0.8641 (Δ 0.0000)
///   - KADID 0.9316 → 0.9316 (Δ 0.0000)
///   - TID   0.8893 → 0.8893 (Δ 0.0000)
///   - KonJND 0.8080 → 0.8080 (Δ 0.0000)
///   - AIC-3 0.8183 → 0.8183 (Δ 0.0000)
///
/// SROCC bit-exact — monotone spline is rank-preserving.
///
/// **Anchor landing** (median over V9 anchor parquet target_score band,
/// post-α-mix + post-spline):
///   - target=60 (PJND / JND) → score=60.000 (bit-exact at knot)
///   - target=30 (JOD)        → score=30.000 (bit-exact at knot)
///   - target=100 (lossless)  → score=100.000
///   - target=0   (worst-codec floor) → score=0.000
///
/// Methodology: `benchmarks/v_compression_v2_2026-05-20_methodology.md`.
pub(crate) fn mlp_bake_preview_v0_5_compression_v2() -> &'static [u8] {
    include_bytes!("../weights/v_compression_v2_2026-05-20.bin")
}

static PROFILE_PREVIEW_V0_5_COMPRESSION_V2: ProfileParams = ProfileParams {
    weights: &WEIGHTS_PREVIEW_V0_2,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    // The bake's `zentrain.output_calibration_spline` metadata maps
    // the per-sample-α mixed output (distance-shaped, range ≈
    // [-27, 20] across V9 anchor parquet) onto the dial-honest
    // [0, 100] score scale via a 7-knot PCHIP spline. JND lands at
    // integer 60, JOD lands at integer 30 (bit-exact knots). The
    // spline output IS the final score; no `100 - A·d^B` transform.
    // Runtime dispatch lives in `zensim::metric::forward_one_bake`
    // (spline applies AFTER per-sample-α mix, AFTER any optional
    // tanh-pin — this bake carries neither tanh_output_head nor
    // feature_transforms).
    skip_score_mapping: true,
    mlp_bytes: Some(mlp_bake_preview_v0_5_compression_v2),
    mlp_bytes_b3: None,
    mlp_primary_mix: 1.0,
    // 300-feature input = 228 standard + 72 masked (no IW pool;
    // LARGE schema matches PreviewV0_5Compression).
    extended_features: true,
    compute_iw_features: false,
    // **No soft-clamp.** The Compression base profile uses
    // `soft_clamp_score: true` to squash the per-sample-α RankNet's
    // unbounded distance-shaped output into a soft logistic — but
    // that squash maps the ≈ [-27, 20] raw range into roughly
    // [2, 18], collapsing the user-facing dial. With the spline
    // mapping raw onto [0, 100] integer anchors, the soft-clamp is
    // not just unnecessary, it would re-squash the dial out of
    // calibration. The production hard clamp `clamp(0, 100)` after
    // the spline is a no-op for in-distribution inputs and catches
    // only the OOD extrapolation tails.
    soft_clamp_score: false,
    extrapolate_score: false,
    ensemble_classifier_bytes: None,
    mlp_bytes_compression: None,
};

// =============================================================================
// EXP-CROSS-CODEC-V10 (2026-05-20): score-space reallocation
// =============================================================================
//
// Per user direction 2026-05-20: the zensim score-space reallocates as
// `lossless = 100, JND = 80, JOD = 50, q=0 floor = 0, pathological < 0`.
// V10 replaces V9's `JND = 60 / JOD = 30 / clamp at 0` table.
//
// Three V10 ship rotations: BalancedV3, CompressionV3, TunerV4. Each
// keeps the underlying V2/V3 network bytes verbatim and only refits
// the `zentrain.output_calibration_spline` against the V10 11-band
// anchor parquet. The `extrapolate_score: true` flag on each profile
// removes the [0, 100] clamp so the spline's linear extrapolation past
// the worst-anchor knot flows through (worst-q codec output maps to a
// negative score, signalling "pathological / unreasonable" rather than
// collapsing to a tie at 0).

/// PreviewV0_5BalancedV3 bake bytes (2026-05-20, EXP-CROSS-CODEC-V10).
///
/// Same V_22-mix-LARGE+iwssim network as
/// [`mlp_bake_preview_v0_5_balanced`] / [`mlp_bake_preview_v0_5_balanced_v2`]
/// with a fresh PCHIP spline calibrated against the V10 11-band anchor
/// parquet. The spline knots after dropping direction-violation bands
/// (8 of 11 band targets survived):
///
/// ```text
/// x= -23.05 → y= 100   (lossless)
/// x= -18.93 → y=  95   (near-lossless)
/// x= -16.45 → y=  90   (visually identical)
/// x=  -5.46 → y=  80   (JND)
/// x=   1.19 → y=  65   (mildly noticeable)
/// x=   7.53 → y=  50   (JOD)
/// x=  10.08 → y=  10   (clear artifacts at scale)
/// x=  13.15 → y=   0   (worst-q floor)
/// ```
pub(crate) fn mlp_bake_preview_v0_5_balanced_v3() -> &'static [u8] {
    include_bytes!("../weights/v_balanced_v3_2026-05-20.bin")
}

static PROFILE_PREVIEW_V0_5_BALANCED_V3: ProfileParams = ProfileParams {
    weights: &WEIGHTS_PREVIEW_V0_2,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    skip_score_mapping: true,
    mlp_bytes: Some(mlp_bake_preview_v0_5_balanced_v3),
    mlp_bytes_b3: None,
    mlp_primary_mix: 1.0,
    extended_features: true,
    compute_iw_features: false,
    soft_clamp_score: false,
    // V10 dial: spline output flows through unclamped so pathological
    // input (butter >> 12) maps below 0.
    extrapolate_score: true,
    ensemble_classifier_bytes: None,
    mlp_bytes_compression: None,
};

/// PreviewV0_5CompressionV3 bake bytes (2026-05-20, EXP-CROSS-CODEC-V10).
///
/// Same V_24-per-sample-α s4 network as
/// [`mlp_bake_preview_v0_5_compression`] / [`mlp_bake_preview_v0_5_compression_v2`]
/// with a fresh PCHIP spline calibrated against the V10 11-band anchor
/// parquet. The spline knots after dropping direction-violation bands
/// (7 of 11 band targets survived):
///
/// ```text
/// x= -26.26 → y= 100   (lossless)
/// x= -20.19 → y=  95   (near-lossless)
/// x= -16.95 → y=  90   (visually identical)
/// x=  -3.58 → y=  80   (JND)
/// x=   3.84 → y=  65   (mildly noticeable)
/// x=   9.35 → y=  10   (clear artifacts at scale)
/// x=  13.89 → y=   0   (worst-q floor)
/// ```
pub(crate) fn mlp_bake_preview_v0_5_compression_v3() -> &'static [u8] {
    include_bytes!("../weights/v_compression_v3_2026-05-20.bin")
}

static PROFILE_PREVIEW_V0_5_COMPRESSION_V3: ProfileParams = ProfileParams {
    weights: &WEIGHTS_PREVIEW_V0_2,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    skip_score_mapping: true,
    mlp_bytes: Some(mlp_bake_preview_v0_5_compression_v3),
    mlp_bytes_b3: None,
    mlp_primary_mix: 1.0,
    extended_features: true,
    compute_iw_features: false,
    soft_clamp_score: false,
    extrapolate_score: true,
    ensemble_classifier_bytes: None,
    mlp_bytes_compression: None,
};

/// PreviewV0_5TunerV4 bake bytes (2026-05-20, EXP-CROSS-CODEC-V10).
///
/// V_24-per-sample-α + tanh-output-head topology (same network as
/// [`mlp_bake_preview_v0_5_tuner_v3`]'s underlying V9 weights) with a
/// fresh PCHIP spline calibrated against the V10 11-band anchor
/// parquet. The V9 spline was stripped via
/// `scripts/v_next/strip_spline_metadata.py` to recover the score-shaped
/// network output, then the V10 spline was fit on top via
/// `scripts/v_next/calibrate_v9_spline.py`. The spline knots after
/// dropping direction-violation bands (9 of 11 band targets survived):
///
/// ```text
/// x=   5.86 → y=   0   (worst-q floor)
/// x=   8.10 → y=  10
/// x=  17.56 → y=  35
/// x=  35.86 → y=  50   (JOD)
/// x=  48.78 → y=  65
/// x=  60.38 → y=  80   (JND)
/// x=  83.10 → y=  90
/// x=  86.87 → y=  95
/// x=  97.16 → y= 100   (lossless)
/// ```
pub(crate) fn mlp_bake_preview_v0_5_tuner_v4() -> &'static [u8] {
    include_bytes!("../weights/v_tuner_v10_2026-05-20.bin")
}

static PROFILE_PREVIEW_V0_5_TUNER_V4: ProfileParams = ProfileParams {
    weights: &WEIGHTS_PREVIEW_V0_2,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    skip_score_mapping: true,
    mlp_bytes: Some(mlp_bake_preview_v0_5_tuner_v4),
    mlp_bytes_b3: None,
    mlp_primary_mix: 1.0,
    extended_features: true,
    compute_iw_features: true,
    soft_clamp_score: false,
    extrapolate_score: true,
    ensemble_classifier_bytes: None,
    mlp_bytes_compression: None,
};

// PreviewV0_5TunerV5 was absorbed into PreviewV0_3 on 2026-05-24 PM
// (the new public canonical bake). The bake bytes live at
// `mlp_bake_preview_v0_3` above; the V_18 lineage bake bytes are
// preserved at `zensim/weights/v0_18_zerobiased_lz4_2026-05-13.bin`.

// --- EXP-CROSS-CODEC-V11-E per-codec-affine variants (task #186, 2026-05-20) ---
//
// Each variant ships the same trained MLP + spline as the corresponding
// V10 ship, with `zentrain.per_codec_calibration` metadata injected on
// top. The metadata is GATED on a codec hint supplied via
// `Zensim::compute_with_codec_hint`; without the hint the output is
// bit-exact to the un-calibrated parent. See
// `benchmarks/v11_e_per_codec_falsification_2026-05-20.md` for the fit
// + falsification numbers.

/// PreviewV0_5TunerV4Calibrated bake bytes (2026-05-20, EXP-CROSS-CODEC-V11-E).
pub(crate) fn mlp_bake_preview_v0_5_tuner_v4_calibrated() -> &'static [u8] {
    include_bytes!("../weights/v_tuner_v4_per_codec_2026-05-20.bin")
}

static PROFILE_PREVIEW_V0_5_TUNER_V4_CALIBRATED: ProfileParams = ProfileParams {
    weights: &WEIGHTS_PREVIEW_V0_2,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    skip_score_mapping: true,
    mlp_bytes: Some(mlp_bake_preview_v0_5_tuner_v4_calibrated),
    mlp_bytes_b3: None,
    mlp_primary_mix: 1.0,
    extended_features: true,
    compute_iw_features: true,
    soft_clamp_score: false,
    extrapolate_score: true,
    ensemble_classifier_bytes: None,
    mlp_bytes_compression: None,
};

/// PreviewV0_5BalancedV3Calibrated bake bytes (2026-05-20, EXP-CROSS-CODEC-V11-E).
pub(crate) fn mlp_bake_preview_v0_5_balanced_v3_calibrated() -> &'static [u8] {
    include_bytes!("../weights/v_balanced_v3_per_codec_2026-05-20.bin")
}

static PROFILE_PREVIEW_V0_5_BALANCED_V3_CALIBRATED: ProfileParams = ProfileParams {
    weights: &WEIGHTS_PREVIEW_V0_2,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    skip_score_mapping: true,
    mlp_bytes: Some(mlp_bake_preview_v0_5_balanced_v3_calibrated),
    mlp_bytes_b3: None,
    mlp_primary_mix: 1.0,
    extended_features: true,
    compute_iw_features: true,
    soft_clamp_score: false,
    extrapolate_score: true,
    ensemble_classifier_bytes: None,
    mlp_bytes_compression: None,
};

/// PreviewV0_5CompressionV3Calibrated bake bytes (2026-05-20, EXP-CROSS-CODEC-V11-E).
pub(crate) fn mlp_bake_preview_v0_5_compression_v3_calibrated() -> &'static [u8] {
    include_bytes!("../weights/v_compression_v3_per_codec_2026-05-20.bin")
}

static PROFILE_PREVIEW_V0_5_COMPRESSION_V3_CALIBRATED: ProfileParams = ProfileParams {
    weights: &WEIGHTS_PREVIEW_V0_2,
    blur_radius: 5,
    blur_passes: 1,
    num_scales: 4,
    score_mapping_a: 18.0,
    score_mapping_b: 0.7,
    skip_score_mapping: true,
    mlp_bytes: Some(mlp_bake_preview_v0_5_compression_v3_calibrated),
    mlp_bytes_b3: None,
    mlp_primary_mix: 1.0,
    extended_features: true,
    compute_iw_features: true,
    soft_clamp_score: false,
    extrapolate_score: true,
    ensemble_classifier_bytes: None,
    mlp_bytes_compression: None,
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
