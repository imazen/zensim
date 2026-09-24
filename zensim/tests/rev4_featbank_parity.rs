// Copyright (c) Imazen LLC.
// Licensed under AGPL-3.0-or-later OR the Imazen commercial license.

//! **Rev4 feature-bank parity gates** (`rev4/FEATBANK_IMPL_brief.md`).
//!
//! This file is deliberately its OWN test executable: `for_each_token_
//! permutation` mutates process-wide SIMD dispatch, so tier-forced probes
//! may never share a process with the parallel unit tests
//! (`attribution_cross_tier.rs` established the convention).
//!
//! * **R4-A (existing-ID bit identity × tiers)** — with every rev4 toggle
//!   ON, slots f0..f985 must emit `to_bits`-identical values to the same
//!   request with all rev4 toggles OFF, on every dispatchable SIMD tier
//!   (native, forced v3, forced scalar among them) and across the
//!   synthetic geometry matrix: odd sizes, sub-64 dims (reflect-padded),
//!   non-tight strides, multi-strip heights, `H_TILE_WIDTH`-crossing
//!   widths.
//! * **R4-B (rev4 tier determinism)** — rev4 segments are bit-identical
//!   across permutations within a SIMD tier. Cross-tier drift is measured
//!   and reported, following the existing era-2 contract.
//! * **R4-C (corpus toggle identity)** — corpus-gated: 64 CID22 TRAIN +
//!   64 SafeSyn + 16 KADID TRAIN pairs, f0..f985 identical with rev4 on
//!   vs off, serial and MT8.
//! * **R4-D (CI smoke)** — 16 deterministic generated pairs run without
//!   external mounts; every existing slot remains bit-identical.

#![cfg(all(feature = "training", feature = "feature-regime-v2"))]

mod common;

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use zensim::feature_v2::{V1PoolsMode, V2NewFeatureToggles, V2Scratch};
use zensim::source::StridedBytes;
use zensim::{PixelFormat, RgbSlice, Zensim, ZensimProfile};

/// First rev4 slot — f986 (the registry's `REV4_BASE`; re-stated here so
/// the gate text survives registry refactors).
const REV4_BASE: usize = 986;
const GMSBANK_BASE: usize = 1322;

fn toggles_on() -> V2NewFeatureToggles {
    V2NewFeatureToggles {
        append_block: true,
        append2_block: true,
        csfw_block: true,
        dvifm_block: true,
        rev4_gridblk: true,
        rev4_ringbasis: true,
        rev4_tailhist: true,
        rev4_arttype: true,
        v1_pools: V1PoolsMode::Full,
        ..V2NewFeatureToggles::default()
    }
}

fn toggles_off() -> V2NewFeatureToggles {
    V2NewFeatureToggles {
        rev4_gridblk: false,
        rev4_ringbasis: false,
        rev4_tailhist: false,
        rev4_arttype: false,
        ..toggles_on()
    }
}

fn pair(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let r = common::generators::gen_value_noise(w, h, 0xC0FFEE ^ w as u32);
    let d = common::generators::distort_block_artifacts(&r, w, h);
    (r, d)
}

fn extract(
    src: &[[u8; 3]],
    dst: &[[u8; 3]],
    w: usize,
    h: usize,
    toggles: V2NewFeatureToggles,
    parallel: bool,
) -> Vec<f64> {
    let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(parallel);
    let mut scratch = V2Scratch::new();
    z.compute_folded720_append_features_streaming(
        &RgbSlice::new(src, w, h),
        &RgbSlice::new(dst, w, h),
        toggles,
        &mut scratch,
    )
    .expect("streaming walk computes")
    .features()
    .to_vec()
}

/// Classify a permutation label into a dispatch tier (mirrors
/// `cross_tier.rs::classify_tier` — same convention, same grouping).
fn classify_tier(disabled: &[&str]) -> &'static str {
    if disabled
        .iter()
        .any(|t| t.contains("NEON") || t.contains("Arm64"))
    {
        if disabled.contains(&"NEON") {
            "scalar"
        } else {
            "NEON"
        }
    } else if disabled.iter().any(|t| t.contains("WASM")) {
        if disabled.contains(&"WASM128") {
            "scalar"
        } else {
            "WASM128"
        }
    } else {
        let v3 = disabled.contains(&"x86-64-v3");
        let v4 = disabled.contains(&"AVX-512");
        let v2 = disabled.contains(&"x86-64-v2");
        if v3 || v2 {
            "scalar/SSE2"
        } else if v4 {
            "AVX2 (v3)"
        } else {
            "AVX-512 (v4)"
        }
    }
}

fn ulp_distance(a: f64, b: f64) -> Option<u64> {
    if a.is_nan() || b.is_nan() {
        return None;
    }
    if a == b {
        return Some(0);
    }
    if a.is_sign_positive() != b.is_sign_positive() && a != 0.0 && b != 0.0 {
        return None;
    }
    Some((a.to_bits() as i64).abs_diff(b.to_bits() as i64))
}

/// **R4-A + R4-B** — the full geometry × tier matrix in one sweep:
/// within each tier the on/off f0..f985 prefix must be bit-identical;
/// the rev4 segments must be bit-identical across permutations mapping
/// to the SAME tier (the within-tier 0-ULP bar `cross_tier.rs` sets for
/// the existing vector); cross-tier drift of the rev4 segments is
/// MEASURED and reported, matching the existing contract that tiers
/// legitimately drift.
#[test]
fn rev4_identity_and_segments_across_all_tiers() {
    // (w, h, strided): odd, sub-64 (reflect-padded), non-tight-stride,
    // multi-strip heights (h>128 → 128+remainder strips), past-64 rows.
    let geoms: [(usize, usize, bool); 8] = [
        (200, 200, false),
        (96, 80, false),
        (300, 200, false),
        (127, 288, true),
        (50, 300, false),
        (300, 50, false),
        (40, 40, false),
        (130, 130, true),
    ];
    for &(w, h, strided) in &geoms {
        let (src, dst) = pair(w, h);
        // Padded copy for the non-tight arm (stride = w*3 + 13).
        let stride = w * 3 + 13;
        let (pad_src, pad_dst) = if strided {
            let mut ps = vec![0u8; stride * h];
            let mut pd = vec![0u8; stride * h];
            for y in 0..h {
                for x in 0..w {
                    let o = y * stride + x * 3;
                    ps[o..o + 3].copy_from_slice(&src[y * w + x]);
                    pd[o..o + 3].copy_from_slice(&dst[y * w + x]);
                }
            }
            (ps, pd)
        } else {
            (Vec::new(), Vec::new())
        };
        let mut tier_first: std::collections::BTreeMap<&'static str, Vec<f64>> =
            std::collections::BTreeMap::new();
        let mut native_on: Option<Vec<f64>> = None;
        let mut max_ulp = 0u64;
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let run = |toggles: V2NewFeatureToggles| -> Vec<f64> {
                if strided {
                    let s = StridedBytes::try_new(&pad_src, w, h, stride, PixelFormat::Srgb8Rgb)
                        .expect("strided src");
                    let d = StridedBytes::try_new(&pad_dst, w, h, stride, PixelFormat::Srgb8Rgb)
                        .expect("strided dst");
                    let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
                    let mut scratch = V2Scratch::new();
                    z.compute_folded720_append_features_streaming(&s, &d, toggles, &mut scratch)
                        .expect("strided walk")
                        .features()
                        .to_vec()
                } else {
                    extract(&src, &dst, w, h, toggles, false)
                }
            };
            let on = run(toggles_on());
            let off = run(toggles_off());
            // R4-A: f0..f985 identical within this tier.
            for i in 0..REV4_BASE {
                assert_eq!(
                    on[i].to_bits(),
                    off[i].to_bits(),
                    "{w}x{h} strided={strided} tier {}: existing slot {i} \
                     moved with rev4 toggles ({:e} -> {:e})",
                    perm.label,
                    off[i],
                    on[i]
                );
            }
            // R4-B within-tier: rev4 segments identical to the first
            // permutation that mapped to this tier (the cross_tier 0-ULP
            // bar), and drift vs the native tier measured.
            let tier = classify_tier(&perm.disabled);
            match tier_first.get(tier) {
                None => {
                    tier_first.insert(tier, on.clone());
                }
                Some(first) => {
                    for i in REV4_BASE..first.len().min(on.len()) {
                        assert_eq!(
                            first[i].to_bits(),
                            on[i].to_bits(),
                            "{w}x{h} strided={strided}: rev4 slot {i} differs \
                             WITHIN tier {} ({} vs {}): {:e} -> {:e}",
                            tier,
                            perm.label,
                            tier,
                            first[i],
                            on[i]
                        );
                    }
                }
            }
            match &native_on {
                None => native_on = Some(on),
                Some(first) => {
                    for i in REV4_BASE..first.len().min(on.len()) {
                        if let Some(u) = ulp_distance(first[i], on[i])
                            && u > max_ulp
                        {
                            max_ulp = u;
                            let rel = (first[i] - on[i]).abs() / first[i].abs().max(1e-30);
                            eprintln!(
                                "  {w}x{h} {strided}: new worst rev4 drift \
                                 f{i} tier {} rel={rel:.3e} ({:e} vs {:e})",
                                perm.label, first[i], on[i]
                            );
                        }
                    }
                }
            }
        });
        eprintln!(
            "{w}x{h} strided={strided}: {} tier permutations, \
             {} tiers, rev4 cross-tier max ULP={max_ulp}",
            report.permutations_run,
            tier_first.len()
        );
        assert!(report.permutations_run >= 3, "tier coverage too thin");
    }
}

/// C8 reuses the rev4 tier harness. The existing 1322 slots must be exact
/// within each tier with C8 on/off; C8 itself is exact on identity and
/// repeatable within a tier. Cross-tier drift follows the rev4 gate's
/// measured-and-reported policy rather than an invented ULP limit.
#[test]
fn gmsbank_prefix_identity_and_tier_consistency() {
    for (w, h) in [(200, 200), (127, 288)] {
        let (src, dst) = pair(w, h);
        let mut first_by_tier: std::collections::BTreeMap<&'static str, Vec<f64>> =
            std::collections::BTreeMap::new();
        let mut scalar: Option<Vec<f64>> = None;
        let mut maximum_relative_drift = 0.0f64;
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            let off = extract(&src, &dst, w, h, toggles_on(), false);
            let on_toggles = V2NewFeatureToggles {
                gmsbank: true,
                ..toggles_on()
            };
            let on = extract(&src, &dst, w, h, on_toggles, false);
            assert_eq!(off.len(), GMSBANK_BASE);
            assert_eq!(on.len(), 1502);
            for i in 0..GMSBANK_BASE {
                assert_eq!(
                    off[i].to_bits(),
                    on[i].to_bits(),
                    "{w}x{h} {}: C8 moved prefix f{i}",
                    perm.label
                );
            }
            let same = extract(&src, &src, w, h, on_toggles, false);
            for (i, value) in same[GMSBANK_BASE..].iter().enumerate() {
                assert_eq!(
                    *value,
                    0.0,
                    "{w}x{h} {} identity f{}",
                    perm.label,
                    GMSBANK_BASE + i
                );
            }
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(8)
                .build()
                .expect("MT8");
            let mt8 = pool.install(|| extract(&src, &dst, w, h, on_toggles, true));
            for i in GMSBANK_BASE..1502 {
                assert_eq!(
                    on[i].to_bits(),
                    mt8[i].to_bits(),
                    "{w}x{h} {} MT8 changed f{i}",
                    perm.label
                );
            }
            let tier = classify_tier(&perm.disabled);
            if let Some(first) = first_by_tier.get(tier) {
                for i in GMSBANK_BASE..1502 {
                    assert_eq!(
                        first[i].to_bits(),
                        on[i].to_bits(),
                        "{w}x{h} {} same-tier f{i}",
                        perm.label
                    );
                }
            } else {
                first_by_tier.insert(tier, on.clone());
            }
            if tier == "scalar/SSE2" || tier == "scalar" {
                scalar = Some(on);
            }
        });
        let scalar = scalar.expect("scalar tier");
        for values in first_by_tier.values() {
            for i in GMSBANK_BASE..1502 {
                let scale = scalar[i].abs().max(1e-12);
                maximum_relative_drift =
                    maximum_relative_drift.max((values[i] - scalar[i]).abs() / scale);
            }
        }
        eprintln!(
            "C8 {w}x{h}: {} tier permutations, {} tiers, max relative drift vs scalar={maximum_relative_drift:.3e}",
            report.permutations_run,
            first_by_tier.len()
        );
        assert!(report.permutations_run >= 3);
    }
}

/// A monotone grayscale ramp has a positive reference gradient, including
/// reflected image borders; the uniform destination has zero gradient.
/// Thus m_d < m_r at every pixel, and every C8 contribution enters loss.
#[test]
fn gmsbank_contrast_reduction_has_exact_zero_gain_in_every_tier() {
    let (w, h) = (127, 128);
    let src: Vec<[u8; 3]> = (0..w * h)
        .map(|i| {
            let x = i % w;
            let y = i / w;
            let gray = (1 + x + y) as u8;
            [gray; 3]
        })
        .collect();
    let dst = vec![[128u8; 3]; w * h];
    let toggles = V2NewFeatureToggles {
        gmsbank: true,
        ..toggles_on()
    };
    let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
        let values = extract(&src, &dst, w, h, toggles, false);
        assert_eq!(values.len(), GMSBANK_BASE + 180);
        let mut total_loss = 0.0;
        for cell in values[GMSBANK_BASE..].as_chunks::<15>().0 {
            for k in 0..5 {
                let loss = cell[k * 3];
                let gain = cell[k * 3 + 1];
                total_loss += loss;
                assert_eq!(
                    gain.to_bits(),
                    0.0f64.to_bits(),
                    "{}: contrast reduction contributed to gain at k={k}",
                    perm.label
                );
            }
        }
        assert!(
            total_loss > 0.0,
            "{}: test pair had no gradients",
            perm.label
        );
    });
    assert!(report.permutations_run >= 3, "tier coverage too thin");
}

/// **R4-D** — mount-free 16-pair CI tier. The corpus gate below covers real
/// TRAIN paths; this catches accidental skips and old-slot regressions in CI.
#[test]
fn rev4_synthetic_16_pair_identity() {
    for n in 0..16usize {
        let (w, h) = (64 + n, 65 + (n * 3) % 17);
        let (src, dst) = pair(w, h);
        let on = extract(&src, &dst, w, h, toggles_on(), false);
        let off = extract(&src, &dst, w, h, toggles_off(), false);
        for i in 0..REV4_BASE {
            assert_eq!(on[i].to_bits(), off[i].to_bits(), "pair {n} f{i}");
        }
    }
}

/// **R4-C** — corpus toggle identity. Loads pairs-tsv rows (PATH COLUMNS
/// ONLY — the lane's no-human-label rule; score columns are never
/// parsed), decodes via `image` (PNG) / `zenjpeg` (JPEG), and asserts
/// f0..f985 `to_bits` equality between the all-on and all-off walks,
/// serial and MT8. The caller supplies corpus access and the expected
/// unsupported SafeSyn count via `just rev4-corpus-tests`; the admitted
/// TRAIN sample is pinned to 11 AVIF and 8 JXL omissions. Missing files,
/// malformed roles and any other unsupported format fail the test.
#[test]
#[ignore = "explicit corpus gate: use just rev4-corpus-tests"]
fn rev4_corpus_toggle_identity() {
    use std::path::Path;

    // Deserialize only role-bearing metadata. Unknown fields, including
    // human targets, are discarded by serde and never materialized here.
    #[derive(serde::Deserialize)]
    struct RoleRow {
        corpus: String,
        reference: String,
        role: String,
    }
    #[derive(serde::Deserialize)]
    struct RoleManifest {
        rows: Vec<RoleRow>,
    }

    let inputs_path = std::env::var("ZENSIM_REV4_KADID_INPUTS")
        .expect("caller must set ZENSIM_REV4_KADID_INPUTS");
    let expected_unsupported: usize = std::env::var("ZENSIM_REV4_EXPECT_UNSUPPORTED_SAFESYN")
        .expect("caller must set ZENSIM_REV4_EXPECT_UNSUPPORTED_SAFESYN")
        .parse()
        .expect("expected unsupported count must be an integer");
    const EXPECTED_SAFESYN_AVIF: usize = 11;
    const EXPECTED_SAFESYN_JXL: usize = 8;
    assert_eq!(
        expected_unsupported,
        EXPECTED_SAFESYN_AVIF + EXPECTED_SAFESYN_JXL,
        "caller count must match the pinned SafeSyn AVIF/JXL census"
    );
    let input_text = std::fs::read_to_string(&inputs_path).expect("read KADID INPUTS.json");
    let input_json: RoleManifest = serde_json::from_str(&input_text).expect("parse INPUTS.json");
    let mut kadid_role = std::collections::HashMap::<String, String>::new();
    for row in input_json.rows {
        if row.corpus != "kadid" {
            continue;
        }
        let old = kadid_role.insert(row.reference.clone(), row.role.clone());
        assert!(
            old.as_deref().is_none_or(|v| v == row.role),
            "conflicting role for {}",
            row.reference
        );
    }
    assert!(!kadid_role.is_empty(), "INPUTS has no KADID roles");

    struct Spec {
        name: &'static str,
        tsv: &'static str,
        n: usize,
        kadid_train_refs: bool,
    }
    let specs = [
        Spec {
            name: "cid22",
            tsv: "/var/tmp/zensim-validation-2026-09-14/baseline-recovery/cid22-train-pairs.tsv",
            n: 64,
            kadid_train_refs: false,
        },
        Spec {
            name: "safesyn",
            tsv: "/var/tmp/zensim-validation-2026-09-14/baseline-recovery/safesyn-train-pairs.tsv",
            n: 64,
            kadid_train_refs: false,
        },
        Spec {
            name: "kadid",
            tsv: "/mnt/v/dataset/kadid10k/kadid_pairs_ab.tsv",
            n: 16,
            kadid_train_refs: true,
        },
    ];

    // Formats this test's dev-deps decode. AVIF/JXL pairs (SafeSyn) are
    // covered by the canonical omni-decode path instead: the same gate ran
    // through `extract_features_372col --full-rev4` (the shipped decode
    // stack) — see benchmarks/rev4_featbank_impl_2026-09-23.md for the
    // recorded 144-pair × serial/MT8 × 3-tier zero-diff matrix.
    let decode = |path: &Path| -> Option<(Vec<[u8; 3]>, usize, usize)> {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if ext == "jpg" || ext == "jpeg" {
            use enough::Unstoppable;
            let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
            let dec = zenjpeg::decoder::Decoder::new()
                .decode(&bytes, Unstoppable)
                .unwrap_or_else(|e| panic!("zenjpeg decode {path:?}: {e}"));
            let (w, h) = dec.dimensions();
            let out = dec.pixels_u8().expect("u8 jpeg");
            let px = out
                .as_chunks::<3>()
                .0
                .iter()
                .map(|c| [c[0], c[1], c[2]])
                .collect();
            Some((px, w as usize, h as usize))
        } else if matches!(
            ext.as_str(),
            "png" | "bmp" | "tiff" | "tif" | "webp" | "ppm" | "pgm"
        ) {
            let img = image::open(path)
                .unwrap_or_else(|e| panic!("decode {path:?}: {e}"))
                .to_rgb8();
            let px = img
                .as_raw()
                .as_chunks::<3>()
                .0
                .iter()
                .map(|c| [c[0], c[1], c[2]])
                .collect();
            Some((px, img.width() as usize, img.height() as usize))
        } else {
            None
        }
    };

    let mut checked = 0usize;
    let mut skipped_fmt = 0usize;
    for spec in &specs {
        let text = std::fs::read_to_string(spec.tsv)
            .unwrap_or_else(|e| panic!("{} corpus TSV {} unavailable: {e}", spec.name, spec.tsv));
        // PATH COLUMNS ONLY: split each line at its first two tabs and
        // never touch column 3 (human labels are outside this lane).
        let mut rows: Vec<(String, String)> = Vec::new();
        for line in text.lines().skip(1) {
            let mut it = line.split('\t');
            let (r, d) = (it.next().unwrap_or(""), it.next().unwrap_or(""));
            assert!(
                !r.is_empty() && !d.is_empty(),
                "{} malformed path row",
                spec.name
            );
            if spec.kadid_train_refs {
                let role = kadid_role
                    .get(r)
                    .unwrap_or_else(|| panic!("missing role for {r}"));
                if role != "train" && role != "fit" {
                    continue;
                }
            }
            rows.push((r.to_string(), d.to_string()));
            if rows.len() >= spec.n {
                break;
            }
        }
        assert_eq!(
            rows.len(),
            spec.n,
            "{}: insufficient admitted TRAIN rows",
            spec.name
        );
        let mut spec_checked = 0usize;
        let mut skipped_avif = 0usize;
        let mut skipped_jxl = 0usize;
        for (r, d) in &rows {
            let (src, sw, sh) = decode(Path::new(r))
                .unwrap_or_else(|| panic!("{}: unsupported reference format: {r}", spec.name));
            let (dst, dw, dh) = match decode(Path::new(d)) {
                Some(v) => v,
                None => {
                    assert_eq!(spec.name, "safesyn", "unsupported distortion: {d}");
                    let ext = Path::new(d)
                        .extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("")
                        .to_ascii_lowercase();
                    match ext.as_str() {
                        "avif" => skipped_avif += 1,
                        "jxl" => skipped_jxl += 1,
                        _ => panic!("unexpected unsupported distortion format: {d}"),
                    }
                    assert!(
                        std::fs::metadata(d)
                            .unwrap_or_else(|e| panic!("missing skipped distortion {d}: {e}"))
                            .is_file(),
                        "skipped distortion is not a file: {d}"
                    );
                    continue;
                }
            };
            assert_eq!((sw, sh), (dw, dh), "dim mismatch {r} vs {d}");
            for parallel in [false, true] {
                let (on, off) = if parallel {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(8)
                        .build()
                        .expect("pool");
                    pool.install(|| {
                        (
                            extract(&src, &dst, sw, sh, toggles_on(), true),
                            extract(&src, &dst, sw, sh, toggles_off(), true),
                        )
                    })
                } else {
                    (
                        extract(&src, &dst, sw, sh, toggles_on(), false),
                        extract(&src, &dst, sw, sh, toggles_off(), false),
                    )
                };
                for i in 0..REV4_BASE {
                    assert_eq!(
                        on[i].to_bits(),
                        off[i].to_bits(),
                        "{} {r} par={parallel}: existing slot {i} moved \
                         with rev4 toggles ({:e} -> {:e})",
                        spec.name,
                        off[i],
                        on[i]
                    );
                }
                checked += 1;
                spec_checked += 1;
            }
        }
        let expected = if spec.name == "safesyn" {
            [EXPECTED_SAFESYN_AVIF, EXPECTED_SAFESYN_JXL]
        } else {
            [0, 0]
        };
        assert_eq!(
            [skipped_avif, skipped_jxl],
            expected,
            "{}: unexpected AVIF/JXL omission counts",
            spec.name
        );
        let spec_skipped = skipped_avif + skipped_jxl;
        skipped_fmt += spec_skipped;
        eprintln!(
            "{}: {} pairs checked (serial + MT8), {spec_skipped} skipped \
             (AVIF={skipped_avif}, JXL={skipped_jxl})",
            spec.name, spec_checked,
        );
    }
    eprintln!(
        "rev4 corpus toggle identity: {checked} pair-mode extractions, \
         {skipped_fmt} pairs skipped (AVIF={EXPECTED_SAFESYN_AVIF}, \
         JXL={EXPECTED_SAFESYN_JXL}; covered by the omni extractor matrix)"
    );
    assert_eq!(checked, 2 * (64 + 64 + 16 - expected_unsupported));
}
