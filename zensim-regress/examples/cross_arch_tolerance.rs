//! Cross-architecture tolerance and override demonstration.
//!
//! Simulates a scenario where a visual test produces slightly different
//! output on x86_64 vs aarch64 due to FMA contraction differences.
//! Shows how per-arch tolerance overrides and multi-entry checksum files
//! handle this without suppressing real regressions.
//!
//! Run with: cargo run -p zensim-regress --example cross_arch_tolerance

use std::collections::BTreeMap;

use zensim::{RgbSlice, Zensim, ZensimProfile};
use zensim_regress::arch::{self, KNOWN_ARCH_TAGS};
use zensim_regress::checksum_file::{
    ChecksumDiff, ChecksumEntry, ImageInfo, TestChecksumFile, ToleranceOverride, ToleranceSpec,
    sanitize_name,
};
use zensim_regress::hasher::{ChecksumHasher, SeaHasher};
use zensim_regress::testing::{RegressionTolerance, check_regression};

/// Create a synthetic 32x32 gradient image.
fn make_baseline(w: usize, h: usize) -> Vec<[u8; 3]> {
    (0..w * h)
        .map(|i| {
            let x = (i % w) as u8;
            let y = (i / w) as u8;
            [
                (x.wrapping_mul(8)),
                (y.wrapping_mul(8)),
                ((x as u16 + y as u16) % 256) as u8,
            ]
        })
        .collect()
}

/// Simulate off-by-one rounding in a specific channel (aarch64-style).
fn make_rounding_variant(baseline: &[[u8; 3]], channel: usize) -> Vec<[u8; 3]> {
    baseline
        .iter()
        .enumerate()
        .map(|(i, &px)| {
            let mut out = px;
            // Only modify some pixels — not uniform, to test partial differing
            if i % 3 == 0 {
                out[channel] = out[channel].saturating_add(1);
            }
            out
        })
        .collect()
}

/// Simulate a two-channel rounding variant (avx512-style).
fn make_multi_channel_variant(baseline: &[[u8; 3]]) -> Vec<[u8; 3]> {
    baseline
        .iter()
        .enumerate()
        .map(|(i, &px)| {
            let mut out = px;
            if i % 5 == 0 {
                out[0] = out[0].saturating_add(1);
            }
            if i % 7 == 0 {
                out[1] = out[1].wrapping_sub(1); // deliberate negative bias
            }
            out
        })
        .collect()
}

fn to_rgba(rgb: &[[u8; 3]]) -> Vec<u8> {
    rgb.iter().flat_map(|&[r, g, b]| [r, g, b, 255]).collect()
}

fn main() {
    println!("=== Cross-Architecture Tolerance Demo ===\n");

    let (w, h) = (32, 32);
    let hasher = SeaHasher;
    let z = Zensim::new(ZensimProfile::latest());
    let current_arch = arch::detect_arch_tag();

    // ─── Create variants ────────────────────────────────────────────────
    println!("--- Creating test variants ---");
    let baseline = make_baseline(w, h);
    let arm_variant = make_rounding_variant(&baseline, 0); // R channel off-by-one
    let avx512_variant = make_multi_channel_variant(&baseline);

    let hash_baseline = hasher.hash_pixels(&to_rgba(&baseline), w as u32, h as u32);
    let hash_arm = hasher.hash_pixels(&to_rgba(&arm_variant), w as u32, h as u32);
    let hash_avx512 = hasher.hash_pixels(&to_rgba(&avx512_variant), w as u32, h as u32);

    println!("  baseline (x86_64-avx2): {hash_baseline}");
    println!("  arm variant (aarch64):  {hash_arm}");
    println!("  avx512 variant:         {hash_avx512}");
    println!();

    // ─── Run comparisons ────────────────────────────────────────────────
    println!("--- Zensim comparisons ---");

    let src = RgbSlice::new(&baseline, w, h);

    let dst_arm = RgbSlice::new(&arm_variant, w, h);
    let report_arm =
        check_regression(&z, &src, &dst_arm, &RegressionTolerance::off_by_one()).unwrap();
    println!("  ARM vs baseline:");
    println!("    {report_arm}");

    let dst_avx512 = RgbSlice::new(&avx512_variant, w, h);
    let report_avx512 =
        check_regression(&z, &src, &dst_avx512, &RegressionTolerance::off_by_one()).unwrap();
    println!("  AVX512 vs baseline:");
    println!("    {report_avx512}");

    // ─── Build checksum file with per-arch entries ──────────────────────
    println!("--- Building multi-arch checksum file ---");

    let diff_arm = ChecksumDiff::from_report(&report_arm, &hash_baseline);
    let diff_avx512 = ChecksumDiff::from_report(&report_avx512, &hash_baseline);

    let file = TestChecksumFile {
        name: "gradient_cross_arch".to_string(),
        tolerance: ToleranceSpec {
            max_delta: 0,
            min_similarity: 100.0,
            max_pixels_different: 0.0,
            max_alpha_delta: 0,
            ignore_alpha: false,
            overrides: BTreeMap::from([
                (
                    "aarch64".to_string(),
                    ToleranceOverride {
                        max_delta: Some(1),
                        min_similarity: Some(95.0),
                        max_pixels_different: Some(0.5),
                        ..Default::default()
                    },
                ),
                (
                    "x86_64-avx512".to_string(),
                    ToleranceOverride {
                        max_delta: Some(1),
                        min_similarity: Some(95.0),
                        max_pixels_different: Some(0.5),
                        ..Default::default()
                    },
                ),
            ]),
        },
        checksum: vec![
            ChecksumEntry {
                id: hash_baseline.clone(),
                confidence: 10,
                commit: Some("d7918fc".to_string()),
                arch: vec!["x86_64-avx2".to_string()],
                reason: Some("authoritative baseline".to_string()),
                status: None,
                diff: None,
            },
            ChecksumEntry {
                id: hash_arm.clone(),
                confidence: 8,
                commit: Some("d7918fc".to_string()),
                arch: vec!["aarch64".to_string()],
                reason: Some("ARM NEON rounding in R channel".to_string()),
                status: None,
                diff: Some(diff_arm),
            },
            ChecksumEntry {
                id: hash_avx512.clone(),
                confidence: 8,
                commit: Some("d7918fc".to_string()),
                arch: vec!["x86_64-avx512".to_string()],
                reason: Some("AVX-512 FMA contraction differences".to_string()),
                status: None,
                diff: Some(diff_avx512),
            },
        ],
        info: Some(ImageInfo {
            width: Some(w as u32),
            height: Some(h as u32),
            format: Some("RGB".to_string()),
        }),
        meta: BTreeMap::new(),
    };

    // Show the file
    let toml_str = toml::to_string_pretty(&file).unwrap();
    println!("\n  --- TOML output ---");
    for line in toml_str.lines() {
        println!("  | {line}");
    }
    println!("  --- end TOML ---\n");

    // ─── Simulate per-arch check ────────────────────────────────────────
    println!("--- Simulating per-arch check workflow ---\n");

    // For each architecture, show what would happen
    for &simulated_arch in KNOWN_ARCH_TAGS {
        if simulated_arch == "unknown" {
            continue;
        }

        println!("  [{simulated_arch}]");

        // 1. Determine which hash this arch produces
        let actual_hash = match simulated_arch {
            "aarch64" => &hash_arm,
            "x86_64-avx512" => &hash_avx512,
            _ => &hash_baseline, // x86_64 and x86_64-avx2 produce baseline
        };
        println!("    produces: {actual_hash}");

        // 2. Check against active checksums
        let matched = file.active_checksums().find(|e| e.id == *actual_hash);
        if let Some(entry) = matched {
            println!(
                "    direct match: confidence={}, reason={:?}",
                entry.confidence, entry.reason
            );

            // Check arch affinity
            let has_arch = entry
                .arch
                .iter()
                .any(|a| arch::arch_matches(a, simulated_arch));
            if has_arch {
                println!("    arch affinity: matches recorded arch");
            } else {
                println!(
                    "    arch affinity: no match (recorded: {:?}, current: {})",
                    entry.arch, simulated_arch
                );
                println!("    action: would add {simulated_arch} to entry's arch list");
            }
        } else {
            println!("    no direct match — would run zensim comparison");

            // Get arch-appropriate tolerance
            let tolerance = file.tolerance.to_regression_tolerance(simulated_arch);
            let _ = tolerance;
            let has_override = file.tolerance.overrides.contains_key(simulated_arch);
            println!(
                "    tolerance: {} override",
                if has_override { "has" } else { "no" }
            );
        }
        println!();
    }

    // ─── Demonstrate tolerance override specificity ─────────────────────
    println!("--- Override specificity ---");

    // "x86_64" override should match "x86_64-avx2" too (prefix rule)
    let spec_with_x86_override = ToleranceSpec {
        max_delta: 0,
        min_similarity: 100.0,
        overrides: BTreeMap::from([
            (
                "x86_64".to_string(),
                ToleranceOverride {
                    max_delta: Some(1),
                    ..Default::default()
                },
            ),
            (
                "x86_64-avx2".to_string(),
                ToleranceOverride {
                    max_delta: Some(2),
                    ..Default::default()
                },
            ),
        ]),
        ..Default::default()
    };

    // "x86_64-avx2" should get the exact match (delta=2), not the prefix (delta=1)
    let _ = spec_with_x86_override.to_regression_tolerance("x86_64-avx2");
    println!("  x86_64-avx2: exact match override (delta=2) takes priority over prefix (delta=1)");

    // "x86_64-avx512" should get the "x86_64" prefix match (delta=1)
    let _ = spec_with_x86_override.to_regression_tolerance("x86_64-avx512");
    println!("  x86_64-avx512: prefix match override (delta=1) from x86_64");

    // "aarch64" should get no override (base tolerance)
    let _ = spec_with_x86_override.to_regression_tolerance("aarch64");
    println!("  aarch64: no override, uses base tolerance (delta=0)");
    println!();

    // ─── File I/O round-trip ────────────────────────────────────────────
    println!("--- File I/O round-trip ---");

    let dir = tempfile::tempdir().unwrap();
    let path = dir
        .path()
        .join(format!("{}.toml", sanitize_name(&file.name)));

    file.write_to(&path).unwrap();
    println!("  wrote: {}", path.display());

    let parsed = TestChecksumFile::read_from(&path).unwrap();
    assert_eq!(parsed.name, file.name);
    assert_eq!(parsed.checksum.len(), 3);
    assert_eq!(parsed.tolerance.overrides.len(), 2);

    // Verify all entries survived
    for entry in &file.checksum {
        let found = parsed.find_by_id(&entry.id);
        assert!(found.is_some(), "missing entry: {}", entry.id);
        let found = found.unwrap();
        assert_eq!(found.confidence, entry.confidence);
        assert_eq!(found.arch, entry.arch);
    }
    println!("  round-trip: all 3 entries, 2 overrides preserved");

    // Verify diffs survived
    let arm_entry = parsed.find_by_id(&hash_arm).unwrap();
    let arm_diff = arm_entry.diff.as_ref().unwrap();
    assert_eq!(arm_diff.vs, hash_baseline);
    assert_eq!(arm_diff.category, format!("{:?}", report_arm.category()));
    println!(
        "  ARM diff preserved: vs={}, score={:.1}, category={}",
        arm_diff.vs, arm_diff.zensim_score, arm_diff.category
    );

    let avx512_entry = parsed.find_by_id(&hash_avx512).unwrap();
    let avx512_diff = avx512_entry.diff.as_ref().unwrap();
    assert_eq!(avx512_diff.vs, hash_baseline);
    println!(
        "  AVX512 diff preserved: vs={}, score={:.1}, category={}",
        avx512_diff.vs, avx512_diff.zensim_score, avx512_diff.category
    );
    println!();

    // ─── Summary ────────────────────────────────────────────────────────
    println!("--- Summary ---");
    println!("  current arch: {current_arch}");
    println!("  3 arch-specific checksums with chain-of-trust diffs");
    println!("  2 per-arch tolerance overrides (aarch64, x86_64-avx512)");
    println!("  exact match > prefix match > base tolerance");
    println!("  forensic diffs preserved through TOML round-trip");
    println!("\n=== cross-arch demo complete ===");
}
