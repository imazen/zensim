//! Full lifecycle demonstration of zensim-regress.
//!
//! This example walks through the complete workflow a developer would use:
//!
//! 1. Create synthetic test images (baseline + variant with rounding differences)
//! 2. Hash them with SeaHasher
//! 3. Run zensim comparison to get a RegressionReport
//! 4. Build ChecksumDiff evidence from the report
//! 5. Assemble a TestChecksumFile with tolerance, entries, diffs, metadata
//! 6. Write to TOML, read back, verify round-trip
//! 7. Query active/authoritative/find_by_id
//! 8. Demonstrate name sanitization and checksum_path
//! 9. Exercise error handling
//!
//! Run with: cargo run -p zensim-regress --example checksum_lifecycle

use std::collections::BTreeMap;

use zensim::{RgbSlice, Zensim, ZensimProfile};
use zensim_regress::arch::{self, KNOWN_ARCH_TAGS};
use zensim_regress::checksum_file::{
    ChecksumDiff, ChecksumEntry, ImageInfo, TestChecksumFile, ToleranceOverride, ToleranceSpec,
    checksum_path, sanitize_name,
};
use zensim_regress::hasher::{ChecksumHasher, SeaHasher};
use zensim_regress::testing::{RegressionTolerance, check_regression};

fn main() {
    println!("=== zensim-regress: full lifecycle demo ===\n");

    // ─── 1. Create synthetic test images ────────────────────────────────
    println!("--- 1. Creating synthetic 32x32 test images ---");

    let (w, h) = (32usize, 32usize);
    let n = w * h;

    // Baseline: smooth gradient
    let baseline_pixels: Vec<[u8; 3]> = (0..n)
        .map(|i| {
            let x = (i % w) as u8 * 8; // 0..248
            let y = (i / w) as u8 * 8;
            [x, y, ((x as u16 + y as u16) / 2) as u8]
        })
        .collect();

    // Variant A: off-by-one rounding (simulates cross-arch difference)
    let variant_a_pixels: Vec<[u8; 3]> = baseline_pixels
        .iter()
        .map(|&[r, g, b]| [r.saturating_add(1), g, b])
        .collect();

    // Variant B: large difference (simulates a real bug)
    let variant_b_pixels: Vec<[u8; 3]> = baseline_pixels
        .iter()
        .map(|&[r, g, b]| [255 - r, g, 255 - b])
        .collect();

    println!("  baseline:  {n} pixels, gradient pattern");
    println!("  variant_a: off-by-one in R channel (rounding)");
    println!("  variant_b: inverted R and B (bug simulation)\n");

    // ─── 2. Hash with SeaHasher ─────────────────────────────────────────
    println!("--- 2. Hashing pixel data with SeaHasher ---");

    let hasher = SeaHasher;

    // SeaHasher expects RGBA bytes, convert RGB→RGBA
    let to_rgba =
        |rgb: &[[u8; 3]]| -> Vec<u8> { rgb.iter().flat_map(|&[r, g, b]| [r, g, b, 255]).collect() };

    let baseline_rgba = to_rgba(&baseline_pixels);
    let variant_a_rgba = to_rgba(&variant_a_pixels);
    let variant_b_rgba = to_rgba(&variant_b_pixels);

    let hash_baseline = hasher.hash_pixels(&baseline_rgba, w as u32, h as u32);
    let hash_variant_a = hasher.hash_pixels(&variant_a_rgba, w as u32, h as u32);
    let hash_variant_b = hasher.hash_pixels(&variant_b_rgba, w as u32, h as u32);

    println!("  baseline:  {hash_baseline}");
    println!("  variant_a: {hash_variant_a}");
    println!("  variant_b: {hash_variant_b}");
    assert_ne!(hash_baseline, hash_variant_a);
    assert_ne!(hash_baseline, hash_variant_b);
    assert_ne!(hash_variant_a, hash_variant_b);

    // Verify determinism
    let hash_baseline_2 = hasher.hash_pixels(&baseline_rgba, w as u32, h as u32);
    assert_eq!(hash_baseline, hash_baseline_2, "hash must be deterministic");

    // Verify dimension sensitivity
    let hash_diff_dims = hasher.hash_pixels(&baseline_rgba, 64, 16);
    assert_ne!(
        hash_baseline, hash_diff_dims,
        "different dims = different hash"
    );
    println!("  determinism: verified");
    println!("  dimension sensitivity: verified\n");

    // ─── 3. Run zensim comparison ───────────────────────────────────────
    println!("--- 3. Running zensim regression comparisons ---");

    let z = Zensim::new(ZensimProfile::latest());
    let src = RgbSlice::new(&baseline_pixels, w, h);

    // Variant A: should be off-by-one rounding
    let dst_a = RgbSlice::new(&variant_a_pixels, w, h);
    let report_a = check_regression(&z, &src, &dst_a, &RegressionTolerance::off_by_one())
        .expect("comparison should succeed");

    println!("  variant_a vs baseline:");
    println!("    passed:   {}", report_a.passed());
    println!("    score:    {:.1}", report_a.score());
    println!("    category: {:?}", report_a.category());
    println!("    max_delta: {:?}", report_a.max_channel_delta());
    println!(
        "    pixels differing: {}/{}",
        report_a.pixels_differing(),
        report_a.pixel_count()
    );

    // Variant B: should be a large difference
    let dst_b = RgbSlice::new(&variant_b_pixels, w, h);
    let report_b = check_regression(&z, &src, &dst_b, &RegressionTolerance::off_by_one())
        .expect("comparison should succeed");

    println!("  variant_b vs baseline:");
    println!("    passed:   {}", report_b.passed());
    println!("    score:    {:.1}", report_b.score());
    println!("    category: {:?}", report_b.category());
    println!("    max_delta: {:?}\n", report_b.max_channel_delta());

    // ─── 4. Build ChecksumDiff evidence ─────────────────────────────────
    println!("--- 4. Building ChecksumDiff from RegressionReports ---");

    let diff_a = ChecksumDiff::from_report(&report_a, &hash_baseline);
    println!("  diff_a (rounding variant):");
    println!("    vs:       {}", diff_a.vs);
    println!("    score:    {:.1}", diff_a.zensim_score);
    println!("    category: {}", diff_a.category);
    println!("    max_delta: {:?}", diff_a.max_channel_delta);
    println!(
        "    pixels_differing_pct: {:.1}%",
        diff_a.pixels_differing_pct.unwrap_or(0.0)
    );
    println!("    rounding_balanced: {:?}", diff_a.rounding_bias_balanced);

    let diff_b = ChecksumDiff::from_report(&report_b, &hash_baseline);
    println!("  diff_b (bug variant):");
    println!("    vs:       {}", diff_b.vs);
    println!("    score:    {:.1}", diff_b.zensim_score);
    println!("    category: {}", diff_b.category);
    println!("    max_delta: {:?}\n", diff_b.max_channel_delta);

    // ─── 5. Assemble TestChecksumFile ───────────────────────────────────
    println!("--- 5. Assembling TestChecksumFile ---");

    let arch_tag = arch::detect_arch_tag();

    let mut file = TestChecksumFile {
        name: "gradient_32x32".to_string(),
        tolerance: ToleranceSpec {
            max_channel_delta: 1,
            min_score: 95.0,
            max_differing_pixel_fraction: 1.0,
            min_identical_channel_fraction: 0.0,
            max_alpha_delta: 0,
            ignore_alpha: false,
            overrides: BTreeMap::from([(
                "aarch64".to_string(),
                ToleranceOverride {
                    max_channel_delta: Some(2),
                    min_score: Some(90.0),
                    ..Default::default()
                },
            )]),
        },
        checksum: vec![
            // Baseline: authoritative reference
            ChecksumEntry {
                id: hash_baseline.clone(),
                confidence: 10,
                commit: Some("d7918fc".to_string()),
                arch: vec![arch_tag.to_string()],
                reason: Some("initial baseline".to_string()),
                status: None,
                diff: None,
            },
            // Variant A: acceptable rounding difference
            ChecksumEntry {
                id: hash_variant_a.clone(),
                confidence: 10,
                commit: Some("d7918fc".to_string()),
                arch: vec!["aarch64".to_string()],
                reason: Some("ARM NEON rounding".to_string()),
                status: None,
                diff: Some(diff_a),
            },
            // Variant B: retired as wrong
            ChecksumEntry {
                id: hash_variant_b.clone(),
                confidence: 0,
                commit: None,
                arch: Vec::new(),
                reason: Some("pre-bugfix output".to_string()),
                status: Some("wrong".to_string()),
                diff: Some(diff_b),
            },
        ],
        info: Some(ImageInfo {
            width: Some(w as u32),
            height: Some(h as u32),
            format: Some("RGB".to_string()),
        }),
        meta: BTreeMap::new(),
    };

    println!("  name: {}", file.name);
    println!("  entries: {} total", file.checksum.len());
    println!("  active:  {} entries", file.active_checksums().count());
    println!(
        "  authoritative: {}",
        file.authoritative()
            .map(|e| e.id.as_str())
            .unwrap_or("none")
    );
    println!();

    // ─── 6. Write to TOML, read back, verify ────────────────────────────
    println!("--- 6. TOML round-trip ---");

    let dir = tempfile::tempdir().expect("create temp dir");
    let toml_path = dir.path().join("checksums/gradient_32x32.toml");

    file.write_to(&toml_path).expect("write TOML");
    println!("  wrote: {}", toml_path.display());

    // Show the TOML content
    let toml_content = std::fs::read_to_string(&toml_path).unwrap();
    println!("\n  --- TOML content ---");
    for line in toml_content.lines() {
        println!("  | {line}");
    }
    println!("  --- end TOML ---\n");

    // Read back
    let parsed = TestChecksumFile::read_from(&toml_path).expect("read TOML");
    assert_eq!(parsed.name, file.name);
    assert_eq!(parsed.checksum.len(), file.checksum.len());
    assert_eq!(parsed.tolerance.max_channel_delta, 1);
    assert_eq!(parsed.tolerance.min_score, 95.0);
    println!("  round-trip verified: name, entries, tolerance all match\n");

    // ─── 7. Query methods ───────────────────────────────────────────────
    println!("--- 7. Querying the checksum file ---");

    // active_checksums
    let active: Vec<_> = parsed.active_checksums().collect();
    println!("  active checksums: {}", active.len());
    for e in &active {
        println!(
            "    {} (confidence={}, arch={:?})",
            e.id, e.confidence, e.arch
        );
    }

    // authoritative
    let auth = parsed.authoritative().expect("should have authoritative");
    println!("  authoritative: {}", auth.id);

    // find_by_id
    let found = parsed.find_by_id(&hash_variant_a);
    assert!(found.is_some());
    println!(
        "  find_by_id(variant_a): found, reason={:?}",
        found.unwrap().reason
    );

    // find_by_id for unknown
    let not_found = parsed.find_by_id("sea:0000000000000000");
    assert!(not_found.is_none());
    println!("  find_by_id(unknown): not found");

    // find_by_id_mut — mutate an entry
    let entry = file.find_by_id_mut(&hash_variant_b).unwrap();
    assert!(!entry.is_active(), "retired entry should not be active");
    entry.reason = Some("confirmed buggy — CICP fix applied".to_string());
    println!(
        "  find_by_id_mut(variant_b): updated reason to {:?}",
        entry.reason
    );

    // is_active checks
    println!(
        "  baseline is_active: {}",
        parsed.find_by_id(&hash_baseline).unwrap().is_active()
    );
    println!(
        "  variant_b is_active: {}",
        parsed.find_by_id(&hash_variant_b).unwrap().is_active()
    );
    println!();

    // ─── 8. Name sanitization ───────────────────────────────────────────
    println!("--- 8. Name sanitization ---");

    let test_cases = [
        ("resize_bicubic_200x200", "resize_bicubic_200x200"),
        ("Resize Bicubic 200", "resize_bicubic_200"),
        ("tests/visual/foo", "tests_visual_foo"),
        ("module::test::case", "module_test_case"),
        ("test@#$%name", "testname"),
        ("__test__", "test"),
        ("a::b::c", "a_b_c"),
        ("", "_unnamed"),
        ("@#$", "_unnamed"),
        ("CamelCase Test-Name", "camelcase_test-name"),
    ];

    for (input, expected) in &test_cases {
        let result = sanitize_name(input);
        let status = if result == *expected {
            "ok"
        } else {
            "MISMATCH"
        };
        println!("  {status}: {input:30} -> {result}");
        assert_eq!(&result, expected, "sanitize({input:?})");
    }

    // checksum_path
    let cs_path = checksum_path(std::path::Path::new("/project/checksums"), "my::test::Name");
    println!(
        "\n  checksum_path(\"/project/checksums\", \"my::test::Name\") = {}",
        cs_path.display()
    );
    assert_eq!(
        cs_path,
        std::path::PathBuf::from("/project/checksums/my_test_name.toml")
    );
    println!();

    // ─── 9. Architecture detection ──────────────────────────────────────
    println!("--- 9. Architecture detection ---");

    println!("  current arch: {arch_tag}");
    println!("  known tags: {KNOWN_ARCH_TAGS:?}");
    assert!(KNOWN_ARCH_TAGS.contains(&arch_tag));

    // arch_matches examples
    let match_cases = [
        ("x86_64-avx2", "x86_64-avx2", true),
        ("x86_64", "x86_64-avx2", true),
        ("x86_64-avx2", "x86_64", false),
        ("aarch64", "x86_64-avx2", false),
        ("x86_64", "x86_64_custom", false),
    ];

    println!("\n  arch_matches examples:");
    for (entry, current, expected) in &match_cases {
        let result = arch::arch_matches(entry, current);
        let status = if result == *expected { "ok" } else { "FAIL" };
        println!("    {status}: arch_matches({entry:20}, {current:20}) = {result}");
        assert_eq!(result, *expected);
    }
    println!();

    // ─── 10. Tolerance override resolution ──────────────────────────────
    println!("--- 10. Tolerance override resolution ---");

    let spec = &parsed.tolerance;

    // x86_64-avx2: no override → base tolerance
    let t_x86 = spec.to_regression_tolerance("x86_64-avx2");
    println!("  x86_64-avx2: uses base tolerance (max_delta=1, min_score=95.0)");
    // Can't inspect private fields, but we can verify it doesn't panic
    let _ = t_x86;

    // aarch64: override applies (max_delta=2, min_score=90.0)
    let t_arm = spec.to_regression_tolerance("aarch64");
    println!("  aarch64: override applied (max_delta=2, min_score=90.0)");
    let _ = t_arm;

    // unknown: no override → base tolerance
    let t_unk = spec.to_regression_tolerance("unknown");
    println!("  unknown: uses base tolerance");
    let _ = t_unk;
    println!();

    // ─── 11. Error handling ─────────────────────────────────────────────
    println!("--- 11. Error handling ---");

    // Missing file
    let err = TestChecksumFile::read_from(std::path::Path::new("/nonexistent/test.toml"));
    match &err {
        Err(zensim_regress::RegressError::Io { path, .. }) => {
            println!("  missing file: Io error at {}", path.display());
        }
        other => panic!("expected Io error, got: {other:?}"),
    }

    // Invalid TOML
    let bad_toml_path = dir.path().join("bad.toml");
    std::fs::write(&bad_toml_path, "this is not { valid toml !!!").unwrap();
    let err = TestChecksumFile::read_from(&bad_toml_path);
    match &err {
        Err(zensim_regress::RegressError::TomlParse { path, .. }) => {
            println!("  bad TOML: parse error at {}", path.display());
        }
        other => panic!("expected TomlParse error, got: {other:?}"),
    }

    // Error display
    println!("  error display: {}", err.unwrap_err());
    println!();

    // ─── 12. Hash file from disk ────────────────────────────────────────
    println!("--- 12. Hash file from disk ---");

    // Save a test image to disk, then hash it
    let img = image::RgbaImage::from_fn(w as u32, h as u32, |x, y| {
        image::Rgba([(x * 8) as u8, (y * 8) as u8, ((x + y) * 4) as u8, 255])
    });
    let img_path = dir.path().join("test_gradient.png");
    img.save(&img_path).expect("save test image");

    let hash_from_file = hasher.hash_file(&img_path).expect("hash file");
    println!("  saved: {}", img_path.display());
    println!("  hash:  {hash_from_file}");
    assert!(hash_from_file.starts_with("sea:"));

    // Hash the same pixels directly — should match
    let hash_from_pixels = hasher.hash_pixels(img.as_raw(), w as u32, h as u32);
    assert_eq!(
        hash_from_file, hash_from_pixels,
        "hash_file and hash_pixels should agree"
    );
    println!("  hash_file == hash_pixels: verified");

    // Hash a nonexistent file
    let err = hasher.hash_file(std::path::Path::new("/nonexistent/image.png"));
    assert!(err.is_err());
    println!("  nonexistent file: {}", err.unwrap_err());
    println!();

    // ─── 13. Simulated match workflow ───────────────────────────────────
    println!("--- 13. Simulated check workflow ---");

    // Simulate what a test harness would do:
    // 1. Run the test → get an actual image
    // 2. Hash it
    // 3. Check against stored checksums

    let actual_hash = &hash_baseline;
    println!("  actual output hash: {actual_hash}");

    // Check: does any active checksum match?
    let matched = parsed.active_checksums().find(|e| e.id == *actual_hash);

    match matched {
        Some(entry) => {
            println!("  MATCH: {} (reason: {:?})", entry.id, entry.reason);
        }
        None => {
            println!("  NO MATCH — would need zensim comparison against authoritative");
            let auth = parsed.authoritative();
            if let Some(auth) = auth {
                println!("  authoritative reference: {}", auth.id);
                println!("  would run: check_regression(&z, auth_image, actual_image, tolerance)");
            } else {
                println!("  no authoritative reference — first run?");
            }
        }
    }

    // Simulate a mismatch
    let unknown_hash = "sea:ffffffffffffffff";
    println!("\n  unknown output hash: {unknown_hash}");
    let matched = parsed.active_checksums().find(|e| e.id == unknown_hash);
    assert!(matched.is_none());
    println!("  NO MATCH — would trigger comparison");

    // Get tolerance for current arch
    let tolerance = parsed.tolerance.to_regression_tolerance(arch_tag);
    println!("  tolerance for {arch_tag}: resolved (with any applicable overrides)");
    let _ = tolerance;

    println!("\n=== lifecycle demo complete ===");
}
