//! Integration tests exercising every workflow path of the ChecksumManager.
//!
//! These tests verify the full lifecycle from first run through multi-arch
//! acceptance, rejection, and tolerance checking. Each test uses an isolated
//! temp directory to avoid interference.

use std::collections::BTreeMap;

use zensim_regress::checksum_file::{
    ChecksumEntry, TestChecksumFile, ToleranceOverride, ToleranceSpec,
};
use zensim_regress::hasher::{ChecksumHasher, SeaHasher};
use zensim_regress::manager::{CheckResult, ChecksumManager};

// ─── Helpers ─────────────────────────────────────────────────────────────

/// Gradient image: deterministic, varies across channels.
fn gradient_rgba(w: u32, h: u32) -> Vec<u8> {
    (0..w * h)
        .flat_map(|i| {
            let x = (i % w) as u8;
            let y = (i / w) as u8;
            [
                x.wrapping_mul(7),
                y.wrapping_mul(11),
                x.wrapping_add(y).wrapping_mul(5),
                255,
            ]
        })
        .collect()
}

/// Off-by-N variant: adds delta to the R channel on some pixels.
fn off_by_n(base: &[u8], delta: u8, frequency: usize) -> Vec<u8> {
    base.chunks(4)
        .enumerate()
        .flat_map(|(i, px)| {
            if i % frequency == 0 {
                [px[0].saturating_add(delta), px[1], px[2], px[3]]
            } else {
                [px[0], px[1], px[2], px[3]]
            }
        })
        .collect()
}

/// Dramatically different image (color-inverted).
fn inverted(base: &[u8]) -> Vec<u8> {
    base.chunks(4)
        .flat_map(|px| [255 - px[0], 255 - px[1], 255 - px[2], px[3]])
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. First run — no baseline exists
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn first_run_normal_mode_returns_no_baseline() {
    let dir = tempfile::tempdir().unwrap();
    let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();

    let px = gradient_rgba(16, 16);
    let result = mgr.check_pixels("first_run", &px, 16, 16).unwrap();

    assert!(
        matches!(
            result,
            CheckResult::NoBaseline {
                auto_accepted: false,
                ..
            }
        ),
        "normal mode should NOT auto-accept: {result:?}"
    );
    assert!(!result.passed());

    // No file should have been created
    assert!(!mgr.test_path("first_run").exists());
}

#[test]
fn first_run_update_mode_creates_baseline() {
    let dir = tempfile::tempdir().unwrap();
    let mgr = ChecksumManager::new(dir.path())
        .with_update_mode_update()
        .with_arch_tag("x86_64-avx2");

    let px = gradient_rgba(16, 16);
    let hash = SeaHasher.hash_pixels(&px, 16, 16);
    let result = mgr.check_pixels("first_run_update", &px, 16, 16).unwrap();

    assert!(
        matches!(
            result,
            CheckResult::NoBaseline {
                auto_accepted: true,
                ..
            }
        ),
        "update mode should auto-accept: {result:?}"
    );
    assert!(result.passed());

    // Verify the created file
    let file = TestChecksumFile::read_from(&mgr.test_path("first_run_update")).unwrap();
    assert_eq!(file.name, "first_run_update");
    assert_eq!(file.checksum.len(), 1);
    assert_eq!(file.checksum[0].id, hash);
    assert_eq!(file.checksum[0].confidence, 10);
    assert_eq!(file.checksum[0].arch, vec!["x86_64-avx2"]);
    assert_eq!(
        file.checksum[0].reason.as_deref(),
        Some("initial baseline")
    );
    assert!(file.checksum[0].diff.is_none()); // no prior reference to diff against
}

#[test]
fn first_run_replace_mode_creates_baseline() {
    let dir = tempfile::tempdir().unwrap();
    let mgr = ChecksumManager::new(dir.path()).with_update_mode_replace();

    let px = gradient_rgba(16, 16);
    let result = mgr
        .check_pixels("first_run_replace", &px, 16, 16)
        .unwrap();

    assert!(result.passed());
    assert!(mgr.test_path("first_run_replace").exists());
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. Exact hash match
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn exact_match_passes() {
    let dir = tempfile::tempdir().unwrap();
    let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();

    let px = gradient_rgba(32, 32);
    let hash = SeaHasher.hash_pixels(&px, 32, 32);

    // Set up baseline
    let mut file = TestChecksumFile::new("exact_match");
    file.checksum.push(ChecksumEntry {
        id: hash.clone(),
        confidence: 10,
        commit: Some("abc123".to_string()),
        arch: vec!["x86_64-avx2".to_string()],
        reason: Some("initial baseline".to_string()),
        status: None,
        diff: None,
    });
    file.write_to(&mgr.test_path("exact_match")).unwrap();

    // Check — should match directly
    let result = mgr.check_pixels("exact_match", &px, 32, 32).unwrap();
    match result {
        CheckResult::Match {
            ref entry_id,
            confidence,
        } => {
            assert_eq!(entry_id, &hash);
            assert_eq!(confidence, 10);
        }
        _ => panic!("expected Match, got {result:?}"),
    }
    assert!(result.passed());
}

#[test]
fn exact_match_adds_arch_tag() {
    let dir = tempfile::tempdir().unwrap();
    let mgr = ChecksumManager::new(dir.path())
        .with_update_mode_normal()
        .with_arch_tag("aarch64");

    let px = gradient_rgba(16, 16);
    let hash = SeaHasher.hash_pixels(&px, 16, 16);

    // Baseline recorded under x86_64-avx2
    let mut file = TestChecksumFile::new("arch_add");
    file.checksum.push(ChecksumEntry {
        id: hash.clone(),
        confidence: 10,
        commit: None,
        arch: vec!["x86_64-avx2".to_string()],
        reason: None,
        status: None,
        diff: None,
    });
    file.write_to(&mgr.test_path("arch_add")).unwrap();

    // Match on aarch64
    let result = mgr.check_pixels("arch_add", &px, 16, 16).unwrap();
    assert!(result.passed());

    // Arch tag should now include aarch64
    let updated = TestChecksumFile::read_from(&mgr.test_path("arch_add")).unwrap();
    assert!(updated.checksum[0].arch.contains(&"aarch64".to_string()));
    assert!(updated.checksum[0]
        .arch
        .contains(&"x86_64-avx2".to_string()));
}

#[test]
fn exact_match_does_not_re_add_same_arch() {
    let dir = tempfile::tempdir().unwrap();
    let mgr = ChecksumManager::new(dir.path())
        .with_update_mode_normal()
        .with_arch_tag("x86_64-avx2");

    let px = gradient_rgba(16, 16);
    let hash = SeaHasher.hash_pixels(&px, 16, 16);

    let mut file = TestChecksumFile::new("arch_nodup");
    file.checksum.push(ChecksumEntry {
        id: hash.clone(),
        confidence: 10,
        commit: None,
        arch: vec!["x86_64-avx2".to_string()],
        reason: None,
        status: None,
        diff: None,
    });
    file.write_to(&mgr.test_path("arch_nodup")).unwrap();

    let _ = mgr.check_pixels("arch_nodup", &px, 16, 16).unwrap();

    let file = TestChecksumFile::read_from(&mgr.test_path("arch_nodup")).unwrap();
    let arch_count = file.checksum[0]
        .arch
        .iter()
        .filter(|a| *a == "x86_64-avx2")
        .count();
    assert_eq!(arch_count, 1, "should not duplicate arch tag");
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. Alternate arch match (same checksum, different arch)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn alternate_arch_produces_same_hash() {
    let dir = tempfile::tempdir().unwrap();

    // First: set up baseline from x86_64-avx2
    let mgr_x86 = ChecksumManager::new(dir.path())
        .with_update_mode_update()
        .with_arch_tag("x86_64-avx2");

    let px = gradient_rgba(16, 16);
    let _ = mgr_x86.check_pixels("same_hash", &px, 16, 16).unwrap();

    // Second: check from aarch64 (same pixels → same hash)
    let mgr_arm = ChecksumManager::new(dir.path())
        .with_update_mode_normal()
        .with_arch_tag("aarch64");

    let result = mgr_arm.check_pixels("same_hash", &px, 16, 16).unwrap();
    assert!(matches!(result, CheckResult::Match { .. }));
    assert!(result.passed());

    // Both arch tags should be recorded
    let file = TestChecksumFile::read_from(&mgr_arm.test_path("same_hash")).unwrap();
    let entry = &file.checksum[0];
    assert!(entry.arch.contains(&"x86_64-avx2".to_string()));
    assert!(entry.arch.contains(&"aarch64".to_string()));
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. Within tolerance — zensim comparison passes
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn within_tolerance_off_by_one() {
    let dir = tempfile::tempdir().unwrap();
    let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();

    let (w, h) = (32u32, 32u32);
    let base = gradient_rgba(w, h);
    let base_hash = SeaHasher.hash_pixels(&base, w, h);

    // Save reference image
    mgr.save_reference_image("within_tol", &base, w, h).unwrap();

    // Set tolerance to allow off-by-1
    let mut file = TestChecksumFile::new("within_tol");
    file.tolerance = ToleranceSpec {
        max_channel_delta: 1,
        min_score: 90.0,
        max_differing_pixel_fraction: 1.0,
        min_identical_channel_fraction: 0.0,
        ..Default::default()
    };
    file.checksum.push(ChecksumEntry::new(base_hash));
    file.write_to(&mgr.test_path("within_tol")).unwrap();

    // Create off-by-1 variant
    let variant = off_by_n(&base, 1, 3);
    let result = mgr.check_pixels("within_tol", &variant, w, h).unwrap();

    match &result {
        CheckResult::WithinTolerance {
            report,
            auto_accepted,
            ..
        } => {
            assert!(report.passed());
            assert!(!auto_accepted);
        }
        _ => panic!("expected WithinTolerance, got {result:?}"),
    }
    assert!(result.passed());
}

#[test]
fn within_tolerance_auto_accept_update_mode() {
    let dir = tempfile::tempdir().unwrap();
    let mgr = ChecksumManager::new(dir.path())
        .with_update_mode_update()
        .with_arch_tag("aarch64");

    let (w, h) = (32u32, 32u32);
    let base = gradient_rgba(w, h);
    let base_hash = SeaHasher.hash_pixels(&base, w, h);

    mgr.save_reference_image("auto_accept", &base, w, h)
        .unwrap();

    let mut file = TestChecksumFile::new("auto_accept");
    file.tolerance = ToleranceSpec {
        max_channel_delta: 1,
        min_score: 90.0,
        max_differing_pixel_fraction: 1.0,
        min_identical_channel_fraction: 0.0,
        ..Default::default()
    };
    file.checksum.push(ChecksumEntry::new(base_hash.clone()));
    file.write_to(&mgr.test_path("auto_accept")).unwrap();

    let variant = off_by_n(&base, 1, 2);
    let variant_hash = SeaHasher.hash_pixels(&variant, w, h);
    let result = mgr.check_pixels("auto_accept", &variant, w, h).unwrap();

    match &result {
        CheckResult::WithinTolerance {
            auto_accepted,
            actual_hash,
            authoritative_id,
            report,
        } => {
            assert!(*auto_accepted);
            assert_eq!(actual_hash, &variant_hash);
            assert_eq!(authoritative_id, &base_hash);
            assert!(report.passed());
        }
        _ => panic!("expected WithinTolerance, got {result:?}"),
    }

    // Verify entry was added with chain-of-trust
    let updated = TestChecksumFile::read_from(&mgr.test_path("auto_accept")).unwrap();
    assert_eq!(updated.checksum.len(), 2);

    let new_entry = updated.find_by_id(&variant_hash).unwrap();
    assert!(new_entry.is_active());
    assert!(new_entry.arch.contains(&"aarch64".to_string()));
    assert_eq!(
        new_entry.reason.as_deref(),
        Some("auto-accepted within tolerance")
    );

    // Chain-of-trust diff
    let diff = new_entry.diff.as_ref().expect("should have diff");
    assert_eq!(diff.vs, base_hash);
    assert!(diff.zensim_score > 90.0);
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. Exceeds tolerance — comparison fails
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fails_tolerance_exact() {
    let dir = tempfile::tempdir().unwrap();
    let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();

    let (w, h) = (32u32, 32u32);
    let base = gradient_rgba(w, h);
    let base_hash = SeaHasher.hash_pixels(&base, w, h);

    mgr.save_reference_image("fails_exact", &base, w, h)
        .unwrap();

    // Exact tolerance (default)
    let mut file = TestChecksumFile::new("fails_exact");
    file.checksum.push(ChecksumEntry::new(base_hash.clone()));
    file.write_to(&mgr.test_path("fails_exact")).unwrap();

    let variant = off_by_n(&base, 1, 1); // every pixel differs by 1
    let result = mgr.check_pixels("fails_exact", &variant, w, h).unwrap();

    match &result {
        CheckResult::Failed {
            report,
            authoritative_id,
            ..
        } => {
            assert!(report.is_some(), "should have a report");
            assert!(!report.as_ref().unwrap().passed());
            assert_eq!(authoritative_id.as_deref(), Some(base_hash.as_str()));
        }
        _ => panic!("expected Failed, got {result:?}"),
    }
    assert!(!result.passed());
}

#[test]
fn fails_tolerance_large_diff() {
    let dir = tempfile::tempdir().unwrap();
    let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();

    let (w, h) = (32u32, 32u32);
    let base = gradient_rgba(w, h);
    let base_hash = SeaHasher.hash_pixels(&base, w, h);

    mgr.save_reference_image("fails_large", &base, w, h)
        .unwrap();

    let mut file = TestChecksumFile::new("fails_large");
    file.tolerance = ToleranceSpec {
        max_channel_delta: 1,
        min_score: 95.0,
        ..Default::default()
    };
    file.checksum.push(ChecksumEntry::new(base_hash));
    file.write_to(&mgr.test_path("fails_large")).unwrap();

    let big_diff = inverted(&base);
    let result = mgr.check_pixels("fails_large", &big_diff, w, h).unwrap();

    assert!(!result.passed());
    assert!(matches!(result, CheckResult::Failed { .. }));
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. Accept workflow
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn accept_creates_new_file() {
    let dir = tempfile::tempdir().unwrap();
    let mgr = ChecksumManager::new(dir.path())
        .with_update_mode_normal()
        .with_arch_tag("x86_64-avx2");

    let hash = "sea:abcdef1234567890";
    mgr.accept("new_test", hash, None, "establishing baseline")
        .unwrap();

    let file = TestChecksumFile::read_from(&mgr.test_path("new_test")).unwrap();
    assert_eq!(file.name, "new_test");
    assert_eq!(file.checksum.len(), 1);
    assert_eq!(file.checksum[0].id, hash);
    assert_eq!(file.checksum[0].confidence, 10);
    assert_eq!(
        file.checksum[0].reason.as_deref(),
        Some("establishing baseline")
    );
    assert!(file.checksum[0]
        .arch
        .contains(&"x86_64-avx2".to_string()));
}

#[test]
fn accept_appends_to_existing() {
    let dir = tempfile::tempdir().unwrap();
    let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();

    mgr.accept("append_test", "sea:aaaa000000000000", None, "first")
        .unwrap();
    mgr.accept("append_test", "sea:bbbb000000000000", None, "second")
        .unwrap();

    let file = TestChecksumFile::read_from(&mgr.test_path("append_test")).unwrap();
    assert_eq!(file.checksum.len(), 2);
    assert_eq!(file.checksum[0].id, "sea:aaaa000000000000");
    assert_eq!(file.checksum[1].id, "sea:bbbb000000000000");
}

#[test]
fn accept_does_not_duplicate_existing_id() {
    let dir = tempfile::tempdir().unwrap();
    let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();

    mgr.accept("nodup_test", "sea:aaaa000000000000", None, "first")
        .unwrap();
    mgr.accept("nodup_test", "sea:aaaa000000000000", None, "updated reason")
        .unwrap();

    let file = TestChecksumFile::read_from(&mgr.test_path("nodup_test")).unwrap();
    assert_eq!(file.checksum.len(), 1);
    assert_eq!(
        file.checksum[0].reason.as_deref(),
        Some("updated reason")
    );
}

#[test]
fn accept_reactivates_retired_entry() {
    let dir = tempfile::tempdir().unwrap();
    let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();

    // Create a retired entry
    let mut file = TestChecksumFile::new("reactivate_test");
    file.checksum.push(ChecksumEntry {
        id: "sea:dead000000000000".to_string(),
        confidence: 0,
        commit: None,
        arch: vec![],
        reason: Some("was wrong".to_string()),
        status: Some("wrong".to_string()),
        diff: None,
    });
    file.write_to(&mgr.test_path("reactivate_test")).unwrap();

    // Accept with same ID → should reactivate
    mgr.accept(
        "reactivate_test",
        "sea:dead000000000000",
        None,
        "actually correct",
    )
    .unwrap();

    let updated = TestChecksumFile::read_from(&mgr.test_path("reactivate_test")).unwrap();
    let entry = &updated.checksum[0];
    assert_eq!(entry.confidence, 10);
    assert!(entry.status.is_none());
    assert_eq!(entry.reason.as_deref(), Some("actually correct"));
    assert!(entry.is_active());
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. Reject workflow
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn reject_retires_entry() {
    let dir = tempfile::tempdir().unwrap();
    let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();

    let hash = "sea:1111111111111111";
    mgr.accept("reject_test", hash, None, "baseline").unwrap();

    let rejected = mgr
        .reject("reject_test", hash, "regression from CICP fix")
        .unwrap();
    assert!(rejected);

    let file = TestChecksumFile::read_from(&mgr.test_path("reject_test")).unwrap();
    let entry = file.find_by_id(hash).unwrap();
    assert_eq!(entry.confidence, 0);
    assert_eq!(entry.status.as_deref(), Some("wrong"));
    assert_eq!(
        entry.reason.as_deref(),
        Some("regression from CICP fix")
    );
    assert!(!entry.is_active());
}

#[test]
fn reject_nonexistent_file_returns_false() {
    let dir = tempfile::tempdir().unwrap();
    let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();

    let result = mgr.reject("no_such_test", "sea:0000", "reason").unwrap();
    assert!(!result);
}

#[test]
fn reject_nonexistent_entry_returns_false() {
    let dir = tempfile::tempdir().unwrap();
    let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();

    mgr.accept("reject_miss", "sea:aaaa000000000000", None, "baseline")
        .unwrap();

    let result = mgr
        .reject("reject_miss", "sea:wrong_hash", "reason")
        .unwrap();
    assert!(!result);
}

#[test]
fn rejected_entry_not_matched_on_check() {
    let dir = tempfile::tempdir().unwrap();
    let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();

    let px = gradient_rgba(16, 16);
    let hash = SeaHasher.hash_pixels(&px, 16, 16);

    mgr.accept("rejected_check", &hash, None, "baseline")
        .unwrap();
    mgr.reject("rejected_check", &hash, "broken").unwrap();

    let result = mgr
        .check_pixels("rejected_check", &px, 16, 16)
        .unwrap();
    // Should NOT match — entry is retired. Falls through to no-active-entries path.
    assert!(!result.passed());
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. REPLACE_CHECKSUMS mode
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn replace_mode_retires_old_adds_new() {
    let dir = tempfile::tempdir().unwrap();
    let mgr = ChecksumManager::new(dir.path())
        .with_update_mode_replace()
        .with_arch_tag("x86_64-avx2");

    // Set up existing entries
    let mut file = TestChecksumFile::new("replace_test");
    file.checksum.push(ChecksumEntry {
        id: "sea:old1111111111111".to_string(),
        confidence: 10,
        commit: None,
        arch: vec!["x86_64-avx2".to_string()],
        reason: Some("old baseline".to_string()),
        status: None,
        diff: None,
    });
    file.checksum.push(ChecksumEntry {
        id: "sea:old2222222222222".to_string(),
        confidence: 8,
        commit: None,
        arch: vec!["aarch64".to_string()],
        reason: Some("arm variant".to_string()),
        status: None,
        diff: None,
    });
    file.write_to(&mgr.test_path("replace_test")).unwrap();

    let px = gradient_rgba(16, 16);
    let new_hash = SeaHasher.hash_pixels(&px, 16, 16);

    let result = mgr
        .check_pixels("replace_test", &px, 16, 16)
        .unwrap();
    assert!(result.passed());

    let updated = TestChecksumFile::read_from(&mgr.test_path("replace_test")).unwrap();

    // Old entries retired
    let old1 = updated.find_by_id("sea:old1111111111111").unwrap();
    assert_eq!(old1.confidence, 0);
    assert_eq!(old1.status.as_deref(), Some("replaced"));

    let old2 = updated.find_by_id("sea:old2222222222222").unwrap();
    assert_eq!(old2.confidence, 0);
    assert_eq!(old2.status.as_deref(), Some("replaced"));

    // New entry active
    let new = updated.find_by_id(&new_hash).unwrap();
    assert!(new.is_active());
    assert!(new.arch.contains(&"x86_64-avx2".to_string()));
    assert_eq!(new.reason.as_deref(), Some("replaced baseline"));
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. Tolerance overrides (per-architecture)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn tolerance_override_applied_to_correct_arch() {
    let dir = tempfile::tempdir().unwrap();

    let (w, h) = (32u32, 32u32);
    let base = gradient_rgba(w, h);
    let base_hash = SeaHasher.hash_pixels(&base, w, h);

    // Strict base tolerance, relaxed for aarch64
    let mut file = TestChecksumFile::new("tol_override");
    file.tolerance = ToleranceSpec {
        max_channel_delta: 0,
        min_score: 100.0,
        max_differing_pixel_fraction: 0.0,
        min_identical_channel_fraction: 1.0,
        overrides: BTreeMap::from([(
            "aarch64".to_string(),
            ToleranceOverride {
                max_channel_delta: Some(1),
                min_score: Some(90.0),
                max_differing_pixel_fraction: Some(1.0),
                min_identical_channel_fraction: Some(0.0),
                ..Default::default()
            },
        )]),
        ..Default::default()
    };
    file.checksum.push(ChecksumEntry::new(base_hash));

    let variant = off_by_n(&base, 1, 3);

    // x86_64 with exact tolerance → should fail
    let mgr_x86 = ChecksumManager::new(dir.path())
        .with_update_mode_normal()
        .with_arch_tag("x86_64-avx2");

    mgr_x86
        .save_reference_image("tol_override", &base, w, h)
        .unwrap();
    file.write_to(&mgr_x86.test_path("tol_override")).unwrap();

    let result_x86 = mgr_x86
        .check_pixels("tol_override", &variant, w, h)
        .unwrap();
    assert!(!result_x86.passed(), "x86 with exact tolerance should fail");

    // aarch64 with relaxed tolerance → should pass
    let mgr_arm = ChecksumManager::new(dir.path())
        .with_update_mode_normal()
        .with_arch_tag("aarch64");

    let result_arm = mgr_arm
        .check_pixels("tol_override", &variant, w, h)
        .unwrap();
    assert!(
        result_arm.passed(),
        "aarch64 with off-by-1 override should pass"
    );
}

#[test]
fn tolerance_prefix_match() {
    let dir = tempfile::tempdir().unwrap();

    let (w, h) = (32u32, 32u32);
    let base = gradient_rgba(w, h);
    let base_hash = SeaHasher.hash_pixels(&base, w, h);

    // Override for "x86_64" (prefix) should match "x86_64-avx2"
    let mut file = TestChecksumFile::new("tol_prefix");
    file.tolerance = ToleranceSpec {
        max_channel_delta: 0,
        min_score: 100.0,
        overrides: BTreeMap::from([(
            "x86_64".to_string(),
            ToleranceOverride {
                max_channel_delta: Some(1),
                min_score: Some(90.0),
                max_differing_pixel_fraction: Some(1.0),
                min_identical_channel_fraction: Some(0.0),
                ..Default::default()
            },
        )]),
        ..Default::default()
    };
    file.checksum.push(ChecksumEntry::new(base_hash));

    let mgr = ChecksumManager::new(dir.path())
        .with_update_mode_normal()
        .with_arch_tag("x86_64-avx2");

    mgr.save_reference_image("tol_prefix", &base, w, h)
        .unwrap();
    file.write_to(&mgr.test_path("tol_prefix")).unwrap();

    let variant = off_by_n(&base, 1, 3);
    let result = mgr
        .check_pixels("tol_prefix", &variant, w, h)
        .unwrap();
    assert!(
        result.passed(),
        "x86_64 prefix override should match x86_64-avx2"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. Hash-only check (no pixel data)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn hash_only_match() {
    let dir = tempfile::tempdir().unwrap();
    let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();

    let hash = "opaque:user-provided-hash-123";
    let mut file = TestChecksumFile::new("hash_only");
    file.checksum.push(ChecksumEntry::new(hash));
    file.write_to(&mgr.test_path("hash_only")).unwrap();

    let result = mgr.check_hash("hash_only", hash).unwrap();
    assert!(matches!(result, CheckResult::Match { .. }));
}

#[test]
fn hash_only_mismatch() {
    let dir = tempfile::tempdir().unwrap();
    let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();

    let mut file = TestChecksumFile::new("hash_miss");
    file.checksum.push(ChecksumEntry::new("sea:aaaa000000000000"));
    file.write_to(&mgr.test_path("hash_miss")).unwrap();

    let result = mgr
        .check_hash("hash_miss", "sea:bbbb000000000000")
        .unwrap();
    match &result {
        CheckResult::Failed {
            report,
            authoritative_id,
            actual_hash,
        } => {
            assert!(report.is_none(), "no pixels = no report");
            assert_eq!(
                authoritative_id.as_deref(),
                Some("sea:aaaa000000000000")
            );
            assert_eq!(actual_hash, "sea:bbbb000000000000");
        }
        _ => panic!("expected Failed, got {result:?}"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 11. Multi-entry checksum files (multiple active entries)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn multi_entry_matches_any_active() {
    let dir = tempfile::tempdir().unwrap();
    let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();

    let mut file = TestChecksumFile::new("multi_entry");
    file.checksum.push(ChecksumEntry {
        id: "sea:aaaa000000000000".to_string(),
        confidence: 10,
        commit: None,
        arch: vec!["x86_64-avx2".to_string()],
        reason: Some("x86 baseline".to_string()),
        status: None,
        diff: None,
    });
    file.checksum.push(ChecksumEntry {
        id: "sea:bbbb000000000000".to_string(),
        confidence: 8,
        commit: None,
        arch: vec!["aarch64".to_string()],
        reason: Some("arm variant".to_string()),
        status: None,
        diff: None,
    });
    file.write_to(&mgr.test_path("multi_entry")).unwrap();

    // Match first entry
    let r1 = mgr
        .check_hash("multi_entry", "sea:aaaa000000000000")
        .unwrap();
    assert!(matches!(
        r1,
        CheckResult::Match {
            confidence: 10,
            ..
        }
    ));

    // Match second entry
    let r2 = mgr
        .check_hash("multi_entry", "sea:bbbb000000000000")
        .unwrap();
    assert!(matches!(
        r2,
        CheckResult::Match {
            confidence: 8,
            ..
        }
    ));
}

// ═══════════════════════════════════════════════════════════════════════════
// 12. Reference image management
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn save_and_load_reference_image() {
    let dir = tempfile::tempdir().unwrap();
    let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();

    let px = gradient_rgba(24, 24);
    let path = mgr
        .save_reference_image("ref_save", &px, 24, 24)
        .unwrap();

    assert!(path.exists());
    assert!(path.to_str().unwrap().ends_with("ref_save.png"));

    // Round-trip through PNG
    let loaded = image::open(&path).unwrap().to_rgba8();
    assert_eq!(loaded.dimensions(), (24, 24));
    assert_eq!(loaded.as_raw(), &px);
}

#[test]
fn reference_image_enables_comparison() {
    let dir = tempfile::tempdir().unwrap();
    let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();

    let (w, h) = (16u32, 16u32);
    let base = gradient_rgba(w, h);
    let base_hash = SeaHasher.hash_pixels(&base, w, h);

    // Without reference image: mismatch returns Failed with no report
    let mut file = TestChecksumFile::new("ref_enables");
    file.tolerance = ToleranceSpec {
        max_channel_delta: 1,
        min_score: 90.0,
        max_differing_pixel_fraction: 1.0,
        min_identical_channel_fraction: 0.0,
        ..Default::default()
    };
    file.checksum.push(ChecksumEntry::new(base_hash.clone()));
    file.write_to(&mgr.test_path("ref_enables")).unwrap();

    let variant = off_by_n(&base, 1, 2);
    let result_no_ref = mgr
        .check_pixels("ref_enables", &variant, w, h)
        .unwrap();
    assert!(
        matches!(result_no_ref, CheckResult::Failed { report: None, .. }),
        "without reference, should fail with no report: {result_no_ref:?}"
    );

    // Save reference image, now comparison should work
    mgr.save_reference_image("ref_enables", &base, w, h)
        .unwrap();
    let result_with_ref = mgr
        .check_pixels("ref_enables", &variant, w, h)
        .unwrap();
    assert!(
        matches!(
            result_with_ref,
            CheckResult::WithinTolerance { .. }
        ),
        "with reference, should pass tolerance: {result_with_ref:?}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 13. Full lifecycle (first run → match → variant → accept → re-check)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn full_lifecycle() {
    let dir = tempfile::tempdir().unwrap();
    let (w, h) = (32u32, 32u32);
    let base = gradient_rgba(w, h);

    // Step 1: First run in UPDATE mode → auto-creates baseline
    let mgr = ChecksumManager::new(dir.path())
        .with_update_mode_update()
        .with_arch_tag("x86_64-avx2");

    let r1 = mgr.check_pixels("lifecycle", &base, w, h).unwrap();
    assert!(r1.passed());
    assert!(matches!(
        r1,
        CheckResult::NoBaseline {
            auto_accepted: true,
            ..
        }
    ));

    // Save reference image for future comparisons
    mgr.save_reference_image("lifecycle", &base, w, h)
        .unwrap();

    // Step 2: Same pixels again → direct match
    let r2 = mgr.check_pixels("lifecycle", &base, w, h).unwrap();
    assert!(matches!(r2, CheckResult::Match { .. }));

    // Step 3: Set tolerance for the test
    let path = mgr.test_path("lifecycle");
    let mut file = TestChecksumFile::read_from(&path).unwrap();
    file.tolerance = ToleranceSpec {
        max_channel_delta: 1,
        min_score: 90.0,
        max_differing_pixel_fraction: 1.0,
        min_identical_channel_fraction: 0.0,
        ..Default::default()
    };
    file.write_to(&path).unwrap();

    // Step 4: ARM variant (off-by-1) → within tolerance, auto-accepted
    let mgr_arm = ChecksumManager::new(dir.path())
        .with_update_mode_update()
        .with_arch_tag("aarch64");

    let arm_variant = off_by_n(&base, 1, 3);
    let arm_hash = SeaHasher.hash_pixels(&arm_variant, w, h);

    let r3 = mgr_arm
        .check_pixels("lifecycle", &arm_variant, w, h)
        .unwrap();
    assert!(r3.passed());
    assert!(matches!(
        r3,
        CheckResult::WithinTolerance {
            auto_accepted: true,
            ..
        }
    ));

    // Step 5: Verify checksum file has two entries
    let file = TestChecksumFile::read_from(&path).unwrap();
    assert_eq!(file.active_checksums().count(), 2);

    let arm_entry = file.find_by_id(&arm_hash).unwrap();
    assert!(arm_entry.arch.contains(&"aarch64".to_string()));
    assert!(arm_entry.diff.is_some());

    // Step 6: ARM variant checks again → now direct match
    let r4 = mgr_arm
        .check_pixels("lifecycle", &arm_variant, w, h)
        .unwrap();
    assert!(matches!(r4, CheckResult::Match { .. }));

    // Step 7: Switch to normal mode, reject the ARM entry
    let mgr_normal = ChecksumManager::new(dir.path())
        .with_update_mode_normal()
        .with_arch_tag("x86_64-avx2");

    mgr_normal
        .reject("lifecycle", &arm_hash, "intentional rejection")
        .unwrap();

    // Step 8: ARM variant now fails in normal mode (entry retired)
    let mgr_arm_normal = ChecksumManager::new(dir.path())
        .with_update_mode_normal()
        .with_arch_tag("aarch64");

    let r5 = mgr_arm_normal
        .check_pixels("lifecycle", &arm_variant, w, h)
        .unwrap();
    // It should be WithinTolerance (not Match, since entry was rejected)
    // but it passes because comparison with reference is still within tolerance
    assert!(r5.passed());
    assert!(matches!(r5, CheckResult::WithinTolerance { auto_accepted: false, .. }));

    // Step 9: Verify final state
    let final_file = TestChecksumFile::read_from(&path).unwrap();
    assert_eq!(final_file.active_checksums().count(), 1); // only x86 baseline
    let rejected = final_file.find_by_id(&arm_hash).unwrap();
    assert!(!rejected.is_active());
    assert_eq!(rejected.status.as_deref(), Some("wrong"));
}

// ═══════════════════════════════════════════════════════════════════════════
// 14. File-based check (image from disk)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn check_file_from_disk() {
    let dir = tempfile::tempdir().unwrap();
    let mgr = ChecksumManager::new(dir.path()).with_update_mode_normal();

    let (w, h) = (16u32, 16u32);
    let px = gradient_rgba(w, h);
    let hash = SeaHasher.hash_pixels(&px, w, h);

    // Write actual output as PNG
    let actual_path = dir.path().join("actual_output.png");
    let img = image::RgbaImage::from_raw(w, h, px.clone()).unwrap();
    img.save(&actual_path).unwrap();

    // Create baseline
    let mut file = TestChecksumFile::new("file_check");
    file.checksum.push(ChecksumEntry::new(hash));
    file.write_to(&mgr.test_path("file_check")).unwrap();

    let result = mgr.check_file("file_check", &actual_path).unwrap();
    assert!(result.passed());
    assert!(matches!(result, CheckResult::Match { .. }));
}

// ═══════════════════════════════════════════════════════════════════════════
// 15. Name sanitization in paths
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_name_sanitization() {
    let dir = tempfile::tempdir().unwrap();
    let mgr = ChecksumManager::new(dir.path()).with_update_mode_update();

    // Names with special characters should be sanitized for file paths
    let px = gradient_rgba(8, 8);

    let _ = mgr
        .check_pixels("module::test::resize (200x200)", &px, 8, 8)
        .unwrap();

    // The file should exist with a sanitized name
    let path = mgr.test_path("module::test::resize (200x200)");
    assert!(path.exists(), "sanitized path should exist: {}", path.display());
    assert!(
        path.to_str().unwrap().contains("module_test_resize"),
        "path should be sanitized: {}",
        path.display()
    );
}
