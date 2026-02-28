//! ChecksumManager workflow demonstration.
//!
//! Shows the high-level API that test harnesses and developers actually use:
//!
//! 1. First run: no baseline → auto-create in UPDATE mode
//! 2. Exact match: same pixels → pass immediately
//! 3. Off-by-1 variant: different hash → zensim comparison → within tolerance
//! 4. Accept: record new checksum with chain-of-trust diff
//! 5. Reject: retire a broken checksum
//! 6. Replace mode: wipe all entries and set new baseline
//! 7. Multi-arch: simulate x86_64 and aarch64 producing different outputs
//!
//! Run with: cargo run -p zensim-regress --example manager_workflow

use zensim_regress::hasher::{ChecksumHasher, SeaHasher};
use zensim_regress::manager::{CheckResult, ChecksumManager};

/// Gradient image.
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

/// Off-by-N variant.
fn off_by_n(base: &[u8], delta: u8) -> Vec<u8> {
    base.chunks(4)
        .enumerate()
        .flat_map(|(i, px)| {
            if i % 3 == 0 {
                [px[0].saturating_add(delta), px[1], px[2], px[3]]
            } else {
                [px[0], px[1], px[2], px[3]]
            }
        })
        .collect()
}

fn main() {
    println!("=== ChecksumManager Workflow Demo ===\n");

    let dir = tempfile::tempdir().unwrap();
    let (w, h) = (32u32, 32u32);
    let base = gradient_rgba(w, h);
    let base_hash = SeaHasher.hash_pixels(&base, w, h);

    // ─── 1. First run in UPDATE mode ─────────────────────────────────────
    println!("--- 1. First run (UPDATE mode) ---");
    let mgr = ChecksumManager::new(dir.path())
        .with_update_mode_update()
        .with_arch_tag("x86_64-avx2");

    let r = mgr.check_pixels("gradient_test", &base, w, h).unwrap();
    println!("  result: {:?}", std::mem::discriminant(&r));
    println!("  passed: {}", r.passed());
    match &r {
        CheckResult::NoBaseline {
            actual_hash,
            auto_accepted,
        } => {
            println!("  hash: {actual_hash}");
            println!("  auto_accepted: {auto_accepted}");
        }
        _ => unreachable!(),
    }
    println!();

    // Save reference image for future comparisons
    let ref_path = mgr
        .save_reference_image("gradient_test", &base, w, h)
        .unwrap();
    println!("  reference saved: {}\n", ref_path.display());

    // ─── 2. Exact match ──────────────────────────────────────────────────
    println!("--- 2. Same pixels again → exact match ---");
    let r = mgr.check_pixels("gradient_test", &base, w, h).unwrap();
    match &r {
        CheckResult::Match {
            entry_id,
            confidence,
        } => {
            println!("  matched: {entry_id} (confidence: {confidence})");
        }
        _ => println!("  unexpected: {r:?}"),
    }
    println!();

    // ─── 3. Off-by-1 variant → within tolerance ─────────────────────────
    println!("--- 3. Off-by-1 variant → zensim comparison ---");

    // First, relax the tolerance
    let path = mgr.test_path("gradient_test");
    let mut file = zensim_regress::checksum_file::TestChecksumFile::read_from(&path).unwrap();
    file.tolerance = zensim_regress::checksum_file::ToleranceSpec {
        max_channel_delta: 1,
        min_score: 90.0,
        max_differing_pixel_fraction: 1.0,
        min_identical_channel_fraction: 0.0,
        ..Default::default()
    };
    file.write_to(&path).unwrap();
    println!("  tolerance set: delta<=1, score>=90");

    let variant = off_by_n(&base, 1);
    let variant_hash = SeaHasher.hash_pixels(&variant, w, h);
    println!("  variant hash: {variant_hash}");

    let r = mgr
        .check_pixels("gradient_test", &variant, w, h)
        .unwrap();
    match &r {
        CheckResult::WithinTolerance {
            report,
            authoritative_id,
            actual_hash,
            auto_accepted,
        } => {
            println!("  authoritative: {authoritative_id}");
            println!("  actual: {actual_hash}");
            println!("  auto_accepted: {auto_accepted}");
            println!("  report: {report}");
        }
        _ => println!("  unexpected: {r:?}"),
    }
    println!();

    // ─── 4. Manual accept ────────────────────────────────────────────────
    println!("--- 4. Manual accept ---");
    let mgr_normal = ChecksumManager::new(dir.path())
        .with_update_mode_normal()
        .with_arch_tag("x86_64-avx2");

    // Accept a new hash manually
    mgr_normal
        .accept(
            "gradient_test",
            "sea:manual_accept_hash",
            None,
            "manually approved by developer",
        )
        .unwrap();

    let file = zensim_regress::checksum_file::TestChecksumFile::read_from(&path).unwrap();
    println!("  active entries: {}", file.active_checksums().count());
    for entry in file.active_checksums() {
        println!(
            "    {} (confidence={}, reason={:?})",
            entry.id, entry.confidence, entry.reason
        );
    }
    println!();

    // ─── 5. Reject ───────────────────────────────────────────────────────
    println!("--- 5. Reject a checksum ---");
    let rejected = mgr_normal
        .reject(
            "gradient_test",
            "sea:manual_accept_hash",
            "turns out it was wrong",
        )
        .unwrap();
    println!("  rejected: {rejected}");

    let file = zensim_regress::checksum_file::TestChecksumFile::read_from(&path).unwrap();
    let entry = file.find_by_id("sea:manual_accept_hash").unwrap();
    println!(
        "  {} → confidence={}, status={:?}",
        entry.id, entry.confidence, entry.status
    );
    println!();

    // ─── 6. Replace mode ─────────────────────────────────────────────────
    println!("--- 6. Replace mode (new baseline) ---");
    let mgr_replace = ChecksumManager::new(dir.path())
        .with_update_mode_replace()
        .with_arch_tag("x86_64-avx2");

    let new_base = gradient_rgba(w, h); // same pixels, but REPLACE mode retires everything
    let r = mgr_replace
        .check_pixels("gradient_test", &new_base, w, h)
        .unwrap();
    println!("  passed: {}", r.passed());

    let file = zensim_regress::checksum_file::TestChecksumFile::read_from(&path).unwrap();
    let active: Vec<_> = file.active_checksums().collect();
    let retired = file.checksum.iter().filter(|e| !e.is_active()).count();
    println!(
        "  active: {}, retired: {} (total: {})",
        active.len(),
        retired,
        file.checksum.len()
    );
    println!();

    // ─── 7. Multi-arch simulation ────────────────────────────────────────
    println!("--- 7. Multi-arch simulation ---");

    // Start fresh
    let dir2 = tempfile::tempdir().unwrap();

    // x86_64 creates baseline
    let mgr_x86 = ChecksumManager::new(dir2.path())
        .with_update_mode_update()
        .with_arch_tag("x86_64-avx2");

    let _ = mgr_x86
        .check_pixels("multi_arch", &base, w, h)
        .unwrap();
    mgr_x86
        .save_reference_image("multi_arch", &base, w, h)
        .unwrap();

    // Set tolerance
    let path2 = mgr_x86.test_path("multi_arch");
    let mut file = zensim_regress::checksum_file::TestChecksumFile::read_from(&path2).unwrap();
    file.tolerance = zensim_regress::checksum_file::ToleranceSpec {
        max_channel_delta: 1,
        min_score: 90.0,
        max_differing_pixel_fraction: 1.0,
        min_identical_channel_fraction: 0.0,
        ..Default::default()
    };
    file.write_to(&path2).unwrap();

    println!("  x86_64: baseline established ({base_hash})");

    // aarch64 produces off-by-1 variant → auto-accepted
    let mgr_arm = ChecksumManager::new(dir2.path())
        .with_update_mode_update()
        .with_arch_tag("aarch64");

    let arm_variant = off_by_n(&base, 1);
    let arm_hash = SeaHasher.hash_pixels(&arm_variant, w, h);

    let r = mgr_arm
        .check_pixels("multi_arch", &arm_variant, w, h)
        .unwrap();
    println!("  aarch64: {arm_hash}");
    match &r {
        CheckResult::WithinTolerance {
            auto_accepted,
            report,
            ..
        } => {
            println!("    auto_accepted: {auto_accepted}");
            println!("    score: {:.1}", report.score());
            println!("    category: {:?}", report.category());
        }
        _ => println!("    unexpected: {r:?}"),
    }

    // Show the TOML
    let file = zensim_regress::checksum_file::TestChecksumFile::read_from(&path2).unwrap();
    println!("\n  --- TOML state ---");
    let toml_str = toml::to_string_pretty(&file).unwrap();
    for line in toml_str.lines() {
        println!("  | {line}");
    }
    println!("  --- end ---");

    // aarch64 second run → direct match
    let r = mgr_arm
        .check_pixels("multi_arch", &arm_variant, w, h)
        .unwrap();
    println!("\n  aarch64 second run: {:?}", std::mem::discriminant(&r));
    println!();

    // ─── Summary ─────────────────────────────────────────────────────────
    println!("--- Summary ---");
    println!("  Demonstrated: first run, exact match, within tolerance,");
    println!("  accept, reject, replace, multi-arch auto-accept.");
    println!("  All workflows use ChecksumManager — the developer-facing API.");
    println!("\n=== done ===");
}
