//! Whole-vector tier-parity gate for `FormulaRevision::Rev5` (the `localwin`
//! era): `research::extract` over the three Rev5 families (basic + peaks +
//! v2, 576 slots) must be BIT-IDENTICAL under every dispatch-tier permutation,
//! including nonzero-remainder geometries.
//!
//! `ZENSIM_FORMULA_REV` is a process-wide `OnceLock`, so the assertions run in
//! a re-exec'd child (same protocol as `featcanon_tier_parity.rs`).

#![cfg(feature = "feature-regime-v2")]
#![allow(deprecated)]

use archmage::testing::{CompileTimePolicy, for_each_token_permutation};
use zensim::RgbSlice;
use zensim::feature_set_id::ComputeToken;
use zensim::research::{self, Request};

const HAS_RUNTIME_TOKENS: bool = cfg!(any(target_arch = "x86_64", target_arch = "aarch64"));

fn test_images(w: usize, h: usize) -> (Vec<[u8; 3]>, Vec<[u8; 3]>) {
    let n = w * h;
    let mut src = vec![[0u8; 3]; n];
    let mut dst = vec![[0u8; 3]; n];
    for y in 0..h {
        for x in 0..w {
            let g = ((x * 7 + y * 13) % 251) as u8;
            let t = (((x * 31) ^ (y * 17)) % 67) as u8;
            let s = g.wrapping_add(t);
            src[y * w + x] = [s, g.wrapping_add(t / 2), 255u8 - s];
            let dv = (s as i32 + ((x * y) % 29) as i32 - 14).clamp(0, 255) as u8;
            dst[y * w + x] = [dv, g.wrapping_add(t / 3), 255u8 - dv];
        }
    }
    (src, dst)
}

fn rev5_request() -> Request {
    let want = research::family_slots(ComputeToken::Basic)
        .union(&research::family_slots(ComputeToken::Peaks))
        .union(&research::family_slots(ComputeToken::V2));
    Request::for_slots(want, research::full_width())
}

fn extract_bits(src: &RgbSlice<'_>, dst: &RgbSlice<'_>) -> Vec<u64> {
    research::extract(&rev5_request(), src, dst)
        .expect("extract")
        .values()
        .iter()
        .map(|v| v.to_bits())
        .collect()
}

fn at_revision(rev: &str, test_path: &str, sentinel: &str) -> bool {
    if std::env::var("ZENSIM_FORMULA_REV").as_deref() == Ok(rev) {
        return true;
    }
    let exe = std::env::current_exe().expect("test binary path");
    let out = std::process::Command::new(exe)
        .args([test_path, "--exact", "--nocapture", "--test-threads=1"])
        .env("ZENSIM_FORMULA_REV", rev)
        .output()
        .expect("re-exec the test binary");
    assert!(
        out.status.success(),
        "child at rev {rev} failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        String::from_utf8_lossy(&out.stdout).contains(sentinel),
        "child at rev {rev} did not run {test_path} (sentinel missing)"
    );
    false
}

#[test]
fn rev5_vector_bit_identical_under_token_permutations() {
    const SENTINEL: &str = "REV5_CANON_OK";
    if !at_revision(
        "5",
        "rev5_vector_bit_identical_under_token_permutations",
        SENTINEL,
    ) {
        return;
    }
    let _guard = archmage::testing::lock_token_testing();
    for &(w, h) in &[(64usize, 64usize), (97, 63), (131, 65), (255, 129)] {
        let (src, dst) = test_images(w, h);
        let s = RgbSlice::new(&src, w, h);
        let d = RgbSlice::new(&dst, w, h);
        let mut vectors: Vec<(String, Vec<u64>)> = Vec::new();
        let report = for_each_token_permutation(CompileTimePolicy::Warn, |perm| {
            vectors.push((perm.label.clone(), extract_bits(&s, &d)));
        });
        if HAS_RUNTIME_TOKENS {
            assert!(report.permutations_run >= 2);
        }
        let base = &vectors[0].1;
        assert!(base.iter().any(|&b| b != 0), "vacuous: all-zero vector");
        for (label, v) in &vectors[1..] {
            let diffs: Vec<usize> = base
                .iter()
                .zip(v.iter())
                .enumerate()
                .filter(|(_, (a, b))| a != b)
                .map(|(i, _)| i)
                .collect();
            assert!(
                diffs.is_empty(),
                "rev5 {w}x{h}: permutation {label} vs {} differs in {} slots (first: f{})",
                vectors[0].0,
                diffs.len(),
                diffs.first().copied().unwrap_or(usize::MAX),
            );
        }
        eprintln!("rev5 {w}x{h}: {} permutations bit-identical", vectors.len());
    }
    println!("{SENTINEL}");
}

/// The Rev5 arithmetic must actually differ from Rev4's (the era is not a
/// relabel): same pair, same slots, at least one slot moves.
#[test]
fn rev5_differs_from_rev4_somewhere() {
    const SENTINEL: &str = "REV5_DIFF_OK";
    if !at_revision("5", "rev5_differs_from_rev4_somewhere", SENTINEL) {
        return;
    }
    let (w, h) = (97usize, 63usize);
    let (src, dst) = test_images(w, h);
    let s = RgbSlice::new(&src, w, h);
    let d = RgbSlice::new(&dst, w, h);
    let r5 = extract_bits(&s, &d);
    // Rev4 reference: run a Rev4 child and compare hashes through stdout.
    let exe = std::env::current_exe().unwrap();
    let out = std::process::Command::new(exe)
        .args([
            "rev4_hash_child",
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("ZENSIM_FORMULA_REV", "4")
        .output()
        .unwrap();
    assert!(out.status.success());
    let txt = String::from_utf8_lossy(&out.stdout).to_string();
    let at = txt.find("REV4HASH ").expect("hash");
    let line = txt[at..].lines().next().unwrap();
    let r4: Vec<u64> = line["REV4HASH ".len()..]
        .split(',')
        .map(|t| t.parse().unwrap())
        .collect();
    // Rev4 cannot serve the same request shape at Rev5-only scope, but the
    // three families' slots are a subset; compare the shared slots.
    assert_eq!(r4.len(), r5.len());
    assert!(r4 != r5, "Rev5 must move values vs Rev4");
    println!("{SENTINEL}");
}

#[test]
fn rev4_hash_child() {
    if std::env::var("ZENSIM_FORMULA_REV").as_deref() != Ok("4") {
        return;
    }
    let (w, h) = (97usize, 63usize);
    let (src, dst) = test_images(w, h);
    let s = RgbSlice::new(&src, w, h);
    let d = RgbSlice::new(&dst, w, h);
    let v: Vec<String> = extract_bits(&s, &d).iter().map(|b| b.to_string()).collect();
    println!("REV4HASH {}", v.join(","));
}
