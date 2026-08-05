//! Integration tests for the `zensim-diff` CLI (issue #14).
//!
//! Runs the actual binary (`CARGO_BIN_EXE_zensim-diff`) on synthetic PNGs
//! written to `CARGO_TARGET_TMPDIR` and checks every mode's contract.

use std::path::PathBuf;
use std::process::Command;

use zensim_regress::Bitmap;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_zensim-diff")
}

/// Per-TEST scratch dir. The tests in this file run on concurrent
/// threads; a shared `diff_cli/expected.png` was a write/read race — one
/// test's `zensim-diff` subprocess read the fixture while another test
/// re-wrote the same path, intermittently observing a truncated PNG
/// ("not a PNG file", surfaced on CI 2026-08-05 once the runner got past
/// the #55 golden-gate fail-fast). Unique-per-test dirs make the fixture
/// race-free; every assertion is unchanged.
fn tmpdir(tag: &str) -> PathBuf {
    let d = PathBuf::from(env!("CARGO_TARGET_TMPDIR"))
        .join("diff_cli")
        .join(tag);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Deterministic gradient pair with a visible perturbation block.
fn write_test_pair(tag: &str) -> (PathBuf, PathBuf) {
    let d = tmpdir(tag);
    let (w, h) = (96u32, 96u32);
    let mut exp = Bitmap::new(w, h);
    let mut act = Bitmap::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let r = (x * 255 / (w - 1)) as u8;
            let g = (y * 255 / (h - 1)) as u8;
            let px = [r, g, 128, 255];
            exp.put_pixel(x, y, px);
            // A 24×24 block shifted by 40 codes — clearly localized.
            let shifted = if (24..48).contains(&x) && (24..48).contains(&y) {
                [r.saturating_add(40), g, 128, 255]
            } else {
                px
            };
            act.put_pixel(x, y, shifted);
        }
    }
    let e = d.join("expected.png");
    let a = d.join("actual.png");
    exp.save(&e).unwrap();
    act.save(&a).unwrap();
    (e, a)
}

#[test]
fn montage_mode_writes_png_and_scores() {
    let (e, a) = write_test_pair("montage");
    let out = tmpdir("montage").join("montage.png");
    let st = Command::new(bin())
        .args([
            e.to_str().unwrap(),
            a.to_str().unwrap(),
            "-o",
            out.to_str().unwrap(),
            "--score",
        ])
        .output()
        .unwrap();
    assert!(
        st.status.success(),
        "stderr: {}",
        String::from_utf8_lossy(&st.stderr)
    );
    let stdout = String::from_utf8_lossy(&st.stdout);
    assert!(
        stdout.contains("zensim_score:"),
        "--score must print the score, got: {stdout}"
    );
    // Output must be a decodable PNG at least as large as 2 panels.
    let img = Bitmap::open(&out).expect("montage output must be a valid PNG");
    assert!(img.width() >= 2 * 96 && img.height() >= 2 * 96);
}

#[test]
fn pixel_and_structural_modes_write_same_size_diffs() {
    let (e, a) = write_test_pair("modes");
    for mode in ["pixel", "structural"] {
        let out = tmpdir("modes").join(format!("{mode}.png"));
        let st = Command::new(bin())
            .args([
                e.to_str().unwrap(),
                a.to_str().unwrap(),
                "--mode",
                mode,
                "-o",
                out.to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(
            st.status.success(),
            "{mode}: {}",
            String::from_utf8_lossy(&st.stderr)
        );
        let img = Bitmap::open(&out).unwrap();
        assert_eq!((img.width(), img.height()), (96, 96), "{mode} diff is 1:1");
    }
}

#[test]
fn spatial_mode_json_localizes_the_perturbed_block() {
    let (e, a) = write_test_pair("spatial");
    let st = Command::new(bin())
        .args([
            e.to_str().unwrap(),
            a.to_str().unwrap(),
            "--mode",
            "spatial",
            "--grid",
            "4x4",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(st.status.success());
    let stdout = String::from_utf8_lossy(&st.stdout);
    assert!(stdout.contains("\"cols\":4") && stdout.contains("\"regions\":["));
    // The perturbation lives in the (1,1) 24px cell of the 4×4 grid over
    // 96px: region (1,1) must report differing pixels; corner (0,0) none.
    let r11 = stdout
        .split("{\"col\":1,\"row\":1,")
        .nth(1)
        .expect("region (1,1) present");
    let differing: f64 = r11
        .split("\"pixels_differing\":")
        .nth(1)
        .unwrap()
        .split(',')
        .next()
        .unwrap()
        .parse()
        .unwrap();
    assert!(
        differing > 0.9,
        "block cell must be ~fully differing, got {differing}"
    );
    let r00 = stdout
        .split("{\"col\":0,\"row\":0,")
        .nth(1)
        .expect("region (0,0) present");
    let clean: f64 = r00
        .split("\"pixels_differing\":")
        .nth(1)
        .unwrap()
        .split(',')
        .next()
        .unwrap()
        .parse()
        .unwrap();
    assert_eq!(clean, 0.0, "clean corner must report zero differing pixels");
}

#[test]
fn mismatched_dims_rejected_outside_montage() {
    let (e, _) = write_test_pair("mismatch");
    let d = tmpdir("mismatch");
    let small = d.join("small.png");
    Bitmap::new(32, 32).save(&small).unwrap();
    let st = Command::new(bin())
        .args([
            e.to_str().unwrap(),
            small.to_str().unwrap(),
            "--mode",
            "pixel",
        ])
        .output()
        .unwrap();
    assert!(
        !st.status.success(),
        "pixel mode on mismatched dims must fail loudly"
    );
    assert!(String::from_utf8_lossy(&st.stderr).contains("dimension mismatch"));
}

#[test]
fn bad_args_fail_with_usage() {
    let st = Command::new(bin()).args(["only-one.png"]).output().unwrap();
    assert!(!st.status.success());
    assert!(String::from_utf8_lossy(&st.stderr).contains("USAGE"));
}
