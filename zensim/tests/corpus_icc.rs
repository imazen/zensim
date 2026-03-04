//! Real-image corpus tests for ICC color primaries.
//!
//! Data-driven tests that download real images from S3, decode them, and
//! verify zensim handles non-sRGB color primaries correctly.
//!
//! **Requires**: `REGRESS_REFERENCE_URL` env var (e.g.,
//! `https://imageflow-resources.s3.amazonaws.com/test_inputs`).
//! Tests are `#[ignore]` when the env var is not set.
//!
//! Run with:
//! ```sh
//! REGRESS_REFERENCE_URL=https://imageflow-resources.s3.amazonaws.com/test_inputs \
//!   cargo test -p zensim --test corpus_icc -- --ignored --nocapture
//! ```

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use image::GenericImageView;
use zensim::{ColorPrimaries, PixelFormat, StridedBytes, Zensim, ZensimProfile};

// ─── Infrastructure ──────────────────────────────────────────────────

fn base_url() -> Option<&'static str> {
    static URL: OnceLock<Option<String>> = OnceLock::new();
    URL.get_or_init(|| std::env::var("REGRESS_REFERENCE_URL").ok())
        .as_deref()
}

fn cache_dir() -> PathBuf {
    let dir = PathBuf::from("/tmp/zensim-corpus-cache");
    std::fs::create_dir_all(&dir).expect("create cache dir");
    dir
}

/// Download a file from base_url/key if not cached. Returns local path.
fn fetch_cached(s3_key: &str) -> PathBuf {
    let url = format!("{}/{}", base_url().expect("REGRESS_REFERENCE_URL"), s3_key);
    let local_name = s3_key.replace('/', "_");
    let dest = cache_dir().join(&local_name);

    if dest.exists() {
        return dest;
    }

    // Try curl first, then wget
    let result = Command::new("curl")
        .args(["-fSL", "-o"])
        .arg(&dest)
        .arg(&url)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .output();

    match result {
        Ok(output) if output.status.success() => return dest,
        _ => {}
    }

    let result = Command::new("wget")
        .args(["-q", "-O"])
        .arg(&dest)
        .arg(&url)
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::piped())
        .output();

    match result {
        Ok(output) if output.status.success() => dest,
        Ok(output) => {
            let _ = std::fs::remove_file(&dest);
            panic!(
                "Failed to download {url}: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
        Err(e) => {
            let _ = std::fs::remove_file(&dest);
            panic!("Failed to download {url}: {e}");
        }
    }
}

/// Decode an image file to RGBA8 pixels.
fn decode_image(path: &Path) -> (Vec<u8>, u32, u32) {
    let img = image::open(path).unwrap_or_else(|e| panic!("decode {}: {e}", path.display()));
    let (w, h) = img.dimensions();
    let rgba = img.into_rgba8();
    (rgba.into_raw(), w, h)
}

/// Map a wide-gamut subfolder name to ColorPrimaries.
fn primaries_from_folder(folder: &str) -> Option<ColorPrimaries> {
    match folder {
        "display-p3" => Some(ColorPrimaries::DisplayP3),
        "rec-2020-pq" => Some(ColorPrimaries::Bt2020),
        "adobe-rgb" => None,    // Adobe RGB not directly supported by zensim
        "prophoto-rgb" => None, // ProPhoto not supported
        "srgb-reference" => Some(ColorPrimaries::Srgb),
        _ => None,
    }
}

fn should_skip() -> bool {
    base_url().is_none()
}

fn zensim() -> Zensim {
    Zensim::new(ZensimProfile::latest())
}

// ─── Wide-gamut corpus tests ─────────────────────────────────────────

/// Wide-gamut images with known ICC profiles. Self-comparison must be 100.0.
const WIDE_GAMUT_SAMPLES: &[(&str, &str)] = &[
    // Display P3 samples
    (
        "wide-gamut/display-p3/flickr_1b94e1228c32cb98.jpg",
        "display-p3",
    ),
    (
        "wide-gamut/display-p3/flickr_2fc1b8c45f922b8e.jpg",
        "display-p3",
    ),
    (
        "wide-gamut/display-p3/flickr_3ac029fc145a8e32.jpg",
        "display-p3",
    ),
    (
        "wide-gamut/display-p3/flickr_403aa5efb8efe6e8.jpg",
        "display-p3",
    ),
    (
        "wide-gamut/display-p3/flickr_47b2cd2c048f29b3.jpg",
        "display-p3",
    ),
    // Rec.2020 samples
    (
        "wide-gamut/rec-2020-pq/flickr_2a68670c58131566.jpg",
        "rec-2020-pq",
    ),
    (
        "wide-gamut/rec-2020-pq/flickr_c2d8824d6ffb6e60.jpg",
        "rec-2020-pq",
    ),
    // sRGB reference (camera samples)
    (
        "wide-gamut/srgb-reference/canon_eos_5d_mark_iv/wmc_81b268fc64ea796c.jpg",
        "srgb-reference",
    ),
    (
        "wide-gamut/srgb-reference/sony-a7r-v/irsample_a141d146726a8314.jpg",
        "srgb-reference",
    ),
    // Adobe RGB (treated as sRGB for decode, just verify no crash)
    (
        "wide-gamut/adobe-rgb/flickr_0119a8378404ece9.jpg",
        "adobe-rgb",
    ),
    (
        "wide-gamut/adobe-rgb/flickr_070040b3922aab8a.jpg",
        "adobe-rgb",
    ),
    // ProPhoto (just verify no crash)
    (
        "wide-gamut/prophoto-rgb/flickr_0d2d634cf46df137.jpg",
        "prophoto-rgb",
    ),
];

#[test]
#[ignore]
fn corpus_wide_gamut_self_comparison() {
    if should_skip() {
        eprintln!("Skipping: REGRESS_REFERENCE_URL not set");
        return;
    }
    let z = zensim();
    let mut failures = Vec::new();

    for &(s3_key, folder) in WIDE_GAMUT_SAMPLES {
        let path = fetch_cached(s3_key);
        let (rgba, w, h) = decode_image(&path);

        // Determine primaries — for unsupported profiles, just use sRGB
        let primaries = primaries_from_folder(folder).unwrap_or(ColorPrimaries::Srgb);

        let src = StridedBytes::new(
            &rgba,
            w as usize,
            h as usize,
            w as usize * 4,
            PixelFormat::Srgb8Rgba,
        )
        .with_color_primaries(primaries);
        let dst = StridedBytes::new(
            &rgba,
            w as usize,
            h as usize,
            w as usize * 4,
            PixelFormat::Srgb8Rgba,
        )
        .with_color_primaries(primaries);

        let result = z.compute(&src, &dst).unwrap();
        println!("  {s3_key:70} {primaries:?} score={:.4}", result.score);

        if result.score != 100.0 {
            failures.push(format!("{s3_key}: expected 100.0, got {}", result.score));
        }
    }

    assert!(
        failures.is_empty(),
        "Self-comparison failures:\n{}",
        failures.join("\n")
    );
}

#[test]
#[ignore]
fn corpus_wide_gamut_wrong_primaries_differs() {
    if should_skip() {
        eprintln!("Skipping: REGRESS_REFERENCE_URL not set");
        return;
    }
    let z = zensim();

    // Test with Display P3 images — comparing with correct P3 vs incorrect sRGB
    let p3_samples = &[
        "wide-gamut/display-p3/flickr_1b94e1228c32cb98.jpg",
        "wide-gamut/display-p3/flickr_2fc1b8c45f922b8e.jpg",
        "wide-gamut/display-p3/flickr_3ac029fc145a8e32.jpg",
    ];

    for s3_key in p3_samples {
        let path = fetch_cached(s3_key);
        let (rgba, w, h) = decode_image(&path);
        let (w, h) = (w as usize, h as usize);

        // Source labeled P3, distorted labeled sRGB
        let src = StridedBytes::new(&rgba, w, h, w * 4, PixelFormat::Srgb8Rgba)
            .with_color_primaries(ColorPrimaries::DisplayP3);
        let dst = StridedBytes::new(&rgba, w, h, w * 4, PixelFormat::Srgb8Rgba)
            .with_color_primaries(ColorPrimaries::Srgb);

        let result = z.compute(&src, &dst).unwrap();
        println!("  {s3_key:70} P3-vs-sRGB score={:.4}", result.score);

        assert!(
            result.score < 100.0,
            "P3 image {s3_key} labeled as sRGB should differ: got {}",
            result.score
        );
    }

    // Test with Rec.2020 images
    let bt2020_samples = &["wide-gamut/rec-2020-pq/flickr_2a68670c58131566.jpg"];

    for s3_key in bt2020_samples {
        let path = fetch_cached(s3_key);
        let (rgba, w, h) = decode_image(&path);
        let (w, h) = (w as usize, h as usize);

        let src = StridedBytes::new(&rgba, w, h, w * 4, PixelFormat::Srgb8Rgba)
            .with_color_primaries(ColorPrimaries::Bt2020);
        let dst = StridedBytes::new(&rgba, w, h, w * 4, PixelFormat::Srgb8Rgba)
            .with_color_primaries(ColorPrimaries::Srgb);

        let result = z.compute(&src, &dst).unwrap();
        println!("  {s3_key:70} BT.2020-vs-sRGB score={:.4}", result.score);

        assert!(
            result.score < 100.0,
            "BT.2020 image {s3_key} labeled as sRGB should differ: got {}",
            result.score
        );
    }
}

#[test]
#[ignore]
fn corpus_repro_icc_decodable() {
    if should_skip() {
        eprintln!("Skipping: REGRESS_REFERENCE_URL not set");
        return;
    }
    let z = zensim();

    // A curated subset of repro-icc images to verify no decode panics
    let repro_samples = &[
        "repro-icc/sharp/1323_115925293-3319d700-a481-11eb-8083-66b5188ee1da.png",
        "repro-icc/python-pillow/1529_9fa6c9ca-8603-11e5-97e7-589cf9e3baaa.jpg",
        "repro-icc/imagemagick/2161_84902501-90046e00-b0b5-11ea-91c6-c220fd29fd44.jpg",
        "repro-icc/libvips/1063_44146319-5742eab6-a08f-11e8-911a-2aaef2a42540.jpg",
    ];

    let mut decoded = 0;
    let mut failed = Vec::new();

    for s3_key in repro_samples {
        let path = match std::panic::catch_unwind(|| fetch_cached(s3_key)) {
            Ok(p) => p,
            Err(_) => {
                eprintln!("  SKIP {s3_key}: download failed");
                continue;
            }
        };

        // Try to decode — some images may be unsupported formats
        let img = match image::open(&path) {
            Ok(img) => img,
            Err(e) => {
                eprintln!("  SKIP {s3_key}: decode error: {e}");
                continue;
            }
        };

        let (w, h) = img.dimensions();
        if w < 8 || h < 8 {
            eprintln!("  SKIP {s3_key}: too small ({w}x{h})");
            continue;
        }

        let rgba = img.into_rgba8().into_raw();
        let (w, h) = (w as usize, h as usize);

        let src = StridedBytes::new(&rgba, w, h, w * 4, PixelFormat::Srgb8Rgba);
        let dst = StridedBytes::new(&rgba, w, h, w * 4, PixelFormat::Srgb8Rgba);

        match z.compute(&src, &dst) {
            Ok(result) => {
                if result.score != 100.0 {
                    failed.push(format!(
                        "{s3_key}: self-comparison score {}, expected 100.0",
                        result.score
                    ));
                }
                decoded += 1;
                println!("  {s3_key:70} score={:.4}", result.score);
            }
            Err(e) => {
                failed.push(format!("{s3_key}: compute error: {e}"));
            }
        }
    }

    println!("  Decoded and scored: {decoded}/{}", repro_samples.len());
    assert!(
        failed.is_empty(),
        "Repro-ICC failures:\n{}",
        failed.join("\n")
    );
}

// ─── Scan-based tests (run on all images in wide-gamut categories) ───

#[test]
#[ignore]
fn corpus_wide_gamut_display_p3_scan() {
    if should_skip() {
        eprintln!("Skipping: REGRESS_REFERENCE_URL not set");
        return;
    }

    // Read manifest to find all display-p3 images
    let manifest_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .join("test-corpus-manifest.tsv");

    if !manifest_path.exists() {
        eprintln!("Skipping: test-corpus-manifest.tsv not found");
        return;
    }

    let manifest = std::fs::read_to_string(&manifest_path).unwrap();
    let z = zensim();
    let mut tested = 0;
    let mut failures = Vec::new();

    for line in manifest.lines().skip(1) {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.len() < 2 {
            continue;
        }
        let s3_key = fields[0];
        let category = fields[1];

        if category != "display-p3" {
            continue;
        }

        let path = fetch_cached(s3_key);
        let (rgba, w, h) = match std::panic::catch_unwind(|| decode_image(&path)) {
            Ok(v) => v,
            Err(_) => {
                eprintln!("  SKIP {s3_key}: decode failed");
                continue;
            }
        };
        let (w, h) = (w as usize, h as usize);

        let src = StridedBytes::new(&rgba, w, h, w * 4, PixelFormat::Srgb8Rgba)
            .with_color_primaries(ColorPrimaries::DisplayP3);
        let dst = StridedBytes::new(&rgba, w, h, w * 4, PixelFormat::Srgb8Rgba)
            .with_color_primaries(ColorPrimaries::DisplayP3);

        let result = z.compute(&src, &dst).unwrap();
        if result.score != 100.0 {
            failures.push(format!("{s3_key}: score {}", result.score));
        }
        tested += 1;
        println!("  [{tested:3}] {s3_key:70} score={:.4}", result.score);
    }

    println!("  Display P3 scan: {tested} images tested");
    assert!(
        failures.is_empty(),
        "P3 self-comparison failures:\n{}",
        failures.join("\n")
    );
    assert!(tested > 0, "No display-p3 images found in manifest");
}
