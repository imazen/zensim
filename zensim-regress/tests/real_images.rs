//! Integration tests using real images from codec-corpus.
//!
//! Tests various distortions on real photographic content to verify
//! that the annotated montage, spatial analysis, and structural diff
//! produce useful diagnostics on non-synthetic images.
//!
//! These tests download images on first run (cached in ~/.cache/codec-corpus).
//! Run with `cargo test --test real_images -- --ignored` to include them.

use image::{imageops, RgbaImage};
use zensim::{RgbaSlice, Zensim, ZensimProfile};
use zensim_regress::diff_image::*;
use zensim_regress::testing::{RegressionTolerance, check_regression};

fn load_corpus_png(max_dim: u32) -> Option<RgbaImage> {
    let corpus = codec_corpus_fetch::Corpus::from_env().ok()?;
    let filter = corpus
        .filter()
        .format("png")
        .max_file_size(200_000); // small PNGs for fast tests

    let entries = filter.entries();
    eprintln!("  corpus: {} PNG entries found", entries.len());

    for (i, entry) in entries.iter().take(20).enumerate() {
        match corpus.fetch_one(entry) {
            Ok(path) => match image::open(&path)
                .or_else(|_| {
                    // Cache files have no extension; try explicit PNG
                    let data = std::fs::read(&path).map_err(|e| image::ImageError::IoError(e))?;
                    image::load_from_memory_with_format(&data, image::ImageFormat::Png)
                }) {
                Ok(img) => {
                    let rgba = img.to_rgba8();
                    let (w, h) = rgba.dimensions();
                    eprintln!("  [{i}] {}x{} from {}", w, h, path.display());
                    if w >= 64 && h >= 64 {
                        let scale = max_dim as f32 / w.max(h) as f32;
                        if scale < 1.0 {
                            let nw = (w as f32 * scale) as u32;
                            let nh = (h as f32 * scale) as u32;
                            return Some(imageops::resize(
                                &rgba,
                                nw,
                                nh,
                                imageops::FilterType::Lanczos3,
                            ));
                        }
                        return Some(rgba);
                    }
                }
                Err(e) => eprintln!("  [{i}] decode failed: {e}"),
            },
            Err(e) => eprintln!("  [{i}] fetch failed: {e}"),
        }
    }
    None
}

fn to_rgba_slice(img: &RgbaImage) -> (Vec<[u8; 4]>, usize, usize) {
    let px: Vec<[u8; 4]> = img
        .as_raw()
        .chunks_exact(4)
        .map(|c| [c[0], c[1], c[2], c[3]])
        .collect();
    (px, img.width() as usize, img.height() as usize)
}

fn save_montage(name: &str, montage: &RgbaImage) {
    let dir = std::path::Path::new("/mnt/v/output/zensim-regress/real-tests");
    std::fs::create_dir_all(dir).ok();
    let path = dir.join(format!("{name}.png"));
    montage.save(&path).unwrap();
    eprintln!("  saved: {}", path.display());
}

#[test]
#[ignore] // requires network + codec-corpus
fn real_image_blur_distortion() {
    let img = load_corpus_png(256).expect("need at least one PNG from corpus");
    let (w, h) = img.dimensions();
    eprintln!("blur test: {}x{}", w, h);

    // Apply Gaussian blur as distortion
    let blurred = imageops::blur(&img, 3.0);

    let z = Zensim::new(ZensimProfile::latest());
    let (exp_px, ew, eh) = to_rgba_slice(&img);
    let (act_px, _, _) = to_rgba_slice(&blurred);
    let tolerance = RegressionTolerance::off_by_one();
    let report = check_regression(
        &z,
        &RgbaSlice::new(&exp_px, ew, eh),
        &RgbaSlice::new(&act_px, ew, eh),
        &tolerance,
    )
    .unwrap();

    let spatial = spatial_analysis(img.as_raw(), blurred.as_raw(), w, h, 3, 3);
    let ann = format_annotation_spatial(&report, &tolerance, Some(&spatial));

    eprintln!("  blur: zdsim={:.4}", zensim::score_to_dissimilarity(report.score()));
    for (text, _) in &ann.primary_lines {
        eprintln!("  {text}");
    }

    let montage = create_annotated_montage(&img, &blurred, 10, 8, &ann);
    save_montage("blur", &montage);

    // Blur should fail off-by-one tolerance (big structural change)
    assert!(!report.passed(), "blur should exceed off-by-one tolerance");
}

#[test]
#[ignore]
fn real_image_brightness_shift() {
    let img = load_corpus_png(256).expect("need at least one PNG from corpus");
    let (w, h) = img.dimensions();
    eprintln!("brightness test: {}x{}", w, h);

    // Uniform brightness increase
    let bright = RgbaImage::from_fn(w, h, |x, y| {
        let p = img.get_pixel(x, y);
        image::Rgba([
            p[0].saturating_add(30),
            p[1].saturating_add(30),
            p[2].saturating_add(30),
            p[3],
        ])
    });

    let z = Zensim::new(ZensimProfile::latest());
    let (exp_px, ew, eh) = to_rgba_slice(&img);
    let (act_px, _, _) = to_rgba_slice(&bright);
    let tolerance = RegressionTolerance::off_by_one();
    let report = check_regression(
        &z,
        &RgbaSlice::new(&exp_px, ew, eh),
        &RgbaSlice::new(&act_px, ew, eh),
        &tolerance,
    )
    .unwrap();

    let spatial = spatial_analysis(img.as_raw(), bright.as_raw(), w, h, 3, 3);
    let ann = format_annotation_spatial(&report, &tolerance, Some(&spatial));

    eprintln!("  bright: zdsim={:.4}", zensim::score_to_dissimilarity(report.score()));

    let montage = create_annotated_montage(&img, &bright, 10, 8, &ann);
    save_montage("brightness", &montage);
}

#[test]
#[ignore]
fn real_image_watermark_added() {
    let img = load_corpus_png(256).expect("need at least one PNG from corpus");
    let (w, h) = img.dimensions();
    eprintln!("watermark test: {}x{}", w, h);

    // Add a semi-transparent watermark rectangle in bottom-right
    let mut watermarked = img.clone();
    let wx0 = w * 3 / 5;
    let wy0 = h * 3 / 5;
    for y in wy0..h.min(wy0 + h / 4) {
        for x in wx0..w.min(wx0 + w / 3) {
            let p = watermarked.get_pixel_mut(x, y);
            // Blend white watermark at 40% opacity
            p[0] = ((p[0] as u16 * 60 + 255 * 40) / 100) as u8;
            p[1] = ((p[1] as u16 * 60 + 255 * 40) / 100) as u8;
            p[2] = ((p[2] as u16 * 60 + 255 * 40) / 100) as u8;
        }
    }

    let z = Zensim::new(ZensimProfile::latest());
    let (exp_px, ew, eh) = to_rgba_slice(&img);
    let (act_px, _, _) = to_rgba_slice(&watermarked);
    let tolerance = RegressionTolerance::off_by_one();
    let report = check_regression(
        &z,
        &RgbaSlice::new(&exp_px, ew, eh),
        &RgbaSlice::new(&act_px, ew, eh),
        &tolerance,
    )
    .unwrap();

    let spatial = spatial_analysis(img.as_raw(), watermarked.as_raw(), w, h, 3, 3);
    let ann = format_annotation_spatial(&report, &tolerance, Some(&spatial));

    eprintln!("  watermark: zdsim={:.4}", zensim::score_to_dissimilarity(report.score()));
    eprintln!("  spatial:\n{}", ann.details);

    let montage = create_annotated_montage(&img, &watermarked, 10, 8, &ann);
    save_montage("watermark", &montage);

    // Spatial should show bottom-right as the affected quadrant
    assert!(
        spatial.regions[8].pixels_differing > spatial.regions[0].pixels_differing,
        "bot-right (idx 8) should have more differences than top-left (idx 0)"
    );
}

#[test]
#[ignore]
fn real_image_missing_region() {
    let img = load_corpus_png(256).expect("need at least one PNG from corpus");
    let (w, h) = img.dimensions();
    eprintln!("missing region test: {}x{}", w, h);

    // Black out bottom-right quadrant (simulates missing feature)
    let mut blanked = img.clone();
    for y in h / 2..h {
        for x in w / 2..w {
            blanked.put_pixel(x, y, image::Rgba([0, 0, 0, 255]));
        }
    }

    let z = Zensim::new(ZensimProfile::latest());
    let (exp_px, ew, eh) = to_rgba_slice(&img);
    let (act_px, _, _) = to_rgba_slice(&blanked);
    let tolerance = RegressionTolerance::off_by_one();
    let report = check_regression(
        &z,
        &RgbaSlice::new(&exp_px, ew, eh),
        &RgbaSlice::new(&act_px, ew, eh),
        &tolerance,
    )
    .unwrap();

    let spatial = spatial_analysis(img.as_raw(), blanked.as_raw(), w, h, 3, 3);
    let ann = format_annotation_spatial(&report, &tolerance, Some(&spatial));

    eprintln!("  missing: zdsim={:.4}", zensim::score_to_dissimilarity(report.score()));
    eprintln!("  spatial:\n{}", ann.details);

    let montage = create_annotated_montage(&img, &blanked, 10, 8, &ann);
    save_montage("missing_region", &montage);

    // Bottom-right should show as MISSING (had content, now uniform)
    assert!(ann.details.contains("MISSING") || ann.details.contains("different"),
        "should detect missing content in bottom-right");
}
