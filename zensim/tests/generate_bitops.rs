/// One-shot test to generate Rust-only distortion fixtures.
/// Run: cargo test -p zensim --test generate_bitops -- --ignored --nocapture
///
/// Loads source PNGs from tests/fixtures/, applies bit-manipulation
/// distortions, saves output PNGs alongside them.

mod common;

use image::{ImageReader, RgbaImage};
use std::path::Path;

fn fixtures_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

fn load_rgb(name: &str) -> (Vec<[u8; 3]>, u32, u32) {
    let path = fixtures_dir().join(name);
    let img = ImageReader::open(&path)
        .unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display()))
        .decode()
        .unwrap_or_else(|e| panic!("Failed to decode {}: {e}", path.display()))
        .to_rgb8();
    let (w, h) = (img.width(), img.height());
    let pixels: Vec<[u8; 3]> = img.pixels().map(|p| p.0).collect();
    (pixels, w, h)
}

fn load_rgba(name: &str) -> (Vec<[u8; 4]>, u32, u32) {
    let path = fixtures_dir().join(name);
    let img = ImageReader::open(&path)
        .unwrap_or_else(|e| panic!("Failed to open {}: {e}", path.display()))
        .decode()
        .unwrap_or_else(|e| panic!("Failed to decode {}: {e}", path.display()))
        .to_rgba8();
    let (w, h) = (img.width(), img.height());
    let pixels: Vec<[u8; 4]> = img.pixels().map(|p| p.0).collect();
    (pixels, w, h)
}

fn save_rgb(name: &str, pixels: &[[u8; 3]], w: u32, h: u32) {
    let path = fixtures_dir().join(name);
    let flat: Vec<u8> = pixels.iter().flat_map(|p| p.iter().copied()).collect();
    image::save_buffer(&path, &flat, w, h, image::ColorType::Rgb8)
        .unwrap_or_else(|e| panic!("Failed to save {}: {e}", path.display()));
    println!("  {name}");
}

fn save_rgba(name: &str, pixels: &[[u8; 4]], w: u32, h: u32) {
    let path = fixtures_dir().join(name);
    let mut img = RgbaImage::new(w, h);
    for (i, p) in pixels.iter().enumerate() {
        let x = i as u32 % w;
        let y = i as u32 / w;
        img.put_pixel(x, y, image::Rgba(*p));
    }
    img.save(&path)
        .unwrap_or_else(|e| panic!("Failed to save {}: {e}", path.display()));
    println!("  {name}");
}

#[test]
#[ignore]
fn generate_bitops_fixtures() {
    use common::distortions::*;

    println!("=== Truncation / rounding ===");
    let (gradient, w, h) = load_rgb("gradient.png");

    save_rgb("gradient_truncate.png", &truncate_lsb(&gradient), w, h);
    save_rgb("gradient_expand256.png", &expand_256(&gradient), w, h);
    save_rgb(
        "gradient_round_half_up.png",
        &round_half_up(&gradient),
        w,
        h,
    );

    println!("\n=== Alpha / premultiply (alpha_patches) ===");
    let (patches, pw, ph) = load_rgba("alpha_patches.png");

    save_rgba(
        "alpha_patches_premul_as_straight.png",
        &premul_as_straight(&patches),
        pw,
        ph,
    );
    save_rgba(
        "alpha_patches_straight_as_premul.png",
        &straight_as_premul(&patches),
        pw,
        ph,
    );

    println!("\n=== Alpha / premultiply (alpha_gradient) ===");
    let (alpha_grad, aw, ah) = load_rgba("alpha_gradient.png");

    save_rgba(
        "alpha_gradient_premul_as_straight.png",
        &premul_as_straight(&alpha_grad),
        aw,
        ah,
    );
    save_rgba(
        "alpha_gradient_wrong_bg_black.png",
        &wrong_bg_black(&alpha_grad),
        aw,
        ah,
    );

    println!("\nDone generating bitops fixtures.");
}
