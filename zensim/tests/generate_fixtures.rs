// One-shot generator for the classification fixtures in tests/fixtures/.
// Run: cargo test -p zensim --test generate_fixtures -- --ignored --nocapture
//
// Emits deterministic 64x64 PNGs: base patterns, transfer-function errors,
// channel ordering, bit-depth quantization, color-matrix errors, alpha
// compositing, seeded noise, and bit-manipulation variants. Everything is
// built from tests/common generators + distortions — pure integer/f64 math,
// no external tools — so output is byte-identical on every platform.

mod common;

use image::RgbaImage;
use std::path::Path;

const W: usize = 64;
const H: usize = 64;

fn fixtures_dir() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

fn save_rgb(name: &str, pixels: &[[u8; 3]]) {
    let path = fixtures_dir().join(name);
    let flat: Vec<u8> = pixels.iter().flat_map(|p| p.iter().copied()).collect();
    image::save_buffer(&path, &flat, W as u32, H as u32, image::ColorType::Rgb8)
        .unwrap_or_else(|e| panic!("Failed to save {}: {e}", path.display()));
    println!("  {name}");
}

fn save_rgba(name: &str, pixels: &[[u8; 4]]) {
    let path = fixtures_dir().join(name);
    let mut img = RgbaImage::new(W as u32, H as u32);
    for (i, p) in pixels.iter().enumerate() {
        let x = i as u32 % W as u32;
        let y = i as u32 / W as u32;
        img.put_pixel(x, y, image::Rgba(*p));
    }
    img.save(&path)
        .unwrap_or_else(|e| panic!("Failed to save {}: {e}", path.display()));
    println!("  {name}");
}

#[test]
#[ignore]
fn write_classification_fixtures() {
    use common::distortions::*;
    use common::generators::*;

    println!("=== Base patterns ===");
    let gradient = gen_rgb_gradient(W, H);
    let blocks = gen_color_blocks(W, H);
    let gray_ramp = gen_gray_ramp(W, H);
    let alpha_patches = gen_alpha_patches(W, H);
    let alpha_gradient = gen_alpha_gradient(W, H);
    save_rgb("gradient.png", &gradient);
    save_rgb("blocks.png", &blocks);
    save_rgb("gray_ramp.png", &gray_ramp);
    save_rgba("alpha_patches.png", &alpha_patches);
    save_rgba("alpha_gradient.png", &alpha_gradient);

    println!("\n=== Transfer function errors ===");
    save_rgb("gradient_gamma22.png", &apply_gamma(&gradient, 2.2));
    save_rgb("gradient_gamma18.png", &apply_gamma(&gradient, 1.8));
    save_rgb("gradient_double_srgb.png", &srgb_encode(&gradient));
    save_rgb("gradient_linear_as_srgb.png", &srgb_decode(&gradient));

    println!("\n=== Channel ordering ===");
    save_rgb("blocks_rgb_bgr.png", &swap_rb(&blocks));
    save_rgb("blocks_cbcr_swap.png", &swap_cbcr(&blocks));

    println!("\n=== Bit depth / quantization ===");
    save_rgb("gradient_depth4.png", &quantize_bits(&gradient, 4));
    save_rgb("gradient_depth2.png", &quantize_bits(&gradient, 2));

    println!("\n=== Color space / ICC matrix ===");
    save_rgb("blocks_adobe_as_srgb.png", &adobe_rgb_to_srgb(&blocks));

    println!("\n=== Alpha / compositing ===");
    save_rgb(
        "alpha_patches_over_black.png",
        &composite_over(&alpha_patches, [0, 0, 0]),
    );
    save_rgb(
        "alpha_patches_over_white.png",
        &composite_over(&alpha_patches, [255, 255, 255]),
    );

    println!("\n=== Negative controls (seeded noise) ===");
    save_rgb(
        "gradient_noise_low.png",
        &gaussian_noise(&gradient, 13.0, 42),
    );
    save_rgb(
        "gradient_noise_high.png",
        &gaussian_noise(&gradient, 77.0, 42),
    );

    println!("\n=== Bit-manipulation variants ===");
    save_rgb("gradient_truncate.png", &truncate_lsb(&gradient));
    save_rgb("gradient_expand256.png", &expand_256(&gradient));
    save_rgb("gradient_round_half_up.png", &round_half_up(&gradient));
    save_rgba(
        "alpha_patches_premul_as_straight.png",
        &premul_as_straight(&alpha_patches),
    );
    save_rgba(
        "alpha_patches_straight_as_premul.png",
        &straight_as_premul(&alpha_patches),
    );
    save_rgba(
        "alpha_gradient_premul_as_straight.png",
        &premul_as_straight(&alpha_gradient),
    );
    save_rgba(
        "alpha_gradient_wrong_bg_black.png",
        &wrong_bg_black(&alpha_gradient),
    );

    println!("\nDone generating classification fixtures.");
}
