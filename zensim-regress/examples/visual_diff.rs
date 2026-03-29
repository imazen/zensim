//! Visual diff with sixel terminal display.
//!
//! Creates synthetic test images with deliberate differences, generates
//! amplified diff images, composes montages, and renders them as sixels
//! directly to stdout.
//!
//! Run with: cargo run -p zensim-regress --example visual_diff
//!
//! Requires a sixel-capable terminal (foot, WezTerm, mlterm, xterm +sixel,
//! or mintty). On unsupported terminals you'll see escape garbage.
//! Save to file instead with: --save-to /tmp/diff.png

#![allow(deprecated)] // TODO: migrate to create_annotated_montage

use std::env;

use image::{Rgba, RgbaImage};
use zensim_regress::diff_image::{create_comparison_montage, create_montage, generate_diff_image};
use zensim_regress::display;

fn main() {
    let args: Vec<String> = env::args().collect();
    let save_path = args
        .windows(2)
        .find(|w| w[0] == "--save-to")
        .map(|w| w[1].clone());
    let max_width: Option<u32> = args
        .windows(2)
        .find(|w| w[0] == "--max-width")
        .and_then(|w| w[1].parse().ok());

    println!("=== Visual Diff Demo ===\n");

    let (w, h) = (64u32, 64u32);

    // ─── Create test images ─────────────────────────────────────────────
    println!("--- Creating test images ({w}x{h}) ---\n");

    // Baseline: colorful gradient
    let expected = RgbaImage::from_fn(w, h, |x, y| {
        Rgba([(x * 4) as u8, (y * 4) as u8, ((x + y) * 2) as u8, 255])
    });

    // Variant 1: off-by-one rounding in R channel
    let rounding = RgbaImage::from_fn(w, h, |x, y| {
        let e = expected.get_pixel(x, y);
        Rgba([e[0].saturating_add(1), e[1], e[2], 255])
    });

    // Variant 2: color shift in center region
    let color_shift = RgbaImage::from_fn(w, h, |x, y| {
        let e = expected.get_pixel(x, y);
        if x > 16 && x < 48 && y > 16 && y < 48 {
            Rgba([e[0], e[1].saturating_add(30), e[2].wrapping_sub(20), 255])
        } else {
            *e
        }
    });

    // Variant 3: channel swap (R↔B)
    let channel_swap = RgbaImage::from_fn(w, h, |x, y| {
        let e = expected.get_pixel(x, y);
        Rgba([e[2], e[1], e[0], 255])
    });

    // ─── Generate diffs ─────────────────────────────────────────────────

    let diff_rounding = generate_diff_image(&expected, &rounding, 50);
    let diff_color = generate_diff_image(&expected, &color_shift, 5);
    let diff_swap = generate_diff_image(&expected, &channel_swap, 1);

    // ─── Display: rounding ──────────────────────────────────────────────
    println!("--- 1. Off-by-one rounding (x50 amplification) ---");
    println!("  Uniform +1 in R channel. Diff should glow red.\n");

    if let Some(ref path) = save_path {
        let montage = create_comparison_montage(&expected, &rounding, 50, 2);
        let save = format!("{}_rounding.png", path.trim_end_matches(".png"));
        montage.save(&save).unwrap();
        println!("  Saved: {save}");
    } else {
        display::print_comparison(&expected, &rounding, 50, max_width);
    }
    println!();

    // ─── Display: color shift ───────────────────────────────────────────
    println!("--- 2. Color shift in center (x5 amplification) ---");
    println!("  +30 green, -20 blue in center 32x32 region.\n");

    if let Some(ref path) = save_path {
        let montage = create_comparison_montage(&expected, &color_shift, 5, 2);
        let save = format!("{}_colorshift.png", path.trim_end_matches(".png"));
        montage.save(&save).unwrap();
        println!("  Saved: {save}");
    } else {
        display::print_comparison(&expected, &color_shift, 5, max_width);
    }
    println!();

    // ─── Display: channel swap ──────────────────────────────────────────
    println!("--- 3. Channel swap R<->B (x1, no amplification needed) ---");
    println!("  RGB/BGR swap — large difference everywhere.\n");

    if let Some(ref path) = save_path {
        let montage = create_comparison_montage(&expected, &channel_swap, 1, 2);
        let save = format!("{}_channelswap.png", path.trim_end_matches(".png"));
        montage.save(&save).unwrap();
        println!("  Saved: {save}");
    } else {
        display::print_comparison(&expected, &channel_swap, 1, max_width);
    }
    println!();

    // ─── Custom montage: all diffs side by side ─────────────────────────
    println!("--- 4. All diffs in one montage ---");
    println!("  Rounding(x50) | ColorShift(x5) | ChannelSwap(x1)\n");

    let all_diffs = create_montage(&[&diff_rounding, &diff_color, &diff_swap], 4);

    if let Some(ref path) = save_path {
        let save = format!("{}_all_diffs.png", path.trim_end_matches(".png"));
        all_diffs.save(&save).unwrap();
        println!("  Saved: {save}");
    } else {
        display::print_image(&all_diffs, max_width);
    }
    println!();

    // ─── Direct sixel_encode for programmatic use ───────────────────────
    println!("--- 5. Sixel encoding stats ---");

    let bytes_expected = display::sixel_encode(&expected, None);
    let bytes_montage = display::sixel_encode(&all_diffs, None);
    let bytes_resized = display::sixel_encode(&expected, Some(32));

    println!("  64x64 image:    {} bytes sixel", bytes_expected.len());
    println!(
        "  {}x{} montage:  {} bytes sixel",
        all_diffs.width(),
        all_diffs.height(),
        bytes_montage.len()
    );
    println!("  64x64 → 32x32:  {} bytes sixel", bytes_resized.len());
    println!(
        "  compression:     {:.0}x vs raw RGBA",
        (64 * 64 * 4) as f64 / bytes_expected.len() as f64
    );

    println!("\n=== visual diff demo complete ===");
}
