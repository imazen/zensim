//! Profile-comparison diagnostic: run a single jpeg round-trip and print the
//! zensim score under every available profile. Reveals the V0_5* bug
//! (their bake produces ~0 for near-identical inputs).
//!
//! `cargo run --release --example smoke_check -p zensim-target -- <path/to/image.png>`
#![allow(deprecated)]
// exercises the deprecated `ZensimProfile::A` (shipped behind the default-on `deprecated-profiles` feature)

use image::ImageReader;
use zensim::{RgbSlice, Zensim, ZensimProfile};

fn main() -> anyhow::Result<()> {
    let path = std::env::args().nth(1).expect("usage: smoke_check <png>");
    let img = ImageReader::open(&path)?
        .with_guessed_format()?
        .decode()?
        .to_rgb8();
    let (w, h) = (img.width(), img.height());
    let rgb = img.into_raw();
    println!("loaded {}x{} ({} bytes)", w, h, rgb.len());

    let backend = zensim_target::codec::backend_for(zensim_target::CodecKind::Jpeg);
    let qs = [75.0f32, 90.0, 95.0, 98.0];
    let mut decoded_pairs = Vec::new();
    for &q in &qs {
        let (encoded, decoded) = backend.encode_decode(&rgb, w, h, q)?;
        println!("q={q:>4.0} encoded {} bytes", encoded.len());
        decoded_pairs.push((q, decoded));
    }

    let src: &[[u8; 3]] = bytemuck::cast_slice(&rgb);

    print!("\n{:>40}  {:>9}", "profile", "identity");
    for &q in &qs {
        print!("  q={:>5.0}", q);
    }
    println!();
    for profile in [
        zensim_experimental::preview_v0_2(),
        ZensimProfile::A,
        zensim_experimental::preview_v0_5(),
        zensim_experimental::preview_v0_5_balanced(),
        zensim_experimental::preview_v0_5_compression(),
        zensim_experimental::preview_v0_5_ensemble(),
    ] {
        let z = Zensim::new(profile);
        let ident = z.compute(
            &RgbSlice::try_new(src, w as usize, h as usize).unwrap(),
            &RgbSlice::try_new(src, w as usize, h as usize).unwrap(),
        )?;
        print!("{:>40}  {:>9.2}", profile.name(), ident.score());
        for (_q, decoded) in &decoded_pairs {
            let dst: &[[u8; 3]] = bytemuck::cast_slice(decoded);
            let real = z.compute(
                &RgbSlice::try_new(src, w as usize, h as usize).unwrap(),
                &RgbSlice::try_new(dst, w as usize, h as usize).unwrap(),
            )?;
            print!("  {:>7.2}", real.score());
        }
        println!();
    }
    Ok(())
}
