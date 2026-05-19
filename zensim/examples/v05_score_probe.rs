//! Smoke probe — print V0_5* score across q=10..95 on a synthetic
//! gradient. Diagnostic only.

use image::codecs::jpeg::JpegEncoder;
use image::{ImageBuffer, Rgb};
use std::io::Cursor;
use zensim::{RgbSlice, Zensim, ZensimProfile};

fn make_gradient_pattern(w: usize, h: usize) -> Vec<[u8; 3]> {
    (0..w * h)
        .map(|i| {
            let x = i % w;
            let y = i / w;
            let r = ((x * 255) / w.max(1)) as u8;
            let g = ((y * 255) / h.max(1)) as u8;
            let b = (((x + y) * 255) / (w + h).max(1)) as u8;
            let hf = ((x ^ y) & 0b1111) as u8 * 8;
            [
                r.saturating_add(hf),
                g.saturating_sub(hf / 2),
                b.saturating_add(hf / 3),
            ]
        })
        .collect()
}

fn jpeg_roundtrip(pixels: &[[u8; 3]], w: u32, h: u32, q: u8) -> Vec<[u8; 3]> {
    let flat: Vec<u8> = pixels.iter().flat_map(|p| p.iter().copied()).collect();
    let img = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_raw(w, h, flat).unwrap();
    let mut bytes = Vec::with_capacity(8192);
    {
        let enc = JpegEncoder::new_with_quality(&mut bytes, q);
        img.write_with_encoder(enc).unwrap();
    }
    let decoded = image::ImageReader::new(Cursor::new(&bytes))
        .with_guessed_format()
        .unwrap()
        .decode()
        .unwrap()
        .into_rgb8();
    decoded.pixels().map(|p| [p[0], p[1], p[2]]).collect()
}

fn main() {
    const W: usize = 64;
    const H: usize = 64;
    let reference = make_gradient_pattern(W, H);
    println!("V0_5 calibration probe — synthetic gradient {W}x{H}\n");
    println!("| profile           |  q   |  raw      |  score   |");
    println!("|-------------------|------|-----------|----------|");
    for profile in [
        ZensimProfile::PreviewV0_5Balanced,
        ZensimProfile::PreviewV0_5Compression,
        ZensimProfile::PreviewV0_5Ensemble,
    ] {
        let label = match profile {
            ZensimProfile::PreviewV0_5Balanced => "PreviewV0_5Balanced",
            ZensimProfile::PreviewV0_5Compression => "PreviewV0_5Compression",
            ZensimProfile::PreviewV0_5Ensemble => "PreviewV0_5Ensemble",
            _ => "other",
        };
        for &q in &[10u8, 20, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95] {
            let distorted = jpeg_roundtrip(&reference, W as u32, H as u32, q);
            let src = RgbSlice::new(&reference, W, H);
            let dst = RgbSlice::new(&distorted, W, H);
            let z = Zensim::new(profile).with_parallel(false);
            let r = z.compute(&src, &dst).unwrap();
            println!(
                "| {label:18} | {q:>3} | {:>9.4} | {:>8.4} |",
                r.raw_distance(),
                r.score()
            );
        }
        println!("|-------------------|------|-----------|----------|");
    }
}
