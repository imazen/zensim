//! Speed comparison: production font path (per-cell Mitchell zenresize,
//! cached scaled strip) vs a prototype SDF sampler (bilinear field
//! sample + threshold + sRGB LUT, no cache).
//!
//! ```text
//! cargo run --release -p zensim-regress --example sdf_speed -- \
//!     /mnt/v/output/zensim-regress/sdf-explainer/sdf_atlas_consolas27.bin
//! ```
//!
//! Atlas bin (baked by benchmarks/sdf_charset_sizes companion tooling):
//! header `<u32 cell_w><u32 cell_h><u32 n_glyphs><f32 spread>` then
//! `n * cell_h * cell_w` bytes of 8-bit SDF, glyph-major.
//!
//! Measures, per size: WARM steady-state render (engine strip cache
//! hot) and a COLD size sweep (every size a cache miss for the engine;
//! the SDF path has no per-size state). Example-grade `Instant` timing
//! for design comparison — port to zenbench when a real Rust sampler
//! lands.

use std::env;
use std::fs;
use std::sync::OnceLock;
use std::time::Instant;

const COMP: f32 = 0.2; // small-size weight compensation (texels at k=0)

struct SdfAtlas {
    cell_w: u32,
    cell_h: u32,
    n: u32,
    spread: f32,
    data: Vec<u8>,
}

impl SdfAtlas {
    fn load(path: &str) -> Self {
        let b = fs::read(path).expect("read atlas bin");
        let u = |i: usize| u32::from_le_bytes(b[i..i + 4].try_into().unwrap());
        let (cell_w, cell_h, n) = (u(0), u(4), u(8));
        let spread = f32::from_le_bytes(b[12..16].try_into().unwrap());
        let data = b[16..].to_vec();
        assert_eq!(data.len(), (n * cell_w * cell_h) as usize);
        SdfAtlas { cell_w, cell_h, n, spread, data }
    }

    #[inline]
    fn texel(&self, glyph: u32, x: u32, y: u32) -> f32 {
        let idx = ((glyph * self.cell_h + y) * self.cell_w + x) as usize;
        self.data[idx] as f32
    }
}

/// coverage (0..=255 linear) -> sRGB byte for white-over-black.
fn srgb_lut() -> &'static [u8; 256] {
    static LUT: OnceLock<[u8; 256]> = OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [0u8; 256];
        for (i, e) in t.iter_mut().enumerate() {
            let a = i as f32 / 255.0;
            let v = if a <= 0.003_130_8 { 12.92 * a } else { 1.055 * a.powf(1.0 / 2.4) - 0.055 };
            *e = (v * 255.0).round() as u8;
        }
        t
    })
}

/// Render `text` at `char_h` into RGBA (white on black), SDF path.
fn render_text_sdf(atlas: &SdfAtlas, text: &str, char_h: u32) -> (Vec<u8>, u32, u32) {
    let lut = srgb_lut();
    let k = char_h as f32 / atlas.cell_h as f32;
    let char_w = ((atlas.cell_w as f32) * k).round().max(1.0) as u32;
    let shift = COMP * (1.0 - k).max(0.0);
    let (out_w, out_h) = (char_w * text.chars().count() as u32, char_h);
    let mut buf = vec![0u8; (out_w * out_h * 4) as usize];
    let inv_k = 1.0 / k;
    let scale = 2.0 * atlas.spread / 255.0;

    for (gi, ch) in text.chars().enumerate() {
        let glyph = (ch as u32).saturating_sub(0x20).min(atlas.n - 1);
        for y in 0..char_h {
            let sy = ((y as f32 + 0.5) * inv_k - 0.5).clamp(0.0, (atlas.cell_h - 1) as f32);
            let y0 = sy as u32;
            let y1 = (y0 + 1).min(atlas.cell_h - 1);
            let fy = sy - y0 as f32;
            for x in 0..char_w {
                let sx = ((x as f32 + 0.5) * inv_k - 0.5).clamp(0.0, (atlas.cell_w - 1) as f32);
                let x0 = sx as u32;
                let x1 = (x0 + 1).min(atlas.cell_w - 1);
                let fx = sx - x0 as f32;
                let q = self_lerp(
                    self_lerp(atlas.texel(glyph, x0, y0), atlas.texel(glyph, x1, y0), fx),
                    self_lerp(atlas.texel(glyph, x0, y1), atlas.texel(glyph, x1, y1), fx),
                    fy,
                );
                let d = q * scale - atlas.spread; // texel units, +inside
                let cov = (0.5 + (d + shift) * k).clamp(0.0, 1.0);
                let g = lut[(cov * 255.0) as usize];
                let o = ((y * out_w + (gi as u32 * char_w + x)) * 4) as usize;
                buf[o..o + 4].copy_from_slice(&[g, g, g, 255]);
            }
        }
    }
    (buf, out_w, out_h)
}

#[inline]
fn self_lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

fn time_ms(mut f: impl FnMut(), iters: u32) -> f64 {
    let t = Instant::now();
    for _ in 0..iters {
        f();
    }
    t.elapsed().as_secs_f64() * 1000.0 / iters as f64
}

fn main() {
    let path = env::args().nth(1).expect("arg 1: atlas bin path");
    let atlas = SdfAtlas::load(&path);
    let text: String = "The quick brown fox jumps over the lazy dog 0123456789 <=>".into();

    println!("text: {} chars", text.chars().count());
    println!("{:>5} {:>10} {:>12} {:>12} {:>8}", "size", "out px", "engine ms", "sdf ms", "ratio");
    for &char_h in &[12u32, 18, 27, 54, 96] {
        // Warm both paths (engine builds+caches its scaled strip here).
        let (e, ew, eh) = zensim_regress::font::render_text_height(
            &text, [255; 4], [0, 0, 0, 255], char_h);
        let (s, sw, sh) = render_text_sdf(&atlas, &text, char_h);
        assert!(!e.is_empty() && !s.is_empty());
        let iters = (200_000 / (ew * eh).max(1)).clamp(20, 2000);
        let em = time_ms(|| {
            let _ = zensim_regress::font::render_text_height(
                &text, [255; 4], [0, 0, 0, 255], char_h);
        }, iters);
        let sm = time_ms(|| { let _ = render_text_sdf(&atlas, &text, char_h); }, iters);
        println!("{:>4}px {:>4}x{:<5} {:>9.3}ms {:>9.3}ms {:>7.2}x  (sdf out {}x{})",
                 char_h, ew, eh, em, sm, em / sm, sw, sh);
    }

    // COLD sweep: 40 distinct sizes the engine has never seen this run —
    // each one rebuilds + caches a full 96-cell scaled strip.
    let sizes: Vec<u32> = (13..53).collect();
    let t = Instant::now();
    for &h in &sizes {
        let _ = zensim_regress::font::render_text_height(&text, [255; 4], [0, 0, 0, 255], h);
    }
    let engine_cold = t.elapsed().as_secs_f64() * 1000.0;
    let t = Instant::now();
    for &h in &sizes {
        let _ = render_text_sdf(&atlas, &text, h);
    }
    let sdf_cold = t.elapsed().as_secs_f64() * 1000.0;
    println!("\ncold sweep, {} unseen sizes: engine {:.1}ms  sdf {:.1}ms  ({:.1}x)",
             sizes.len(), engine_cold, sdf_cold, engine_cold / sdf_cold);
}
