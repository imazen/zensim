//! Dump the production font path's output at a ladder of sizes, as PGM,
//! for visual comparison against alternative glyph representations
//! (e.g. the SDF-atlas prototype benchmarked 2026-07-13).
//!
//! ```text
//! cargo run -p zensim-regress --example font_size_ladder -- \
//!     /mnt/v/output/zensim-regress/sdf-explainer "Rag7" 12 18 27 54
//! ```
//!
//! Writes `engine_<size>px.pgm` (single channel — text is rendered
//! white-on-black so R carries the full signal) per requested size,
//! through the exact shipping path: Mitchell-Netravali per-cell
//! zenresize downscale of the embedded strip, gamma-correct blend.

use std::env;
use std::fs;
use std::io::Write as _;
use std::path::PathBuf;

fn main() {
    let mut args = env::args().skip(1);
    let out_dir = PathBuf::from(args.next().expect("arg 1: output dir"));
    let text = args.next().expect("arg 2: text to render");
    let sizes: Vec<u32> = args
        .map(|s| s.parse().expect("size args must be u32 px"))
        .collect();
    assert!(!sizes.is_empty(), "pass at least one size");
    fs::create_dir_all(&out_dir).expect("create out dir");

    for &char_h in &sizes {
        let (rgba, w, h) =
            zensim_regress::font::render_text_height(&text, [255; 4], [0, 0, 0, 255], char_h);
        let gray: Vec<u8> = rgba.chunks_exact(4).map(|p| p[0]).collect();
        let path = out_dir.join(format!("engine_{char_h}px.pgm"));
        let mut f = fs::File::create(&path).expect("create pgm");
        writeln!(f, "P5 {w} {h} 255").expect("header");
        f.write_all(&gray).expect("pixels");
        println!("{} ({w}x{h})", path.display());
    }
}
