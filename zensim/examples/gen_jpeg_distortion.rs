//! Tiny helper: decode a PNG (any zen-crate-supported source), JPEG-encode
//! it at a given quality via zenjpeg (4:2:0 chroma subsampling), write the
//! result to disk. Used to build real-content (reference, distorted) pairs
//! from single real reference images -- e.g. the imazen-26 non-photo corpus
//! images identified as pathology-prone (real content, not a synthetic
//! fixture) for `v2_bounds_smoke`.
//!
//! ```sh
//! cargo run --release -p zensim --features feature-regime-v2 \
//!   --example gen_jpeg_distortion -- <in.png> <quality 0-100> <out.jpg> \
//!   [--max-dim N] [--ref-out <ref.png>]
//! ```
//!
//! `--max-dim N` Lanczos-downscales the source so `max(w, h) <= N` BEFORE
//! encoding, and `--ref-out` writes that downscaled source back out as a PNG.
//! Together they make this the one owner of "turn a corpus original into a
//! matched (reference, distorted) pair at a chosen display size" — which is
//! what the docstring above already claimed this helper was for, and what the
//! `diffmap_heatmap` demo gallery needs. Resizing has to happen *before* the
//! encode: downscaling a full-resolution JPEG averages its 8x8 blocking away,
//! so a pair built the other way round shows artefacts the viewer would never
//! see at that size. Both flags are optional; the three positional arguments
//! are unchanged.

#[path = "support/zen_io.rs"]
mod zen_io;

fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut positional: Vec<String> = Vec::new();
    let mut max_dim: Option<usize> = None;
    let mut ref_out: Option<String> = None;
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--max-dim" => {
                i += 1;
                max_dim = Some(
                    argv.get(i)
                        .and_then(|v| v.parse::<usize>().ok())
                        .filter(|&n| n > 0)
                        .unwrap_or_else(|| usage("--max-dim needs a positive integer")),
                );
            }
            "--ref-out" => {
                i += 1;
                ref_out = Some(
                    argv.get(i)
                        .cloned()
                        .unwrap_or_else(|| usage("--ref-out needs a path")),
                );
            }
            other if other.starts_with("--") => usage(other),
            _ => positional.push(argv[i].clone()),
        }
        i += 1;
    }
    if positional.len() != 3 {
        usage("expected <in.png> <quality> <out.jpg>");
    }
    let in_path = positional[0].clone();
    let quality: u8 = positional[1].parse().expect("quality 0-100");
    let out_path = positional[2].clone();

    let (mut px, mut w, mut h) = zen_io::decode_rgb8(std::path::Path::new(&in_path));
    if let Some(n) = max_dim
        && w.max(h) > n
    {
        let scale = n as f64 / w.max(h) as f64;
        let (tw, th) = (
            ((w as f64 * scale).round() as usize).max(1),
            ((h as f64 * scale).round() as usize).max(1),
        );
        px = zen_io::resize_rgb8(&px, w, h, tw, th);
        w = tw;
        h = th;
    }
    if let Some(path) = &ref_out {
        std::fs::write(path, zen_io::encode_png_rgb8(&px, w, h)).expect("write reference png");
    }
    let jpeg = zen_io::encode_jpeg_q(&px, w, h, quality);
    std::fs::write(&out_path, &jpeg).expect("write jpeg");
    println!(
        "{in_path} ({w}x{h}) -> {out_path} q={quality} ({} bytes)",
        jpeg.len()
    );
}

fn usage(what: &str) -> ! {
    eprintln!("bad argument: {what}");
    eprintln!(
        "usage: gen_jpeg_distortion <in.png> <quality> <out.jpg> \
         [--max-dim N] [--ref-out <ref.png>]"
    );
    std::process::exit(2);
}
