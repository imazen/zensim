//! Tiny helper: decode a PNG (any zen-crate-supported source), JPEG-encode
//! it at a given quality via zenjpeg (4:2:0 chroma subsampling), write the
//! result to disk. Used to build real-content (reference, distorted) pairs
//! from single real reference images -- e.g. the imazen-26 non-photo corpus
//! images identified as pathology-prone (real content, not a synthetic
//! fixture) for `v2_bounds_smoke`.
//!
//! ```sh
//! cargo run --release -p zensim --features feature-regime-v2 \
//!   --example gen_jpeg_distortion -- <in.png> <quality 0-100> <out.jpg>
//! ```

#[path = "support/zen_io.rs"]
mod zen_io;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 3 {
        eprintln!("usage: gen_jpeg_distortion <in.png> <quality> <out.jpg>");
        std::process::exit(2);
    }
    let (in_path, quality, out_path) = (
        &args[0],
        args[1].parse::<u8>().expect("quality 0-100"),
        &args[2],
    );
    let (px, w, h) = zen_io::decode_rgb8(std::path::Path::new(in_path));
    let jpeg = zen_io::encode_jpeg_q(&px, w, h, quality);
    std::fs::write(out_path, &jpeg).expect("write jpeg");
    println!(
        "{in_path} ({w}x{h}) -> {out_path} q={quality} ({} bytes)",
        jpeg.len()
    );
}
