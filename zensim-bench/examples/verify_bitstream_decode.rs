//! verify_bitstream_decode — prove `qNN.png` == zencodec-decode(`qNN.<bitstream>`).
//!
//! The ~403 GiB of `qNN.png` files under `/mnt/v/input/zensim/images` are a
//! lossless DECODE CACHE: the synthetic-training generator
//! (`coefficient/examples/generate_zensim_training.rs`) encoded each source
//! tile to a compressed bitstream (`qNN.jpg` / `.webp` / `.avif` / `.jxl`),
//! decoded it back to RGB8, and saved that RGB8 as `qNN.png`. The 372-feature
//! extractor then reads the PNG. So the PNG is just "the pixels you get by
//! decoding the adjacent bitstream", frozen at generation time.
//!
//! This tool re-decodes the kept bitstreams through the unified **`zencodec`**
//! API and checks whether the result matches the stored PNG byte-for-byte
//! (RGB8). If it does, the PNG cache is fully regenerable from the ~38 GiB of
//! bitstreams and can be deleted. Where it does NOT match (notably JXL — the
//! generator decoded with `jxl-oxide`, while zencodec's `zenjxl` path uses
//! `zenjxl-decoder`), the per-codec mismatch is reported so re-extraction can
//! be planned through one consistent decoder rather than relying on stale PNGs.
//!
//! The reference PNG is read with the `image` crate — exactly how the
//! canonical feature extractor (`extract_features_372col.rs`) reads it — so an
//! exact match here means re-extraction from bitstreams via zencodec would
//! produce byte-identical features to the canonical parquets.
//!
//! ## Build
//!
//! JPEG only (lightest — `zenjpeg` is already a workspace dep):
//! ```text
//! cargo build --release -p zensim-bench --example verify_bitstream_decode \
//!     --features verify-decode
//! ```
//! Add codecs (heavier — pulls rav1d / jxl / webp):
//! ```text
//! --features verify-decode,verify-avif,verify-jxl,verify-webp   # or verify-all
//! ```
//!
//! ## Run
//! ```text
//! ./target/release/examples/verify_bitstream_decode \
//!     --images-root /mnt/v/input/zensim/images --tiles 25 [--max-per-codec 0] [--codec <substr>]
//! ```

#[cfg(not(feature = "verify-decode"))]
fn main() {
    eprintln!(
        "verify_bitstream_decode requires `--features verify-decode` \
         (optionally + verify-avif,verify-jxl,verify-webp, or verify-all)."
    );
    std::process::exit(2);
}

#[cfg(feature = "verify-decode")]
fn main() {
    real::run();
}

#[cfg(feature = "verify-decode")]
mod real {
    use std::borrow::Cow;
    use std::collections::BTreeMap;
    use std::path::{Path, PathBuf};

    use zencodec::decode::{Decode, DecodeJob, DecoderConfig};
    use zenpixels::{PixelBuffer, PixelDescriptor};

    /// Bitstream extensions we attempt to verify (the `.png` sibling is the
    /// reference). Anything else in a codec dir is ignored.
    const BITSTREAM_EXTS: &[&str] = &["jpg", "jpeg", "avif", "jxl", "webp"];

    struct Args {
        images_root: PathBuf,
        tiles: usize,
        max_per_codec: usize,
        codec_filter: Option<String>,
    }

    fn parse_args() -> Args {
        let mut a = Args {
            images_root: PathBuf::from("/mnt/v/input/zensim/images"),
            tiles: 25,
            max_per_codec: 0,
            codec_filter: None,
        };
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "--images-root" => a.images_root = PathBuf::from(it.next().expect("--images-root <path>")),
                "--tiles" => a.tiles = it.next().expect("--tiles <n>").parse().expect("int"),
                "--max-per-codec" => {
                    a.max_per_codec = it.next().expect("--max-per-codec <n>").parse().expect("int")
                }
                "--codec" => a.codec_filter = Some(it.next().expect("--codec <substr>")),
                "-h" | "--help" => {
                    eprintln!(
                        "verify_bitstream_decode --images-root <dir> [--tiles N] \
                         [--max-per-codec N] [--codec <substr>]"
                    );
                    std::process::exit(0);
                }
                other => {
                    eprintln!("unknown argument: {other}");
                    std::process::exit(2);
                }
            }
        }
        a
    }

    /// Per-codec rollup of the comparison.
    #[derive(Default)]
    struct Stat {
        n: u64,
        exact: u64,
        diff: u64,
        dim_mismatch: u64,
        decode_err: u64,
        not_built: u64,
        worst_max_abs: u8,
        worst_mean_abs: f64,
        worst_path: String,
        first_err: String,
    }

    enum Outcome {
        /// RGB8 pixels decoded from the bitstream.
        Rgb(Vec<u8>),
        /// Decoder for this extension was not compiled in.
        NotBuilt,
        /// Decode failed.
        Err(String),
    }

    pub fn run() {
        let args = parse_args();
        eprintln!("verify_bitstream_decode (zencodec decode path)");
        eprintln!("  images-root : {}", args.images_root.display());
        eprintln!("  tiles       : {}", args.tiles);
        eprintln!("  max/codec   : {}", args.max_per_codec);
        eprintln!(
            "  codecs built: jpeg{}{}{}",
            if cfg!(feature = "verify-avif") { " avif" } else { "" },
            if cfg!(feature = "verify-jxl") { " jxl" } else { "" },
            if cfg!(feature = "verify-webp") { " webp" } else { "" },
        );

        let mut tiles: Vec<PathBuf> = match std::fs::read_dir(&args.images_root) {
            Ok(rd) => rd
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.is_dir())
                .collect(),
            Err(e) => {
                eprintln!("cannot read images-root: {e}");
                std::process::exit(1);
            }
        };
        tiles.sort();
        if args.tiles > 0 && tiles.len() > args.tiles {
            tiles.truncate(args.tiles);
        }
        eprintln!("  scanning {} tile dirs\n", tiles.len());

        let mut stats: BTreeMap<String, Stat> = BTreeMap::new();
        let mut per_codec_count: BTreeMap<String, usize> = BTreeMap::new();

        for tile in &tiles {
            let codec_dirs = match std::fs::read_dir(tile) {
                Ok(rd) => rd.filter_map(|e| e.ok()).map(|e| e.path()).filter(|p| p.is_dir()),
                Err(_) => continue,
            };
            for codec_dir in codec_dirs {
                let codec = codec_dir
                    .file_name()
                    .map(|s| s.to_string_lossy().into_owned())
                    .unwrap_or_default();
                if let Some(f) = &args.codec_filter {
                    if !codec.contains(f.as_str()) {
                        continue;
                    }
                }
                let entries = match std::fs::read_dir(&codec_dir) {
                    Ok(rd) => rd,
                    Err(_) => continue,
                };
                for entry in entries.filter_map(|e| e.ok()) {
                    let p = entry.path();
                    let ext = p
                        .extension()
                        .and_then(|e| e.to_str())
                        .map(|s| s.to_ascii_lowercase())
                        .unwrap_or_default();
                    if !BITSTREAM_EXTS.contains(&ext.as_str()) {
                        continue;
                    }
                    if args.max_per_codec > 0 {
                        let c = per_codec_count.entry(codec.clone()).or_default();
                        if *c >= args.max_per_codec {
                            continue;
                        }
                        *c += 1;
                    }
                    let png = p.with_extension("png");
                    if !png.exists() {
                        continue; // orphan bitstream (unscored) — no reference to check
                    }
                    let stat = stats.entry(codec.clone()).or_default();
                    stat.n += 1;
                    check_pair(&p, &ext, &png, stat);
                }
            }
        }

        report(&stats);
    }

    fn check_pair(bitstream: &Path, ext: &str, png: &Path, stat: &mut Stat) {
        // Reference: read the stored PNG exactly how the feature extractor does.
        let (rw, rh, ref_rgb) = match image::open(png) {
            Ok(img) => {
                let rgb = img.to_rgb8();
                let (w, h) = rgb.dimensions();
                (w, h, rgb.into_raw())
            }
            Err(e) => {
                stat.decode_err += 1;
                if stat.first_err.is_empty() {
                    stat.first_err = format!("ref png {}: {e}", png.display());
                }
                return;
            }
        };

        let bytes = match std::fs::read(bitstream) {
            Ok(b) => b,
            Err(e) => {
                stat.decode_err += 1;
                if stat.first_err.is_empty() {
                    stat.first_err = format!("read {}: {e}", bitstream.display());
                }
                return;
            }
        };

        let dec = match decode_bitstream(&bytes, ext) {
            Outcome::Rgb(v) => v,
            Outcome::NotBuilt => {
                stat.not_built += 1;
                return;
            }
            Outcome::Err(e) => {
                stat.decode_err += 1;
                if stat.first_err.is_empty() {
                    stat.first_err = format!("{}: {e}", bitstream.display());
                }
                return;
            }
        };

        if dec.len() != ref_rgb.len() {
            stat.dim_mismatch += 1;
            if stat.first_err.is_empty() {
                stat.first_err = format!(
                    "{}: decoded {} bytes vs png {}x{}={} bytes",
                    bitstream.display(),
                    dec.len(),
                    rw,
                    rh,
                    ref_rgb.len()
                );
            }
            return;
        }

        let mut max_abs = 0u8;
        let mut n_diff = 0u64;
        let mut sum_abs = 0u64;
        for (a, b) in dec.iter().zip(ref_rgb.iter()) {
            let d = a.abs_diff(*b);
            if d != 0 {
                n_diff += 1;
                sum_abs += d as u64;
                if d > max_abs {
                    max_abs = d;
                }
            }
        }

        if n_diff == 0 {
            stat.exact += 1;
        } else {
            stat.diff += 1;
            let mean_abs = sum_abs as f64 / dec.len() as f64;
            if max_abs > stat.worst_max_abs
                || (max_abs == stat.worst_max_abs && mean_abs > stat.worst_mean_abs)
            {
                stat.worst_max_abs = max_abs;
                stat.worst_mean_abs = mean_abs;
                stat.worst_path = bitstream.display().to_string();
            }
        }
    }

    fn decode_bitstream(bytes: &[u8], ext: &str) -> Outcome {
        match ext {
            "jpg" | "jpeg" => wrap(decode_jpeg(bytes)),
            "avif" => decode_avif(bytes),
            "jxl" => decode_jxl(bytes),
            "webp" => decode_webp(bytes),
            _ => Outcome::NotBuilt,
        }
    }

    fn wrap(r: Result<Vec<u8>, String>) -> Outcome {
        match r {
            Ok(v) => Outcome::Rgb(v),
            Err(e) => Outcome::Err(e),
        }
    }

    /// Decode a bitstream through the zencodec `DecoderConfig → DecodeJob →
    /// Decode` path, asking for the codec's native format (`&[]`), then flatten
    /// the resulting `PixelBuffer` to packed RGB8.
    fn decode_jpeg(bytes: &[u8]) -> Result<Vec<u8>, String> {
        let cfg = zenjpeg::JpegDecoderConfig::new();
        let out = cfg
            .job()
            .decoder(Cow::Borrowed(bytes), &[])
            .map_err(|e| format!("jpeg job: {e}"))?
            .decode()
            .map_err(|e| format!("jpeg decode: {e}"))?;
        let pb = out.into_buffer();
        pixelbuffer_to_rgb8(&pb)
    }

    #[cfg(feature = "verify-avif")]
    fn decode_avif(bytes: &[u8]) -> Outcome {
        let cfg = zenavif::AvifDecoderConfig::new();
        let res = (|| -> Result<Vec<u8>, String> {
            let out = cfg
                .job()
                .decoder(Cow::Borrowed(bytes), &[])
                .map_err(|e| format!("avif job: {e}"))?
                .decode()
                .map_err(|e| format!("avif decode: {e}"))?;
            pixelbuffer_to_rgb8(&out.into_buffer())
        })();
        wrap(res)
    }
    #[cfg(not(feature = "verify-avif"))]
    fn decode_avif(_bytes: &[u8]) -> Outcome {
        Outcome::NotBuilt
    }

    #[cfg(feature = "verify-jxl")]
    fn decode_jxl(bytes: &[u8]) -> Outcome {
        let cfg = zenjxl::JxlDecoderConfig::new();
        let res = (|| -> Result<Vec<u8>, String> {
            let out = cfg
                .job()
                .decoder(Cow::Borrowed(bytes), &[])
                .map_err(|e| format!("jxl job: {e}"))?
                .decode()
                .map_err(|e| format!("jxl decode: {e}"))?;
            pixelbuffer_to_rgb8(&out.into_buffer())
        })();
        wrap(res)
    }
    #[cfg(not(feature = "verify-jxl"))]
    fn decode_jxl(_bytes: &[u8]) -> Outcome {
        Outcome::NotBuilt
    }

    #[cfg(feature = "verify-webp")]
    fn decode_webp(bytes: &[u8]) -> Outcome {
        let cfg = zenwebp::zencodec::WebpDecoderConfig::new();
        let res = (|| -> Result<Vec<u8>, String> {
            let out = cfg
                .job()
                .decoder(Cow::Borrowed(bytes), &[])
                .map_err(|e| format!("webp job: {e}"))?
                .decode()
                .map_err(|e| format!("webp decode: {e}"))?;
            pixelbuffer_to_rgb8(&out.into_buffer())
        })();
        wrap(res)
    }
    #[cfg(not(feature = "verify-webp"))]
    fn decode_webp(_bytes: &[u8]) -> Outcome {
        Outcome::NotBuilt
    }

    /// Flatten a `PixelBuffer` (RGB8 or RGBA8, possibly strided) to tightly
    /// packed RGB8. Mirrors `extract_features_372col_omni::pixelbuffer_to_rgb8`.
    fn pixelbuffer_to_rgb8(pb: &PixelBuffer) -> Result<Vec<u8>, String> {
        let desc = pb.descriptor();
        let w = pb.width() as usize;
        let h = pb.height() as usize;
        let slice = pb.as_slice();
        let stride = slice.stride();
        let data = slice.as_strided_bytes();

        if desc.layout_compatible(PixelDescriptor::RGB8) || desc.layout_compatible(PixelDescriptor::RGB8_SRGB)
        {
            let bpr = w * 3;
            let mut out = Vec::with_capacity(bpr * h);
            for row in 0..h {
                let start = row * stride;
                out.extend_from_slice(&data[start..start + bpr]);
            }
            Ok(out)
        } else if desc.layout_compatible(PixelDescriptor::RGBA8)
            || desc.layout_compatible(PixelDescriptor::RGBA8_SRGB)
        {
            let bpr_in = w * 4;
            let bpr_out = w * 3;
            let mut out = Vec::with_capacity(bpr_out * h);
            for row in 0..h {
                let start = row * stride;
                for px in data[start..start + bpr_in].chunks_exact(4) {
                    out.extend_from_slice(&px[..3]);
                }
            }
            Ok(out)
        } else {
            Err(format!("decoded descriptor {desc:?} is not RGB8/RGBA8"))
        }
    }

    fn report(stats: &BTreeMap<String, Stat>) {
        println!("\n===================== PER-CODEC RESULTS =====================");
        println!(
            "{:<24} {:>6} {:>6} {:>6} {:>5} {:>5} {:>5}  verdict",
            "codec", "n", "exact", "diff", "dim", "err", "nb"
        );
        let mut tot = Stat::default();
        for (codec, s) in stats {
            let checked = s.exact + s.diff;
            let pct = if checked > 0 {
                100.0 * s.exact as f64 / checked as f64
            } else {
                0.0
            };
            let verdict = if s.not_built == s.n {
                "decoder not built".to_string()
            } else if checked == 0 {
                "no comparable pairs".to_string()
            } else if s.exact == checked {
                "REPRODUCIBLE (byte-exact)".to_string()
            } else {
                format!(
                    "MISMATCH {:.1}% exact, worst max_abs={} mean_abs={:.4}",
                    pct, s.worst_max_abs, s.worst_mean_abs
                )
            };
            println!(
                "{:<24} {:>6} {:>6} {:>6} {:>5} {:>5} {:>5}  {}",
                codec, s.n, s.exact, s.diff, s.dim_mismatch, s.decode_err, s.not_built, verdict
            );
            if !s.worst_path.is_empty() {
                println!("    worst: {}", s.worst_path);
            }
            if !s.first_err.is_empty() {
                println!("    first err: {}", s.first_err);
            }
            tot.n += s.n;
            tot.exact += s.exact;
            tot.diff += s.diff;
            tot.dim_mismatch += s.dim_mismatch;
            tot.decode_err += s.decode_err;
            tot.not_built += s.not_built;
        }
        println!("-------------------------------------------------------------");
        let checked = tot.exact + tot.diff;
        let pct = if checked > 0 {
            100.0 * tot.exact as f64 / checked as f64
        } else {
            0.0
        };
        println!(
            "{:<24} {:>6} {:>6} {:>6} {:>5} {:>5} {:>5}  {:.1}% exact of {} compared",
            "TOTAL", tot.n, tot.exact, tot.diff, tot.dim_mismatch, tot.decode_err, tot.not_built, pct, checked
        );
        println!("=============================================================");
        println!(
            "\nByte-exact codecs are safe to delete-and-regenerate from bitstreams via zencodec.\n\
             MISMATCH codecs: the stored PNG was made by a different decoder than zencodec uses\n\
             (e.g. JXL: generator=jxl-oxide vs zencodec=zenjxl-decoder) — plan re-extraction\n\
             through one consistent decoder instead of trusting the stale PNG."
        );
    }
}
