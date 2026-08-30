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
//! ## `--decode-list` mode (added 2026-08-30, R1b keyed rebuild)
//!
//! The same zencodec decoders, driven off an explicit list instead of the
//! images-root walk, writing each decoded bitstream out as RGB8 PNG:
//!
//! ```text
//! verify_bitstream_decode --decode-list <tsv> --out-dir <dir> [--jobs N]
//! ```
//!
//! `<tsv>` is one input path per line (a `dist_path` header line is skipped);
//! the output is `<dir>/<input file stem>.png`. This exists because the zensim
//! feature extractors (`v2_ab_extract`, `extract_features_372col`) read PNG /
//! JPEG only, so re-extracting features for corpus rows whose distorted side
//! is an `.avif` / `.jxl` / `.webp` bitstream needs a decode step through the
//! SAME decoders the fleet uses. It reuses this file's per-codec decode
//! functions verbatim — no second decode path.
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

    use enough::Unstoppable;
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
        /// RGB8 pixels decoded from the bitstream, with its dimensions.
        Rgb(Vec<u8>, u32, u32),
        /// Decoder for this extension was not compiled in.
        NotBuilt,
        /// Decode failed.
        Err(String),
    }

    pub fn run() {
        let argv: Vec<String> = std::env::args().collect();
        if let Some(i) = argv.iter().position(|a| a == "--decode-list") {
            let list = PathBuf::from(argv.get(i + 1).expect("--decode-list <tsv>"));
            let out_dir = argv
                .iter()
                .position(|a| a == "--out-dir")
                .and_then(|j| argv.get(j + 1))
                .map(PathBuf::from)
                .expect("--out-dir <dir> required with --decode-list");
            let jobs = argv
                .iter()
                .position(|a| a == "--jobs")
                .and_then(|j| argv.get(j + 1))
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(8);
            std::process::exit(run_decode_list(&list, &out_dir, jobs));
        }
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

    /// Decode every path in `list` and write `<out_dir>/<stem>.png`. Returns a
    /// process exit code: 0 only if EVERY input decoded and was written.
    fn run_decode_list(list: &Path, out_dir: &Path, jobs: usize) -> i32 {
        use std::sync::Mutex;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let text = match std::fs::read_to_string(list) {
            Ok(t) => t,
            Err(e) => {
                eprintln!("ABORT: read {}: {e}", list.display());
                return 2;
            }
        };
        let inputs: Vec<PathBuf> = text
            .lines()
            .map(|l| l.split('\t').next().unwrap_or("").trim())
            .filter(|l| !l.is_empty() && *l != "dist_path" && *l != "path")
            .map(PathBuf::from)
            .collect();
        if let Err(e) = std::fs::create_dir_all(out_dir) {
            eprintln!("ABORT: mkdir {}: {e}", out_dir.display());
            return 2;
        }
        eprintln!(
            "decode-list: {} inputs -> {} ({} threads)",
            inputs.len(),
            out_dir.display(),
            jobs
        );
        let done = AtomicUsize::new(0);
        let skipped = AtomicUsize::new(0);
        let errs: Mutex<Vec<String>> = Mutex::new(Vec::new());
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(jobs)
            .build()
            .expect("rayon pool");
        pool.install(|| {
            use rayon::prelude::*;
            inputs.par_iter().for_each(|src| {
                let stem = src.file_stem().and_then(|s| s.to_str()).unwrap_or("");
                let dst = out_dir.join(format!("{stem}.png"));
                if dst.is_file() && std::fs::metadata(&dst).map(|m| m.len()).unwrap_or(0) > 0 {
                    skipped.fetch_add(1, Ordering::Relaxed);
                    return;
                }
                let ext = src
                    .extension()
                    .and_then(|e| e.to_str())
                    .unwrap_or("")
                    .to_ascii_lowercase();
                let bytes = match std::fs::read(src) {
                    Ok(b) => b,
                    Err(e) => {
                        errs.lock().unwrap().push(format!("{}: read {e}", src.display()));
                        return;
                    }
                };
                let (rgb, w, h) = match decode_bitstream(&bytes, &ext) {
                    Outcome::Rgb(v, w, h) => (v, w, h),
                    Outcome::NotBuilt => {
                        errs.lock().unwrap().push(format!(
                            "{}: decoder for .{ext} NOT COMPILED IN — rebuild with \
                             --features verify-all",
                            src.display()
                        ));
                        return;
                    }
                    Outcome::Err(e) => {
                        errs.lock().unwrap().push(format!("{}: {e}", src.display()));
                        return;
                    }
                };
                if rgb.len() != (w as usize) * (h as usize) * 3 {
                    errs.lock().unwrap().push(format!(
                        "{}: {} bytes for {w}x{h}",
                        src.display(),
                        rgb.len()
                    ));
                    return;
                }
                match write_png_rgb8(&dst, &rgb, w, h) {
                    Ok(()) => {
                        let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                        if n % 2000 == 0 {
                            eprintln!("  decoded {n}/{}", inputs.len());
                        }
                    }
                    Err(e) => errs.lock().unwrap().push(format!("{}: {e}", dst.display())),
                }
            });
        });
        let errs = errs.into_inner().unwrap();
        eprintln!(
            "decode-list: wrote {}, skipped-existing {}, errors {}",
            done.load(Ordering::Relaxed),
            skipped.load(Ordering::Relaxed),
            errs.len()
        );
        for e in errs.iter().take(10) {
            eprintln!("  ERR {e}");
        }
        if errs.is_empty() { 0 } else { 1 }
    }

    /// Write packed RGB8 as a PNG through zenpng (the zen encoder, not `image`).
    fn write_png_rgb8(dst: &Path, px: &[u8], w: u32, h: u32) -> Result<(), String> {
        use rgb::FromSlice;
        let img = imgref::ImgRef::new(px.as_rgb(), w as usize, h as usize);
        let cfg = zenpng::EncodeConfig::default();
        let bytes = zenpng::encode_rgb8(img, None, &cfg, &Unstoppable, &Unstoppable)
            .map_err(|e| format!("png encode: {e:?}"))?;
        std::fs::write(dst, bytes).map_err(|e| format!("write: {e}"))
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
            Outcome::Rgb(v, _, _) => v,
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

    fn wrap(r: Result<(Vec<u8>, u32, u32), String>) -> Outcome {
        match r {
            Ok((v, w, h)) => Outcome::Rgb(v, w, h),
            Err(e) => Outcome::Err(e),
        }
    }

    /// Decode a bitstream through the zencodec `DecoderConfig → DecodeJob →
    /// Decode` path, asking for the codec's native format (`&[]`), then flatten
    /// the resulting `PixelBuffer` to packed RGB8.
    fn decode_jpeg(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32), String> {
        let cfg = zenjpeg::JpegDecoderConfig::new();
        let out = cfg
            .job()
            .decoder(Cow::Borrowed(bytes), &[])
            .map_err(|e| format!("jpeg job: {e}"))?
            .decode()
            .map_err(|e| format!("jpeg decode: {e}"))?;
        let pb = out.into_buffer();
        let (w, h) = (pb.width(), pb.height());
        pixelbuffer_to_rgb8(&pb).map(|v| (v, w, h))
    }

    #[cfg(feature = "verify-avif")]
    fn decode_avif(bytes: &[u8]) -> Outcome {
        let cfg = zenavif::AvifDecoderConfig::new();
        let res = (|| -> Result<(Vec<u8>, u32, u32), String> {
            let out = cfg
                .job()
                .decoder(Cow::Borrowed(bytes), &[])
                .map_err(|e| format!("avif job: {e}"))?
                .decode()
                .map_err(|e| format!("avif decode: {e}"))?;
            let pb = out.into_buffer();
            let (w, h) = (pb.width(), pb.height());
            pixelbuffer_to_rgb8(&pb).map(|v| (v, w, h))
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
        let res = (|| -> Result<(Vec<u8>, u32, u32), String> {
            let out = cfg
                .job()
                .decoder(Cow::Borrowed(bytes), &[])
                .map_err(|e| format!("jxl job: {e}"))?
                .decode()
                .map_err(|e| format!("jxl decode: {e}"))?;
            let pb = out.into_buffer();
            let (w, h) = (pb.width(), pb.height());
            pixelbuffer_to_rgb8(&pb).map(|v| (v, w, h))
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
        let res = (|| -> Result<(Vec<u8>, u32, u32), String> {
            let out = cfg
                .job()
                .decoder(Cow::Borrowed(bytes), &[])
                .map_err(|e| format!("webp job: {e}"))?
                .decode()
                .map_err(|e| format!("webp decode: {e}"))?;
            let pb = out.into_buffer();
            let (w, h) = (pb.width(), pb.height());
            pixelbuffer_to_rgb8(&pb).map(|v| (v, w, h))
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
            // Everything else (10/12-bit AVIF -> Rgb16, gray, f32, premultiplied,
            // channel-reordered) goes through the CANONICAL pixel-format owner,
            // `zenpixels_convert::RowConverter` -> RGB8_SRGB — the exact path
            // zenmetrics' `decode.rs` uses, so a bitstream decoded here matches
            // what the fleet extractor saw. Measured need (R1b, 2026-08-30):
            // 4,417 of 20,655 canonical-picker AVIF members are `bd10` cells
            // that decode to Rgb16 and hit this branch.
            use zenpixels_convert::converter::RowConverter;
            let dst_stride = w * 3;
            let mut out = vec![0u8; dst_stride * h];
            let mut conv = RowConverter::new(desc, PixelDescriptor::RGB8_SRGB)
                .map_err(|e| format!("cannot plan {desc:?} -> RGB8_SRGB: {e}"))?;
            conv.convert_rows(
                data,
                stride,
                &mut out,
                dst_stride,
                w as u32,
                h as u32,
            )
            .map_err(|e| format!("row conversion {desc:?} -> RGB8_SRGB: {e}"))?;
            Ok(out)
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
