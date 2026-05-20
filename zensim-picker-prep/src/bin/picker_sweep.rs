//! picker_sweep — per-codec encode + decode + zensim_Tuner score, in-memory.
//!
//! For each source image in `--sources`, encodes with the selected codec
//! across q ∈ [5, 95] step 5, decodes the encoded bytes back to RGB8, scores
//! the (source, dist) pair with `PreviewV0_5Tuner`, and writes one parquet
//! row per (source, codec, q) cell.
//!
//! Output schema:
//!   ref_basename : Utf8
//!   codec        : Utf8
//!   q            : Int64
//!   achieved_zensim_tuner : Float64 (nullable on encode/decode/score fail)
//!   encoded_bytes : Int64
//!   width, height : Int64
//!
//! Designed for the per-codec picker training data-prep pipeline; see
//! ~/.claude/projects/-home-lilith-work-zen/memory/project_per_codec_picker_design.md.
//!
//! Usage:
//!   picker_sweep --codec zenjpeg \
//!                --sources /mnt/v/output/zen/picker-data-prep/sources \
//!                --q-grid 5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95 \
//!                --output /mnt/v/zen/picker-training/2026-05-19/zenjpeg.parquet

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use arrow::array::{ArrayRef, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use clap::{Parser, ValueEnum};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;

use zensim::{PixelFormat, StridedBytes, Zensim, ZensimProfile};

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq)]
enum CodecKind {
    Zenjpeg,
    Zenwebp,
    Zenavif,
    Zenjxl,
    Zenpng,
}

impl CodecKind {
    fn name(self) -> &'static str {
        match self {
            Self::Zenjpeg => "zenjpeg",
            Self::Zenwebp => "zenwebp",
            Self::Zenavif => "zenavif",
            Self::Zenjxl => "zenjxl",
            Self::Zenpng => "zenpng",
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "picker_sweep")]
struct Args {
    /// Codec to drive.
    #[arg(long, value_enum)]
    codec: CodecKind,

    /// Directory of source images (PNG / JPEG).
    #[arg(long)]
    sources: PathBuf,

    /// Comma-separated list of integer qualities (0..=100).
    /// Default matches the picker design: q ∈ [5, 95] step 5.
    #[arg(
        long,
        default_value = "5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95"
    )]
    q_grid: String,

    /// Output parquet path.
    #[arg(long)]
    output: PathBuf,

    /// Number of rayon threads (default: rayon auto).
    #[arg(long)]
    threads: Option<usize>,

    /// Max sources to process (for smoke testing). 0 = all.
    #[arg(long, default_value_t = 0)]
    max_sources: usize,
}

#[derive(Clone, Debug)]
struct Source {
    basename: String,
    pixels: Vec<u8>,
    width: u32,
    height: u32,
}

fn parse_q_grid(s: &str) -> Result<Vec<u32>> {
    let mut out = Vec::new();
    for tok in s.split(',') {
        let t = tok.trim();
        if t.is_empty() {
            continue;
        }
        let q: u32 = t.parse().with_context(|| format!("parsing q '{t}'"))?;
        if q > 100 {
            return Err(anyhow!("q '{q}' out of range 0..=100"));
        }
        out.push(q);
    }
    if out.is_empty() {
        return Err(anyhow!("empty q-grid"));
    }
    Ok(out)
}

fn load_source(p: &Path) -> Result<(Vec<u8>, u32, u32)> {
    let img = image::open(p).with_context(|| format!("decoding {:?}", p))?;
    let rgb = img.to_rgb8();
    let (w, h) = rgb.dimensions();
    Ok((rgb.into_raw(), w, h))
}

// ── Encoders ────────────────────────────────────────────────────────────────

fn encode_jpeg(src: &Source, q: u32) -> Result<Vec<u8>> {
    use zencodec::encode::{EncodeJob as _, Encoder as _, EncoderConfig as _};
    use zenjpeg::JpegEncoderConfig;
    use zenpixels::{PixelDescriptor, PixelSlice};

    let cfg = JpegEncoderConfig::new().with_generic_quality(q as f32);
    let stride = src.width as usize * 3;
    let slice = PixelSlice::new(
        &src.pixels,
        src.width,
        src.height,
        stride,
        PixelDescriptor::RGB8_SRGB,
    )
    .map_err(|e| anyhow!("zenjpeg slice: {e}"))?;

    let encoder = cfg
        .job()
        .encoder()
        .map_err(|e| anyhow!("zenjpeg ctor: {e}"))?;
    let output = encoder
        .encode(slice)
        .map_err(|e| anyhow!("zenjpeg encode: {e}"))?;
    Ok(output.into_vec())
}

fn encode_webp(src: &Source, q: u32) -> Result<Vec<u8>> {
    use zenwebp::{EncodeRequest, EncoderConfig, LossyConfig, PixelLayout};
    let cfg = EncoderConfig::Lossy(LossyConfig::new().with_quality(q as f32));
    let bytes = EncodeRequest::new(&cfg, &src.pixels, PixelLayout::Rgb8, src.width, src.height)
        .encode()
        .map_err(|e| anyhow!("zenwebp encode: {e}"))?;
    Ok(bytes)
}

fn encode_avif(src: &Source, q: u32) -> Result<Vec<u8>> {
    use imgref::ImgRef;
    use zenavif::EncoderConfig;

    let pixels: &[rgb::Rgb<u8>] = bytemuck_cast_rgb(&src.pixels);
    let img = ImgRef::new(pixels, src.width as usize, src.height as usize);
    let cfg = EncoderConfig::new().quality(q as f32);
    let encoded = zenavif::encode_rgb8(
        img,
        &cfg,
        almost_enough::StopToken::new(enough::Unstoppable),
    )
    .map_err(|e| anyhow!("zenavif encode: {e}"))?;
    Ok(encoded.avif_file)
}

fn encode_jxl(src: &Source, q: u32) -> Result<Vec<u8>> {
    use zencodec::encode::{EncodeJob as _, Encoder as _, EncoderConfig as _};
    use zenjxl::JxlEncoderConfig;
    use zenpixels::{PixelDescriptor, PixelSlice};

    // Per picker-data-prep spec: map q ∈ [5, 95] → distance ∈ (15.0 → 0.0]
    // via distance = (95 − q) / 95 * 15.0. At q=95 distance ≈ 0 (near-lossless),
    // q=5 distance ≈ 14.2 (heavy compression). jxl-encoder 0.3.1 panics
    // (divide-by-zero in vardct/ac_context.rs:231) when distance is
    // exactly 0.0, so floor at 0.01 to keep the lossy VarDCT path alive.
    let raw_distance = (95.0_f32 - q as f32) / 95.0 * 15.0;
    let distance = raw_distance.max(0.01);
    let cfg = JxlEncoderConfig::new().with_distance(distance);
    let stride = src.width as usize * 3;
    let slice = PixelSlice::new(
        &src.pixels,
        src.width,
        src.height,
        stride,
        PixelDescriptor::RGB8_SRGB,
    )
    .map_err(|e| anyhow!("zenjxl slice: {e}"))?;
    let encoder = cfg
        .job()
        .encoder()
        .map_err(|e| anyhow!("zenjxl ctor: {e}"))?;
    let output = encoder
        .encode(slice)
        .map_err(|e| anyhow!("zenjxl encode: {e}"))?;
    Ok(output.into_vec())
}

fn encode_png(src: &Source, _q: u32) -> Result<Vec<u8>> {
    use imgref::ImgRef;
    use zenpng::EncodeConfig;
    let cfg = EncodeConfig::default();
    let pixels: &[rgb::Rgb<u8>] = bytemuck_cast_rgb(&src.pixels);
    let img = ImgRef::new(pixels, src.width as usize, src.height as usize);
    let bytes = zenpng::encode_rgb8(img, None, &cfg, &enough::Unstoppable, &enough::Unstoppable)
        .map_err(|e| anyhow!("zenpng encode: {e}"))?;
    Ok(bytes)
}

fn bytemuck_cast_rgb(buf: &[u8]) -> &[rgb::Rgb<u8>] {
    // rgb::Rgb<u8> is repr(C) of three u8s, so a flat RGB8 slice can be
    // viewed as an array-of-RGB struct without copying.
    debug_assert!(buf.len() % 3 == 0);
    let n = buf.len() / 3;
    let ptr = buf.as_ptr() as *const rgb::Rgb<u8>;
    // SAFETY: rgb::Rgb<u8> is repr(C) (u8,u8,u8); aliasing model unchanged.
    // Required because rgb 0.8 doesn't expose a safe cast helper and
    // bytemuck doesn't cover external types by default.
    #[allow(unsafe_code)]
    unsafe {
        std::slice::from_raw_parts(ptr, n)
    }
}

fn encode(codec: CodecKind, src: &Source, q: u32) -> Result<Vec<u8>> {
    match codec {
        CodecKind::Zenjpeg => encode_jpeg(src, q),
        CodecKind::Zenwebp => encode_webp(src, q),
        CodecKind::Zenavif => encode_avif(src, q),
        CodecKind::Zenjxl => encode_jxl(src, q),
        CodecKind::Zenpng => encode_png(src, q),
    }
}

// ── Decoders ────────────────────────────────────────────────────────────────

fn decode_png_bytes(data: &[u8]) -> Result<(Vec<u8>, u32, u32)> {
    use zenpng::{PngDecodeConfig, decode};
    let cancel: Box<dyn enough::Stop + Send + Sync> = Box::new(enough::Unstoppable);
    let output = decode(data, &PngDecodeConfig::default(), &*cancel)
        .map_err(|e| anyhow!("zenpng decode: {e}"))?;
    pixel_buffer_to_rgb8(&output.pixels)
}

fn decode_jpeg_bytes(data: &[u8]) -> Result<(Vec<u8>, u32, u32)> {
    use zenjpeg::JpegDecoderConfig;
    let output = JpegDecoderConfig::new()
        .decode(data)
        .map_err(|e| anyhow!("zenjpeg decode: {e}"))?;
    pixel_slice_to_rgb8(&output.pixels())
}

fn decode_webp_bytes(data: &[u8]) -> Result<(Vec<u8>, u32, u32)> {
    let (pixels, width, height) =
        zenwebp::decoder::decode_rgb(data).map_err(|e| anyhow!("zenwebp decode: {e}"))?;
    Ok((pixels, width, height))
}

fn decode_avif_bytes(data: &[u8]) -> Result<(Vec<u8>, u32, u32)> {
    let pixels = zenavif::decode(data).map_err(|e| anyhow!("zenavif decode: {e}"))?;
    pixel_buffer_to_rgb8(&pixels)
}

fn decode_jxl_bytes(data: &[u8]) -> Result<(Vec<u8>, u32, u32)> {
    use zenjxl::decode;
    let output = decode(data, None, &[]).map_err(|e| anyhow!("zenjxl decode: {e}"))?;
    pixel_buffer_to_rgb8(&output.pixels)
}

fn decode(codec: CodecKind, data: &[u8]) -> Result<(Vec<u8>, u32, u32)> {
    match codec {
        CodecKind::Zenjpeg => decode_jpeg_bytes(data),
        CodecKind::Zenwebp => decode_webp_bytes(data),
        CodecKind::Zenavif => decode_avif_bytes(data),
        CodecKind::Zenjxl => decode_jxl_bytes(data),
        CodecKind::Zenpng => decode_png_bytes(data),
    }
}

fn pixel_buffer_to_rgb8(buf: &zenpixels::PixelBuffer) -> Result<(Vec<u8>, u32, u32)> {
    pixel_slice_to_rgb8(&buf.as_slice())
}

fn pixel_slice_to_rgb8(pixels: &zenpixels::PixelSlice<'_>) -> Result<(Vec<u8>, u32, u32)> {
    use zenpixels::PixelDescriptor;
    use zenpixels_convert::converter::RowConverter;

    let width = pixels.width();
    let height = pixels.rows();
    let src_stride = pixels.stride();
    let src_desc = pixels.descriptor();
    let src_bytes = pixels.as_strided_bytes();

    let dst_desc = PixelDescriptor::RGB8_SRGB;
    let dst_stride = width as usize * 3;
    let mut dst = vec![0u8; dst_stride * height as usize];

    let mut conv = RowConverter::new(src_desc, dst_desc)
        .map_err(|e| anyhow!("decode plan {src_desc:?} → RGB8_SRGB: {e}"))?;
    conv.convert_rows(src_bytes, src_stride, &mut dst, dst_stride, width, height)
        .map_err(|e| anyhow!("decode row conversion: {e}"))?;
    Ok((dst, width, height))
}

// ── Pipeline ────────────────────────────────────────────────────────────────

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(t) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .map_err(|e| anyhow!("rayon init: {e}"))?;
    }

    // Silence the default panic printer — jxl-encoder 0.3.1 panics
    // on legitimate content + we catch them per-cell. Without this
    // the stderr storm makes the progress log unreadable.
    std::panic::set_hook(Box::new(|_info| {}));

    let q_grid = parse_q_grid(&args.q_grid)?;
    eprintln!("picker_sweep");
    eprintln!("  codec:   {}", args.codec.name());
    eprintln!("  sources: {:?}", args.sources);
    eprintln!("  q-grid:  {:?}", q_grid);
    eprintln!("  output:  {:?}", args.output);
    eprintln!("  profile: PreviewV0_5Tuner");

    let t_start = Instant::now();

    // ─── walk sources ─────────────────────────────────────────────
    let mut src_paths: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(&args.sources)? {
        let entry = entry?;
        let p = entry.path();
        // Follow symlinks (we use them) so is_file() works.
        if p.is_file() {
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
            if matches!(ext.to_ascii_lowercase().as_str(), "png" | "jpg" | "jpeg") {
                src_paths.push(p);
            }
        }
    }
    src_paths.sort();
    if args.max_sources > 0 && src_paths.len() > args.max_sources {
        src_paths.truncate(args.max_sources);
    }
    let n_sources = src_paths.len();
    eprintln!("  sources: {} files", n_sources);
    if n_sources == 0 {
        return Err(anyhow!("no sources found in {:?}", args.sources));
    }

    // ─── per-source loop (parallel) ──────────────────────────────
    let n_qs = q_grid.len();
    let total = n_sources * n_qs;
    let progress = AtomicUsize::new(0);
    let t_score = Instant::now();

    let zensim = Zensim::new(ZensimProfile::PreviewV0_5Tuner).with_parallel(false);

    type Row = (String, u32, Option<f64>, Option<usize>, Option<(u32, u32)>);
    // (ref_basename, q, score, encoded_bytes, (w,h))

    let rows: Vec<Row> = src_paths
        .par_iter()
        .flat_map(|src_path| -> Vec<Row> {
            let basename = src_path
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| src_path.display().to_string());

            // Load source once per outer-parallel slot.
            let (sp_pixels, sp_w, sp_h) = match load_source(src_path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("  WARN: source load fail {:?}: {e}", src_path);
                    let n_added = q_grid.len();
                    let n = progress.fetch_add(n_added, Ordering::Relaxed) + n_added;
                    log_progress(n, total, &t_score);
                    return q_grid
                        .iter()
                        .map(|&q| (basename.clone(), q, None, None, None))
                        .collect();
                }
            };
            let src = Source {
                basename: basename.clone(),
                pixels: sp_pixels,
                width: sp_w,
                height: sp_h,
            };

            // For each q: encode → decode → score.
            let mut out: Vec<Row> = Vec::with_capacity(n_qs);
            for &q in &q_grid {
                let cell = encode_and_score(&zensim, args.codec, &src, q);
                out.push((
                    basename.clone(),
                    q,
                    cell.score,
                    cell.encoded_bytes,
                    cell.dims,
                ));
                let n = progress.fetch_add(1, Ordering::Relaxed) + 1;
                if n % 500 == 0 || n == total {
                    log_progress(n, total, &t_score);
                }
            }
            out
        })
        .collect();

    eprintln!(
        "  scoring: {} cells across {} sources in {:.1}s",
        total,
        n_sources,
        t_score.elapsed().as_secs_f64()
    );

    // ─── parquet output ──────────────────────────────────────────
    if let Some(parent) = args.output.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent).ok();
    }
    let fields: Vec<Arc<Field>> = vec![
        Arc::new(Field::new("ref_basename", DataType::Utf8, false)),
        Arc::new(Field::new("codec", DataType::Utf8, false)),
        Arc::new(Field::new("q", DataType::Int64, false)),
        Arc::new(Field::new("achieved_zensim_tuner", DataType::Float64, true)),
        Arc::new(Field::new("encoded_bytes", DataType::Int64, true)),
        Arc::new(Field::new("width", DataType::Int64, true)),
        Arc::new(Field::new("height", DataType::Int64, true)),
    ];
    let schema = Arc::new(Schema::new(fields));
    let codec_name = args.codec.name();
    let basenames: Vec<String> = rows.iter().map(|r| r.0.clone()).collect();
    let codecs: Vec<String> = (0..rows.len()).map(|_| codec_name.to_string()).collect();
    let qs: Vec<i64> = rows.iter().map(|r| r.1 as i64).collect();
    let scores: Vec<Option<f64>> = rows.iter().map(|r| r.2).collect();
    let encbytes: Vec<Option<i64>> = rows.iter().map(|r| r.3.map(|b| b as i64)).collect();
    let widths: Vec<Option<i64>> = rows.iter().map(|r| r.4.map(|(w, _)| w as i64)).collect();
    let heights: Vec<Option<i64>> = rows.iter().map(|r| r.4.map(|(_, h)| h as i64)).collect();
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(basenames)),
        Arc::new(StringArray::from(codecs)),
        Arc::new(Int64Array::from(qs)),
        Arc::new(Float64Array::from(scores)),
        Arc::new(Int64Array::from(encbytes)),
        Arc::new(Int64Array::from(widths)),
        Arc::new(Int64Array::from(heights)),
    ];
    let batch = RecordBatch::try_new(schema.clone(), arrays)?;
    let out_file = std::fs::File::create(&args.output)
        .with_context(|| format!("creating {:?}", args.output))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let mut writer = ArrowWriter::try_new(out_file, schema.clone(), Some(props))?;
    writer.write(&batch)?;
    writer.close()?;

    // ─── sanity stats ────────────────────────────────────────────
    let n_score_ok = rows.iter().filter(|r| r.2.is_some()).count();
    let n_score_nan = rows.iter().filter(|r| r.2.is_none()).count();
    let scored: Vec<f64> = rows.iter().filter_map(|r| r.2).collect();
    let (mn, mx, mean) = if scored.is_empty() {
        (f64::NAN, f64::NAN, f64::NAN)
    } else {
        let mn = scored.iter().copied().fold(f64::INFINITY, f64::min);
        let mx = scored.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mean: f64 = scored.iter().sum::<f64>() / scored.len() as f64;
        (mn, mx, mean)
    };
    eprintln!(
        "  wrote {} rows → {:?}\n  achieved_zensim_tuner: min={:.2} mean={:.2} max={:.2} n_ok={} n_fail={}",
        rows.len(),
        args.output,
        mn,
        mean,
        mx,
        n_score_ok,
        n_score_nan
    );
    eprintln!("  total wall: {:.1}s", t_start.elapsed().as_secs_f64());
    Ok(())
}

struct CellResult {
    score: Option<f64>,
    encoded_bytes: Option<usize>,
    dims: Option<(u32, u32)>,
}

fn encode_and_score(zensim: &Zensim, codec: CodecKind, src: &Source, q: u32) -> CellResult {
    // Catch panics in the codec encoder (jxl-encoder 0.3.1 panics on
    // some content + low-distance combos with "attempt to divide by
    // zero" in vardct/ac_context.rs:231) so a single bad cell doesn't
    // sink the whole sweep.
    let encode_result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| encode(codec, src, q)));
    let bytes = match encode_result {
        Ok(Ok(b)) => b,
        Ok(Err(e)) => {
            eprintln!("  WARN: encode fail {} q={}: {e}", src.basename, q);
            return CellResult {
                score: None,
                encoded_bytes: None,
                dims: None,
            };
        }
        Err(_panic) => {
            eprintln!("  WARN: encode PANIC {} q={}", src.basename, q);
            return CellResult {
                score: None,
                encoded_bytes: None,
                dims: None,
            };
        }
    };
    let n_enc = bytes.len();
    let decode_result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| decode(codec, &bytes)));
    let (dpx, dw, dh) = match decode_result {
        Ok(Ok(t)) => t,
        Ok(Err(e)) => {
            eprintln!("  WARN: decode fail {} q={}: {e}", src.basename, q);
            return CellResult {
                score: None,
                encoded_bytes: Some(n_enc),
                dims: None,
            };
        }
        Err(_panic) => {
            eprintln!("  WARN: decode PANIC {} q={}", src.basename, q);
            return CellResult {
                score: None,
                encoded_bytes: Some(n_enc),
                dims: None,
            };
        }
    };
    if dw != src.width || dh != src.height || dpx.len() != src.pixels.len() {
        eprintln!(
            "  WARN: dim mismatch {} q={}: src={}x{} dec={}x{}",
            src.basename, q, src.width, src.height, dw, dh
        );
        return CellResult {
            score: None,
            encoded_bytes: Some(n_enc),
            dims: Some((dw, dh)),
        };
    }

    let w = src.width as usize;
    let h = src.height as usize;
    let stride = w * 3;
    let s_src = match StridedBytes::try_new(&src.pixels, w, h, stride, PixelFormat::Srgb8Rgb) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("  WARN: src slice {} q={}: {e:?}", src.basename, q);
            return CellResult {
                score: None,
                encoded_bytes: Some(n_enc),
                dims: Some((dw, dh)),
            };
        }
    };
    let s_dst = match StridedBytes::try_new(&dpx, w, h, stride, PixelFormat::Srgb8Rgb) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("  WARN: dist slice {} q={}: {e:?}", src.basename, q);
            return CellResult {
                score: None,
                encoded_bytes: Some(n_enc),
                dims: Some((dw, dh)),
            };
        }
    };
    let score = match zensim.compute(&s_src, &s_dst) {
        Ok(r) => Some(r.score()),
        Err(e) => {
            eprintln!("  WARN: zensim {} q={}: {e:?}", src.basename, q);
            None
        }
    };
    CellResult {
        score,
        encoded_bytes: Some(n_enc),
        dims: Some((dw, dh)),
    }
}

fn log_progress(n: usize, total: usize, t_start: &Instant) {
    let elapsed = t_start.elapsed().as_secs_f64();
    let rate = n as f64 / elapsed;
    let eta = if rate > 0.0 {
        (total - n) as f64 / rate
    } else {
        0.0
    };
    eprintln!(
        "  cells: {}/{} ({:.1}%) {:.1}s rate={:.2}/s eta={:.0}s",
        n,
        total,
        100.0 * n as f64 / total as f64,
        elapsed,
        rate,
        eta
    );
}
