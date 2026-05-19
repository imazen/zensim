//! cross_codec_butter_features — encode + decode + butter score + 372-feat
//! extraction for cross-codec equivalence training.
//!
//! Reuses the encode/decode pipeline from `picker_sweep` but instead of
//! scoring with PreviewV0_5Tuner, computes:
//!  - butteraugli max + pnorm_3 (against the original source)
//!  - 372-dim zensim feature vector (extended + IW pool features)
//!
//! Output: one parquet per codec with one row per (ref_basename, q) cell:
//!   ref_basename : Utf8
//!   codec        : Utf8
//!   q            : Int64
//!   butter_max   : Float64
//!   butter_pnorm3: Float64
//!   encoded_bytes: Int64
//!   width, height: Int64
//!   f0..f371     : Float32 (zensim 372-dim feature vector)
//!
//! Designed to feed the cross-codec equivalence pair builder downstream:
//! given the butter table we can find q values that hit the same
//! perceptual distance across codecs, then construct training pairs
//! where the metric must learn to score them identically.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use anyhow::{Context, Result, anyhow};
use arrow::array::{ArrayRef, Float32Array, Float64Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use clap::{Parser, ValueEnum};
use parquet::arrow::ArrowWriter;
use parquet::basic::Compression;
use parquet::file::properties::WriterProperties;
use rayon::prelude::*;

use butteraugli::{ButteraugliParams, Img as BImg, RGB8 as BRGB8, butteraugli};
use zensim::{ZensimConfig, compute_zensim_with_config};

const NUM_FEATURES: usize = 372;

#[derive(Copy, Clone, Debug, ValueEnum, PartialEq, Eq)]
enum CodecKind {
    Zenjpeg,
    Zenwebp,
    Zenavif,
    Zenjxl,
}

impl CodecKind {
    fn name(self) -> &'static str {
        match self {
            Self::Zenjpeg => "zenjpeg",
            Self::Zenwebp => "zenwebp",
            Self::Zenavif => "zenavif",
            Self::Zenjxl => "zenjxl",
        }
    }
}

#[derive(Parser, Debug)]
#[command(name = "cross_codec_butter_features")]
struct Args {
    /// Codec to drive.
    #[arg(long, value_enum)]
    codec: CodecKind,

    /// Directory of source images (PNG / JPEG).
    #[arg(long)]
    sources: PathBuf,

    /// Comma-separated list of integer qualities (0..=100).
    #[arg(long, default_value = "5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95")]
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

    /// Skip features extraction (butter only). Speeds up sweep by ~5x
    /// when only butter values are needed for the equivalence-pair
    /// builder.
    #[arg(long, default_value_t = false)]
    skip_features: bool,
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
    let encoder = cfg.job().encoder().map_err(|e| anyhow!("zenjpeg ctor: {e}"))?;
    let output = encoder.encode(slice).map_err(|e| anyhow!("zenjpeg encode: {e}"))?;
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
    let encoder = cfg.job().encoder().map_err(|e| anyhow!("zenjxl ctor: {e}"))?;
    let output = encoder.encode(slice).map_err(|e| anyhow!("zenjxl encode: {e}"))?;
    Ok(output.into_vec())
}

fn bytemuck_cast_rgb(buf: &[u8]) -> &[rgb::Rgb<u8>] {
    debug_assert!(buf.len() % 3 == 0);
    let n = buf.len() / 3;
    let ptr = buf.as_ptr() as *const rgb::Rgb<u8>;
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
    }
}

// ── Decoders ────────────────────────────────────────────────────────────────

fn decode_jpeg_bytes(data: &[u8]) -> Result<(Vec<u8>, u32, u32)> {
    use zenjpeg::JpegDecoderConfig;
    let output = JpegDecoderConfig::new()
        .decode(data)
        .map_err(|e| anyhow!("zenjpeg decode: {e}"))?;
    pixel_slice_to_rgb8(&output.pixels())
}

fn decode_webp_bytes(data: &[u8]) -> Result<(Vec<u8>, u32, u32)> {
    let (pixels, width, height) = zenwebp::decoder::decode_rgb(data)
        .map_err(|e| anyhow!("zenwebp decode: {e}"))?;
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

// ── Scoring ─────────────────────────────────────────────────────────────────

/// Compute butteraugli max + pnorm_3 on (ref, dist) RGB8 buffers.
fn score_butter(
    ref_rgb: &[u8],
    dist_rgb: &[u8],
    width: u32,
    height: u32,
) -> Result<(f64, f64)> {
    let w = width as usize;
    let h = height as usize;
    let n = w * h;
    if ref_rgb.len() != n * 3 || dist_rgb.len() != n * 3 {
        return Err(anyhow!(
            "buffer size mismatch: ref={} dist={} expected={}",
            ref_rgb.len(),
            dist_rgb.len(),
            n * 3
        ));
    }
    // Convert to imgref::Img<RGB8>
    let ref_pixels: Vec<BRGB8> = ref_rgb
        .chunks_exact(3)
        .map(|c| BRGB8 {
            r: c[0],
            g: c[1],
            b: c[2],
        })
        .collect();
    let dist_pixels: Vec<BRGB8> = dist_rgb
        .chunks_exact(3)
        .map(|c| BRGB8 {
            r: c[0],
            g: c[1],
            b: c[2],
        })
        .collect();
    let ref_img = BImg::new(ref_pixels, w, h);
    let dist_img = BImg::new(dist_pixels, w, h);
    let params = ButteraugliParams::default();
    let res = butteraugli(ref_img.as_ref(), dist_img.as_ref(), &params)
        .map_err(|e| anyhow!("butteraugli: {e}"))?;
    Ok((res.score, res.pnorm_3))
}

/// Compute 372-dim zensim feature vector on (ref, dist).
fn score_features(
    ref_rgb: &[u8],
    dist_rgb: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<f32>> {
    let w = width as usize;
    let h = height as usize;
    let n = w * h;
    if ref_rgb.len() != n * 3 || dist_rgb.len() != n * 3 {
        return Err(anyhow!(
            "buffer size mismatch: ref={} dist={} expected={}",
            ref_rgb.len(),
            dist_rgb.len(),
            n * 3
        ));
    }
    // Build [[u8; 3]] views from interleaved RGB8.
    let ref_view: Vec<[u8; 3]> = ref_rgb
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let dist_view: Vec<[u8; 3]> = dist_rgb
        .chunks_exact(3)
        .map(|c| [c[0], c[1], c[2]])
        .collect();
    let mut config = ZensimConfig::default();
    config.extended_features = true;
    config.compute_iw_features = true;
    let res = compute_zensim_with_config(&ref_view, &dist_view, w, h, config)
        .map_err(|e| anyhow!("zensim features: {:?}", e))?;
    // Vec<f64> → Vec<f32>, pad/truncate to 372.
    let mut feats: Vec<f32> = res.features().iter().map(|v| *v as f32).collect();
    feats.resize(NUM_FEATURES, 0.0);
    Ok(feats)
}

// ── Pipeline ────────────────────────────────────────────────────────────────

#[derive(Clone)]
struct CellRow {
    basename: String,
    q: u32,
    butter_max: Option<f64>,
    butter_pnorm3: Option<f64>,
    encoded_bytes: Option<usize>,
    dims: Option<(u32, u32)>,
    features: Option<Vec<f32>>,
}

fn encode_and_score(
    codec: CodecKind,
    src: &Source,
    q: u32,
    skip_features: bool,
) -> CellRow {
    let basename = src.basename.clone();
    let encode_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| encode(codec, src, q)));
    let bytes = match encode_result {
        Ok(Ok(b)) => b,
        Ok(Err(e)) => {
            eprintln!("  WARN: encode fail {} q={}: {e}", basename, q);
            return CellRow {
                basename,
                q,
                butter_max: None,
                butter_pnorm3: None,
                encoded_bytes: None,
                dims: None,
                features: None,
            };
        }
        Err(_) => {
            eprintln!("  WARN: encode PANIC {} q={}", basename, q);
            return CellRow {
                basename,
                q,
                butter_max: None,
                butter_pnorm3: None,
                encoded_bytes: None,
                dims: None,
                features: None,
            };
        }
    };
    let encoded_bytes = bytes.len();

    let dec = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| decode(codec, &bytes)));
    let (dist_rgb, dw, dh) = match dec {
        Ok(Ok(t)) => t,
        Ok(Err(e)) => {
            eprintln!("  WARN: decode fail {} q={}: {e}", basename, q);
            return CellRow {
                basename,
                q,
                butter_max: None,
                butter_pnorm3: None,
                encoded_bytes: Some(encoded_bytes),
                dims: None,
                features: None,
            };
        }
        Err(_) => {
            eprintln!("  WARN: decode PANIC {} q={}", basename, q);
            return CellRow {
                basename,
                q,
                butter_max: None,
                butter_pnorm3: None,
                encoded_bytes: Some(encoded_bytes),
                dims: None,
                features: None,
            };
        }
    };
    if dw != src.width || dh != src.height {
        eprintln!(
            "  WARN: dim mismatch {} q={}: src {}×{} dist {}×{}",
            basename, q, src.width, src.height, dw, dh
        );
        return CellRow {
            basename,
            q,
            butter_max: None,
            butter_pnorm3: None,
            encoded_bytes: Some(encoded_bytes),
            dims: Some((dw, dh)),
            features: None,
        };
    }

    let (bmax, bpn3) = match score_butter(&src.pixels, &dist_rgb, dw, dh) {
        Ok(p) => (Some(p.0), Some(p.1)),
        Err(e) => {
            eprintln!("  WARN: butter fail {} q={}: {e}", basename, q);
            (None, None)
        }
    };

    let features = if skip_features {
        None
    } else {
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            score_features(&src.pixels, &dist_rgb, dw, dh)
        })) {
            Ok(Ok(f)) => Some(f),
            Ok(Err(e)) => {
                eprintln!("  WARN: features fail {} q={}: {e}", basename, q);
                None
            }
            Err(_) => {
                eprintln!("  WARN: features PANIC {} q={}", basename, q);
                None
            }
        }
    };

    CellRow {
        basename,
        q,
        butter_max: bmax,
        butter_pnorm3: bpn3,
        encoded_bytes: Some(encoded_bytes),
        dims: Some((dw, dh)),
        features,
    }
}

fn log_progress(done: usize, total: usize, start: &Instant) {
    let elapsed = start.elapsed().as_secs_f64();
    let rate = done as f64 / elapsed.max(0.001);
    let remaining = (total - done) as f64 / rate.max(0.001);
    eprintln!(
        "    {}/{} ({:.1}%) elapsed {:.0}s @ {:.1}/s, eta {:.0}s",
        done,
        total,
        100.0 * done as f64 / total as f64,
        elapsed,
        rate,
        remaining
    );
}

fn main() -> Result<()> {
    let args = Args::parse();
    if let Some(t) = args.threads {
        rayon::ThreadPoolBuilder::new()
            .num_threads(t)
            .build_global()
            .map_err(|e| anyhow!("rayon init: {e}"))?;
    }
    std::panic::set_hook(Box::new(|_info| {}));

    let q_grid = parse_q_grid(&args.q_grid)?;
    eprintln!("cross_codec_butter_features");
    eprintln!("  codec:   {}", args.codec.name());
    eprintln!("  sources: {:?}", args.sources);
    eprintln!("  q-grid:  {:?}", q_grid);
    eprintln!("  output:  {:?}", args.output);
    eprintln!("  skip_features: {}", args.skip_features);

    let t_start = Instant::now();

    // walk sources
    let mut src_paths: Vec<PathBuf> = Vec::new();
    for entry in fs::read_dir(&args.sources)? {
        let entry = entry?;
        let p = entry.path();
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
        return Err(anyhow!("no sources in {:?}", args.sources));
    }

    let n_qs = q_grid.len();
    let total = n_sources * n_qs;
    let progress = AtomicUsize::new(0);
    let t_score = Instant::now();
    let codec = args.codec;
    let skip_features = args.skip_features;

    let rows: Vec<CellRow> = src_paths
        .par_iter()
        .flat_map(|src_path| -> Vec<CellRow> {
            let basename = src_path
                .file_name()
                .map(|s| s.to_string_lossy().into_owned())
                .unwrap_or_else(|| src_path.display().to_string());

            let (sp_pixels, sp_w, sp_h) = match load_source(src_path) {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("  WARN: source load fail {:?}: {e}", src_path);
                    let n_added = q_grid.len();
                    let n = progress.fetch_add(n_added, Ordering::Relaxed) + n_added;
                    log_progress(n, total, &t_score);
                    return q_grid
                        .iter()
                        .map(|&q| CellRow {
                            basename: basename.clone(),
                            q,
                            butter_max: None,
                            butter_pnorm3: None,
                            encoded_bytes: None,
                            dims: None,
                            features: None,
                        })
                        .collect();
                }
            };
            let src = Source {
                basename: basename.clone(),
                pixels: sp_pixels,
                width: sp_w,
                height: sp_h,
            };

            let mut out: Vec<CellRow> = Vec::with_capacity(n_qs);
            for &q in &q_grid {
                let cell = encode_and_score(codec, &src, q, skip_features);
                out.push(cell);
                let n = progress.fetch_add(1, Ordering::Relaxed) + 1;
                if n % 200 == 0 || n == total {
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

    let mut fields: Vec<Arc<Field>> = vec![
        Arc::new(Field::new("ref_basename", DataType::Utf8, false)),
        Arc::new(Field::new("codec", DataType::Utf8, false)),
        Arc::new(Field::new("q", DataType::Int64, false)),
        Arc::new(Field::new("butter_max", DataType::Float64, true)),
        Arc::new(Field::new("butter_pnorm3", DataType::Float64, true)),
        Arc::new(Field::new("encoded_bytes", DataType::Int64, true)),
        Arc::new(Field::new("width", DataType::Int64, true)),
        Arc::new(Field::new("height", DataType::Int64, true)),
    ];
    if !skip_features {
        for i in 0..NUM_FEATURES {
            fields.push(Arc::new(Field::new(
                &format!("f{i}"),
                DataType::Float32,
                true,
            )));
        }
    }
    let schema = Arc::new(Schema::new(fields));
    let codec_name = codec.name();

    let basenames: Vec<String> = rows.iter().map(|r| r.basename.clone()).collect();
    let codecs: Vec<String> = (0..rows.len()).map(|_| codec_name.to_string()).collect();
    let qs: Vec<i64> = rows.iter().map(|r| r.q as i64).collect();
    let bmax: Vec<Option<f64>> = rows.iter().map(|r| r.butter_max).collect();
    let bpn3: Vec<Option<f64>> = rows.iter().map(|r| r.butter_pnorm3).collect();
    let encbytes: Vec<Option<i64>> = rows
        .iter()
        .map(|r| r.encoded_bytes.map(|b| b as i64))
        .collect();
    let widths: Vec<Option<i64>> = rows.iter().map(|r| r.dims.map(|(w, _)| w as i64)).collect();
    let heights: Vec<Option<i64>> = rows.iter().map(|r| r.dims.map(|(_, h)| h as i64)).collect();

    let mut arrays: Vec<ArrayRef> = vec![
        Arc::new(StringArray::from(basenames)),
        Arc::new(StringArray::from(codecs)),
        Arc::new(Int64Array::from(qs)),
        Arc::new(Float64Array::from(bmax)),
        Arc::new(Float64Array::from(bpn3)),
        Arc::new(Int64Array::from(encbytes)),
        Arc::new(Int64Array::from(widths)),
        Arc::new(Int64Array::from(heights)),
    ];
    if !skip_features {
        for i in 0..NUM_FEATURES {
            let col: Vec<Option<f32>> = rows
                .iter()
                .map(|r| r.features.as_ref().map(|f| f[i]))
                .collect();
            arrays.push(Arc::new(Float32Array::from(col)));
        }
    }
    let batch = RecordBatch::try_new(schema.clone(), arrays)?;

    let out_file = std::fs::File::create(&args.output)
        .with_context(|| format!("creating {:?}", args.output))?;
    let props = WriterProperties::builder()
        .set_compression(Compression::ZSTD(Default::default()))
        .build();
    let mut writer = ArrowWriter::try_new(out_file, schema.clone(), Some(props))?;
    writer.write(&batch)?;
    writer.close()?;

    // stats
    let n_butter_ok = rows.iter().filter(|r| r.butter_max.is_some()).count();
    let n_feat_ok = rows.iter().filter(|r| r.features.is_some()).count();
    let butter_vals: Vec<f64> = rows.iter().filter_map(|r| r.butter_max).collect();
    let (bmin, bmax_v, bmean) = if butter_vals.is_empty() {
        (f64::NAN, f64::NAN, f64::NAN)
    } else {
        let mn = butter_vals.iter().copied().fold(f64::INFINITY, f64::min);
        let mx = butter_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mean: f64 = butter_vals.iter().sum::<f64>() / butter_vals.len() as f64;
        (mn, mx, mean)
    };
    eprintln!(
        "  wrote {} rows → {:?}\n  butter_max: min={:.3} mean={:.3} max={:.3} n_butter_ok={} n_feat_ok={}",
        rows.len(),
        args.output,
        bmin,
        bmean,
        bmax_v,
        n_butter_ok,
        n_feat_ok
    );
    eprintln!("  total wall: {:.1}s", t_start.elapsed().as_secs_f64());
    Ok(())
}
