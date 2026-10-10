//! HDRVID stimulus reconstruction for the external HDR video panels
//! (HDR-VDC, AVT-VQDB-UHD-1-HDR). Owner direction 2026-10-10: AV1 decodes
//! with rav1d-safe; HEVC/VVC/FFVHUFF decode with a pinned ffmpeg 8.1 build and
//! arrive here as raw yuv420p10le on stdin. Colour conversion and resampling are
//! imazen code; ffmpeg never converts or scales.
//!
//! Chain (the July registered display-frame convention, PROTOCOL.md of
//! `scripts/external_reads/asrun/{hdrvdc,avthdr}`):
//! 1. decode every frame; require exactly `--frames N`; keep frames
//!    `k_j = floor((j + 0.5) * N / 8)`, j = 0..7;
//! 2. BT.2020 non-constant-luminance, limited-range 10-bit YCbCr 4:2:0 ->
//!    full-range R'G'B' PQ code values with zenavif's canonical decode recipe
//!    (exact {9,3,3,1}/16 chroma interpolation, one fixed-point rounding to
//!    10-bit codes). PQ is not decoded;
//! 3. Lanczos a=3 (zenresize `Filter::Lanczos`) on the code values to the
//!    display frame; a frame already at display size passes through;
//! 4. clamp to [0, 1] and store as 16-bit RGB PNG (`round(v * 65535)`).
//!
//! The receipt records the decoded-plane SHA-256 of every kept frame and of
//! the whole decoded stream, so an independent decoder can be compared bit for
//! bit, plus code-value and PQ luminance statistics for each kept frame.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use rav1d_safe::{
    ColorPrimaries, ColorRange, Decoder, MatrixCoefficients, PixelLayout, Planes, Settings,
    TransferCharacteristics,
};
use sha2::{Digest, Sha256};
use zenavif::yuv_convert::{YuvMatrix, YuvRange, yuv420_to_rgb16_strip};
use zenresize::{Filter, PixelDescriptor, ResizeConfig, Resizer};

const DEPENDENCY_PINS: &str = "rav1d-safe f3132ee6f9c37310291168b28751f2926ceedb8e; \
zenavif 85dd0d2b609ba0d49ebf70330b6fcf4f337bbd67 (yuv_convert); \
zenresize e3975fb9d6d6b7baa96038a0eb8e27febb37c012; \
zenpng 0.1.4 (crates.io); \
linear-srgb c56e7940f4a123fb653bc60593376cf2eaed3e70";

/// One decoded 10-bit 4:2:0 frame, tightly packed.
struct Yuv {
    w: usize,
    h: usize,
    y: Vec<u16>,
    u: Vec<u16>,
    v: Vec<u16>,
}

impl Yuv {
    fn hash_into(&self, hasher: &mut Sha256) {
        for plane in [&self.y, &self.u, &self.v] {
            for s in plane.iter() {
                hasher.update(s.to_le_bytes());
            }
        }
    }

    fn sha256(&self) -> String {
        let mut h = Sha256::new();
        self.hash_into(&mut h);
        hex(&h.finalize())
    }

    fn check_range(&self) -> Result<(), String> {
        if [&self.y, &self.u, &self.v]
            .iter()
            .any(|p| p.iter().any(|&s| s > 1023))
        {
            return Err("sample above 10-bit range".into());
        }
        Ok(())
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

/// Uniform frame indices of the July protocol: floor((j + 0.5) N / 8).
fn indices(n: usize) -> [usize; 8] {
    core::array::from_fn(|j| (2 * j + 1) * n / 16)
}

struct Args {
    kind: String,
    input: String,
    width: usize,
    height: usize,
    frames: usize,
    display: (usize, usize),
    out_dir: PathBuf,
    stem: String,
    threads: u32,
    receipt: PathBuf,
    /// Optional far-viewing frame: Lanczos-3 downscale of the stored (16-bit)
    /// display frame, as the July HDR-VDC chain's second filter leg.
    far: Option<((usize, usize), PathBuf)>,
}

fn parse_args() -> Result<Args, String> {
    let args: Vec<String> = std::env::args().collect();
    let get = |key: &str| -> Result<String, String> {
        let mut hits = args.iter().enumerate().filter(|(_, a)| *a == key);
        let (i, _) = hits.next().ok_or_else(|| format!("missing {key}"))?;
        if hits.next().is_some() {
            return Err(format!("duplicate {key}"));
        }
        args.get(i + 1)
            .filter(|v| !v.starts_with("--"))
            .cloned()
            .ok_or_else(|| format!("missing value for {key}"))
    };
    let opt = |key: &str| get(key).ok();
    let num = |s: String| s.parse::<usize>().map_err(|e| e.to_string());
    let kind = get("--kind")?;
    let (width, height) = match kind.as_str() {
        "av1-ivf" => (0, 0),
        "raw-yuv420p10le" => (num(get("--width")?)?, num(get("--height")?)?),
        other => return Err(format!("unknown --kind {other}")),
    };
    let dims = |s: String| -> Result<(usize, usize), String> {
        let (w, h) = s.split_once('x').ok_or("expected WxH")?;
        Ok((num(w.into())?, num(h.into())?))
    };
    let far = match (opt("--far-display"), opt("--far-out-dir")) {
        (Some(d), Some(dir)) => Some((dims(d)?, PathBuf::from(dir))),
        (None, None) => None,
        _ => return Err("--far-display and --far-out-dir go together".into()),
    };
    Ok(Args {
        kind,
        input: get("--input")?,
        width,
        height,
        frames: num(get("--frames")?)?,
        display: dims(get("--display")?)?,
        out_dir: get("--out-dir")?.into(),
        stem: get("--stem")?,
        threads: opt("--threads").map_or(Ok(1), |t| t.parse().map_err(|e| format!("{e}")))?,
        receipt: get("--receipt")?.into(),
        far,
    })
}

/// What a decoder reports about the stream, checked against the registered
/// format (10-bit 4:2:0, BT.2020 primaries, PQ, BT.2020 NCL, limited range).
#[derive(Default)]
struct StreamFacts {
    decoded: usize,
    stream_sha256: String,
    color: Option<serde_json::Value>,
}

/// Decode an AV1 IVF stream with rav1d-safe. `keep(k)` copies frame k.
fn decode_ivf(
    path: &str,
    threads: u32,
    keep: &dyn Fn(usize) -> bool,
    kept: &mut Vec<(usize, Yuv)>,
) -> Result<StreamFacts, String> {
    let data = std::fs::read(path).map_err(|e| format!("{path}: {e}"))?;
    if data.len() < 32 || &data[0..4] != b"DKIF" || &data[8..12] != b"AV01" {
        return Err("not an AV1 IVF file".into());
    }
    let header_len = u16::from_le_bytes([data[6], data[7]]) as usize;
    let mut settings = Settings::default();
    settings.threads = threads;
    settings.max_frame_delay = 1; // synchronous: decode() returns frames in order
    let mut decoder = Decoder::with_settings(settings).map_err(|e| format!("{e:?}"))?;
    let mut facts = StreamFacts::default();
    let mut stream = Sha256::new();
    let mut take = |frame: rav1d_safe::Frame, facts: &mut StreamFacts| -> Result<(), String> {
        let info = frame.color_info();
        let color = serde_json::json!({
            "bit_depth": frame.bit_depth(),
            "layout": format!("{:?}", frame.pixel_layout()),
            "primaries": format!("{:?}", info.primaries),
            "transfer": format!("{:?}", info.transfer_characteristics),
            "matrix": format!("{:?}", info.matrix_coefficients),
            "range": format!("{:?}", info.color_range),
        });
        if frame.bit_depth() != 10
            || frame.pixel_layout() != PixelLayout::I420
            || info.primaries != ColorPrimaries::BT2020
            || info.transfer_characteristics != TransferCharacteristics::SMPTE2084
            || info.matrix_coefficients != MatrixCoefficients::BT2020NCL
            || info.color_range != ColorRange::Limited
        {
            return Err(format!("unregistered AV1 stream format: {color}"));
        }
        match &facts.color {
            Some(prev) if prev != &color => return Err("stream format changed".into()),
            None => facts.color = Some(color),
            _ => {}
        }
        let Planes::Depth16(p) = frame.planes() else {
            return Err("10-bit frame without 16-bit planes".into());
        };
        let tight = |view: rav1d_safe::PlaneView16| -> Vec<u16> {
            let mut out = Vec::with_capacity(view.width() * view.height());
            for row in view.rows() {
                out.extend_from_slice(&row[..view.width()]);
            }
            out
        };
        let y = p.y();
        let (w, h) = (y.width(), y.height());
        let yuv = Yuv {
            w,
            h,
            y: tight(y),
            u: tight(p.u().ok_or("missing U plane")?),
            v: tight(p.v().ok_or("missing V plane")?),
        };
        if (yuv.u.len(), yuv.v.len()) != (w.div_ceil(2) * h.div_ceil(2), w.div_ceil(2) * h.div_ceil(2)) {
            return Err("unexpected chroma plane geometry".into());
        }
        yuv.check_range()?;
        yuv.hash_into(&mut stream);
        if keep(facts.decoded) {
            kept.push((facts.decoded, yuv));
        }
        facts.decoded += 1;
        Ok(())
    };
    let mut pos = header_len;
    while pos < data.len() {
        if pos + 12 > data.len() {
            return Err("truncated IVF frame header".into());
        }
        let size = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 12;
        let payload = data.get(pos..pos + size).ok_or("truncated IVF frame")?;
        pos += size;
        if let Some(f) = decoder.decode(payload).map_err(|e| format!("{e:?}"))? {
            take(f, &mut facts)?;
        }
        while let Some(f) = decoder.get_frame().map_err(|e| format!("{e:?}"))? {
            take(f, &mut facts)?;
        }
    }
    for f in decoder.flush().map_err(|e| format!("{e:?}"))? {
        take(f, &mut facts)?;
    }
    facts.stream_sha256 = hex(&stream.finalize());
    Ok(facts)
}

/// Read raw yuv420p10le frames (ffmpeg rawvideo output) from a file or stdin.
fn decode_raw(
    args: &Args,
    keep: &dyn Fn(usize) -> bool,
    kept: &mut Vec<(usize, Yuv)>,
) -> Result<StreamFacts, String> {
    let (w, h) = (args.width, args.height);
    let (cw, ch) = (w.div_ceil(2), h.div_ceil(2));
    let samples = w * h + 2 * cw * ch;
    let mut reader: Box<dyn Read> = if args.input == "-" {
        Box::new(std::io::stdin().lock())
    } else {
        Box::new(std::fs::File::open(&args.input).map_err(|e| e.to_string())?)
    };
    let mut buf = vec![0u8; samples * 2];
    let mut facts = StreamFacts::default();
    let mut stream = Sha256::new();
    loop {
        let mut filled = 0;
        while filled < buf.len() {
            match reader.read(&mut buf[filled..]) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e.to_string()),
            }
        }
        if filled == 0 {
            break;
        }
        if filled != buf.len() {
            return Err(format!("partial raw frame ({filled} of {} bytes)", buf.len()));
        }
        stream.update(&buf);
        if keep(facts.decoded) {
            let s: Vec<u16> = buf
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect();
            let yuv = Yuv {
                w,
                h,
                y: s[..w * h].to_vec(),
                u: s[w * h..w * h + cw * ch].to_vec(),
                v: s[w * h + cw * ch..].to_vec(),
            };
            yuv.check_range()?;
            kept.push((facts.decoded, yuv));
        }
        facts.decoded += 1;
    }
    facts.stream_sha256 = hex(&stream.finalize());
    Ok(facts)
}

/// YCbCr -> full-range PQ code values in [0, 1], interleaved RGB f32.
fn to_rgb_codes(yuv: &Yuv) -> Vec<f32> {
    let mut rgb = vec![rgb::Rgb::<u16>::new(0, 0, 0); yuv.w * yuv.h];
    let cw = yuv.w.div_ceil(2);
    yuv420_to_rgb16_strip(
        &yuv.y,
        yuv.w,
        &yuv.u,
        cw,
        &yuv.v,
        cw,
        yuv.w,
        yuv.h,
        0,
        yuv.h,
        YuvRange::Limited,
        YuvMatrix::Bt2020,
        10,
        &mut rgb,
    );
    rgb.iter()
        .flat_map(|p| [p.r, p.g, p.b])
        .map(|v| v as f32 / 1023.0)
        .collect()
}

fn display_frame(codes: Vec<f32>, w: usize, h: usize, display: (usize, usize)) -> Vec<f32> {
    if (w, h) == display {
        return codes;
    }
    let config = ResizeConfig::builder(w as u32, h as u32, display.0 as u32, display.1 as u32)
        .filter(Filter::Lanczos)
        .format(PixelDescriptor::RGBF32_LINEAR)
        .build();
    Resizer::new(&config).resize_f32(&codes)
}

/// Code-value and PQ luminance statistics of one display frame (pre-clamp
/// overshoot counted; luminance from BT.2020 weights on PQ-decoded channels at
/// the 10 000 cd/m^2 spec peak).
fn stats(display: &[f32]) -> serde_json::Value {
    let mut lo = [f32::INFINITY; 3];
    let mut hi = [f32::NEG_INFINITY; 3];
    let mut sum = [0f64; 3];
    let (mut under, mut over) = (0usize, 0usize);
    let mut nits: Vec<f32> = Vec::with_capacity(display.len() / 3);
    for px in display.chunks_exact(3) {
        for c in 0..3 {
            lo[c] = lo[c].min(px[c]);
            hi[c] = hi[c].max(px[c]);
            sum[c] += px[c] as f64;
            under += (px[c] < 0.0) as usize;
            over += (px[c] > 1.0) as usize;
        }
        let lin: [f32; 3] =
            core::array::from_fn(|c| linear_srgb::tf::pq_to_linear(px[c].clamp(0.0, 1.0)) * 10000.0);
        nits.push(0.2627 * lin[0] + 0.6780 * lin[1] + 0.0593 * lin[2]);
    }
    let n = nits.len();
    let mut pct = |q: f64| -> f32 {
        let k = ((q * (n - 1) as f64).round() as usize).min(n - 1);
        *nits.select_nth_unstable_by(k, |a, b| a.total_cmp(b)).1
    };
    let (p50, p95, p999, max) = (pct(0.5), pct(0.95), pct(0.999), pct(1.0));
    serde_json::json!({
        "code_min": lo, "code_max": hi,
        "code_mean": sum.map(|s| s / n as f64),
        "samples_below_0": under, "samples_above_1": over,
        "luminance_nits": {"p50": p50, "p95": p95, "p99_9": p999, "max": max},
    })
}

/// Storage quantization: clamp to [0, 1], `round(v * 65535)`.
fn quantize(frame: &[f32]) -> Vec<rgb::Rgb<u16>> {
    frame
        .chunks_exact(3)
        .map(|p| {
            let q = |v: f32| (v.clamp(0.0, 1.0) * 65535.0).round() as u16;
            rgb::Rgb::new(q(p[0]), q(p[1]), q(p[2]))
        })
        .collect()
}

fn write_png(path: &Path, pixels: &[rgb::Rgb<u16>], w: usize, h: usize) -> Result<String, String> {
    let config = zenpng::EncodeConfig::default().with_compression(zenpng::Compression::Fast);
    let png = zenpng::encode_rgb16(
        imgref::ImgRef::new(pixels, w, h),
        None,
        &config,
        &enough::Unstoppable,
        &enough::Unstoppable,
    )
    .map_err(|e| format!("{e:?}"))?;
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map_err(|e| format!("{path:?}: {e}"))?;
    file.write_all(&png).map_err(|e| e.to_string())?;
    Ok(hex(&Sha256::digest(&png)))
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    if args.frames < 8 {
        return Err("--frames must be at least 8".into());
    }
    if args.receipt.exists() {
        return Err("fresh receipt path required".into());
    }
    let wanted = indices(args.frames);
    let keep = |k: usize| wanted.contains(&k);
    let mut kept = Vec::new();
    let input_sha256 = if args.input == "-" {
        None
    } else {
        Some(hex(&Sha256::digest(
            std::fs::read(&args.input).map_err(|e| e.to_string())?,
        )))
    };
    let facts = match args.kind.as_str() {
        "av1-ivf" => decode_ivf(&args.input, args.threads, &keep, &mut kept)?,
        _ => decode_raw(&args, &keep, &mut kept)?,
    };
    if facts.decoded != args.frames {
        return Err(format!(
            "decoded {} frames, registered frame count {}",
            facts.decoded, args.frames
        ));
    }
    let mut frames = Vec::new();
    std::fs::create_dir_all(&args.out_dir).map_err(|e| e.to_string())?;
    if let Some((_, dir)) = &args.far {
        std::fs::create_dir_all(dir).map_err(|e| e.to_string())?;
    }
    for (j, (k, yuv)) in kept.iter().enumerate() {
        assert_eq!(*k, wanted[j]);
        let display = display_frame(to_rgb_codes(yuv), yuv.w, yuv.h, args.display);
        let png = args.out_dir.join(format!("{}_f{j}.png", args.stem));
        let stored = quantize(&display);
        let png_sha256 = write_png(&png, &stored, args.display.0, args.display.1)?;
        let mut record = serde_json::json!({
            "j": j, "frame_index": k, "coded": [yuv.w, yuv.h],
            "yuv_sha256": yuv.sha256(), "png": png, "png_sha256": png_sha256,
            "stats": stats(&display),
        });
        if let Some((far, dir)) = &args.far {
            let codes: Vec<f32> = stored
                .iter()
                .flat_map(|p| [p.r, p.g, p.b])
                .map(|v| v as f32 / 65535.0)
                .collect();
            let small = display_frame(codes, args.display.0, args.display.1, *far);
            let path = dir.join(format!("{}_f{j}.png", args.stem));
            let sha = write_png(&path, &quantize(&small), far.0, far.1)?;
            record["far"] = serde_json::json!({"png": path, "png_sha256": sha, "display": [far.0, far.1]});
        }
        frames.push(record);
    }
    let receipt = serde_json::json!({
        "schema": "hdrvid-decode-receipt-v1",
        "tool": "zensim tools/hdrvid_decode",
        "dependencies": DEPENDENCY_PINS,
        "kind": args.kind,
        "decoder": if args.kind == "av1-ivf" {
            "rav1d-safe (Strictness::Strict default, max_frame_delay 1)"
        } else {
            "external raw yuv420p10le (see driver receipt)"
        },
        "input": args.input, "input_sha256": input_sha256,
        "frames_decoded": facts.decoded, "indices": wanted,
        "decoded_stream_sha256": facts.stream_sha256,
        "stream_format": facts.color,
        "csc": "zenavif yuv_convert canonical recipe: BT.2020 NCL, limited range, 10-bit, {9,3,3,1}/16 chroma, fixed-point to 10-bit R'G'B' codes / 1023",
        "resample": "zenresize Filter::Lanczos (window 3) on PQ code values, RGBF32; pass-through at display size",
        "display": [args.display.0, args.display.1],
        "far_display": args.far.as_ref().map(|(d, _)| [d.0, d.1]),
        "storage": "16-bit RGB PNG, full-range PQ code values round(clamp(v,0,1)*65535), BT.2020 primaries (untagged)",
        "frames": frames,
    });
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&args.receipt)
        .map_err(|e| e.to_string())?;
    serde_json::to_writer_pretty(&mut file, &receipt).map_err(|e| e.to_string())?;
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("hdrvid_decode: {e}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn indices_match_the_july_protocol() {
        // AVT: N = 600 -> {37, 112, ..., 562} (avthdr PROTOCOL.md).
        assert_eq!(indices(600), [37, 112, 187, 262, 337, 412, 487, 562]);
        for n in 8..2000 {
            for (j, k) in indices(n).iter().enumerate() {
                assert_eq!(*k, ((j as f64 + 0.5) * n as f64 / 8.0).floor() as usize);
            }
        }
    }

    #[test]
    fn limited_range_black_and_white_map_to_code_extremes() {
        let frame = |y: u16| Yuv {
            w: 4,
            h: 4,
            y: vec![y; 16],
            u: vec![512; 4],
            v: vec![512; 4],
        };
        assert!(to_rgb_codes(&frame(64)).iter().all(|&v| v == 0.0));
        assert!(to_rgb_codes(&frame(940)).iter().all(|&v| v == 1.0));
        let mid = to_rgb_codes(&frame(502));
        assert!(mid.windows(2).all(|w| w[0] == w[1]), "neutral chroma stays gray");
    }

    #[test]
    fn display_sized_frames_pass_through_and_others_are_resampled() {
        let codes = vec![0.25f32; 8 * 6 * 3];
        assert_eq!(display_frame(codes.clone(), 8, 6, (8, 6)), codes);
        let up = display_frame(codes, 8, 6, (16, 12));
        assert_eq!(up.len(), 16 * 12 * 3);
        assert!(up.iter().all(|v| (v - 0.25).abs() < 1e-5), "flat stays flat");
    }
}
