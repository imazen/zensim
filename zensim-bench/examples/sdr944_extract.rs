//! sdr944_extract — SDR 944-regime (Folded720Append2) feature extraction for
//! arbitrary (ref, dist) PNG pairs. The local SDR twin of `hdr944_extract`
//! (added 2026-08-28, balance-campaign era-lessons revisit: the identity-check
//! instrument needed a local 944 extractor; the fleet's `zensim-foldapp2`
//! metric string is jobexec-only).
//!
//! Input TSV: `ref_path\tdist_path` (+ optional extra columns, ignored).
//! Output TSV: `ref_path\tdist_path\tfeat_0..feat_943`.
//!
//!   cargo run --release -p zensim-bench --example sdr944_extract \
//!       --features feature-regime-v2 -- --pairs-tsv in.tsv --out out.tsv
//!
//! `--hdr-pq16`: inputs are 16-bit PQ PNGs (the srgb_to_pq_png.py output);
//! features come from the HDR route (`compute_folded720_append2_features_hdr`,
//! Pq peak 10,000) — the PU21-single-model experiment path.
use std::io::Write;
use std::path::Path;
use zensim::feature_v2::{HdrEncoding, V2NewFeatureToggles, V2Scratch};
use zensim::{AlphaMode, PixelFormat, StridedBytes, Zensim, ZensimProfile};

/// 16-bit PQ image as an ImageSource declaring HDR (mirrors hdr944_extract's
/// local impl — the PQ code values ride the Srgb16Rgba byte layout with
/// `is_hdr() = true`, which routes the HDR entry's PQ interpretation).
struct Pq16Image {
    data: Vec<[u16; 4]>,
    w: usize,
    h: usize,
}
impl Pq16Image {
    fn from_rgb16(px: &[u16], w: usize, h: usize) -> Self {
        Self {
            data: px.chunks_exact(3).map(|c| [c[0], c[1], c[2], 65535]).collect(),
            w,
            h,
        }
    }
}
impl zensim::source::ImageSource for Pq16Image {
    fn width(&self) -> usize { self.w }
    fn height(&self) -> usize { self.h }
    fn pixel_format(&self) -> zensim::source::PixelFormat {
        zensim::source::PixelFormat::Srgb16Rgba
    }
    fn row_bytes(&self, y: usize) -> &[u8] {
        bytemuck::cast_slice(&self.data[y * self.w..(y + 1) * self.w])
    }
    fn alpha_mode(&self) -> zensim::source::AlphaMode {
        zensim::source::AlphaMode::Opaque
    }
    fn is_hdr(&self) -> bool { true }
}

fn decode_png_rgb16(p: &Path) -> Result<(Vec<u16>, usize, usize), String> {
    let f = std::fs::File::open(p).map_err(|e| format!("open {p:?}: {e}"))?;
    let dec = png::Decoder::new(std::io::BufReader::new(f));
    let mut reader = dec.read_info().map_err(|e| format!("png info {p:?}: {e}"))?;
    let mut buf = vec![0u8; reader.output_buffer_size().expect("png buffer size")];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| format!("png frame {p:?}: {e}"))?;
    buf.truncate(info.buffer_size());
    let (w, h) = (info.width as usize, info.height as usize);
    if info.bit_depth != png::BitDepth::Sixteen {
        return Err(format!("{p:?}: expected 16-bit PQ png, got {:?}", info.bit_depth));
    }
    let px: Vec<u16> = buf
        .chunks_exact(2)
        .map(|c| u16::from_be_bytes([c[0], c[1]]))
        .collect();
    let rgb = match info.color_type {
        png::ColorType::Rgb => px,
        png::ColorType::Rgba => {
            let mut o = Vec::with_capacity(w * h * 3);
            for q in px.chunks_exact(4) {
                o.extend_from_slice(&q[..3]);
            }
            o
        }
        ct => return Err(format!("{p:?}: unsupported 16-bit color {ct:?}")),
    };
    Ok((rgb, w, h))
}

fn decode_png_rgb8(p: &Path) -> Result<(Vec<u8>, usize, usize), String> {
    let f = std::fs::File::open(p).map_err(|e| format!("open {p:?}: {e}"))?;
    let dec = png::Decoder::new(std::io::BufReader::new(f));
    let mut reader = dec.read_info().map_err(|e| format!("png info {p:?}: {e}"))?;
    let mut buf = vec![0u8; reader.output_buffer_size().expect("png buffer size")];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| format!("png frame {p:?}: {e}"))?;
    buf.truncate(info.buffer_size());
    let (w, h) = (info.width as usize, info.height as usize);
    let rgb = match (info.color_type, info.bit_depth) {
        (png::ColorType::Rgb, png::BitDepth::Eight) => buf,
        (png::ColorType::Rgba, png::BitDepth::Eight) => {
            let mut o = Vec::with_capacity(w * h * 3);
            for px in buf.chunks_exact(4) {
                o.extend_from_slice(&px[..3]);
            }
            o
        }
        (png::ColorType::Grayscale, png::BitDepth::Eight) => {
            let mut o = Vec::with_capacity(w * h * 3);
            for &g in &buf {
                o.extend_from_slice(&[g, g, g]);
            }
            o
        }
        (png::ColorType::Rgb, png::BitDepth::Sixteen) => {
            // magick-normalized sources can stay 16-bit; take the high byte
            // (exact for values that were 8-bit-scaled up; ±1/2 LSB otherwise)
            buf.chunks_exact(2).map(|c| c[0]).collect()
        }
        (png::ColorType::Rgba, png::BitDepth::Sixteen) => {
            let mut o = Vec::with_capacity(w * h * 3);
            for px in buf.chunks_exact(8) {
                o.extend_from_slice(&[px[0], px[2], px[4]]);
            }
            o
        }
        (ct, bd) => return Err(format!("{p:?}: unsupported png {ct:?}/{bd:?}")),
    };
    Ok((rgb, w, h))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |flag: &str| -> Option<String> {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let pairs = get("--pairs-tsv").expect("--pairs-tsv <tsv>");
    let out = get("--out").expect("--out <tsv>");
    let hdr_pq16 = args.iter().any(|a| a == "--hdr-pq16");
    let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(true);
    let mut scratch = V2Scratch::new();
    let mut w = std::io::BufWriter::new(std::fs::File::create(&out).expect("create out"));
    let mut n = 0usize;
    for (li, line) in std::fs::read_to_string(&pairs)
        .expect("read pairs tsv")
        .lines()
        .enumerate()
    {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() < 2 || f[0] == "ref_path" {
            continue;
        }
        let (rp, dp) = (Path::new(f[0]), Path::new(f[1]));
        let r = if hdr_pq16 {
            let (rb, rw, rh) = decode_png_rgb16(rp).unwrap_or_else(|e| panic!("line {li}: {e}"));
            let (db, dw, dh) = decode_png_rgb16(dp).unwrap_or_else(|e| panic!("line {li}: {e}"));
            assert_eq!((rw, rh), (dw, dh), "dim mismatch line {li}");
            let rs = Pq16Image::from_rgb16(&rb, rw, rh);
            let ds = Pq16Image::from_rgb16(&db, dw, dh);
            z.compute_folded720_append2_features_hdr(
                &rs, &ds,
                HdrEncoding::Pq { peak_nits: 10_000.0 },
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap_or_else(|e| panic!("hdr compute line {li}: {e:?}"))
        } else {
            let (rb, rw, rh) = decode_png_rgb8(rp).unwrap_or_else(|e| panic!("line {li}: {e}"));
            let (db, dw, dh) = decode_png_rgb8(dp).unwrap_or_else(|e| panic!("line {li}: {e}"));
            assert_eq!((rw, rh), (dw, dh), "dim mismatch line {li}");
            let rs = StridedBytes::with_alpha_mode(
                &rb, rw, rh, rw * 3, PixelFormat::Srgb8Rgb, AlphaMode::Opaque,
            );
            let ds = StridedBytes::with_alpha_mode(
                &db, dw, dh, dw * 3, PixelFormat::Srgb8Rgb, AlphaMode::Opaque,
            );
            z.compute_folded720_append2_features(&rs, &ds)
                .unwrap_or_else(|e| panic!("compute line {li}: {e:?}"))
        };
        let feats = r.features();
        assert_eq!(feats.len(), 944, "regime width");
        write!(w, "{}\t{}", f[0], f[1]).unwrap();
        for v in feats {
            write!(w, "\t{v}").unwrap();
        }
        writeln!(w).unwrap();
        n += 1;
    }
    w.flush().unwrap();
    eprintln!("sdr944_extract: {n} pairs -> {out}");
}
