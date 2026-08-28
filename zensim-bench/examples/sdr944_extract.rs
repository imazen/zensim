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
use std::io::Write;
use std::path::Path;
use zensim::{AlphaMode, PixelFormat, StridedBytes, Zensim, ZensimProfile};

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
    let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(true);
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
        let (rb, rw, rh) = decode_png_rgb8(rp).unwrap_or_else(|e| panic!("line {li}: {e}"));
        let (db, dw, dh) = decode_png_rgb8(dp).unwrap_or_else(|e| panic!("line {li}: {e}"));
        assert_eq!((rw, rh), (dw, dh), "dim mismatch line {li}");
        let rs = StridedBytes::with_alpha_mode(
            &rb, rw, rh, rw * 3, PixelFormat::Srgb8Rgb, AlphaMode::Opaque,
        );
        let ds = StridedBytes::with_alpha_mode(
            &db, dw, dh, dw * 3, PixelFormat::Srgb8Rgb, AlphaMode::Opaque,
        );
        let r = z
            .compute_folded720_append2_features(&rs, &ds)
            .unwrap_or_else(|e| panic!("compute line {li}: {e:?}"));
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
