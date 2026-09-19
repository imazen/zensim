//! Pixel-level adjudication for dHash flags: decode both sides with
//! zen_decode (imazen-only), Mitchell-resize the holdout to the ref's
//! geometry (sharpen=0, the core kernel), and report luminance RMSE and
//! RMSE/std(ref luma). A genuine same-source pair lands near ratio 0;
//! unrelated content lands near 1.
//!
//! Usage: flag_confirm <pairs.tsv: ref_path\tdist_path> — prints TSV.

#[path = "shared/zen_decode.rs"]
mod zen_decode;
use zen_decode::decode_rgb8_path;
use zenresize::{Filter, PixelDescriptor, ResizeConfig, Resizer};

struct Rgb8 {
    w: u32,
    h: u32,
    px: Vec<u8>,
}

fn mitchell_to(src: &Rgb8, ow: u32, oh: u32) -> Rgb8 {
    let mut rgba = Vec::with_capacity(src.px.len() / 3 * 4);
    for p in src.px.as_chunks::<3>().0 {
        rgba.extend_from_slice(&[p[0], p[1], p[2], 255]);
    }
    let cfg = ResizeConfig::builder(src.w, src.h, ow, oh)
        .filter(Filter::Mitchell)
        .resize_sharpen(0.0)
        .format(PixelDescriptor::RGBA8_SRGB)
        .build();
    let out = Resizer::new(&cfg).resize(&rgba);
    let px: Vec<u8> = out
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| [p[0], p[1], p[2]])
        .collect();
    Rgb8 { w: ow, h: oh, px }
}

fn luma(p: &[u8]) -> f64 {
    0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64
}

fn main() {
    let tsv = std::env::args().nth(1).expect("pairs.tsv");
    let txt = std::fs::read_to_string(tsv).unwrap();
    println!("ref\tholdout\trmse\tstd_ratio\tncc");
    for (i, line) in txt.lines().enumerate() {
        if i == 0 || line.is_empty() {
            continue;
        }
        let mut c = line.split('\t');
        let rp = c.next().unwrap();
        let dp = c.next().unwrap();
        let a = decode_rgb8_path(std::path::Path::new(rp))
            .map(|d| Rgb8 { w: d.width, h: d.height, px: d.pixels })
            .unwrap_or_else(|e| panic!("{rp}: {e}"));
        let b0 = decode_rgb8_path(std::path::Path::new(dp))
            .map(|d| Rgb8 { w: d.width, h: d.height, px: d.pixels })
            .unwrap_or_else(|e| panic!("{dp}: {e}"));
        let b = if b0.w == a.w && b0.h == a.h {
            b0
        } else {
            mitchell_to(&b0, a.w, a.h)
        };
        let n = a.px.len() / 3;
        let (mut sa, mut sb, mut se, mut s2a, mut s2b, mut sab) =
            (0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for i in 0..n {
            let (la, lb) = (luma(&a.px[3 * i..]), luma(&b.px[3 * i..]));
            sa += la;
            sb += lb;
            se += (la - lb).powi(2);
            s2a += la * la;
            s2b += lb * lb;
            sab += la * lb;
        }
        let nf = n as f64;
        let (ma, mb) = (sa / nf, sb / nf);
        let rmse = (se / nf).sqrt();
        let va = s2a / nf - ma * ma;
        let vb = s2b / nf - mb * mb;
        let ncc = (sab / nf - ma * mb) / (va.sqrt() * vb.sqrt()).max(1e-9);
        println!(
            "{}\t{}\t{:.3}\t{:.3}\t{:.4}",
            rp.rsplit('/').next().unwrap(),
            dp.rsplit('/').next().unwrap(),
            rmse,
            rmse / va.sqrt().max(1e-9),
            ncc
        );
    }
}
