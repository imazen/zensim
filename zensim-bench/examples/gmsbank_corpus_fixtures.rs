//! Four canonical imazen-26 TRAIN references: deterministic blur, noise and
//! zenjpeg 4:2:0 q95..q5 step-5 response fixtures. Inputs are raw-file SHA
//! checked against the canonical TRAIN manifest before any decode.
//! Usage: gmsbank_corpus_fixtures <refs.tsv> <output-dir>

#[path = "shared/score_input.rs"]
mod score_input;
#[path = "shared/zen_decode.rs"]
mod zen_decode;

use enough::Unstoppable;
use score_input::{InputContract, ScoreInput};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::Path};
use zenjpeg::encoder::{ChromaSubsampling, EncoderConfig, PixelLayout};
use zenpng::{EncodeConfig, encode_rgb8};

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn png(path: &Path, pixels: &[u8], w: usize, h: usize) -> Result<(), Box<dyn std::error::Error>> {
    let view = imgref::ImgRef::new(rgb::FromSlice::as_rgb(pixels), w, h);
    let bytes = encode_rgb8(
        view,
        None,
        &EncodeConfig::default(),
        &Unstoppable,
        &Unstoppable,
    )?;
    fs::write(path, bytes)?;
    Ok(())
}

fn blur(pixels: &[u8], w: usize, h: usize) -> Vec<u8> {
    let mut out = vec![0; pixels.len()];
    for y in 0..h {
        for x in 0..w {
            for ch in 0..3 {
                let mut sum = 0u32;
                for oy in -1isize..=1 {
                    for ox in -1isize..=1 {
                        let yy = y.saturating_add_signed(oy).min(h - 1);
                        let xx = x.saturating_add_signed(ox).min(w - 1);
                        sum += u32::from(pixels[3 * (yy * w + xx) + ch]);
                    }
                }
                out[3 * (y * w + x) + ch] = ((sum + 4) / 9) as u8;
            }
        }
    }
    out
}

fn noise(pixels: &[u8]) -> Vec<u8> {
    let mut state = 0x5e0f_1a15_u64;
    pixels
        .iter()
        .map(|&p| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let delta = ((state >> 24) % 25) as i16 - 12;
            (i16::from(p) + delta).clamp(0, 255) as u8
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 3, "<refs.tsv> <output-dir>");
    let root = Path::new(&args[2]);
    fs::create_dir_all(root)?;
    let mut pairs = fs::File::create(root.join("pairs.tsv"))?;
    let mut meta = fs::File::create(root.join("metadata.tsv"))?;
    writeln!(pairs, "ref_path\tdist_path\tpair_index")?;
    writeln!(
        meta,
        "id\tdistortion\tquality\tref_path\tdist_path\tdist_sha256"
    )?;
    let mut nref = 0usize;
    let mut npair = 0usize;
    for (line_no, line) in fs::read_to_string(&args[1])?.lines().enumerate() {
        if line_no == 0 {
            assert_eq!(line, "id\tpath\tsha256\tclass");
            continue;
        }
        let f: Vec<_> = line.split('\t').collect();
        assert_eq!(f.len(), 4);
        let original = fs::read(f[1])?;
        assert_eq!(sha(&original), f[2], "canonical TRAIN raw SHA: {}", f[0]);
        let decoded = ScoreInput::decode(Path::new(f[1]), InputContract::LegacyRgb8)?;
        let (w, h) = (decoded.width as usize, decoded.height as usize);
        let pixels = decoded.bytes();
        assert_eq!(pixels.len(), w * h * 3);
        let mut emit = |kind: &str,
                        quality: &str,
                        bytes: Vec<u8>,
                        ext: &str|
         -> Result<(), Box<dyn std::error::Error>> {
            let path = root.join(format!("{}_{}_{}.{}", f[0], kind, quality, ext));
            fs::write(&path, &bytes)?;
            writeln!(pairs, "{}\t{}\t{}", f[1], path.display(), npair)?;
            writeln!(
                meta,
                "{}\t{kind}\t{quality}\t{}\t{}\t{}",
                f[0],
                f[1],
                path.display(),
                sha(&bytes)
            )?;
            npair += 1;
            Ok(())
        };
        for (kind, image) in [("blur", blur(pixels, w, h)), ("noise", noise(pixels))] {
            let path = root.join(format!("{}_{}_0.png", f[0], kind));
            png(&path, &image, w, h)?;
            let encoded = fs::read(&path)?;
            emit(kind, "0", encoded, "png")?;
        }
        for q in (1..=19).rev().map(|step| step * 5) {
            let cfg = EncoderConfig::ycbcr(q as f32, ChromaSubsampling::Quarter);
            let mut enc = cfg.encode_from_bytes(w as u32, h as u32, PixelLayout::Rgb8Srgb)?;
            enc.push_packed(pixels, Unstoppable)?;
            emit("zenjpeg420", &q.to_string(), enc.finish()?, "jpg")?;
        }
        nref += 1;
        eprintln!("gmsbank fixtures {nref}/4: {} {w}x{h}", f[0]);
    }
    assert_eq!(nref, 4);
    assert_eq!(npair, 4 * 21);
    eprintln!("gmsbank fixtures complete refs={nref} pairs={npair}");
    Ok(())
}
