//! Label-free RGB8 decode and content classification for GMSBANK calibration.
//! Reuses the exact legacy-rgb8 decode owner used by the bank extractor.
//! `classify refs.tsv classes.tsv`: refs TSV = ref_group, ref_path.
//! `dump pairs.tsv out_dir`: pairs TSV = pair_key, ref_sha, dist_sha, ref_path, dist_path.
//! Class and size sampling is performed by the preregistered Python driver.

#[path = "shared/score_input.rs"]
mod score_input;
#[path = "shared/zen_decode.rs"]
mod zen_decode;

use score_input::{InputContract, ScoreInput};
use sha2::{Digest, Sha256};
use std::{collections::HashMap, fs, io::Write, path::Path};

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn classify(bytes: &[u8], w: usize, h: usize) -> (&'static str, usize, f64, f64) {
    let mut colors = std::collections::HashSet::new();
    let mut flat = 0usize;
    let mut edge = 0usize;
    let mut total = 0usize;
    for y in (0..h.saturating_sub(8)).step_by(8) {
        for x in (0..w.saturating_sub(8)).step_by(8) {
            let i = 3 * (y * w + x);
            let j = i + 24;
            let rgb = &bytes[i..i + 3];
            let rgb_next = &bytes[j..j + 3];
            colors.insert(
                ((rgb[0] >> 3) as u16) << 10 | ((rgb[1] >> 3) as u16) << 5 | (rgb[2] >> 3) as u16,
            );
            flat += usize::from(rgb.iter().zip(rgb_next).all(|(a, b)| (a >> 3) == (b >> 3)));
            let luma = |p: &[u8]| -> i32 {
                (299 * i32::from(p[0]) + 587 * i32::from(p[1]) + 114 * i32::from(p[2]) + 500) / 1000
            };
            edge += usize::from((luma(rgb) - luma(rgb_next)).abs() > 24);
            total += 1;
        }
    }
    let flat_rate = flat as f64 / total.max(1) as f64;
    let edge_rate = edge as f64 / total.max(1) as f64;
    let class = if colors.len() <= 64 && edge_rate > 0.08 {
        "line_art"
    } else if flat_rate >= 0.45 && colors.len() > 64 {
        "screen"
    } else if flat_rate >= 0.20 {
        "mixed"
    } else {
        "photo"
    };
    (class, colors.len(), flat_rate, edge_rate)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(
        args.len(),
        4,
        "classify refs.tsv classes.tsv | dump pairs.tsv out_dir"
    );
    let lines = fs::read_to_string(&args[2])?;
    if args[1] == "classify" {
        let mut out = fs::File::create(&args[3])?;
        writeln!(
            out,
            "ref_group\twidth\theight\tclass\tunique_q5\tflat_rate\tedge_rate"
        )?;
        for (n, line) in lines.lines().enumerate() {
            let mut fields = line.split('\t');
            let group = fields.next().ok_or("group")?;
            let path = fields.next().ok_or("path")?;
            let img = ScoreInput::decode(Path::new(path), InputContract::LegacyRgb8)?;
            let (class, unique, flat, edge) =
                classify(img.bytes(), img.width as usize, img.height as usize);
            writeln!(
                out,
                "{group}\t{}\t{}\t{class}\t{unique}\t{flat:.8}\t{edge:.8}",
                img.width, img.height
            )?;
            if n % 100 == 0 {
                eprintln!("classify {n}");
            }
        }
    } else if args[1] == "dump" {
        fs::create_dir_all(&args[3])?;
        let mut cache: HashMap<String, (u32, u32)> = HashMap::new();
        let mut out = fs::File::create(Path::new(&args[3]).join("planes.tsv"))?;
        for (n, line) in lines.lines().enumerate() {
            let f: Vec<_> = line.split('\t').collect();
            assert_eq!(f.len(), 5);
            let mut dumps = Vec::new();
            for (expected, path) in [(f[1], f[3]), (f[2], f[4])] {
                let name = format!("{expected}.rgb");
                if let Some((w, h)) = cache.get(expected) {
                    dumps.push((*w, *h, name));
                    continue;
                }
                let img = ScoreInput::decode(Path::new(path), InputContract::LegacyRgb8)?;
                assert_eq!(sha(img.bytes()), expected, "pixel sha: {path}");
                fs::write(Path::new(&args[3]).join(&name), img.bytes())?;
                cache.insert(expected.to_owned(), (img.width, img.height));
                dumps.push((img.width, img.height, name));
            }
            assert_eq!(dumps[0].0, dumps[1].0);
            assert_eq!(dumps[0].1, dumps[1].1);
            assert_eq!(sha(format!("{}{}legacy-rgb8", f[1], f[2]).as_bytes()), f[0]);
            writeln!(
                out,
                "{}\t{}\t{}\t{}\t{}",
                f[0], dumps[0].0, dumps[0].1, dumps[0].2, dumps[1].2
            )?;
            if n % 25 == 0 {
                eprintln!("dump {n}");
            }
        }
    } else {
        return Err("unknown mode".into());
    }
    Ok(())
}
