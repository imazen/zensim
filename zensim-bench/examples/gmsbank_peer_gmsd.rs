//! Exact GMSD/GMSM peer columns for the promoted Rev4 pixel bank.
//! Usage: gmsbank_peer_gmsd <keys.tsv> <scores.tsv>
//! The TSV has pair_key, ref SHA, dist SHA, width, height, ref path, dist path.
//! No label file is opened. Decode uses the bank extractor's shared owner.

#[path = "shared/score_input.rs"]
mod score_input;
#[path = "shared/zen_decode.rs"]
mod zen_decode;

use score_input::{InputContract, ScoreInput};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::Path};

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn decode(
    path: &str,
    expected: &str,
    w: usize,
    h: usize,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let img = ScoreInput::decode(Path::new(path), InputContract::LegacyRgb8)?;
    assert_eq!(
        (img.width as usize, img.height as usize),
        (w, h),
        "dimensions: {path}"
    );
    assert_eq!(sha(img.bytes()), expected, "pixel digest: {path}");
    Ok(img.bytes().to_vec())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 3);
    let lines = fs::read_to_string(&args[1])?;
    let mut out = fs::File::create(&args[2])?;
    writeln!(out, "pair_key\tgmsd\tgmsm")?;
    let mut last_ref: Option<(String, String, Vec<u8>)> = None;
    let mut checked = 0usize;
    for (n, line) in lines.lines().enumerate() {
        let f: Vec<_> = line.split('\t').collect();
        assert_eq!(f.len(), 7, "bad row {n}");
        let w: usize = f[3].parse()?;
        let h: usize = f[4].parse()?;
        if !last_ref
            .as_ref()
            .is_some_and(|(p, digest, _)| p == f[5] && digest == f[1])
        {
            last_ref = Some((f[5].to_owned(), f[1].to_owned(), decode(f[5], f[1], w, h)?));
        }
        let ref_pixels = &last_ref.as_ref().unwrap().2;
        let dst_pixels = decode(f[6], f[2], w, h)?;
        assert_eq!(
            sha(format!("{}{}legacy-rgb8", f[1], f[2]).as_bytes()),
            f[0],
            "pair key row {n}"
        );
        checked += 1;
        let score = gmsd::gmsd_rgb8(ref_pixels, &dst_pixels, w, h, w * 3)
            .map_err(|e| format!("GMSD row {n}: {e:?}"))?;
        assert!(score.gmsd.is_finite() && score.mean_gms.is_finite());
        writeln!(
            out,
            "{}\t{:.17e}\t{:.17e}",
            f[0], score.gmsd, score.mean_gms
        )?;
        if n % 500 == 0 {
            eprintln!("gmsbank peer {n} key_checked={checked}");
            out.flush()?;
        }
    }
    eprintln!("gmsbank peer complete rows={checked} key_checked={checked}");
    Ok(())
}
