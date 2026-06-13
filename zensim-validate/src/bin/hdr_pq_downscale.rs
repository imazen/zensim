//! Downscale 16-bit PQ HDR PNG references to a target long edge, resampling
//! in **linear light** (PQ EOTF → resize → PQ OETF), preserving the input
//! `cICP` (transfer 16 = PQ; primaries passed through).
//!
//! Why: the imazen-26 HDR refs are 12–14 MP. zenmetrics' HDR JXL encode
//! exceeds the encoder's 2 GiB memory budget at that size, AND the UPIQ HDR
//! validation set is 1920×1080 — so the training corpus must match ~2 MP for
//! the feature distributions to line up. Resampling PQ code values directly
//! would be photometrically wrong (PQ is non-linear); we EOTF to absolute
//! cd/m², resample the light, then OETF back. Round-trip is <1 LSB at 16-bit
//! (see `pq_roundtrip` test).
//!
//! Output PNGs carry a hand-written `cICP` chunk (transfer 16) so zenmetrics'
//! `png_to_rgb16_pq` (which requires cICP transfer == 16) accepts them.
//!
//! Usage:
//!   hdr_pq_downscale --in-dir <refs> --out-dir <out> [--long-edge 1920]

use std::path::{Path, PathBuf};

// ── PQ / SMPTE ST 2084 (constants from docs/HDR_PLAN.md §3) ─────────────
const PQ_M1: f64 = 2610.0 / 16384.0; // 0.1593017578125
const PQ_M2: f64 = 2523.0 / 4096.0 * 128.0; // 78.84375
const PQ_C1: f64 = 3424.0 / 4096.0; // 0.8359375
const PQ_C2: f64 = 2413.0 / 4096.0 * 32.0; // 18.8515625
const PQ_C3: f64 = 2392.0 / 4096.0 * 32.0; // 18.6875
const PQ_LMAX: f64 = 10000.0;

/// PQ code value in `[0,1]` → absolute luminance (cd/m², `[0, 10000]`).
fn pq_eotf(v: f64) -> f64 {
    let v = v.clamp(0.0, 1.0);
    let vp = v.powf(1.0 / PQ_M2);
    let num = (vp - PQ_C1).max(0.0);
    let den = PQ_C2 - PQ_C3 * vp;
    PQ_LMAX * (num / den).powf(1.0 / PQ_M1)
}

/// Absolute luminance (cd/m²) → PQ code value in `[0,1]`.
fn pq_oetf(nits: f64) -> f64 {
    let y = (nits / PQ_LMAX).clamp(0.0, 1.0);
    let yp = y.powf(PQ_M1);
    ((PQ_C1 + PQ_C2 * yp) / (1.0 + PQ_C3 * yp)).powf(PQ_M2)
}

// ── PNG cICP chunk read/write (image crate can't do cICP) ──────────────
const PNG_SIG: [u8; 8] = [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1a, b'\n'];

/// Read the source PNG's cICP color-primaries byte (the corpus uses 1 / 9 / 12),
/// defaulting to 1 (BT.709) if absent. Transfer is assumed PQ (16).
fn read_cicp_primaries(data: &[u8]) -> u8 {
    if data.len() < 8 || data[..8] != PNG_SIG {
        return 1;
    }
    let mut pos = 8;
    while pos + 8 <= data.len() {
        let len = u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        let typ = &data[pos + 4..pos + 8];
        if typ == b"cICP" && pos + 8 + 4 <= data.len() {
            return data[pos + 8]; // color_primaries
        }
        if typ == b"IDAT" || typ == b"IEND" {
            break;
        }
        pos += 12 + len; // length(4) + type(4) + data(len) + crc(4)
    }
    1
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFF_FFFF;
    for &b in bytes {
        crc ^= b as u32;
        for _ in 0..8 {
            crc = if crc & 1 != 0 { (crc >> 1) ^ 0xEDB8_8320 } else { crc >> 1 };
        }
    }
    crc ^ 0xFFFF_FFFF
}

/// Insert a `cICP` chunk (primaries, transfer=16 PQ, matrix=0, full-range=1)
/// immediately after IHDR in an existing PNG byte stream.
fn inject_cicp(png: &[u8], primaries: u8) -> Vec<u8> {
    // IHDR is always the first chunk: 8 (sig) + 4 (len) + 4 (type) + 13 (data) + 4 (crc) = 33.
    let insert_at = 8 + 4 + 4 + 13 + 4;
    let payload = [primaries, 16u8, 0u8, 1u8];
    let mut chunk = Vec::with_capacity(12 + 4);
    chunk.extend_from_slice(&4u32.to_be_bytes());
    chunk.extend_from_slice(b"cICP");
    chunk.extend_from_slice(&payload);
    let mut crc_input = Vec::with_capacity(8);
    crc_input.extend_from_slice(b"cICP");
    crc_input.extend_from_slice(&payload);
    chunk.extend_from_slice(&crc32(&crc_input).to_be_bytes());

    let mut out = Vec::with_capacity(png.len() + chunk.len());
    out.extend_from_slice(&png[..insert_at]);
    out.extend_from_slice(&chunk);
    out.extend_from_slice(&png[insert_at..]);
    out
}

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter().position(|a| a == key).and_then(|i| args.get(i + 1).cloned())
}

fn downscale_one(path: &Path, out: &Path, long_edge: u32) -> Result<(u32, u32), String> {
    let data = std::fs::read(path).map_err(|e| format!("read {path:?}: {e}"))?;
    let primaries = read_cicp_primaries(&data);
    let img = image::load_from_memory(&data).map_err(|e| format!("decode {path:?}: {e}"))?;
    let rgb16 = img.to_rgb16(); // ImageBuffer<Rgb<u16>>
    let (w, h) = (rgb16.width(), rgb16.height());

    // Target dims: scale so the long edge == long_edge (never upscale).
    let scale = (long_edge as f64) / (w.max(h) as f64);
    if scale >= 1.0 {
        // Already <= target: copy through unchanged (still re-tag cICP for safety).
        std::fs::write(out, inject_cicp(&data, primaries)).map_err(|e| format!("write {out:?}: {e}"))?;
        return Ok((w, h));
    }
    let nw = ((w as f64 * scale).round() as u32).max(1);
    let nh = ((h as f64 * scale).round() as u32).max(1);

    // PQ code (u16) → absolute nits (f32), resize in linear light, → PQ code.
    let mut lin = image::ImageBuffer::<image::Rgb<f32>, Vec<f32>>::new(w, h);
    for (px_in, px_out) in rgb16.pixels().zip(lin.pixels_mut()) {
        for c in 0..3 {
            px_out[c] = pq_eotf(px_in[c] as f64 / 65535.0) as f32;
        }
    }
    let resized = image::imageops::resize(&lin, nw, nh, image::imageops::FilterType::Lanczos3);
    let mut out16 = image::ImageBuffer::<image::Rgb<u16>, Vec<u16>>::new(nw, nh);
    for (px_in, px_out) in resized.pixels().zip(out16.pixels_mut()) {
        for c in 0..3 {
            let nits = (px_in[c] as f64).max(0.0); // clamp Lanczos ring undershoot
            px_out[c] = (pq_oetf(nits) * 65535.0).round().clamp(0.0, 65535.0) as u16;
        }
    }

    // Encode PNG (image, no cICP) → inject cICP chunk.
    let mut png_bytes: Vec<u8> = Vec::new();
    {
        let mut cur = std::io::Cursor::new(&mut png_bytes);
        out16
            .write_to(&mut cur, image::ImageFormat::Png)
            .map_err(|e| format!("encode {out:?}: {e}"))?;
    }
    std::fs::write(out, inject_cicp(&png_bytes, primaries)).map_err(|e| format!("write {out:?}: {e}"))?;
    Ok((nw, nh))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let in_dir = PathBuf::from(arg(&args, "--in-dir").expect("--in-dir required"));
    let out_dir = PathBuf::from(arg(&args, "--out-dir").expect("--out-dir required"));
    let long_edge: u32 = arg(&args, "--long-edge").and_then(|s| s.parse().ok()).unwrap_or(1920);
    std::fs::create_dir_all(&out_dir).expect("create out-dir");

    let mut files: Vec<PathBuf> = std::fs::read_dir(&in_dir)
        .expect("read in-dir")
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some("png"))
        .collect();
    files.sort();
    let total = files.len();
    println!("downscaling {total} PQ-PNG refs to long-edge {long_edge} → {}", out_dir.display());

    let (mut ok, mut err) = (0usize, 0usize);
    for (i, f) in files.iter().enumerate() {
        // Resolve symlinks so we read the real PQ-PNG bytes.
        let real = std::fs::canonicalize(f).unwrap_or_else(|_| f.clone());
        let name = f.file_name().unwrap().to_string_lossy().into_owned();
        let out = out_dir.join(&name);
        match downscale_one(&real, &out, long_edge) {
            Ok((nw, nh)) => {
                ok += 1;
                if (i + 1) % 10 == 0 || i + 1 == total {
                    println!("  [{}/{total}] {name} → {nw}x{nh}", i + 1);
                }
            }
            Err(e) => {
                eprintln!("  FAIL {name}: {e}");
                err += 1;
            }
        }
    }
    println!("done: {ok} downscaled, {err} failed");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pq_roundtrip() {
        // EOTF∘OETF and OETF∘EOTF identity across the PQ range.
        for &nits in &[0.0, 0.005, 0.1, 1.0, 10.0, 100.0, 203.0, 1000.0, 4000.0, 10000.0] {
            let v = pq_oetf(nits);
            let back = pq_eotf(v);
            let rel = (back - nits).abs() / nits.max(1e-6);
            assert!(rel < 1e-4 || (back - nits).abs() < 1e-3, "nits {nits} → {v} → {back}");
        }
        // EOTF anchors from docs/HDR_PLAN.md §4.
        assert!((pq_eotf(1.0) - 10000.0).abs() < 1.0, "pq_eotf(1.0)={}", pq_eotf(1.0));
        assert!((pq_eotf(0.5) - 92.245).abs() < 0.5, "pq_eotf(0.5)={}", pq_eotf(0.5));
    }

    #[test]
    fn cicp_crc_known() {
        // CRC of the IEND chunk type (no data) is a well-known PNG constant.
        assert_eq!(crc32(b"IEND"), 0xAE42_6082);
    }
}
