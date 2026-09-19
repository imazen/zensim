//! joint-core-v1 fresh-leg generator — the imazen owner for
//! rendition + encode + score of the core dataset's fresh legs.
//!
//! Pipeline per rendition (from `plan/renditions_fresh.tsv`):
//!   1. decode source (zenpng / zenjpeg through `shared/zen_decode.rs`)
//!   2. verify decoded dims vs manifest (EXIF-rotated sources transpose)
//!   3. zenresize `Filter::Mitchell` + `resize_sharpen(0)` to the rung's
//!      long-edge box, aspect preserved, never upscaling
//!   4. write `refs/<class>/<rendition>.png` via zenpng, record sha256
//!
//! Per cell (from `plan/cells_fresh.tsv`):
//!   5. encode the rendition through the cell's imazen codec
//!      (`zenjpeg-420-e2` / `zenwebp-m4` / `zenavif-s6` / `zenjxl-e7`)
//!   6. write `dists/<rendition>/<codec>/q<q>.<ext>`, record sha256
//!   7. decode the encoded bytes back, score `fast_ssim2` + `butteraugli`
//!   8. emit one `pairs_fresh.tsv` row (raw teacher scores only — the
//!      target blend is applied at assembly so the rule is auditable)
//!
//! Usage:
//!   cargo run --release -p zensim-bench --example core_variant_gen \
//!       --features core-gen -- \
//!       --renditions plan/renditions_fresh.tsv --cells plan/cells_fresh.tsv \
//!       --out-root /mnt/v/output/zensim/joint-core-v1 --threads 24
//!
//! Era rule (same as m3_fixture_gen): outputs are era-stamped by out-root;
//! a new render rule gets a new root, never in-place regeneration.

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use imgref::Img;
use rayon::prelude::*;
use rgb::RGB8;
use sha2::{Digest, Sha256};

use enough::Unstoppable;
use zenpng::{EncodeConfig, encode_rgb8 as png_encode};
use zenresize::{Filter, PixelDescriptor, ResizeConfig, Resizer};

#[path = "shared/zen_decode.rs"]
mod zen_decode;

#[derive(Clone)]
struct Rgb8 {
    w: u32,
    h: u32,
    px: Vec<u8>,
}

#[derive(Clone)]
struct Rendition {
    name: String,
    src_id: String,
    group: String,
    src_class: String,
    rung: u32,
    band: String,
    src_path: String,
    src_sha256: String,
    src_w: u32,
    src_h: u32,
    kernel: String,
}

struct Cell {
    rendition: String,
    codec: String,
    q: u32,
}

fn sha256_hex(b: &[u8]) -> String {
    let mut h = Sha256::new();
    h.update(b);
    hex(&h.finalize())
}

fn hex(b: &[u8]) -> String {
    let mut s = String::with_capacity(b.len() * 2);
    for x in b {
        s.push_str(&format!("{x:02x}"));
    }
    s
}

/// Mitchell-resize to fit inside rung x rung, aspect preserved; never upscales.
/// sRGB-domain u8 resample — the "plain Mitchell" core rule (sharpen 0).
/// When the rung meets or exceeds the source long edge the source passes
/// through untouched — there is no resample to attribute a kernel to, so the
/// rendition's effective kernel is reported as `native,no-resample(rung>=src)`.
fn mitchell_rung(src: &Rgb8, rung: u32) -> Result<(Rgb8, bool), String> {
    let le = src.w.max(src.h);
    if rung >= le {
        return Ok((Rgb8 { w: src.w, h: src.h, px: src.px.clone() }, true));
    }
    let scale = rung as f64 / le as f64;
    let (ow, oh) = (
        ((src.w as f64 * scale).round() as u32).max(1),
        ((src.h as f64 * scale).round() as u32).max(1),
    );
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
    Ok((Rgb8 { w: ow, h: oh, px }, false))
}

fn encode_cell(codec: &str, q: u32, img: &Rgb8) -> Result<(Vec<u8>, &'static str), String> {
    match codec {
        "zenjpeg-420-e2" => {
            let cfg = zenjpeg::encoder::EncoderConfig::ycbcr(
                q as f32,
                zenjpeg::encoder::ChromaSubsampling::Quarter,
            );
            let mut enc = cfg
                .encode_from_bytes(img.w, img.h, zenjpeg::encoder::PixelLayout::Rgb8Srgb)
                .map_err(|e| format!("jpeg init: {e}"))?;
            enc.push_packed(&img.px, Unstoppable)
                .map_err(|e| format!("jpeg push: {e}"))?;
            Ok((enc.finish().map_err(|e| format!("jpeg: {e}"))?, "jpg"))
        }
        "zenwebp-m4" => {
            let cfg = zenwebp::LossyConfig::new()
                .with_quality(q as f32)
                .with_method(4);
            let out = zenwebp::EncodeRequest::lossy(
                &cfg,
                &img.px,
                zenwebp::PixelLayout::Rgb8,
                img.w,
                img.h,
            )
            .encode()
            .map_err(|e| format!("webp: {e:?}"))?;
            Ok((out, "webp"))
        }
        "zenavif-s6" => {
            let cfg = zenavif::AvifEncoderConfig::new()
                .with_quality(q as f32)
                .with_effort_u32(6);
            let view = imgref::ImgRef::new(
                rgb::FromSlice::as_rgb(&img.px[..]),
                img.w as usize,
                img.h as usize,
            );
            let out = cfg.encode_rgb8(view).map_err(|e| format!("avif: {e:?}"))?;
            Ok((out.data().to_vec(), "avif"))
        }
        "zenjxl-e7" => {
            let d = zenjxl::quality_to_distance(q as f32);
            let cfg = zenjxl::LossyConfig::new(d).with_effort(7);
            let view = imgref::ImgRef::new(
                rgb::FromSlice::as_rgb(&img.px[..]),
                img.w as usize,
                img.h as usize,
            );
            let out = zenjxl::encode_rgb8(view, &cfg).map_err(|e| format!("jxl: {e:?}"))?;
            Ok((out, "jxl"))
        }
        _ => Err(format!("unknown codec {codec}")),
    }
}

fn score(refi: &Rgb8, dist: &Rgb8) -> Result<(f64, f64), String> {
    if refi.w != dist.w || refi.h != dist.h {
        return Err(format!(
            "dim mismatch ref {}x{} vs dist {}x{}",
            refi.w, refi.h, dist.w, dist.h
        ));
    }
    let (w, h) = (refi.w as usize, refi.h as usize);
    let s3: Vec<[u8; 3]> = refi.px.as_chunks::<3>().0.to_vec();
    let d3: Vec<[u8; 3]> = dist.px.as_chunks::<3>().0.to_vec();
    let ssim2 = fast_ssim2::compute_ssimulacra2(Img::new(s3.as_slice(), w, h), Img::new(d3.as_slice(), w, h))
        .map_err(|e| format!("ssim2: {e:?}"))?;
    let s8: &[RGB8] = bytemuck::cast_slice(&s3);
    let d8: &[RGB8] = bytemuck::cast_slice(&d3);
    let butter = butteraugli::butteraugli(
        Img::new(s8, w, h),
        Img::new(d8, w, h),
        &butteraugli::ButteraugliParams::default(),
    )
    .map_err(|e| format!("butter: {e:?}"))?
    .score;
    Ok((ssim2, butter))
}

fn arg(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let rend_path = arg(&args, "--renditions").expect("--renditions <tsv>");
    let cells_path = arg(&args, "--cells").expect("--cells <tsv>");
    let out_root = PathBuf::from(arg(&args, "--out-root").expect("--out-root <dir>"));
    let threads: usize = arg(&args, "--threads")
        .and_then(|s| s.parse().ok())
        .unwrap_or(24);
    let limit: usize = arg(&args, "--limit")
        .and_then(|s| s.parse().ok())
        .unwrap_or(usize::MAX);
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .ok();

    std::fs::create_dir_all(out_root.join("refs")).unwrap();
    std::fs::create_dir_all(out_root.join("dists")).unwrap();
    std::fs::create_dir_all(out_root.join("pairs/rows")).unwrap();

    // resume: renditions with an existing rows file are already complete
    let done_set: std::collections::HashSet<String> = std::fs::read_dir(out_root.join("pairs/rows"))
        .map(|d| {
            d.flatten()
                .filter_map(|e| e.file_name().to_str().map(|s| s.replace(".tsv", "")))
                .collect()
        })
        .unwrap_or_default();

    // --- load plan ---
    let mut rends: Vec<Rendition> = Vec::new();
    for (i, line) in std::fs::read_to_string(&rend_path).unwrap().lines().enumerate() {
        if i == 0 {
            continue;
        }
        let c: Vec<&str> = line.split('\t').collect();
        if c.len() < 11 {
            continue;
        }
        rends.push(Rendition {
            name: c[0].into(),
            src_id: c[1].into(),
            group: c[2].into(),
            src_class: c[3].into(),
            rung: c[4].parse().unwrap(),
            band: c[5].into(),
            src_path: c[6].into(),
            src_sha256: c[7].into(),
            src_w: c[8].parse().unwrap_or(0),
            src_h: c[9].parse().unwrap_or(0),
            kernel: c[10].into(),
        });
    }
    let mut cells: Vec<Cell> = Vec::new();
    for (i, line) in std::fs::read_to_string(&cells_path).unwrap().lines().enumerate() {
        if i == 0 {
            continue;
        }
        let c: Vec<&str> = line.split('\t').collect();
        if c.len() < 3 {
            continue;
        }
        cells.push(Cell {
            rendition: c[0].into(),
            codec: c[1].into(),
            q: c[2].parse().unwrap(),
        });
    }
    let rends: Vec<Rendition> = rends
        .into_iter()
        .filter(|r| !done_set.contains(&r.name))
        .take(limit)
        .collect();
    let rmap: HashMap<String, &Rendition> =
        rends.iter().map(|r| (r.name.clone(), r)).collect();
    let cells: Vec<&Cell> = cells
        .iter()
        .filter(|c| rmap.contains_key(&c.rendition))
        .collect();
    let mut by_rend: HashMap<String, Vec<&Cell>> = HashMap::new();
    for c in &cells {
        by_rend.entry(c.rendition.clone()).or_default().push(*c);
    }
    eprintln!(
        "core-gen: {} renditions, {} cells, {} threads",
        rends.len(),
        cells.len(),
        threads
    );

    let rend_done_path = out_root.join("plan/renditions_done.tsv");
    let need_header = !rend_done_path.exists()
        || std::fs::metadata(&rend_done_path).map(|m| m.len() == 0).unwrap_or(true);
    let rend_out = Mutex::new(std::io::BufWriter::new(
        std::fs::File::options()
            .create(true)
            .append(true)
            .open(&rend_done_path)
            .unwrap(),
    ));
    if need_header {
        let mut w = rend_out.lock().unwrap();
        writeln!(
            w,
            "rendition\tsrc_id\tgroup\tsrc_class\trung\tband\tkernel\tref_path\tref_sha256\tref_w\tref_h\tsrc_path\tsrc_file_sha256\tsrc_manifest_sha256\tsrc_w\tsrc_h"
        )
        .unwrap();
    }
    let err_out = Mutex::new(std::io::BufWriter::new(
        std::fs::File::options()
            .create(true)
            .append(true)
            .open(out_root.join("pairs/errors.tsv"))
            .unwrap(),
    ));

    let n_done = AtomicUsize::new(0);
    let n_err = AtomicUsize::new(0);

    // Phase 1+2 fused per rendition: render ref once, then all its cells.
    rends.par_iter().for_each(|r| {
        let ref_rel = format!("refs/{}/{}.png", r.src_class, r.name);
        let ref_path = out_root.join(&ref_rel);
        // Provenance chain for a source file: the imazen-26 manifest sha256
        // covers the ORIGINAL asset (the png-v3 render has no published file
        // sha), so verification = decoded dims == manifest dims (EXIF
        // transpose allowed) + measured file sha recorded beside the
        // manifest sha. A dims mismatch means the cache file is not the
        // manifest's row — the rendition is dropped, not warned.
        let src_bytes = match std::fs::read(&r.src_path) {
            Ok(b) => b,
            Err(e) => {
                let mut w = err_out.lock().unwrap();
                writeln!(w, "{}\tsrc-read\t{}", r.name, e).unwrap();
                n_err.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };
        let src_sha = sha256_hex(&src_bytes);
        // decode source
        let src = match zen_decode::decode_rgb8_bytes(&src_bytes, &r.src_path) {
            Ok(d) => d,
            Err(e) => {
                let mut w = err_out.lock().unwrap();
                writeln!(w, "{}\tsrc-decode\t{:?}", r.name, e).unwrap();
                n_err.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };
        // manifest dims check (EXIF rotation can transpose stored vs rendered)
        if r.src_w > 0
            && !(src.width == r.src_w && src.height == r.src_h)
            && !(src.width == r.src_h && src.height == r.src_w)
        {
            let mut w = err_out.lock().unwrap();
            writeln!(
                w,
                "{}\tsrc-dims\tmanifest {}x{} decoded {}x{}",
                r.name, r.src_w, r.src_h, src.width, src.height
            )
            .unwrap();
            n_err.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let src = Rgb8 {
            w: src.width,
            h: src.height,
            px: src.pixels,
        };
        // resize (rung == long edge target; aspect preserved). rung >= src
        // passes the source through with an honest no-resample kernel tag.
        let (refi, passthrough) = match mitchell_rung(&src, r.rung) {
            Ok(t) => t,
            Err(e) => {
                let mut w = err_out.lock().unwrap();
                writeln!(w, "{}\tresize\t{}", r.name, e).unwrap();
                n_err.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };
        let eff_kernel = if passthrough {
            "native,no-resample(rung>=src)"
        } else {
            r.kernel.as_str()
        };
        // write ref png
        let view = imgref::ImgRef::new(
            rgb::FromSlice::as_rgb(&refi.px[..]),
            refi.w as usize,
            refi.h as usize,
        );
        let png_bytes = match png_encode(
            view,
            None,
            &EncodeConfig::default(),
            &Unstoppable,
            &Unstoppable,
        ) {
            Ok(b) => b,
            Err(e) => {
                let mut w = err_out.lock().unwrap();
                writeln!(w, "{}\tpng-enc\t{:?}", r.name, e).unwrap();
                n_err.fetch_add(1, Ordering::Relaxed);
                return;
            }
        };
        std::fs::create_dir_all(ref_path.parent().unwrap()).unwrap();
        if std::fs::write(&ref_path, &png_bytes).is_err() {
            return;
        }
        let ref_sha = sha256_hex(&png_bytes);
        {
            let mut w = rend_out.lock().unwrap();
            writeln!(
                w,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                r.name, r.src_id, r.group, r.src_class, r.rung, r.band, eff_kernel,
                ref_rel, ref_sha, refi.w, refi.h, r.src_path, src_sha, r.src_sha256, r.src_w, r.src_h
            )
            .unwrap();
        }

        // Phase 2: cells for this rendition (serial within rendition —
        // parallelism is across renditions; cells share the decoded ref).
        // Pair rows accumulate per rendition and land in one rows file so a
        // restart can skip fully-completed renditions.
        let mut rows: Vec<String> = Vec::new();
        for c in by_rend.get(&r.name).cloned().unwrap_or_default() {
            let res = (|| -> Result<(), String> {
                // reuse a previously written bitstream when resuming a
                // partially-completed rendition (encode is the expensive step)
                let dir = out_root.join(format!("dists/{}/{}", r.name, c.codec));
                let mut bytes_ext: Option<(Vec<u8>, &'static str)> = None;
                for ext in ["jpg", "webp", "avif", "jxl"] {
                    let p = dir.join(format!("q{}.{}", c.q, ext));
                    if let Ok(b) = std::fs::read(&p) {
                        bytes_ext = Some((b, ext));
                        break;
                    }
                }
                let (bytes, ext) = match bytes_ext {
                    Some(be) => be,
                    None => {
                        let (b, e) = encode_cell(c.codec.as_str(), c.q, &refi)?;
                        let p = dir.join(format!("q{}.{}", c.q, e));
                        std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
                        std::fs::write(&p, &b).map_err(|e| e.to_string())?;
                        (b, e)
                    }
                };
                let dist_rel = format!("dists/{}/{}/q{}.{}", r.name, c.codec, c.q, ext);
                let dist_sha = sha256_hex(&bytes);
                let dist = zen_decode::decode_rgb8_bytes(&bytes, &dist_rel)
                    .map_err(|e| format!("dist-decode: {e:?}"))?;
                let dist = Rgb8 { w: dist.width, h: dist.height, px: dist.pixels };
                let (ssim2, butter) = score(&refi, &dist)?;
                rows.push(format!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.6}\t{:.6}\t{}",
                    r.name, r.src_id, r.group, r.src_class, r.rung, r.band, eff_kernel,
                    ref_rel, ref_sha, refi.w, refi.h, dist_rel, dist_sha, c.codec, c.q,
                    ssim2, butter, bytes.len()
                ));
                Ok(())
            })();
            if let Err(e) = res {
                let mut w = err_out.lock().unwrap();
                writeln!(w, "{}\t{}\tq{}\t{}", r.name, c.codec, c.q, e).unwrap();
                n_err.fetch_add(1, Ordering::Relaxed);
            }
        }
        // commit this rendition's rows only when every cell succeeded —
        // a missing rows file makes a restart redo it (dists are reused,
        // so a retry costs decode+score, not re-encode)
        let n_cells = by_rend.get(&r.name).map(|v| v.len()).unwrap_or(0);
        if rows.len() == n_cells && n_cells > 0 {
            let rows_path = out_root.join(format!("pairs/rows/{}.tsv", r.name));
            if let Ok(mut f) = std::fs::File::create(&rows_path) {
                for row in &rows {
                    let _ = writeln!(f, "{row}");
                }
            }
        }
        let d = n_done.fetch_add(1, Ordering::Relaxed) + 1;
        if d % 25 == 0 {
            eprintln!("  {d}/{} renditions ({} cell errors)", rends.len(), n_err.load(Ordering::Relaxed));
        }
    });
    eprintln!(
        "core-gen done: {} renditions, {} cell errors",
        n_done.load(Ordering::Relaxed),
        n_err.load(Ordering::Relaxed)
    );
}
