//! hdr944_extract — 944-regime HDR-PQ feature extraction for the hdr_v3mix
//! datagen pairs (SOTA-944 amendment; benchmarks/sota944_campaign_2026-08-03.md
//! "B-gap resolution").
//!
//! Input: the datagen pairs TSVs (`image_path codec q knob_tuple_json
//! ref_path dist_path` — fleet-container paths; only the BASENAMES are used),
//! with `--ref-root` (the imazen-26-hdr-grid PQ-PNG refs) and a paired
//! `--enc-root` per TSV (the datagen `enc/zenjxl` bitstore).
//!
//! Per pair: ref = 16-bit PNG (PQ code values), dist = zenjxl-decoded to
//! RGB16 BT.2100-PQ; features = the CANONICAL
//! `Zensim::compute_folded720_append2_features_hdr` (944; `HdrEncoding::Pq
//! { peak_nits: 10_000 }`, default toggles — dst-activity OFF per the P1.5
//! adjudication), profile `codec_target`, per-pair single-threaded compute
//! with pair-level std threads. Output CSV: `dist_basename,q,f0..f943`.
//!
//! FRONT-END NOTE (documented in the leg's manifest): this is the CURRENT
//! HDR route (PU21 chunk-2 lineage) at 944 — a NEW-REGIME leg. The v3-era
//! `compute_pu_linear_extended_features` 372 front-end is superseded; the
//! carried asset from the 2026-07-03 corpus is the TARGET (cvvdp-mix), not
//! the features.

use std::io::{BufWriter, Write as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use zensim::feature_v2::{HdrEncoding, V2NewFeatureToggles, V2Scratch};
use zensim::{Zensim, ZensimProfile};

struct Pq16Image {
    data: Vec<[u16; 4]>,
    w: usize,
    h: usize,
}

impl Pq16Image {
    fn from_rgb16(px: &[[u16; 3]], w: usize, h: usize) -> Self {
        Self {
            data: px.iter().map(|&[r, g, b]| [r, g, b, 65535]).collect(),
            w,
            h,
        }
    }
}

impl zensim::source::ImageSource for Pq16Image {
    fn width(&self) -> usize {
        self.w
    }
    fn height(&self) -> usize {
        self.h
    }
    fn pixel_format(&self) -> zensim::source::PixelFormat {
        zensim::source::PixelFormat::Srgb16Rgba
    }
    fn row_bytes(&self, y: usize) -> &[u8] {
        bytemuck::cast_slice(&self.data[y * self.w..(y + 1) * self.w])
    }
    fn alpha_mode(&self) -> zensim::source::AlphaMode {
        zensim::source::AlphaMode::Opaque
    }
    fn is_hdr(&self) -> bool {
        true
    }
}

#[derive(Clone)]
struct Cell {
    ref_file: PathBuf,
    dist_file: PathBuf,
    dist_base: String,
    q: String,
}

fn decode_ref_png16(path: &Path) -> Result<(Vec<[u16; 3]>, usize, usize), String> {
    let img = image::open(path).map_err(|e| format!("ref {path:?}: {e}"))?;
    let rgb = img.to_rgb16();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let px: Vec<[u16; 3]> = rgb.pixels().map(|p| [p.0[0], p.0[1], p.0[2]]).collect();
    Ok((px, w, h))
}

fn decode_dist_jxl16(path: &Path) -> Result<(Vec<[u16; 3]>, usize, usize), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {path:?}: {e}"))?;
    let out = zenjxl::decode(
        &bytes,
        None,
        &[zenpixels::PixelDescriptor::RGB16_BT2100_PQ],
    )
    .map_err(|e| format!("jxl decode {path:?}: {e:?}"))?;
    let buf = out.pixels;
    let w = buf.width() as usize;
    let h = buf.height() as usize;
    let slice = buf.as_slice();
    let bytes = slice.contiguous_bytes();
    let u16s: &[u16] = bytemuck::try_cast_slice(bytes.as_ref())
        .map_err(|e| format!("jxl {path:?}: pixel cast: {e}"))?;
    let ch = u16s.len() / (w * h);
    if ch < 3 {
        return Err(format!("jxl {path:?}: {ch} channels"));
    }
    let px: Vec<[u16; 3]> = (0..w * h)
        .map(|i| [u16s[i * ch], u16s[i * ch + 1], u16s[i * ch + 2]])
        .collect();
    Ok((px, w, h))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut pairs_tsvs: Vec<PathBuf> = Vec::new();
    let mut enc_roots: Vec<PathBuf> = Vec::new();
    let mut ref_root = PathBuf::new();
    let mut out_path = PathBuf::new();
    let mut n_threads = 8usize;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--pairs" => {
                pairs_tsvs.push(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--enc-root" => {
                enc_roots.push(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            "--ref-root" => {
                ref_root = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--out" => {
                out_path = PathBuf::from(&args[i + 1]);
                i += 2;
            }
            "--threads" => {
                n_threads = args[i + 1].parse().expect("threads");
                i += 2;
            }
            other => panic!("unknown arg {other}"),
        }
    }
    assert_eq!(
        pairs_tsvs.len(),
        enc_roots.len(),
        "--pairs and --enc-root must pair up"
    );
    assert!(!pairs_tsvs.is_empty() && ref_root.exists() && !out_path.as_os_str().is_empty());

    // Load cells from every TSV.
    let mut cells: Vec<Cell> = Vec::new();
    for (tsv, enc) in pairs_tsvs.iter().zip(&enc_roots) {
        let txt = std::fs::read_to_string(tsv).expect("pairs tsv");
        let mut lines = txt.lines();
        let header: Vec<&str> = lines.next().expect("header").split('\t').collect();
        let col = |n: &str| header.iter().position(|h| *h == n).expect("column");
        let (c_q, c_ref, c_dist) = (col("q"), col("ref_path"), col("dist_path"));
        for line in lines {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() <= c_dist {
                continue;
            }
            let rb = Path::new(f[c_ref]).file_name().expect("ref base");
            let db = Path::new(f[c_dist]).file_name().expect("dist base");
            cells.push(Cell {
                ref_file: ref_root.join(rb),
                dist_file: enc.join(db),
                dist_base: db.to_string_lossy().into_owned(),
                q: f[c_q].to_string(),
            });
        }
    }
    eprintln!("hdr944_extract: {} cells, {} threads", cells.len(), n_threads);

    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let mut rows: Vec<Option<String>> = vec![None; cells.len()];
    let rows_ptr = std::sync::Mutex::new(&mut rows);
    std::thread::scope(|s| {
        for _ in 0..n_threads {
            s.spawn(|| {
                let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
                let mut scratch = V2Scratch::new();
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= cells.len() {
                        break;
                    }
                    let c = &cells[i];
                    let row = (|| -> Result<String, String> {
                        let (r16, rw, rh) = decode_ref_png16(&c.ref_file)?;
                        let (d16, dw, dh) = decode_dist_jxl16(&c.dist_file)?;
                        if (rw, rh) != (dw, dh) {
                            return Err(format!(
                                "dim mismatch {}: ref {rw}x{rh} vs dist {dw}x{dh}",
                                c.dist_base
                            ));
                        }
                        let r = z
                            .compute_folded720_append2_features_hdr(
                                &Pq16Image::from_rgb16(&r16, rw, rh),
                                &Pq16Image::from_rgb16(&d16, dw, dh),
                                HdrEncoding::Pq {
                                    peak_nits: 10_000.0,
                                },
                                V2NewFeatureToggles::default(),
                                &mut scratch,
                            )
                            .map_err(|e| format!("compute {}: {e:?}", c.dist_base))?;
                        let feats = r.features();
                        assert_eq!(feats.len(), 944, "regime width");
                        let mut line =
                            String::with_capacity(16 + c.dist_base.len() + feats.len() * 20);
                        line.push_str(&c.dist_base);
                        line.push('\t');
                        line.push_str(&c.q);
                        for f in feats {
                            line.push('\t');
                            line.push_str(&format!("{f:?}"));
                        }
                        Ok(line)
                    })();
                    match row {
                        Ok(line) => {
                            rows_ptr.lock().unwrap()[i] = Some(line);
                        }
                        Err(e) => eprintln!("SKIP {e}"),
                    }
                    let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if d % 500 == 0 {
                        eprintln!("  {d}/{} cells", cells.len());
                    }
                }
            });
        }
    });

    let f = std::fs::File::create(&out_path).expect("out");
    let mut w = BufWriter::new(f);
    let mut header = String::from("dist_basename\tq");
    for i in 0..944 {
        header.push_str(&format!("\tf{i}"));
    }
    writeln!(w, "{header}").unwrap();
    let mut n_ok = 0usize;
    for r in rows.iter().flatten() {
        writeln!(w, "{r}").unwrap();
        n_ok += 1;
    }
    eprintln!("hdr944_extract: wrote {n_ok}/{} rows -> {out_path:?}", cells.len());
    assert_eq!(n_ok, cells.len(), "SKIPped cells present — investigate before use");
}
