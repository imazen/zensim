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
//! native RGB16 PQ with codestream-declared primaries; features = the CANONICAL
//! `Zensim::compute_folded720_append2_features_hdr` (944; `HdrEncoding::Pq
//! { peak_nits: 10_000 }`, default toggles — dst-activity OFF per the P1.5
//! adjudication), profile `codec_target`, per-pair single-threaded compute
//! with pair-level std threads. Output CSV: `dist_basename,q,f0..f943`.
//!
//! FRONT-END NOTE (documented in the leg's manifest): this is the CURRENT
//! HDR route (PU21 chunk-2 lineage) at 944 — a NEW-REGIME leg. The v3-era
//! `compute_pu_linear_extended_features` 372 front-end is superseded; the
//! legacy targets require separate admission and are never copied by this tool.
//! Optional `--audit-composition JSON` audits a hash-bound Rust ensemble against
//! canonical features, native scalar scoring and prepared HDR maps.

use std::io::{BufWriter, Write as _};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use zensim::feature_v2::{HdrEncoding, V2NewFeatureToggles, V2Scratch};
use zensim::{Zensim, ZensimProfile};

struct Pq16Image {
    data: Vec<[u16; 4]>,
    w: usize,
    h: usize,
    primaries: zensim::ColorPrimaries,
}

impl Pq16Image {
    fn from_rgb16(px: &[[u16; 3]], w: usize, h: usize, primaries: zensim::ColorPrimaries) -> Self {
        Self {
            data: px.iter().map(|&[r, g, b]| [r, g, b, 65535]).collect(),
            w,
            h,
            primaries,
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
    fn color_primaries(&self) -> zensim::ColorPrimaries {
        self.primaries
    }
}

#[derive(Clone)]
struct Cell {
    ref_file: PathBuf,
    dist_file: PathBuf,
    dist_base: String,
    q: String,
}

type DecodedPq16 = (Vec<[u16; 3]>, usize, usize, zensim::ColorPrimaries);

fn decode_ref_png16(path: &Path, declared_cicp: bool) -> Result<DecodedPq16, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let out = zenpng::decode(
        &bytes,
        &zenpng::PngDecodeConfig::default(),
        &enough::Unstoppable,
    )
    .map_err(|e| format!("ref {path:?}: {e}"))?;
    let info = &out.info;
    if info.bit_depth != 16
        || info.icc_profile.is_some()
        || info.srgb_intent.is_some()
        || info.source_gamma.is_some()
        || info.chromaticities.is_some()
    {
        return Err(format!(
            "ref {path:?}: requires native16 BT.2020 PQ or explicitly declared untagged datagen input; conflicting/ICC color unsupported"
        ));
    }
    let primaries = if declared_cicp {
        match info.cicp {
            Some(c)
                if c.transfer_characteristics == 16
                    && c.matrix_coefficients == 0
                    && c.full_range =>
            {
                match c.color_primaries {
                    1 => zensim::ColorPrimaries::Srgb,
                    9 => zensim::ColorPrimaries::Bt2020,
                    12 => zensim::ColorPrimaries::DisplayP3,
                    _ => return Err("unsupported HDR reference primaries".into()),
                }
            }
            _ => return Err("declared HDR reference requires full-range RGB PQ cICP".into()),
        }
    } else {
        if info
            .cicp
            .is_some_and(|c| c != zenpixels::Cicp::new(9, 16, 0, true))
        {
            return Err("reference conflicts with explicit BT.2020 PQ datagen contract".into());
        }
        zensim::ColorPrimaries::Bt2020
    };
    let buf = out.pixels;
    let (w, h) = (buf.width() as usize, buf.height() as usize);
    let bytes = buf.copy_to_contiguous_bytes();
    let channels = bytes.len() / (w * h * 2);
    if !matches!(channels, 3 | 4) {
        return Err("HDR reference requires RGB/RGBA16".into());
    }
    let mut pixels = Vec::with_capacity(w * h);
    for p in bytes.chunks_exact(channels * 2) {
        let c = |i: usize| u16::from_ne_bytes([p[i * 2], p[i * 2 + 1]]);
        if channels == 4 && c(3) != 65535 {
            return Err("HDR reference requires opaque alpha".into());
        }
        pixels.push([c(0), c(1), c(2)]);
    }
    Ok((pixels, w, h, primaries))
}

fn decode_dist_jxl16(path: &Path) -> Result<DecodedPq16, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("read {path:?}: {e}"))?;
    let out = zenjxl::decode(&bytes, None, &[zenpixels::PixelDescriptor::RGB16_BT2100_PQ])
        .map_err(|e| format!("jxl decode {path:?}: {e:?}"))?;
    // A preferred descriptor chooses sample storage, not a color conversion.
    // zenjxl keeps native code values and reports their color in JxlInfo.
    let primaries = match out.info.cicp {
        Some((p, 16, 0, true)) => match p {
            1 => zensim::ColorPrimaries::Srgb,
            9 => zensim::ColorPrimaries::Bt2020,
            12 => zensim::ColorPrimaries::DisplayP3,
            _ => return Err(format!("jxl {path:?}: unsupported PQ primaries {p}")),
        },
        _ => {
            return Err(format!(
                "jxl {path:?}: explicit full-range RGB PQ codestream required, got {:?}",
                out.info.cicp
            ));
        }
    };
    let buf = out.pixels;
    if buf.descriptor().channel_type() != zenpixels::ChannelType::U16 {
        return Err(format!(
            "jxl {path:?}: expected native16 storage, got {:?}",
            buf.descriptor()
        ));
    }
    let w = buf.width() as usize;
    let h = buf.height() as usize;
    let slice = buf.as_slice();
    let bytes = slice.contiguous_bytes();
    let u16s: &[u16] = bytemuck::try_cast_slice(bytes.as_ref())
        .map_err(|e| format!("jxl {path:?}: pixel cast: {e}"))?;
    let ch = u16s.len() / (w * h);
    if !matches!(ch, 3 | 4) {
        return Err(format!("jxl {path:?}: {ch} channels"));
    }
    if ch == 4 && u16s.as_chunks::<4>().0.iter().any(|p| p[3] != 65535) {
        return Err("HDR distorted image requires opaque alpha".into());
    }
    let px: Vec<[u16; 3]> = (0..w * h)
        .map(|i| [u16s[i * ch], u16s[i * ch + 1], u16s[i * ch + 2]])
        .collect();
    Ok((px, w, h, primaries))
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut pairs_tsvs: Vec<PathBuf> = Vec::new();
    let mut enc_roots: Vec<PathBuf> = Vec::new();
    let mut ref_root = PathBuf::new();
    let mut out_path = PathBuf::new();
    let mut n_threads = 8usize;
    let mut input_contract = None;
    let mut audit_composition: Option<PathBuf> = None;
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
            "--input-contract" => {
                assert!(input_contract.is_none(), "duplicate input contract");
                input_contract = Some(args[i + 1].clone());
                i += 2;
            }
            "--threads" => {
                n_threads = args[i + 1].parse().expect("threads");
                i += 2;
            }
            "--audit-composition" => {
                audit_composition = Some(PathBuf::from(&args[i + 1]));
                i += 2;
            }
            other => panic!("unknown arg {other}"),
        }
    }
    assert!(
        matches!(
            input_contract.as_deref(),
            Some("hdr-common-primaries-v2-bt2020-pq10000" | "hdr-common-primaries-v2-cicp-pq10000")
        ),
        "explicit current HDR input contract required; old caches are incompatible"
    );
    let declared_cicp = input_contract.as_deref() == Some("hdr-common-primaries-v2-cicp-pq10000");
    assert!(
        n_threads > 0 && !out_path.exists(),
        "nonzero threads and fresh output required"
    );
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
    eprintln!(
        "hdr944_extract: {} cells, {} threads",
        cells.len(),
        n_threads
    );

    let composition: Option<serde_json::Value> = audit_composition.as_ref().map(|path| {
        serde_json::from_slice(&std::fs::read(path).expect("composition"))
            .expect("composition JSON")
    });
    let model_bytes: Vec<Vec<u8>> = composition.as_ref().map_or_else(Vec::new, |c| {
        use sha2::{Digest, Sha256};
        c["members"]
            .as_array()
            .expect("members")
            .iter()
            .map(|m| {
                let bytes =
                    std::fs::read(m["path"].as_str().expect("model path")).expect("model bytes");
                assert_eq!(
                    Sha256::digest(&bytes)
                        .iter()
                        .map(|b| format!("{b:02x}"))
                        .collect::<String>(),
                    m["sha256"].as_str().expect("model hash")
                );
                bytes
            })
            .collect()
    });
    assert!(
        composition.is_none() || !model_bytes.is_empty(),
        "empty composition"
    );
    let weights: Option<Vec<f64>> = composition.as_ref().map(|c| {
        c["weights"]
            .as_array()
            .expect("weights")
            .iter()
            .map(|x| x.as_f64().expect("weight"))
            .collect()
    });
    let audits = std::sync::Mutex::new(vec![None; cells.len()]);
    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let mut rows: Vec<Option<String>> = vec![None; cells.len()];
    let rows_ptr = std::sync::Mutex::new(&mut rows);
    std::thread::scope(|s| {
        for _ in 0..n_threads {
            s.spawn(|| {
                let z = Zensim::new(ZensimProfile::codec_target()).with_parallel(false);
                let mut scratch = V2Scratch::new();
                let models: Vec<_> = model_bytes.iter().map(|b| zenpredict::Model::from_bytes(b).expect("model")).collect();
                let mut scorer = (!models.is_empty()).then(|| zensim::BakeScorer::ensemble(&models, weights.as_deref()).expect("servable composition").with_parallel(false).with_finite_moment_refinement(true));
                loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= cells.len() {
                        break;
                    }
                    let c = &cells[i];
                    let row = (|| -> Result<String, String> {
                        let (r16, rw, rh, primaries) =
                            decode_ref_png16(&c.ref_file, declared_cicp)?;
                        let (d16, dw, dh, dist_primaries) = decode_dist_jxl16(&c.dist_file)?;
                        if (rw, rh) != (dw, dh) {
                            return Err(format!(
                                "dim mismatch {}: ref {rw}x{rh} vs dist {dw}x{dh}",
                                c.dist_base
                            ));
                        }
                        let source = Pq16Image::from_rgb16(&r16, rw, rh, primaries);
                        let distorted = Pq16Image::from_rgb16(&d16, dw, dh, dist_primaries);
                        let encoding = HdrEncoding::Pq { peak_nits: 10_000.0 };
                        let r = z
                            .compute_folded720_append2_features_hdr(
                                &source,
                                &distorted,
                                encoding,
                                V2NewFeatureToggles {
                                    v1_pools: zensim::feature_v2::V1PoolsMode::Full,
                                    ..Default::default()
                                },
                                &mut scratch,
                            )
                            .map_err(|e| format!("compute {}: {e:?}", c.dist_base))?;
                        let feats = r.features();
                        assert_eq!(feats.len(), 944, "regime width");
                        if let Some(scorer) = &mut scorer {
                            let ids = scorer.consumed_feature_ids().map_err(|e| e.to_string())?;
                            let identical = primaries == dist_primaries && r16 == d16;
                            let cached = scorer.score_features_with_identity(feats, rw as u32, rh as u32, None, identical).map_err(|e|e.to_string())?;
                            let scalar = scorer.compute_hdr(&source, &distorted, encoding, None).map_err(|e|e.to_string())?;
                            let mut worker = scorer.prepare_steering_hdr(&source, encoding, 8).map_err(|e|e.to_string())?;
                            let mapped = worker.compute(&distorted, None).map_err(|e|e.to_string())?;
                            let max_feature_error = ids.iter().map(|&i| (feats[i as usize]-mapped.result().features()[i as usize]).abs()).fold(0.0_f64,f64::max);
                            for &id in &ids {
                                let id=id as usize;
                                if (feats[id]-mapped.result().features()[id]).abs()>2e-9*feats[id].abs().max(1.) { return Err(format!("canonical feature mismatch f{id}")); }
                            }
                            if (cached-scalar).abs()>1e-5 || (scalar-mapped.result().score()).abs()>1e-5 { return Err("public HDR score parity failure".into()); }
                            if !mapped.unsupported_refinement_feature_ids().is_empty() || !mapped.refinement_gain(0,0,rw,rh).is_finite() { return Err("incomplete/nonfinite HDR refinement".into()); }
                            let audit=serde_json::json!({"row_id":i,"dist_basename":c.dist_base,"q":c.q,"width":rw,"height":rh,"cached_score":cached,"scalar_score":scalar,"map_score":mapped.result().score(),"consumed_ids":ids,"max_feature_error":max_feature_error,"unsupported_density_ids":mapped.unsupported_feature_ids(),"unsupported_refinement_ids":mapped.unsupported_refinement_feature_ids(),"full_image_gain":mapped.refinement_gain(0,0,rw,rh)});
                            let identity=worker.compute(&source,None).map_err(|e|e.to_string())?;
                            if identity.result().score()!=100. || identity.refinement_gain(0,0,rw,rh)!=0. { return Err("HDR identity failure".into()); }
                            audits.lock().unwrap()[i]=Some(audit);
                        }
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
                    if d.is_multiple_of(500) {
                        eprintln!("  {d}/{} cells", cells.len());
                    }
                }
            });
        }
    });

    assert!(
        rows.iter().all(Option::is_some),
        "failed HDR rows: refusing partial output"
    );
    let manifest = serde_json::json!({
        "input_contract":input_contract,
        "formula_revision":format!("{:?}",zensim::feature_v2::active_formula_revision()),
        "rows":cells.len(), "feature_count":944,
        "reference_primaries":if declared_cicp { "per-image cICP" } else { "BT.2020" },
        "v1_pools":"full", "distorted_primaries":"per-image codestream cICP",
        "transfer":"PQ", "display_peak_nits":10000,
        "untagged_reference_policy":"explicit datagen declaration; never inferred from bit depth",
        "source_manifests":pairs_tsvs, "decoder":"zenpng native16 + zenjxl RGB16_BT2100_PQ",
        "feature_input_era":"hdr-common-primaries-v2"
    });
    let manifest_path = out_path.with_extension("manifest.json");
    let mut manifest_file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(manifest_path)
        .expect("fresh manifest");
    serde_json::to_writer_pretty(&mut manifest_file, &manifest).unwrap();
    writeln!(manifest_file).unwrap();
    if composition.is_some() {
        let audits = audits.into_inner().unwrap();
        assert!(audits.iter().all(Option::is_some), "incomplete HDR audit");
        let f = std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(out_path.with_extension("audit.json"))
            .expect("fresh audit");
        serde_json::to_writer_pretty(f,&serde_json::json!({"composition":composition,"rows":audits,"contract":input_contract,"scope":"native scalar/cached/prepared feature parity and identity; not HDR human or encoder RD qualification"})).unwrap();
    }
    let f = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&out_path)
        .expect("fresh out");
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
    eprintln!(
        "hdr944_extract: wrote {n_ok}/{} rows -> {out_path:?}",
        cells.len()
    );
    assert_eq!(
        n_ok,
        cells.len(),
        "SKIPped cells present — investigate before use"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn hdr_datagen_keeps_low_bits_and_refuses_conflicting_transfer() {
        let root = std::env::temp_dir().join(format!(
            "zensim-hdr-native-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir(&root).unwrap();
        let pixels: Vec<_> = (0..256)
            .map(|i| rgb::Rgb::new(12000 + i, 22000, 33000))
            .collect();
        for (tag, metadata, valid) in [
            ("declared-untagged", None, true),
            (
                "pq",
                Some(zencodec::Metadata::none().with_cicp(zenpixels::Cicp::new(9, 16, 0, true))),
                true,
            ),
            (
                "hlg",
                Some(zencodec::Metadata::none().with_cicp(zenpixels::Cicp::new(9, 18, 0, true))),
                false,
            ),
            (
                "bt709",
                Some(zencodec::Metadata::none().with_cicp(zenpixels::Cicp::new(1, 16, 0, true))),
                false,
            ),
            (
                "p3",
                Some(zencodec::Metadata::none().with_cicp(zenpixels::Cicp::new(12, 16, 0, true))),
                false,
            ),
        ] {
            let bytes = zenpng::encode_rgb16(
                imgref::Img::new(pixels.as_slice(), 16, 16),
                metadata.as_ref(),
                &zenpng::EncodeConfig::default(),
                &enough::Unstoppable,
                &enough::Unstoppable,
            )
            .unwrap();
            let path = root.join(format!("{tag}.png"));
            std::fs::write(&path, bytes).unwrap();
            let declared = decode_ref_png16(&path, true);
            assert_eq!(
                declared.is_ok(),
                matches!(tag, "pq" | "p3" | "bt709"),
                "declared {tag}"
            );
            if let Ok((_, _, _, primaries)) = declared {
                assert_eq!(
                    primaries,
                    match tag {
                        "p3" => zensim::ColorPrimaries::DisplayP3,
                        "bt709" => zensim::ColorPrimaries::Srgb,
                        _ => zensim::ColorPrimaries::Bt2020,
                    }
                );
            }
            let result = decode_ref_png16(&path, false);
            assert_eq!(result.is_ok(), valid, "{tag}");
            if let Ok((actual, w, h, primaries)) = result {
                assert_eq!((w, h), (16, 16));
                for (a, b) in actual.iter().zip(&pixels) {
                    assert_eq!(*a, [b.r, b.g, b.b]);
                }
                assert_eq!(
                    zensim::ImageSource::color_primaries(&Pq16Image::from_rgb16(
                        &actual, w, h, primaries
                    )),
                    zensim::ColorPrimaries::Bt2020
                );
            }
        }
        let pixels8 = vec![[128u8; 3]; 256];
        let rgb8: &[rgb::Rgb<u8>] = bytemuck::cast_slice(&pixels8);
        let bytes = zenpng::encode_rgb8(
            imgref::Img::new(rgb8, 16, 16),
            None,
            &zenpng::EncodeConfig::default(),
            &enough::Unstoppable,
            &enough::Unstoppable,
        )
        .unwrap();
        let path = root.join("eight.png");
        std::fs::write(&path, bytes).unwrap();
        assert!(decode_ref_png16(&path, false).is_err());
        std::fs::remove_dir_all(root).unwrap();
    }
}
