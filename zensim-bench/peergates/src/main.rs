//! Label-free peer calls on the immutable engineering packet. No inference,
//! calibration, gate thresholds or rank statistics are reimplemented here.
#![forbid(unsafe_code)]

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

mod stats {
    include!("../../../zensim/examples/support/coherence_stats.rs");
}

fn load(path: &str, digest: &str) -> (Vec<[u8; 3]>, usize, usize) {
    let bytes = fs::read(path).unwrap();
    assert_eq!(hex(&bytes), digest, "{path}");
    // Same explicit PNG decoder as the frozen E34 engineering packet.
    let im = image::load_from_memory(&bytes).unwrap().to_rgb8();
    let (w, h) = (im.width() as usize, im.height() as usize);
    let px = im.as_raw().as_chunks::<3>().0.to_vec();
    (px, w, h)
}

fn slice(px: &[[u8; 3]], w: usize, h: usize) -> fast_ssim2::PixelSlice<'_> {
    // The source is tightly packed; API carries dimensions, stride and color.
    fast_ssim2::PixelSlice::new(
        rgb_bytes(px),
        w as u32,
        h as u32,
        w * 3,
        fast_ssim2::PixelDescriptor::RGB8_SRGB,
    )
    .unwrap()
}

fn rgb_bytes(px: &[[u8; 3]]) -> &[u8] {
    // Arrays have a safe flattening operation on the supported Rust toolchain.
    px.as_flattened()
}

fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect()
}

fn butter(px: &[[u8; 3]], w: usize, h: usize) -> imgref::ImgVec<rgb::Rgb<u8>> {
    imgref::ImgVec::new(
        px.iter()
            .map(|p| rgb::RGB8::new(p[0], p[1], p[2]))
            .collect(),
        w,
        h,
    )
}

fn save_map(path: &Path, values: &[f32]) {
    assert!(values.iter().all(|v| v.is_finite()));
    let file = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap(),
    );
    let mut out = zstd::stream::write::Encoder::new(file, 1).unwrap();
    for v in values {
        out.write_all(&v.to_le_bytes()).unwrap();
    }
    out.finish().unwrap().flush().unwrap();
}

fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 4, "mode input.json output-directory");
    assert!(matches!(args[1].as_str(), "steering" | "pairs"));
    let input: Value = serde_json::from_slice(&fs::read(&args[2]).unwrap()).unwrap();
    let root = Path::new(&args[3]);
    fs::create_dir_all(root).unwrap();
    let mut out = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(root.join("results.jsonl"))
            .unwrap(),
    );
    let cases = input["cases"].as_array().unwrap();
    for (i, case) in cases.iter().enumerate() {
        let (r, w, h) = load(
            case["reference"].as_str().unwrap(),
            case["reference_sha256"].as_str().unwrap(),
        );
        let (d, dw, dh) = load(
            case["distorted"].as_str().unwrap(),
            case["distorted_sha256"].as_str().unwrap(),
        );
        assert_eq!((w, h), (dw, dh));
        if let Some(hash) = case["reference_pixels_sha256"].as_str() {
            assert_eq!(hex(rgb_bytes(&r)), hash, "decoded reference {i}");
        }
        if let Some(hash) = case["distorted_pixels_sha256"].as_str() {
            assert_eq!(hex(rgb_bytes(&d)), hash, "decoded distorted {i}");
        }
        let rs = slice(&r, w, h);
        let ds = slice(&d, w, h);
        let rb = butter(&r, w, h);
        let db = butter(&d, w, h);
        let bp = butteraugli::ButteraugliParams::default().with_compute_diffmap(true);
        let bbase = butteraugli::butteraugli(rb.as_ref(), db.as_ref(), &bp).unwrap();
        let sm = fast_ssim2::compute_ssimulacra2_maps(&rs, &ds).unwrap();
        let scalar = fast_ssim2::compute_ssimulacra2(&rs, &ds).unwrap();
        assert_eq!(
            sm.score.to_bits(),
            scalar.to_bits(),
            "scalar/map parity {i}"
        );
        assert!(scalar.is_finite() && bbase.pnorm_3.is_finite());
        let bm: Vec<_> = bbase
            .diffmap
            .as_ref()
            .unwrap()
            .rows()
            .flat_map(|r| r.iter().copied())
            .collect();
        let smap = sm.diffmap();
        assert_eq!(smap.len(), w * h);
        assert_eq!(bm.len(), w * h);
        save_map(&root.join(format!("{i:05}-butter.f32le.zst")), &bm);
        save_map(&root.join(format!("{i:05}-ssim2.f32le.zst")), &smap);
        let mut record = json!({"key":case["key"],"index":i,"width":w,"height":h,
            "butter_pnorm3":bbase.pnorm_3,"butter_max":bbase.score,"ssim2":scalar,
            "scalar_map_score_bits_equal":true});
        if args[1] == "steering" {
            let bs = case["block"].as_u64().unwrap() as usize;
            let mut altered = d.clone();
            let mut blocks = Vec::new();
            for y0 in (0..h).step_by(bs) {
                for x0 in (0..w).step_by(bs) {
                    let (x1, y1) = ((x0 + bs).min(w), (y0 + bs).min(h));
                    let (mut pb, mut ps) = (0., 0.);
                    for y in y0..y1 {
                        let range = y * w + x0..y * w + x1;
                        altered[range.clone()].copy_from_slice(&r[range.clone()]);
                        for x in range {
                            pb += f64::from(bm[x]);
                            ps += f64::from(smap[x]);
                        }
                    }
                    let ab = butter(&altered, w, h);
                    let ba = butteraugli::butteraugli(
                        rb.as_ref(),
                        ab.as_ref(),
                        &butteraugli::ButteraugliParams::default(),
                    )
                    .unwrap();
                    let ss = fast_ssim2::compute_ssimulacra2(&rs, &slice(&altered, w, h)).unwrap();
                    assert!(ss.is_finite() && ba.pnorm_3.is_finite());
                    blocks.push(
                        json!({"bounds":[x0,y0,x1,y1],"butter_map_sum":pb,"ssim2_map_sum":ps,
                        "butter_gain":bbase.pnorm_3-ba.pnorm_3,"ssim2_gain":ss-scalar,
                        "butter_repair_pnorm3":ba.pnorm_3,"ssim2_repair_score":ss}),
                    );
                    for y in y0..y1 {
                        let range = y * w + x0..y * w + x1;
                        altered[range.clone()].copy_from_slice(&d[range]);
                    }
                }
            }
            let vec = |key: &str| {
                blocks
                    .iter()
                    .map(|b| b[key].as_f64().unwrap())
                    .collect::<Vec<_>>()
            };
            record["butter_rho"] =
                json!(stats::spearman(&vec("butter_map_sum"), &vec("butter_gain")));
            record["ssim2_rho"] = json!(stats::spearman(&vec("ssim2_map_sum"), &vec("ssim2_gain")));
            let cb = case["candidate_blocks"].as_array().unwrap();
            assert_eq!(cb.len(), blocks.len());
            for (a, b) in cb.iter().zip(&blocks) {
                assert_eq!(a["bounds"], b["bounds"]);
            }
            let candidate_vec = |key: &str| {
                cb.iter()
                    .map(|b| b[key].as_f64().unwrap())
                    .collect::<Vec<_>>()
            };
            let mut matrix = serde_json::Map::new();
            for (name, map) in [
                ("diff410", candidate_vec("a_map")),
                ("diff_frag820", candidate_vec("c_map")),
                ("butter", vec("butter_map_sum")),
                ("ssim2", vec("ssim2_map_sum")),
            ] {
                let mut row = serde_json::Map::new();
                for (judge, gain) in [
                    ("fixed_zensim", candidate_vec("a_gain")),
                    ("butter", vec("butter_gain")),
                    ("ssim2", vec("ssim2_gain")),
                ] {
                    row.insert(judge.into(), json!(stats::spearman(&map, &gain)));
                }
                matrix.insert(name.into(), json!(row));
            }
            record["cross_judge"] = json!(matrix);
            record["blocks"] = json!(blocks);
        }
        writeln!(out, "{record}").unwrap();
        out.flush().unwrap();
        println!(
            "{}/{} {} butter={} ssim2={} rho={}/{}",
            i + 1,
            cases.len(),
            case["key"],
            bbase.pnorm_3,
            scalar,
            record["butter_rho"],
            record["ssim2_rho"]
        );
        // Bound storage before starting another observation; no cache deletion.
        let space = std::process::Command::new("df")
            .args(["--output=avail", "-B1", "/home"])
            .output()
            .unwrap();
        let avail = String::from_utf8(space.stdout)
            .unwrap()
            .split_whitespace()
            .last()
            .unwrap()
            .parse::<u64>()
            .unwrap();
        assert!(avail >= 20_000_000_000, "below20GB /home safety floor");
    }
}
