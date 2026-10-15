//! Explicit, label-free E34 pixel preparation. Decode/resize/PNG I/O and
//! feature extraction stay in their existing native owners. Outputs are fresh.
#[path = "support/zen_io.rs"]
mod zen_io;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, io::Write, path::Path};
use zensim::{BakeScorer, RgbSlice};

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}
fn text<'a>(v: &'a Value, key: &str) -> &'a str {
    v[key].as_str().unwrap()
}
fn read(v: &Value, key: &str) -> (Vec<[u8; 3]>, usize, usize) {
    let path = Path::new(text(v, key));
    assert_eq!(
        sha(&fs::read(path).unwrap()),
        text(v, &format!("{key}_sha256"))
    );
    zen_io::decode_rgb8(path)
}
fn fresh(path: &str) -> fs::File {
    fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .unwrap()
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 2, "e34_pixels JOB.json");
    let job: Value = serde_json::from_slice(&fs::read(&args[1]).unwrap()).unwrap();
    let mut output = fresh(text(&job, "output"));
    match text(&job, "mode") {
        "normalize" => {
            for row in job["rows"].as_array().unwrap() {
                let (mut pixels, mut w, mut h) = read(row, "source");
                let edge = job["longest_side"].as_u64().unwrap() as usize;
                if edge > 0 {
                    let factor = edge as f64 / w.max(h) as f64;
                    let nw = (w as f64 * factor).round().max(1.) as usize;
                    let nh = (h as f64 * factor).round().max(1.) as usize;
                    pixels = zen_io::resize_rgb8(&pixels, w, h, nw, nh);
                    w = nw;
                    h = nh;
                }
                let bytes = zen_io::encode_png_rgb8(&pixels, w, h);
                fresh(text(row, "destination")).write_all(&bytes).unwrap();
                writeln!(output,"{}",json!({"source":row["source"],"source_sha256":row["source_sha256"],
                "destination":row["destination"],"sha256":sha(&bytes),"pixels_sha256":sha(bytemuck::cast_slice(&pixels)),
                "width":w,"height":h})).unwrap();
            }
        }
        "repairs" => {
            assert_eq!(std::env::var("ZENSIM_FORMULA_REV").as_deref(), Ok("5"));
            assert_eq!(std::env::var("ZENSIM_MAX_TIER").as_deref(), Ok("v3"));
            #[cfg(target_arch = "x86_64")]
            {
                archmage::X64V4xToken::dangerously_disable_token_process_wide(true).unwrap();
                archmage::X64V4Token::dangerously_disable_token_process_wide(true).unwrap();
                eprintln!("E34 extraction: v4/v4x disabled, registered v3 cap");
            }
            let blob = fs::read(text(&job, "model")).unwrap();
            assert_eq!(sha(&blob), text(&job, "model_sha256"));
            let model = zenpredict::Model::from_bytes(&blob).unwrap();
            let mut scorer = BakeScorer::new(&model).unwrap().with_parallel(false);
            let ids: Vec<usize> = job["feature_ids"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap() as usize)
                .collect();
            let width = job["width"].as_u64().unwrap() as usize;
            let extract = |features: &[f64]| -> Vec<f32> {
                let mut row = vec![0.; width];
                for &id in &ids {
                    assert!(features[id].is_finite());
                    row[id] = features[id] as f32;
                }
                row
            };
            for row in job["rows"].as_array().unwrap() {
                let (r, w, h) = read(row, "reference");
                let (d, dw, dh) = read(row, "distorted");
                assert_eq!((w, h), (dw, dh));
                assert_eq!(w.max(h), 256);
                let rs = RgbSlice::new(&r, w, h);
                let base = extract(
                    scorer
                        .compute(&rs, &RgbSlice::new(&d, w, h), None)
                        .unwrap()
                        .features(),
                );
                let mut repaired = d.clone();
                for block in row["blocks"].as_array().unwrap() {
                    let b: Vec<usize> = block
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|x| x.as_u64().unwrap() as usize)
                        .collect();
                    assert_eq!(b.len(), 3);
                    let (x, y, size) = (b[0], b[1], b[2]);
                    assert!(
                        [8, 16, 32, 64].contains(&size)
                            && x < w
                            && y < h
                            && x % size == 0
                            && y % size == 0
                    );
                    for yy in y..(y + size).min(h) {
                        let range = yy * w + x..yy * w + (x + size).min(w);
                        repaired[range.clone()].copy_from_slice(&r[range]);
                    }
                    let features = extract(
                        scorer
                            .compute(&rs, &RgbSlice::new(&repaired, w, h), None)
                            .unwrap()
                            .features(),
                    );
                    let identical = base
                        .iter()
                        .zip(&features)
                        .all(|(a, b)| a.to_bits() == b.to_bits());
                    writeln!(output,"{}",json!({"base_id":row["base_id"],"block_size":size,"block":[x,y],
                        "base_features":base,"repair_features":features,"drop_bit_identical":identical})).unwrap();
                    for yy in y..(y + size).min(h) {
                        let range = yy * w + x..yy * w + (x + size).min(w);
                        repaired[range.clone()].copy_from_slice(&d[range]);
                    }
                }
                output.flush().unwrap();
            }
        }
        _ => panic!("unknown pixel preparation mode"),
    }
}
