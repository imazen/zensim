//! Label-free reference content features through the pinned canonical analyzer.
#![forbid(unsafe_code)]
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, io::Write};
use zenanalyze::feature::{AnalysisQuery, FeatureSet, FeatureValue};
fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}
fn main() {
    assert_eq!(std::env::var("ZENSIM_MAX_TIER").as_deref(), Ok("v3"));
    #[cfg(target_arch = "x86_64")]
    {
        archmage::X64V4xToken::dangerously_disable_token_process_wide(true).unwrap();
        archmage::X64V4Token::dangerously_disable_token_process_wide(true).unwrap();
    }
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 2);
    let job: Value = serde_json::from_slice(&fs::read(&args[1]).unwrap()).unwrap();
    let mut output = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(job["output"].as_str().unwrap())
        .unwrap();
    let query = AnalysisQuery::new(FeatureSet::SUPPORTED);
    for row in job["rows"].as_array().unwrap() {
        let bytes = fs::read(row["source"].as_str().unwrap()).unwrap();
        assert_eq!(sha(&bytes), row["source_sha256"].as_str().unwrap());
        let decoded = zenpng::decode(
            &bytes,
            &zenpng::PngDecodeConfig::default(),
            &enough::Unstoppable,
        )
        .unwrap();
        let (w, h) = (decoded.info.width, decoded.info.height);
        assert_eq!(w.max(h), 256);
        assert_eq!(decoded.info.bit_depth, 8);
        assert_eq!(decoded.pixels.descriptor().channels(), 3);
        let mut pixels = Vec::with_capacity((w * h * 3) as usize);
        let view = decoded.pixels.as_slice();
        for y in 0..h {
            pixels.extend_from_slice(&view.row(y)[..w as usize * 3]);
        }
        let features = zenanalyze::analyze_features_rgb8(&pixels, w, h, &query);
        let values: Vec<_> = FeatureSet::SUPPORTED
            .iter()
            .map(|id| {
                let v = match features.get(id).expect("requested native feature") {
                    FeatureValue::F32(x) => f64::from(x),
                    FeatureValue::U32(x) => f64::from(x),
                    FeatureValue::U64(x) => x as f64,
                    FeatureValue::Bool(x) => f64::from(u8::from(x)),
                    _ => panic!("unregistered native feature value type"),
                };
                assert!(v.is_finite(), "nonfinite feature {id:?}");
                (id.id(), v)
            })
            .collect();
        writeln!(
            output,
            "{}",
            json!({"origin":row["origin"],"source_sha256":row["source_sha256"],
            "pixels_sha256":sha(&pixels),"width":w,"height":h,"features":values})
        )
        .unwrap();
    }
}
