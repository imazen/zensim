//! Fixed broad-panel scalar JXL recipe. No scoring, calibration or model fit.
#![forbid(unsafe_code)]
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, io::Write};
use zenpixels::{PixelBuffer, PixelDescriptor};
use zenpixels_convert::PixelBufferConvertTypedExt;
fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}
fn main() {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(args.len(), 2);
    let job: Value = serde_json::from_slice(&fs::read(&args[1]).unwrap()).unwrap();
    let mut out = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(job["output"].as_str().unwrap())
        .unwrap();
    for row in job["rows"].as_array().unwrap() {
        let bytes = fs::read(row["reference"].as_str().unwrap()).unwrap();
        assert_eq!(sha(&bytes), row["reference_sha256"].as_str().unwrap());
        let png = zenpng::decode(
            &bytes,
            &zenpng::PngDecodeConfig::default(),
            &enough::Unstoppable,
        )
        .unwrap();
        let (w, h) = (png.info.width, png.info.height);
        assert_eq!(png.info.bit_depth, 8);
        assert_eq!(png.pixels.descriptor().channels(), 3);
        let mut pixels = Vec::new();
        let view = png.pixels.as_slice();
        for y in 0..h {
            pixels.extend_from_slice(&view.row(y)[..w as usize * 3]);
        }
        let d = row["distance"].as_f64().unwrap() as f32;
        let encoded = jxl_encoder::LossyConfig::new(d)
            .with_strategy(jxl_encoder::api::EncoderStrategy::Zenjxl)
            .with_effort(8)
            .with_auto_resampling(false)
            .with_perceptual_metric(jxl_encoder::api::PerceptualMetric::Zensim)
            .with_butteraugli_iters(0)
            .with_zensim_iters(0)
            .encode(&pixels, w, h, jxl_encoder::PixelLayout::Rgb8)
            .unwrap();
        let decoded = zenjxl_decoder::decode_with(
            &encoded,
            zenjxl_decoder::api::JxlDecoderOptions::default().with_dither_u8(true),
        )
        .unwrap();
        assert_eq!((decoded.width, decoded.height), (w as usize, h as usize));
        assert!(matches!(decoded.channels, 2 | 4));
        assert!(
            decoded
                .data
                .iter()
                .skip(decoded.channels - 1)
                .step_by(decoded.channels)
                .all(|&x| x == 255)
        );
        let descriptor = if decoded.channels == 2 {
            PixelDescriptor::GRAYA8_SRGB
        } else {
            PixelDescriptor::RGBA8_SRGB
        };
        let buffer = PixelBuffer::from_vec(decoded.data, w, h, descriptor).unwrap();
        let rgb = buffer.to_rgb8();
        let image = rgb.as_imgref();
        let mut rgb_bytes = Vec::new();
        for row in image.rows() {
            for p in row {
                rgb_bytes.extend_from_slice(&[p.r, p.g, p.b]);
            }
        }
        let png = zenpng::encode_rgb8(
            image,
            None,
            &zenpng::EncodeConfig::default(),
            &enough::Unstoppable,
            &enough::Unstoppable,
        )
        .unwrap();
        for (key, data) in [("encoded", &encoded), ("distorted", &png)] {
            fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(row[key].as_str().unwrap())
                .unwrap()
                .write_all(data)
                .unwrap();
        }
        writeln!(out,"{}",json!({"origin":row["origin"],"distance":f64::from(d),"encoded":row["encoded"],"encoded_sha256":sha(&encoded),
            "distorted":row["distorted"],"distorted_sha256":sha(&png),"width":w,"height":h,"decoded_pixels_sha256":sha(&rgb_bytes)})).unwrap();
    }
}
