//! Gates for the imazen-only decode owner (`examples/shared/zen_decode.rs`).
//!
//! These are the FAILING-FIRST tests for the 2026-09-04 fix. Against the
//! previous `image::open(..).ok()?` decode path every one of them fails:
//!
//! * `avif_row_decodes` / `jxl_row_decodes` — `image` 0.25's default features
//!   carry no AVIF and no JXL decoder, so those rows silently vanished
//!   (measured: 30.8 % of the safesyn corpus).
//! * `xyb_jpeg_decodes_through_the_xyb_transform` — `image` decodes an XYB
//!   JPEG as an ordinary YCbCr JPEG and never applies the inverse XYB→sRGB
//!   transform, so it returns *wrong pixels with no error*. This is the test
//!   that matters most: a missing decoder is loud, a wrong one is not.
//! * `undecodable_input_fails_loud` / `detected_but_unbuilt_format_fails_loud`
//!   — the old path returned `None`, which the extractor turned into a
//!   dropped row.
//!
//! Fixtures are synthesised with the imazen ENCODERS at test time; nothing
//! reads a corpus from disk, so there is no path where a missing file turns
//! into a skipped test.

use std::path::Path;

#[path = "../examples/shared/zen_decode.rs"]
mod zen_decode;

/// A deterministic 96×64 RGB8 test image with structure at several scales:
/// a smooth gradient, a hard vertical edge, and a high-frequency checker.
/// Enough signal that a colour-space mistake cannot hide in it.
fn fixture_rgb8(w: usize, h: usize) -> Vec<u8> {
    let mut px = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        for x in 0..w {
            let grad = (x * 255 / w.max(1)) as u8;
            let edge = if x > w / 2 { 200u8 } else { 40u8 };
            let checker = if (x / 4 + y / 4) % 2 == 0 { 30u8 } else { 0u8 };
            px.push(grad.saturating_add(checker));
            px.push(edge);
            px.push(((y * 255 / h.max(1)) as u8).saturating_sub(checker));
        }
    }
    px
}

/// Mean absolute per-channel difference between two packed RGB8 buffers.
fn mean_abs_diff(a: &[u8], b: &[u8]) -> f64 {
    assert_eq!(a.len(), b.len(), "buffers differ in length");
    let sum: u64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| (x as i32 - y as i32).unsigned_abs() as u64)
        .sum();
    sum as f64 / a.len() as f64
}

/// Encode the fixture as JPEG through zenjpeg, optionally in **XYB** mode.
///
/// XYB is the arm that matters: the bitstream carries RGB component IDs and an
/// XYB ICC profile, so a decoder that is not XYB-aware returns wrong pixels
/// with no error.
fn encode_jpeg(src: &[u8], xyb: bool) -> Vec<u8> {
    use zenjpeg::encode::EncoderConfig;
    use zenjpeg::encode::encoder_types::XybSubsampling;
    let cfg = if xyb {
        EncoderConfig::xyb(92, XybSubsampling::BQuarter)
    } else {
        EncoderConfig::ycbcr(
            92,
            zenjpeg::encode::encoder_types::ChromaSubsampling::Quarter,
        )
    };
    cfg.encode_bytes(
        src,
        W as u32,
        H as u32,
        zenjpeg::encode::encoder_types::PixelLayout::Rgb8Srgb,
    )
    .expect("zenjpeg encode")
}

const W: usize = 96;
const H: usize = 64;

fn decode_and_check(bytes: &[u8], label: &str, tolerance: f64) {
    let src = fixture_rgb8(W, H);
    let out = zen_decode::decode_rgb8_bytes(bytes, label)
        .unwrap_or_else(|e| panic!("{label}: decode FAILED (this is the bug): {e}"));
    assert_eq!(
        (out.width as usize, out.height as usize),
        (W, H),
        "{label}: dimensions"
    );
    assert_eq!(out.pixels.len(), W * H * 3, "{label}: packed RGB8 length");
    let mad = mean_abs_diff(&src, &out.pixels);
    assert!(
        mad < tolerance,
        "{label}: decoded pixels are {mad:.2} mean-abs from the source \
         (tolerance {tolerance}). A large value here means the decoder ran but \
         produced the WRONG COLOUR SPACE — the exact failure mode that made an \
         XYB JPEG read 0.659 stored vs 2875.0 fresh."
    );
}

#[test]
fn png_row_decodes() {
    let src = fixture_rgb8(W, H);
    let img = imgref::ImgRef::new(bytemuck::cast_slice::<u8, rgb::RGB8>(&src), W, H);
    let bytes = zenpng::encode_rgb8(
        img,
        None,
        &zenpng::EncodeConfig::default(),
        &enough::Unstoppable,
        &enough::Unstoppable,
    )
    .expect("zenpng encode");

    // PNG is lossless: the round-trip must be byte-exact.
    let out = zen_decode::decode_rgb8_bytes(&bytes, "fixture.png").expect("png decode");
    assert_eq!(out.pixels, src, "PNG round-trip must be lossless");
}

#[test]
fn jpeg_row_decodes() {
    let src = fixture_rgb8(W, H);
    decode_and_check(&encode_jpeg(&src, false), "fixture.jpg", 12.0);
}

/// THE regression test for the retracted 2026-09-04 probe.
///
/// An XYB JPEG carries RGB component IDs (82, 71, 66) and an XYB ICC profile;
/// zenjpeg detects that and runs the inverse XYB→sRGB transform in its output
/// stage. A decoder that does not (the `image` crate) returns a plausible
/// image in the wrong space — no error, wrong pixels. The tolerance below is
/// loose enough for lossy JPEG and far tighter than an un-transformed decode,
/// which lands O(100) mean-abs away.
#[test]
fn xyb_jpeg_decodes_through_the_xyb_transform() {
    let src = fixture_rgb8(W, H);
    decode_and_check(&encode_jpeg(&src, true), "fixture_xyb.jpg", 12.0);
}

#[test]
fn webp_row_decodes() {
    let src = fixture_rgb8(W, H);
    let cfg = zenwebp::LossyConfig::new();
    let bytes =
        zenwebp::EncodeRequest::lossy(&cfg, &src, zenwebp::PixelLayout::Rgb8, W as u32, H as u32)
            .encode()
            .expect("zenwebp encode");
    decode_and_check(&bytes, "fixture.webp", 12.0);
}

/// FAILING-FIRST: `image` 0.25 has no AVIF decoder in its default features, so
/// every `zenavif-*` row (34,001 in safesyn) was silently dropped.
#[test]
fn avif_row_decodes() {
    use zencodec::encode::{EncodeJob as _, Encoder as _, EncoderConfig as _};
    use zenpixels::{PixelDescriptor, PixelSlice};

    let src = fixture_rgb8(W, H);
    let cfg = zenavif::AvifEncoderConfig::new().with_quality(92.0);
    let slice = PixelSlice::new(&src, W as u32, H as u32, W * 3, PixelDescriptor::RGB8_SRGB)
        .expect("zenavif slice");
    let bytes = cfg
        .job()
        .encoder()
        .expect("zenavif job")
        .encode(slice)
        .expect("zenavif encode")
        .into_vec();
    decode_and_check(&bytes, "fixture.avif", 14.0);
}

/// FAILING-FIRST: `image` 0.25 has no JXL decoder, so every `zenjxl-*` row
/// (26,362 in safesyn) was silently dropped.
#[test]
fn jxl_row_decodes() {
    use zencodec::encode::{EncodeJob as _, Encoder as _, EncoderConfig as _};
    use zenpixels::{PixelDescriptor, PixelSlice};

    let src = fixture_rgb8(W, H);
    let cfg = zenjxl::JxlEncoderConfig::new().with_distance(1.0);
    let slice = PixelSlice::new(&src, W as u32, H as u32, W * 3, PixelDescriptor::RGB8_SRGB)
        .expect("zenjxl slice");
    let bytes = cfg
        .job()
        .encoder()
        .expect("zenjxl job")
        .encode(slice)
        .expect("zenjxl encode")
        .into_vec();
    decode_and_check(&bytes, "fixture.jxl", 14.0);
}

/// Bytes that are not an image must produce an ERROR, never a skipped row.
#[test]
fn undecodable_input_fails_loud() {
    let junk = b"this is not an image, it is a sentence about not being one.".to_vec();
    let err =
        zen_decode::decode_rgb8_bytes(&junk, "junk.png").expect_err("garbage must not decode");
    let msg = err.to_string();
    assert!(
        msg.contains("could not detect"),
        "expected an explicit detection failure, got: {msg}"
    );
}

/// CHARACTERIZATION, not a wish: **a truncated JPEG partial-decodes to full
/// dimensions and returns `Ok`.** MEASURED 2026-09-04 — one third of a valid
/// JPEG yields a complete 96×64 buffer with no error (zenjpeg fills the MCUs
/// it never received, which is what libjpeg-family decoders do).
///
/// This is pinned as a test because it is a **corpus hazard**, not because it
/// is desirable: "the extractor decoded it" is NOT evidence that a corpus file
/// is intact, so integrity has to come from a checksum or a byte count, never
/// from a successful decode. If zenjpeg ever starts rejecting truncation this
/// test fires and the hazard note gets deleted — that is the point.
#[test]
fn truncated_jpeg_partial_decodes_which_is_a_corpus_hazard() {
    let src = fixture_rgb8(W, H);
    let bytes = encode_jpeg(&src, false);
    let truncated = &bytes[..bytes.len() / 3];
    let out = zen_decode::decode_rgb8_bytes(truncated, "truncated.jpg")
        .expect("MEASURED 2026-09-04: zenjpeg partial-decodes a truncated JPEG");
    assert_eq!((out.width as usize, out.height as usize), (W, H));
    // The recovered prefix must still resemble the source — a partial decode
    // that returned noise would be a different (real) defect.
    assert!(
        mean_abs_diff(&src, &out.pixels) < 120.0,
        "partial decode returned unrelated pixels"
    );
}

/// A bitstream that is *only* a magic number, with no parsable structure
/// behind it, must be a loud codec error — the decoder gets to reject, and
/// `zen_decode` must surface that rather than swallow it.
#[test]
fn header_only_bitstream_fails_loud() {
    let stub = vec![0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10];
    let err = zen_decode::decode_rgb8_bytes(&stub, "stub.jpg")
        .expect_err("a JPEG with no frame must not decode");
    let msg = err.to_string();
    assert!(
        msg.contains("JPEG"),
        "expected a JPEG-attributed error, got: {msg}"
    );
}

/// FAILING-FIRST for the corpora that are not PNG/JPEG: TID2013 ships `.BMP`
/// and the PIPAL extraction reads BMP, so a decode owner without it turns two
/// real corpora into loud aborts. BMP is lossless, so this round-trip must be
/// byte-exact.
#[test]
fn bmp_row_decodes_losslessly() {
    use zencodec::encode::{EncodeJob as _, Encoder as _, EncoderConfig as _};
    use zenpixels::{PixelDescriptor, PixelSlice};

    let src = fixture_rgb8(W, H);
    let slice = PixelSlice::new(&src, W as u32, H as u32, W * 3, PixelDescriptor::RGB8_SRGB)
        .expect("bmp slice");
    let bytes = zenbitmaps::BmpEncoderConfig::new()
        .job()
        .encoder()
        .expect("bmp job")
        .encode(slice)
        .expect("bmp encode")
        .into_vec();
    let out = zen_decode::decode_rgb8_bytes(&bytes, "fixture.bmp").expect("bmp decode");
    assert_eq!((out.width as usize, out.height as usize), (W, H));
    assert_eq!(out.pixels, src, "BMP round-trip must be lossless");
}

/// A format zencodec recognises but this module has no imazen decoder for is a
/// DISTINCT, loud error — so a missing arm can never be mistaken for a corrupt
/// file (or, worse, quietly routed to a third-party fallback).
#[test]
fn detected_but_unsupported_format_fails_loud() {
    // Minimal GIF87a header — enough for magic-byte detection. GIF is a real
    // format with an imazen codec (zengif); it simply has no arm HERE, and
    // that must read as "add the arm", never as "the file is corrupt".
    let gif = b"GIF87a\x01\x00\x01\x00\x00\x00\x00".to_vec();
    let err = zen_decode::decode_rgb8_bytes(&gif, "fixture.gif")
        .expect_err("GIF has no arm in zen_decode");
    let msg = err.to_string();
    assert!(
        msg.contains("no imazen decoder"),
        "expected the unsupported-format message, got: {msg}"
    );
}

/// Detection is by MAGIC BYTES, not by extension: a JPEG named `.png` still
/// decodes as a JPEG. (The safesyn decode cache wrote decoded bitstreams as
/// `.png`; trusting the extension is how a corpus silently mis-decodes.)
#[test]
fn detection_ignores_a_lying_extension() {
    let src = fixture_rgb8(W, H);
    decode_and_check(&encode_jpeg(&src, false), "actually_a_jpeg_named.png", 12.0);
}

/// Every format the safesyn / imazen-26 corpora contain must be BUILT in this
/// configuration. A build where one is missing is exactly the silent-drop
/// state this fix removes, so it fails the suite rather than skipping.
#[test]
fn every_corpus_format_is_built() {
    use zencodec::ImageFormat;
    for (f, name) in [
        (ImageFormat::Jpeg, "JPEG"),
        (ImageFormat::Png, "PNG"),
        (ImageFormat::WebP, "WebP"),
        (ImageFormat::Avif, "AVIF"),
        (ImageFormat::Jxl, "JXL"),
        (ImageFormat::Bmp, "BMP"),
    ] {
        assert!(
            zen_decode::is_supported(f),
            "{name} has no decoder in this build — enable the `zen-decode` feature"
        );
    }
}

/// The path entry point must surface a missing file as an IO error, not as a
/// decode failure and not as a skip.
#[test]
fn missing_file_fails_loud() {
    let err = zen_decode::decode_rgb8_path(Path::new("/nonexistent/zen_decode_gate.png"))
        .expect_err("a missing file must error");
    assert!(err.to_string().contains("read "), "got: {err}");
}
