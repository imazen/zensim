//! Canonical corruption math with the existing native fixture IO/anchor owners.
use super::*;
use corruption_corpus::{ContentClass, Rgb8 as CorpusRgb8, manifest_for_reference};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::time::Instant;
use zenpixels_convert::PixelBufferConvertTypedExt;

fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn write_verified_png(path: &PathBuf, img: &Rgb8) -> Res<String> {
    write_png_rgb8(path, img)?;
    let roundtrip = read_png_rgb8(path)?;
    if (roundtrip.w, roundtrip.h) != (img.w, img.h) || roundtrip.px != img.px {
        return Err("native PNG roundtrip changed pixels or dimensions".into());
    }
    Ok(sha(&fs::read(path)?))
}

pub(super) fn run(args: &[String]) -> Res<()> {
    let mut flags = BTreeSet::new();
    for pair in args[1..].chunks(2) {
        if pair.len() != 2
            || !["--in", "--out", "--ref-id", "--class", "--seed"].contains(&pair[0].as_str())
            || pair[1].starts_with("--")
            || !flags.insert(&pair[0])
        {
            return Err("invalid, missing, or duplicate corruption argument".into());
        }
    }
    let input = PathBuf::from(need(args, "--in")?);
    let out = PathBuf::from(need(args, "--out")?);
    let origin = need(args, "--ref-id")?;
    let class_text = need(args, "--class")?;
    let class = match class_text.as_str() {
        "photo" => ContentClass::Photo,
        "screen" => ContentClass::Screen,
        "graphic" | "line_art" => ContentClass::LineArt,
        "document" | "text" => ContentClass::Text,
        "gradient" => ContentClass::Gradient,
        _ => return Err("unsupported content class".into()),
    };
    if origin.is_empty()
        || !origin
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        return Err("--ref-id must be a nonempty simple origin identifier".into());
    }
    if out.exists() {
        return Err("output must be fresh".into());
    }
    let seed: u64 = arg(args, "--seed").unwrap_or_else(|| "1".into()).parse()?;
    let source = fs::read(&input)?;
    let decoded = decode(&source, &PngDecodeConfig::default(), &Unstoppable)?;
    if decoded.info.bit_depth != 8 || ![2, 6].contains(&decoded.info.color_type) {
        return Err("corruption sources require RGB8 or opaque RGBA8 PNG".into());
    }
    // This registered corpus contains already normalized sRGB renditions.
    // Refuse color/orientation metadata needing a transform, rather than strip it.
    let info = &decoded.info;
    if info.sequence.is_animation()
        || info.sequence.is_multi()
        || info.icc_profile.is_some()
        || info.cicp.is_some()
        || info.exif.is_some()
        || info.chromaticities.is_some()
        || info.content_light_level.is_some()
        || info.mastering_display.is_some()
        || info
            .source_gamma
            .is_some_and(|g| g != 45455 || info.srgb_intent.is_none())
    {
        return Err(
            "source requires color/orientation handling outside this sRGB corpus contract".into(),
        );
    }
    let rgba = decoded.pixels.to_rgba8();
    let view = rgba.as_imgref();
    let mut px = Vec::with_capacity(decoded.info.width as usize * decoded.info.height as usize * 3);
    for row in view.rows() {
        for p in row {
            if p.a != 255 {
                return Err("transparent corruption source is unsupported".into());
            }
            px.extend_from_slice(&[p.r, p.g, p.b]);
        }
    }
    let img = Rgb8 {
        w: decoded.info.width,
        h: decoded.info.height,
        px,
    };
    if img.w < 8 || img.h < 8 {
        return Err("corruption source must be at least 8x8".into());
    }
    let reference = CorpusRgb8::from_raw(img.w, img.h, img.px.clone());
    let entries = manifest_for_reference(&origin, class, seed);
    let mut names = BTreeSet::new();
    for entry in &entries {
        if !names.insert(entry.params.slug()) {
            return Err("duplicate corruption catalog slug".into());
        }
    }
    if entries.is_empty() {
        return Err("empty corruption catalog".into());
    }
    fs::create_dir_all(&out)?;
    let started = Instant::now();
    let reference_png_sha = write_verified_png(&out.join("reference.png"), &img)?;
    let mut records = Vec::new();
    // Anchors depend only on source/quality: two full encodes, not two per corruption.
    for quality in [20, 10] {
        let start = Instant::now();
        let encoded = encode_jpeg(&img, quality as f32)?;
        let encode_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let decoded = zenjpeg::decoder::Decoder::new().decode(&encoded, Unstoppable)?;
        if (decoded.width(), decoded.height()) != (img.w, img.h) {
            return Err("native JPEG anchor dimensions changed".into());
        }
        let pixels = decoded
            .into_pixels_u8()
            .ok_or("anchor decoder returned non-u8 output")?;
        if pixels.len() != img.px.len() {
            return Err("anchor decoder returned non-RGB output".into());
        }
        let decode_seconds = start.elapsed().as_secs_f64();
        let filename = format!("anchor-q{quality}.png");
        fs::write(out.join(format!("anchor-q{quality}.jpg")), &encoded)?;
        let png_sha = write_verified_png(
            &out.join(&filename),
            &Rgb8 {
                w: img.w,
                h: img.h,
                px: pixels.clone(),
            },
        )?;
        records.push(
            json!({"kind":"honest_anchor","filename":filename,"quality":quality,
            "pixels_sha256":sha(&pixels),"png_sha256":png_sha,"encoded_sha256":sha(&encoded),"bytes":encoded.len(),
            "encode_seconds":encode_seconds,"decode_seconds":decode_seconds,
            "is_corruption":false}),
        );
    }
    let mut inert = 0;
    for (index, entry) in entries.iter().enumerate() {
        let mut distorted = reference.clone();
        entry.params.apply(&mut distorted, entry.seed);
        let changed = distorted
            .as_bytes()
            .as_chunks::<3>()
            .0
            .iter()
            .zip(img.px.as_chunks::<3>().0)
            .filter(|(a, b)| a != b)
            .count();
        inert += usize::from(changed == 0);
        let filename = format!("corruption-{index:04}.png");
        let png_sha = write_verified_png(
            &out.join(&filename),
            &Rgb8 {
                w: img.w,
                h: img.h,
                px: distorted.as_bytes().to_vec(),
            },
        )?;
        records.push(
            json!({"kind":"corruption","filename":filename,"entry":entry,
            "pixels_sha256":sha(distorted.as_bytes()),"png_sha256":png_sha,"pixels_changed":changed,
            "inert":changed==0,"is_corruption":changed>0}),
        );
    }
    if source != fs::read(&input)? {
        return Err("source changed during generation".into());
    }
    let manifest = json!({"schema":"native-corruption-fixtures-v1","source":input,
        "source_sha256":sha(&source),"source_pixels_sha256":sha(&img.px),
        "reference_png_sha256":reference_png_sha,"pixel_contract":"packed opaque RGB8 sRGB; no color/orientation transform",
        "origin":origin,"content_class":class_text,"width":img.w,"height":img.h,
        "base_seed":seed,"generator_revision":"8e10d4d765667c1c49d74413878fc4bfb46dcf8d",
        "anchor_config":"zenjpeg YCbCr 4:2:0 q20/q10; independently decoded",
        "records":records});
    fs::write(
        out.join("_MANIFEST.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    fs::write(
        out.join("COMPLETE.json"),
        serde_json::to_vec_pretty(&json!({
            "source_sha256":sha(&source),"catalog_entries":entries.len(),"inert_entries":inert,
            "full_encodes":2,"independent_anchor_decodes":2,"png_files":entries.len()+3,
            "manifest_sha256":sha(&fs::read(out.join("_MANIFEST.json"))?),
            "elapsed_seconds":started.elapsed().as_secs_f64()
        }))?,
    )?;
    eprintln!(
        "{origin}: {} corruption entries ({inert} inert), two native anchors",
        entries.len()
    );
    Ok(())
}
