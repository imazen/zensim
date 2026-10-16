//! Fixed CPU judge packet for label-free native RD experiments.
//! SSIMULACRA2's scalar-only artifact policy is an explicit caller decision.
#![forbid(unsafe_code)]
use rgb::FromSlice;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
};
use zenpixels_convert::PixelBufferConvertTypedExt;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
fn sha(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect()
}
fn file_sha(path: &Path) -> Result<String> {
    let mut hash = Sha256::new();
    let mut file = fs::File::open(path)?;
    let mut block = [0; 65536];
    loop {
        let n = file.read(&mut block)?;
        if n == 0 {
            break;
        }
        hash.update(&block[..n]);
    }
    Ok(hash.finalize().iter().map(|v| format!("{v:02x}")).collect())
}
fn load(path: &Path) -> Result<(Vec<[u8; 3]>, usize, usize)> {
    let decoded = zenpng::decode(
        &fs::read(path)?,
        &zenpng::PngDecodeConfig::default(),
        &enough::Unstoppable,
    )?;
    let info = &decoded.info;
    if info.bit_depth != 8
        || info.sequence.is_animation()
        || info.icc_profile.is_some()
        || info.exif.is_some()
        || info.content_light_level.is_some()
        || info.mastering_display.is_some()
        || info.cicp.is_some()
        || info.chromaticities.is_some()
        || info.source_gamma.is_some_and(|v| v != 45455)
    {
        return Err("outside opaque sRGB8 judge contract".into());
    }
    let rgba = decoded.pixels.to_rgba8().copy_to_contiguous_bytes();
    if rgba.as_chunks::<4>().0.iter().any(|p| p[3] != 255) {
        return Err("opaque judge input required".into());
    }
    let pixels = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .map(|p| [p[0], p[1], p[2]])
        .collect();
    Ok((pixels, info.width as usize, info.height as usize))
}
fn map(
    root: &Path,
    name: &str,
    reference: imgref::ImgRef<'_, [u8; 3]>,
    distorted: imgref::ImgRef<'_, [u8; 3]>,
    values: imgref::ImgRef<'_, f32>,
    driver: &str,
) -> Result<serde_json::Value> {
    if values.width() != reference.width()
        || values.height() != reference.height()
        || reference.width() != distorted.width()
        || reference.height() != distorted.height()
    {
        return Err("map geometry differs".into());
    }
    let id = format!(
        "{name};rgb8-srgb;{}x{};driver-sha256={driver}",
        values.width(),
        values.height()
    );
    let mut key = Sha256::new();
    for row in reference.rows() {
        key.update(row.as_flattened());
    }
    for row in distorted.rows() {
        key.update(row.as_flattened());
    }
    key.update(id.as_bytes());
    let dedup_key = key
        .finalize()
        .iter()
        .map(|v| format!("{v:02x}"))
        .collect::<String>();
    let mut bytes = b"SMAP1\0".to_vec();
    bytes.extend_from_slice(&(values.width() as u64).to_le_bytes());
    bytes.extend_from_slice(&(values.height() as u64).to_le_bytes());
    for row in values.rows() {
        for value in row {
            if !value.is_finite() {
                return Err("nonfinite map".into());
            }
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    // Scalar map data: RGB8 PNG is a lossless byte container, not a heatmap.
    let logical_sha = sha(&bytes);
    let raw_header = 6 + 8 + 8;
    let packed_width = (values.width() * 4).div_ceil(3);
    let mut packed = vec![0u8; packed_width * values.height() * 3];
    for (y, row) in bytes[raw_header..]
        .chunks_exact(values.width() * 4)
        .enumerate()
    {
        packed[y * packed_width * 3..y * packed_width * 3 + row.len()].copy_from_slice(row);
    }
    let png = zenpng::encode_rgb8(
        imgref::ImgRef::new(packed.as_rgb(), packed_width, values.height()),
        None,
        &zenpng::EncodeConfig::default().with_compression(zenpng::Compression::Fastest),
        &enough::Unstoppable,
        &enough::Unstoppable,
    )?;
    let check = zenpng::decode(
        &png,
        &zenpng::PngDecodeConfig::default(),
        &enough::Unstoppable,
    )?;
    let rgba = check.pixels.to_rgba8().copy_to_contiguous_bytes();
    let unpacked: Vec<u8> = rgba
        .as_chunks::<4>()
        .0
        .iter()
        .flat_map(|p| p[..3].iter().copied())
        .collect();
    if unpacked != packed {
        return Err("float-map storage changed bits".into());
    }
    let digest = sha(&png);
    let directory = root.join("maps");
    fs::create_dir_all(&directory)?;
    let path = directory.join(format!("{digest}.data.png"));
    match fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)
    {
        Ok(mut f) => f.write_all(&png)?,
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            if fs::read(&path)? != png {
                return Err("existing CAS map differs".into());
            }
        }
        Err(e) => return Err(e.into()),
    }
    Ok(
        json!({"metric_id":id,"dedup_key":dedup_key,"map_sha256":digest,"logical_smap1_sha256":logical_sha,
        "path":path,"width":values.width(),"height":values.height(),"storage_rgb_width":packed_width,
        "raw_smap1_bytes":bytes.len(),"stored_bytes":png.len(),
        "format":"f32le rows packed into opaque RGB8 PNG; row padding to3bytes; logical SMAP1"}),
    )
}

fn main() -> Result<()> {
    let inspect: Vec<String> = std::env::args().collect();
    if inspect.get(1).map(String::as_str) == Some("--inspect-corruption-binding") {
        if inspect.len() != 4 {
            return Err("expected model and corruption-head paths".into());
        }
        let bytes = fs::read(&inspect[2])?;
        let model = zenpredict::Model::from_bytes(&bytes)?;
        let head_bytes = fs::read(&inspect[3])?;
        let head = zensim::corruption_head::CorruptionHead::from_bytes(&head_bytes)?;
        let declared =
            zensim::declared_feature_ids(&model).ok_or("candidate has no explicit declared IDs")?;
        let head_ids = head.declared_feature_ids();
        let native = zensim::BakeScorer::new(&model)?.with_corruption_head(&head, None);
        let report = json!({"model_sha256":sha(&bytes),"head_sha256":sha(&head_bytes),
          "candidate_declared_feature_ids":declared,"head_declared_feature_ids":head_ids,
          "exact_declared_id_match":declared.as_slice()==head_ids,
          "native_attachment_admitted":native.is_ok(),"native_attachment_error":native.err().map(|e|e.to_string()),
          "baked_probability_threshold":head.deadband(),"native_score_deadband":head.deadband_score()});
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    let mut args = std::env::args().skip(1);
    let mut pairs = None;
    let mut output = None;
    let mut policy = None;
    let mut requested = "ssim2,butter,zensim".to_string();
    while let Some(arg) = args.next() {
        let value = args.next().ok_or("missing value")?;
        match arg.as_str() {
            "--pairs" => pairs = Some(PathBuf::from(value)),
            "--out-dir" => output = Some(PathBuf::from(value)),
            "--ssim2-map-policy" => policy = Some(value),
            "--metrics" => requested = value,
            _ => return Err(format!("unknown flag {arg}").into()),
        }
    }
    let requested: std::collections::BTreeSet<_> =
        requested.split(',').map(str::to_string).collect();
    if requested.is_empty()
        || !requested
            .iter()
            .all(|m| matches!(m.as_str(), "ssim2" | "butter" | "zensim"))
    {
        return Err("invalid judge selection".into());
    }
    if requested.contains("ssim2") && policy.as_deref() != Some("scalar-only") {
        return Err(
            "SSIM2 requires explicit --ssim2-map-policy scalar-only; CPU API emits no map".into(),
        );
    }
    let pairs = pairs.ok_or("--pairs required")?;
    let output = output.ok_or("--out-dir required")?;
    if output.exists() {
        return Err("fresh judge output required".into());
    }
    fs::create_dir_all(&output)?;
    let driver = file_sha(&std::env::current_exe()?)?;
    fs::write(
        output.join("INPUTS.json"),
        serde_json::to_vec_pretty(&json!({"driver_sha256":driver,
        "pairs_sha256":file_sha(&pairs)?,"requested_metrics":requested,"ssim2_map_policy":policy,"fixed_zensim_profile":"B",
        "formula_revision_environment":std::env::var("ZENSIM_FORMULA_REV").ok(),
        "metric_map_key":"sha256(ref RGB8 rows || distorted RGB8 rows || metric_id)",
        "map_storage":"lossless opaque RGB8 PNG packing of f32LE bytes; every map bit-readback checked"}))?,
    )?;
    let metric = zensim::Zensim::new(zensim::ZensimProfile::B).with_parallel(false);
    let mut sink = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output.join("judges.jsonl"))?;
    let mut reader = csv::ReaderBuilder::new()
        .delimiter(b'\t')
        .from_path(&pairs)?;
    let headers = reader.headers()?.clone();
    let mut count = 0;
    let mut cached = 0;
    let mut cache: BTreeMap<String, serde_json::Value> = BTreeMap::new();
    for row in reader.records() {
        let row = row?;
        let fields: BTreeMap<_, _> = headers
            .iter()
            .zip(row.iter())
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let ref_path = Path::new(fields.get("ref_path").ok_or("missing ref_path")?);
        let dist_path = Path::new(fields.get("dist_path").ok_or("missing dist_path")?);
        let (reference, w, h) = load(ref_path)?;
        let (distorted, dw, dh) = load(dist_path)?;
        if (w, h) != (dw, dh) {
            return Err("judge pair dimensions differ".into());
        }
        let rh = sha(reference.as_flattened());
        let dhash = sha(distorted.as_flattened());
        let key = format!("{rh}/{dhash}/{w}/{h}");
        let mut value = if let Some(value) = cache.get(&key) {
            cached += 1;
            value.clone()
        } else {
            let r = imgref::ImgRef::new(reference.as_slice(), w, h);
            let d = imgref::ImgRef::new(distorted.as_slice(), w, h);
            let ss = if requested.contains("ssim2") {
                Some(fast_ssim2::compute_ssimulacra2(r, d)?)
            } else {
                None
            };
            let mut value = json!({"reference_pixels_sha256":rh,"distorted_pixels_sha256":dhash,"width":w,"height":h,
                "ssim2":ss,"ssim2_map_policy":if requested.contains("ssim2") {policy.as_deref()} else {None}});
            let mut maps = Vec::new();
            if let Some(score) = ss
                && !score.is_finite()
            {
                return Err("nonfinite SSIM2".into());
            }
            if requested.contains("butter") {
                let result = butteraugli::butteraugli(
                    imgref::ImgRef::new(reference.as_flattened().as_rgb(), w, h),
                    imgref::ImgRef::new(distorted.as_flattened().as_rgb(), w, h),
                    &butteraugli::ButteraugliParams::new().with_compute_diffmap(true),
                )?;
                let p2 = result.pnorm(2.).ok_or("missing pnorm2")?;
                let p4 = result.pnorm(4.).ok_or("missing pnorm4")?;
                if ![result.score, result.pnorm_3, p2, p4]
                    .iter()
                    .all(|v| v.is_finite())
                {
                    return Err("nonfinite butteraugli".into());
                }
                value["butteraugli_max"] = json!(result.score);
                value["butteraugli_pnorm3"] = json!(result.pnorm_3);
                value["butteraugli_pnorm2"] = json!(p2);
                value["butteraugli_pnorm4"] = json!(p4);
                maps.push(map(
                    &output,
                    "butteraugli",
                    r,
                    d,
                    result
                        .diffmap
                        .as_ref()
                        .ok_or("missing butteraugli map")?
                        .as_ref(),
                    &driver,
                )?);
            }
            if requested.contains("zensim") {
                let z = metric.compute_with_diffmap(
                    &zensim::RgbSlice::new(&reference, w, h),
                    &zensim::RgbSlice::new(&distorted, w, h),
                    zensim::DiffmapOptions::default(),
                )?;
                if ![z.score(), z.result().raw_distance()]
                    .iter()
                    .chain(z.result().features())
                    .all(|v| v.is_finite())
                {
                    return Err("nonfinite fixed zensim".into());
                }
                value["fixed_zensim"] = json!(z.score());
                value["zensim_raw_distance"] = json!(z.result().raw_distance());
                value["zensim_features"] = json!(z.result().features());
                value["zensim_mean_offset"] = json!(z.result().mean_offset());
                maps.push(map(
                    &output,
                    "zensim-B-default-diffmap",
                    r,
                    d,
                    imgref::ImgRef::new(z.diffmap(), w, h),
                    &driver,
                )?);
            }
            value["maps"] = json!(maps);
            cache.insert(key, value.clone());
            value
        };
        value["input"] = json!(fields);
        value["ref_path"] = json!(ref_path);
        value["dist_path"] = json!(dist_path);
        writeln!(sink, "{}", serde_json::to_string(&value)?)?;
        sink.flush()?;
        count += 1;
        println!("judged {count}: {}", dist_path.display());
    }
    if count == 0 {
        return Err("empty judge matrix".into());
    }
    fs::write(
        output.join("COMPLETE.json"),
        serde_json::to_vec_pretty(&json!({"pairs":count,"unique_pixel_pairs":count-cached,
        "cached_pixel_pairs":cached,"driver_sha256":driver,"requested_metrics":requested}))?,
    )?;
    Ok(())
}
