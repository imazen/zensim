//! Registered severe quantization probe. No model fitting or corruption head.
#![forbid(unsafe_code)]
use anyhow::{Context, Result, ensure};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{fs, path::Path};
use zenjpeg::encoder::{
    ChromaSubsampling, EncoderConfig, ExpertConfig, OptimizationPreset, PixelLayout,
    QuantTableConfig,
};

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn save_json(path: &Path, value: &serde_json::Value) -> Result<()> {
    ensure!(!path.exists(), "preserve existing output");
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
fn encode(source: &Path, dc: u16, ac: u16, output: &Path) -> Result<()> {
    ensure!(!output.exists(), "fresh cell required");
    fs::create_dir_all(output)?;
    let image = image::open(source)?.to_rgb8();
    let (w, h) = image.dimensions();
    let mut quant = [f32::from(ac); 64];
    quant[0] = f32::from(dc);
    // The public expert constructor supplies Exact scaling and neutral
    // zero-bias. Override every quant coefficient; none of its preset table
    // values participate in this registered instrument.
    let expert = ExpertConfig::from_preset(OptimizationPreset::MozjpegBaseline, 85.);
    ensure!(!expert.uses_quality_scaling(), "exact scaling required");
    let mut tables = expert.tables;
    for component in 0..3 {
        *tables.quant.get_mut(component) = quant;
        *tables.zero_bias_mul.get_mut(component) = [0.; 64];
    }
    tables.zero_bias_offset_dc = [0.5; 3];
    tables.zero_bias_offset_ac = [0.5; 3];
    let config = EncoderConfig::ycbcr(85., ChromaSubsampling::None)
        .quant_table_config(QuantTableConfig::Custom(Box::new(tables)))
        .allow_16bit_quant_tables(true)
        .progressive(false)
        .aq_enabled(false)
        .deringing(false)
        .auto_optimize(false);
    ensure!(config.get_trellis().is_none(), "unexpected trellis");
    let mut encoder = config.encode_from_bytes(w, h, PixelLayout::Rgb8Srgb)?;
    encoder.push(
        image.as_raw(),
        h as usize,
        w as usize * 3,
        enough::Unstoppable,
    )?;
    let initially_encoded = encoder.finish()?;
    let decoder = zenjpeg::decoder::Decoder::new();
    let mut preserved = decoder.decode_coefficients(&initially_encoded, enough::Unstoppable)?;
    let mut restored = 0usize;
    for component in &preserved.components {
        let table = preserved.quant_tables[usize::from(component.quant_table_idx)]
            .context("missing original table")?;
        for (k, actual) in table.iter().enumerate() {
            let desired = if k == 0 { dc } else { ac };
            if *actual != desired {
                // DQT arrays are natural order; AC entries are all identical
                // in this probe, so every changed AC entry is checked below.
                if k == 0 {
                    ensure!(
                        component
                            .coeffs
                            .as_chunks::<64>()
                            .0
                            .iter()
                            .all(|b| b[0] == 0),
                        "nonzero changed DC divisor"
                    );
                } else {
                    ensure!(
                        component
                            .coeffs
                            .as_chunks::<64>()
                            .0
                            .iter()
                            .all(|b| b[1..].iter().all(|v| *v == 0)),
                        "nonzero changed AC divisor"
                    );
                }
                restored += 1;
            }
        }
    }
    for component in &preserved.components {
        let table = preserved.quant_tables[usize::from(component.quant_table_idx)]
            .as_mut()
            .context("missing table to preserve")?;
        table.fill(ac);
        table[0] = dc;
    }
    let encoded = if restored > 0 {
        zenjpeg::recompress::expert::emit_preserved(
            &preserved,
            zenjpeg::decoder::Subsampling::S444,
            &zenjpeg::recompress::expert::EmitConfig::uniform_scale(
                zenjpeg::recompress::expert::QuantScale::IDENTITY,
            ),
        )?
    } else {
        initially_encoded.clone()
    };
    let before = decoder
        .decode(&initially_encoded, enough::Unstoppable)?
        .into_pixels_u8()
        .context("original RGB")?;
    let after = decoder
        .decode(&encoded, enough::Unstoppable)?
        .into_pixels_u8()
        .context("preserved RGB")?;
    ensure!(before == after, "DQT preservation changed pixels");
    fs::write(output.join("initial.jpg"), &initially_encoded)?;
    fs::write(output.join("encoded.jpg"), &encoded)?;
    let decoder = zenjpeg::decoder::Decoder::new();
    let coefficients = decoder.decode_coefficients(&encoded, enough::Unstoppable)?;
    ensure!(
        coefficients
            .components
            .iter()
            .zip(&preserved.components)
            .all(|(a, b)| a.coeffs == b.coeffs),
        "preserved coefficient drift"
    );
    ensure!(coefficients.components.len() == 3, "YCbCr components");
    let mut nonzero_ac = 0usize;
    let mut blocks = 0usize;
    for component in &coefficients.components {
        ensure!(
            component.h_samp == 1 && component.v_samp == 1,
            "444 sampling"
        );
        let table = coefficients.quant_tables[usize::from(component.quant_table_idx)]
            .context("missing DQT")?;
        ensure!(
            table[0] == dc && table[1..].iter().all(|v| *v == ac),
            "DQT drift"
        );
        for block in component.coeffs.as_chunks::<64>().0 {
            nonzero_ac += block[1..].iter().filter(|v| **v != 0).count();
            blocks += 1;
        }
    }
    if ac == 32767 {
        ensure!(nonzero_ac == 0, "DC-only instrument has AC");
    }
    let pixels = decoder.decode(&encoded, enough::Unstoppable)?;
    ensure!(
        (pixels.width(), pixels.height()) == (w, h),
        "decode geometry"
    );
    let rgb = pixels.into_pixels_u8().context("RGB8 decode")?;
    fs::write(output.join("decoded.rgb"), &rgb)?;
    image::save_buffer(
        output.join("decoded.png"),
        &rgb,
        w,
        h,
        image::ColorType::Rgb8,
    )?;
    save_json(
        &output.join("ENCODE.json"),
        &json!({
            "width":w,"height":h,"dc_quant":dc,"ac_quant":ac,"blocks":blocks,
            "nonzero_ac_coefficients":nonzero_ac,"encoded_sha256":sha(&encoded),
            "reference_file_sha256":sha(&fs::read(source)?),"reference_rgb_sha256":sha(image.as_raw()),
            "decoded_rgb_sha256":sha(&rgb),"encoded_bytes":encoded.len(),
            "encoder":"zenjpeg4196fa93","sampling":"444","aq":false,"deringing":false,
            "trellis":false,"exact_tables":true,"full_encodes":1,"coefficient_serializations":usize::from(restored>0),"restored_zero_coefficient_divisors":restored,"initial_encoded_sha256":sha(&initially_encoded),"preserved_pixels_exact":true
        }),
    )?;
    Ok(())
}
fn score(bake: &Path, reference: &Path, distorted: &Path, output: &Path) -> Result<()> {
    ensure!(!output.exists(), "fresh score required");
    fs::create_dir_all(output)?;
    let source = image::open(reference)?.to_rgb8();
    let target = image::open(distorted)?.to_rgb8();
    ensure!(source.dimensions() == target.dimensions(), "score geometry");
    let (w, h) = source.dimensions();
    let model_bytes = fs::read(bake)?;
    let model = zenpredict::Model::from_bytes(&model_bytes)?;
    let mut scorer = zensim::BakeScorer::new(&model)?.with_parallel(false);
    let rs = zensim::RgbSlice::new(source.as_raw().as_chunks::<3>().0, w as usize, h as usize);
    let ds = zensim::RgbSlice::new(target.as_raw().as_chunks::<3>().0, w as usize, h as usize);
    let result = scorer.compute(&rs, &ds, None)?;
    let value = result.score();
    ensure!(value.is_finite(), "finite base score");
    let features: Vec<u8> = result
        .features()
        .iter()
        .flat_map(|v| (*v as f32).to_le_bytes())
        .collect();
    fs::write(output.join("features.f32le"), &features)?;
    // Retain the native per-pixel attribution, with its actual coverage. This
    // map is not substituted for the independently obtained scalar result.
    let mut prepared = scorer.prepare_steering(&rs, 1)?;
    let mapped = prepared.compute(&ds, None)?;
    ensure!(!mapped.has_corruption_gate(), "detector must be absent");
    let map_score = mapped.result().score();
    ensure!(
        (map_score - value).abs() <= 1e-4,
        "prepared/scalar precision contract"
    );
    let density = mapped.attribution().density();
    ensure!(
        density.len() == w as usize * h as usize,
        "per-pixel map geometry"
    );
    ensure!(density.iter().all(|v| v.is_finite()), "finite map");
    let map: Vec<u8> = density.iter().flat_map(|v| v.to_le_bytes()).collect();
    fs::write(output.join("attribution.f32le"), &map)?;
    save_json(
        &output.join("SCORE.json"),
        &json!({
            "model_sha256":sha(&model_bytes),"base_score":value,"prepared_score":map_score,
            "detector_attached":false,"reference_rgb_sha256":sha(source.as_raw()),
            "distorted_rgb_sha256":sha(target.as_raw()),"width":w,"height":h,
            "features_sha256":sha(&features),"attribution_sha256":sha(&map),
            "map_semantics":"native signed score-gain density, bin1; coarse replay supplied separately by API",
            "unsupported_feature_ids":mapped.unsupported_feature_ids(),
            "unsupported_refinement_feature_ids":mapped.unsupported_refinement_feature_ids(),
            "negative_scores_preserved":true,"scalar_comparisons":1,"map_comparisons":1
        }),
    )?;
    Ok(())
}
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    ensure!(
        std::env::var("ZENSIM_FORMULA_REV")?.as_str() == "5",
        "Rev5 required"
    );
    match args.get(1).map(String::as_str) {
        Some("encode") if args.len() == 6 => encode(
            Path::new(&args[2]),
            args[3].parse()?,
            args[4].parse()?,
            Path::new(&args[5]),
        ),
        Some("score") if args.len() == 6 => score(
            Path::new(&args[2]),
            Path::new(&args[3]),
            Path::new(&args[4]),
            Path::new(&args[5]),
        ),
        _ => anyhow::bail!("encode SOURCE DC AC OUT | score BAKE REF DIST OUT"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_tables_and_dc_only_survive_native_jpeg_roundtrip() -> Result<()> {
        let scratch = std::env::var_os("TMPDIR").context("explicit disk scratch required")?;
        let root = Path::new(&scratch).join(format!("r8-quant-fixture-{}", std::process::id()));
        ensure!(!root.exists(), "preserve previous fixture");
        fs::create_dir(&root)?;
        let source = root.join("source.png");
        let pixels: Vec<u8> = (0..17 * 19 * 3)
            .map(|i| ((i * 73 + i / 7) % 256) as u8)
            .collect();
        image::save_buffer(&source, &pixels, 17, 19, image::ColorType::Rgb8)?;
        for (dc, ac) in [
            (16, 32767),
            (64, 32767),
            (256, 32767),
            (16, 256),
            (16, 512),
            (16, 1024),
        ] {
            let cell = root.join(format!("{dc}-{ac}"));
            encode(&source, dc, ac, &cell)?;
            let row: serde_json::Value =
                serde_json::from_slice(&fs::read(cell.join("ENCODE.json"))?)?;
            assert_eq!(row["dc_quant"], dc);
            assert_eq!(row["ac_quant"], ac);
            assert_eq!(row["width"], 17);
            assert_eq!(row["height"], 19);
            if ac == 32767 {
                assert_eq!(row["nonzero_ac_coefficients"], 0);
            }
        }
        fs::remove_dir_all(root)?;
        Ok(())
    }
}
