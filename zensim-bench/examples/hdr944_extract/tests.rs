use super::*;
#[test]
fn e26_native_readset_matches_canonical_hdr_and_refuses_sdr() {
    if std::env::var("E26_HDR_TEST_CHILD").is_err() {
        let result = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "tests::e26_native_readset_matches_canonical_hdr_and_refuses_sdr",
                "--nocapture",
            ])
            .env("E26_HDR_TEST_CHILD", "1")
            .env("ZENSIM_FORMULA_REV", "5")
            .env("ZENSIM_ROOT_FORM", "sqrt")
            .output()
            .unwrap();
        assert!(
            result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stderr)
        );
        return;
    }
    let registry: serde_json::Value = serde_json::from_str(include_str!(
        "../../../benchmarks/costset2_2026-10-03.candidate_ids.json"
    ))
    .unwrap();
    let ids: Vec<usize> = registry["candidates"]["by_v2fy"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(ids.len(), 420);
    let req = zensim::research::Request::for_slots(
        zensim::feature_set_id::SlotSet::from_slots(ids.iter().copied()),
        zensim::research::full_width(),
    );
    let encoding = HdrEncoding::Pq { peak_nits: 10000.0 };
    for (w, h) in [(16, 16), (65, 97)] {
        let refpx: Vec<_> = (0..w * h)
            .map(|i| [10000 + (i % 9000) as u16, 20000 + (i % 5000) as u16, 30000])
            .collect();
        let mut distpx = refpx.clone();
        for (i, pixel) in distpx.iter_mut().enumerate() {
            pixel[i % 3] += 37;
        }
        let src = Pq16Image::from_rgb16(&refpx, w, h, zensim::ColorPrimaries::Srgb);
        let dst = Pq16Image::from_rgb16(&distpx, w, h, zensim::ColorPrimaries::Srgb);
        let extracted = zensim::research::extract_hdr(&req, &src, &dst, encoding).unwrap();
        let full = Zensim::new(ZensimProfile::codec_target())
            .with_parallel(false)
            .compute_folded720_features_hdr(
                &src,
                &dst,
                encoding,
                V2NewFeatureToggles {
                    v1_pools: zensim::feature_v2::V1PoolsMode::Peaks,
                    ..Default::default()
                },
                &mut V2Scratch::new(),
            )
            .unwrap();
        for &id in &ids {
            assert_eq!(
                extracted.values()[id].to_bits(),
                full.features()[id].to_bits(),
                "f{id} {w}x{h}"
            );
        }
        assert!(zensim::research::extract(&req, &src, &dst).is_err());
    }
}

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
