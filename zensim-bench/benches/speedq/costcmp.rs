//! Extra COSTCMP arms; all scores use the existing public serving surfaces.
use serde_json::{Value, json};
use zenpredict::Model;
use zensim::{RgbSlice, Zensim, ZensimProfile};

type Action = Box<dyn FnMut() -> f64>;
const FAST_MAIN: &str = "09ec3e7c78bd230b1545572a188e162313c59fda";

#[allow(deprecated)] // Profile A is explicitly requested as the historical control.
pub(super) fn action(
    arm: &str,
    src: &'static [[u8; 3]],
    dst: &'static [[u8; 3]],
    w: usize,
    h: usize,
) -> Option<(Action, Value)> {
    match arm {
        "zensim_A" | "zensim_B" => {
            // These are the exact slots named by profile.rs and serving.rs.
            // Pin their bytes as metadata; scoring still calls Zensim::compute.
            let (profile, bytes, name): (ZensimProfile, &[u8], &str) = if arm == "zensim_A" {
                (
                    ZensimProfile::A,
                    include_bytes!(
                        "../../../zensim/weights/v47_strict_qat_native_byid_2026-09-06.bin"
                    ),
                    "A",
                )
            } else {
                (
                    ZensimProfile::B,
                    include_bytes!(
                        "../../../zensim/weights/b_sdr_linear_cid80_inclwinsor_dense_dial_byid_2026-09-06.bin"
                    ),
                    "B",
                )
            };
            let model = Model::from_bytes(bytes).unwrap();
            let declared = model.metadata().get_utf8("zentrain.formula_revision").ok();
            // Missing metadata is the registered shipped Rev1 era. No stamp.
            let revision = declared.unwrap_or("1");
            assert_eq!(revision, "1", "serving profile arithmetic changed");
            assert_eq!(std::env::var("ZENSIM_FORMULA_REV").unwrap(), revision);
            let info = json!({"source_sha256":super::digest(bytes),"revision":revision,
                "declared_revision":declared,"profile":name,"n_inputs":model.n_inputs(),
                "surface":"Zensim::compute; complete named serving profile"});
            let z = Zensim::new(profile).with_parallel(true);
            let action: Action = Box::new(move || {
                z.compute(&RgbSlice::new(src, w, h), &RgbSlice::new(dst, w, h))
                    .unwrap()
                    .score()
            });
            Some((action, info))
        }
        "fast_ssim2_main" => {
            let source = fast_ssim2_main::PixelSlice::new(
                bytemuck::cast_slice(src),
                w as u32,
                h as u32,
                w * 3,
                fast_ssim2_main::PixelDescriptor::RGB8_SRGB,
            )
            .unwrap();
            let distorted = fast_ssim2_main::PixelSlice::new(
                bytemuck::cast_slice(dst),
                w as u32,
                h as u32,
                w * 3,
                fast_ssim2_main::PixelDescriptor::RGB8_SRGB,
            )
            .unwrap();
            let action: Action = Box::new(move || {
                fast_ssim2_main::compute_ssimulacra2(&source, &distorted).unwrap()
            });
            Some((
                action,
                json!({"commit":FAST_MAIN,"version":"0.9.0","surface":"compute_ssimulacra2 RGB8 one-shot","rayon":true}),
            ))
        }
        "fast_ssim2" => {
            let action: Action = Box::new(move || {
                fast_ssim2::compute_ssimulacra2(
                    imgref::Img::new(src, w, h),
                    imgref::Img::new(dst, w, h),
                )
                .unwrap()
            });
            Some((
                action,
                json!({"version":"0.8.2","surface":"compute_ssimulacra2 RGB8 one-shot","rayon":cfg!(feature="ssim2-rayon")}),
            ))
        }
        "e33_a" | "e33_c" => {
            // E33 section 9.3: a registered full-data seed-0 candidate, pinned by path and SHA from the caller,
            // served through the same dynamic surface as production (BakeScorer, Rev5, Rayon enabled).
            let label = arm.trim_start_matches("e33_").to_uppercase();
            let path = std::env::var(format!("ZEN_S2_E33_BAKE_{label}"))
                .expect("pinned E33 candidate path");
            let expected =
                std::env::var(format!("ZEN_S2_E33_SHA_{label}")).expect("pinned E33 candidate SHA");
            let bytes = std::fs::read(path).unwrap();
            assert_eq!(
                super::digest(&bytes),
                expected,
                "E33 candidate bytes changed"
            );
            assert_eq!(
                std::env::var("ZENSIM_FORMULA_REV").unwrap(),
                "5",
                "E33 candidates serve Rev5"
            );
            let model = Box::leak(Box::new(Model::from_bytes(&bytes).unwrap()));
            assert_eq!(model.layers().next().unwrap().out_dim, 128, "H128 shape");
            assert_eq!(model.n_outputs(), 1);
            let mut scorer = zensim::BakeScorer::new(model).unwrap().with_parallel(true);
            let consumed = scorer.consumed_feature_ids().unwrap();
            let info = json!({"source_sha256":expected,"revision":"5","hidden":128,
                "consumed_ids":consumed.len(),"n_inputs":model.n_inputs(),"source_bytes":bytes.len(),
                "surface":"BakeScorer::compute; E33 registered candidate"});
            let action: Action = Box::new(move || {
                scorer
                    .compute(&RgbSlice::new(src, w, h), &RgbSlice::new(dst, w, h), None)
                    .unwrap()
                    .score()
            });
            Some((action, info))
        }
        _ => None,
    }
}
