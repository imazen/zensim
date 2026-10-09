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
        _ => None,
    }
}
