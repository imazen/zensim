//! CHdr through the PU-linear HDR entry points (CHDR_PU, 2026-10-10).
//!
//! `ZensimProfile::CHdr` is the HDR candidate-of-record and declares 697 ids
//! of the 944 folded layout. Its training and validation features came from
//! the fleet's `zensim-foldapp2` HDR route: zenmetrics
//! `hdr::hdr_foldapp_features`, which calls
//! [`Zensim::compute_folded720_append2_features_hdr`] with
//! `HdrEncoding::Linear` (absolute cd/m²) and default toggles
//! (`benchmarks/hdr944_bake_wave_2026-08-27.md`, "hdrfeat944 (944-regime,
//! `zensim-foldapp2`, HdrEncoding::Linear arms)"). The PU entries must serve
//! CHdr, every input layout must agree bit for bit, and the served features
//! and score must equal that validation extraction.
#![cfg(feature = "candidate-profiles")]

use zensim::feature_v2::{HdrEncoding, V2NewFeatureToggles, V2Scratch};
use zensim::source::{AlphaMode, ImageSource, PixelFormat};
use zensim::{Zensim, ZensimProfile};

const CHDR_BAKE: &[u8] = include_bytes!("../weights/c_hdr_l1t1944_byid_2026-09-07.bin");

/// Absolute-linear cd/m² RGBA f32 source: the zenmetrics extractor's adapter
/// shape (`LinearF32Rgba`, opaque, declared HDR).
struct Nits {
    bytes: Vec<u8>,
    w: usize,
    h: usize,
}

impl Nits {
    fn new(rgb: &[f32], w: usize, h: usize) -> Self {
        let mut bytes = Vec::with_capacity(w * h * 16);
        for p in rgb.as_chunks::<3>().0 {
            for v in [p[0], p[1], p[2], 1.0f32] {
                bytes.extend_from_slice(&v.to_ne_bytes());
            }
        }
        Self { bytes, w, h }
    }
}

impl ImageSource for Nits {
    fn width(&self) -> usize {
        self.w
    }
    fn height(&self) -> usize {
        self.h
    }
    fn pixel_format(&self) -> PixelFormat {
        PixelFormat::LinearF32Rgba
    }
    fn alpha_mode(&self) -> AlphaMode {
        AlphaMode::Opaque
    }
    fn is_hdr(&self) -> bool {
        true
    }
    fn row_bytes(&self, y: usize) -> &[u8] {
        &self.bytes[y * self.w * 16..(y + 1) * self.w * 16]
    }
}

/// Deterministic synthetic HDR pair: a 0.1–1,800 cd/m² ramp with texture,
/// and a distortion that brightens every seventh sample by 8%.
fn pair(w: usize, h: usize, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let mut s = seed;
    let mut rnd = || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 40) as f32 / (1u64 << 24) as f32
    };
    let r: Vec<f32> = (0..w * h * 3)
        .map(|i| {
            let x = (i / 3) % w;
            let base = 0.1 + 1200.0 * (x as f32 / w as f32).powi(2);
            base * (0.5 + rnd())
        })
        .collect();
    let d = r
        .iter()
        .enumerate()
        .map(|(i, v)| if i % 7 == 0 { v * 1.08 } else { *v })
        .collect();
    (r, d)
}

fn planes(rgb: &[f32], w: usize, h: usize, stride: usize) -> [Vec<f32>; 3] {
    core::array::from_fn(|c| {
        let mut p = vec![-1.0f32; stride * h];
        for y in 0..h {
            for x in 0..w {
                p[y * stride + x] = rgb[(y * w + x) * 3 + c];
            }
        }
        p
    })
}

fn padded(rgb: &[f32], w: usize, h: usize, stride: usize) -> Vec<f32> {
    let mut v = vec![-1.0f32; stride * h];
    for y in 0..h {
        v[y * stride..y * stride + 3 * w].copy_from_slice(&rgb[y * 3 * w..(y + 1) * 3 * w]);
    }
    v
}

#[test]
fn chdr_scores_through_the_pu_entries_and_matches_its_validation_extraction() {
    let model = zenpredict::Model::from_bytes(CHDR_BAKE).expect("CHdr bake parses");
    let declared = zensim::declared_feature_ids(&model).expect("CHdr declares explicit ids");
    assert_eq!(declared.len(), 697);
    for (w, h, seed) in [
        (17usize, 19usize, 0x9e37_79b9_7f4a_7c15u64),
        (64, 64, 0x2545_f491_4f6c_dd1d),
        (97, 65, 0x1234_5678_9abc_def1),
        (256, 128, 0x0bad_c0de_dead_beef),
    ] {
        let tag = format!("{w}x{h}");
        let (r, d) = pair(w, h, seed);
        let z = Zensim::new(ZensimProfile::CHdr);

        let served = z
            .compute_pu_linear(&r, &d, w, h, 3 * w, 3 * w)
            .unwrap_or_else(|e| panic!("{tag}: CHdr refused by compute_pu_linear: {e:?}"));
        let score = served.score();
        assert!(score.is_finite(), "{tag}");

        // Every input layout serves the same bits.
        let (rs, ds) = (3 * w + 5, 3 * w + 5);
        let strided = z
            .compute_pu_linear(&padded(&r, w, h, rs), &padded(&d, w, h, ds), w, h, rs, ds)
            .unwrap();
        let (rp, dp) = (planes(&r, w, h, w), planes(&d, w, h, w));
        let planar = z
            .compute_pu_linear_planar([&rp[0], &rp[1], &rp[2]], [&dp[0], &dp[1], &dp[2]], w, h, w)
            .unwrap_or_else(|e| panic!("{tag}: CHdr refused by compute_pu_linear_planar: {e:?}"));
        let ps = w + 3;
        let (rq, dq) = (planes(&r, w, h, ps), planes(&d, w, h, ps));
        let planar_strided = z
            .compute_pu_linear_planar([&rq[0], &rq[1], &rq[2]], [&dq[0], &dq[1], &dq[2]], w, h, ps)
            .unwrap();
        let descriptor = z
            .compute(&Nits::new(&r, w, h), &Nits::new(&d, w, h))
            .unwrap();
        for (name, other) in [
            ("strided interleaved", &strided),
            ("planar", &planar),
            ("strided planar", &planar_strided),
            ("descriptor-HDR compute", &descriptor),
        ] {
            assert_eq!(
                other.score().to_bits(),
                score.to_bits(),
                "{tag}: {name} disagrees with interleaved"
            );
            assert_eq!(
                other.features(),
                served.features(),
                "{tag}: {name} features"
            );
        }

        // The validation extraction: zenmetrics `hdr_foldapp_features`.
        let mut scratch = V2Scratch::new();
        let reference = z
            .compute_folded720_append2_features_hdr(
                &Nits::new(&r, w, h),
                &Nits::new(&d, w, h),
                HdrEncoding::Linear,
                V2NewFeatureToggles::default(),
                &mut scratch,
            )
            .unwrap()
            .into_features();
        assert_eq!(reference.len(), 944);
        for &id in &declared {
            let id = usize::from(id);
            assert_eq!(
                served.features()[id].to_bits(),
                reference[id].to_bits(),
                "{tag}: served f{id} {} vs validation extraction {}",
                served.features()[id],
                reference[id]
            );
        }
        let recomputed = zensim::score_features_with_profile(
            ZensimProfile::CHdr,
            &reference,
            w as u32,
            h as u32,
        )
        .unwrap();
        assert_eq!(
            score.to_bits(),
            recomputed.to_bits(),
            "{tag}: served {score} vs feature-path recomputation {recomputed}"
        );

        // Identity stays exactly 100 on the PU entry.
        assert_eq!(
            z.compute_pu_linear(&r, &r, w, h, 3 * w, 3 * w)
                .unwrap()
                .score(),
            100.0
        );
    }
}
