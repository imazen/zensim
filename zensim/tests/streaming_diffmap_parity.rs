//! Parity tests for `StreamingDiffmap`: verify it matches
//! `Zensim::compute_with_ref_and_diffmap` within ≤1e-4 across multiple
//! synthetic images, sizes, and strip alignments.

mod common;

use common::generators::{
    distort_block_artifacts, distort_blur, distort_color_shift, distort_sharpen, gen_checkerboard,
    gen_color_blocks, gen_mandelbrot, gen_value_noise,
};

use zensim::{
    DiffmapOptions, DiffmapWeighting, RgbSlice, StreamingDiffmap, Zensim, ZensimProfile,
};

const STRIP_ALIGNED: usize = 16; // STRIP_INNER from streaming.rs
const STRIP_OFFSET: usize = 7; // intentionally misaligned

type Case = (String, usize, usize, Vec<[u8; 3]>, Vec<[u8; 3]>);

/// Sweep test cases: (label, width, height, source pixels, distorted pixels).
fn cases() -> Vec<Case> {
    let sizes: &[(usize, usize)] = &[(256, 256), (600, 450), (320, 240)];

    let mut out = Vec::new();
    for &(w, h) in sizes {
        let mb = gen_mandelbrot(w, h);
        let mb_d = distort_blur(&mb, w, h, 2);
        out.push((format!("mandelbrot-{w}x{h}"), w, h, mb, mb_d));

        let cb = gen_checkerboard(w, h, 32);
        let cb_d = distort_block_artifacts(&cb, w, h);
        out.push((format!("checker-{w}x{h}"), w, h, cb, cb_d));

        let nz = gen_value_noise(w, h, 7);
        let nz_d = distort_sharpen(&nz, w, h);
        out.push((format!("noise-{w}x{h}"), w, h, nz, nz_d));

        let cl = gen_color_blocks(w, h);
        let cl_d = distort_color_shift(&cl, w, h);
        out.push((format!("color-{w}x{h}"), w, h, cl, cl_d));

        // Smooth gradient + tiny photometric shift (small absolute differences).
        let mut grad = vec![[0u8; 3]; w * h];
        for y in 0..h {
            for x in 0..w {
                let r = ((x * 255) / w.max(1)) as u8;
                let g = ((y * 255) / h.max(1)) as u8;
                let b = (((x + y) * 127) / (w + h).max(1)) as u8;
                grad[y * w + x] = [r, g, b];
            }
        }
        let grad_d: Vec<[u8; 3]> = grad
            .iter()
            .map(|p| [p[0].saturating_add(3), p[1].saturating_sub(2), p[2].saturating_add(1)])
            .collect();
        out.push((format!("grad-{w}x{h}"), w, h, grad, grad_d));

        // Identical pair — score should be ~100, diffmap ~0 everywhere.
        let id = gen_mandelbrot(w, h);
        let id_d = id.clone();
        out.push((format!("identical-{w}x{h}"), w, h, id, id_d));
    }
    out
}

fn drive_streaming(
    sd: &mut StreamingDiffmap<'_>,
    distorted: &RgbSlice<'_>,
    height: usize,
    strip_rows: usize,
) -> Option<zensim::StripContribution> {
    let mut last = None;
    let mut y = 0;
    while y < height {
        let r = strip_rows.min(height - y);
        last = sd.push_distorted_strip(distorted, y, r);
        y += r;
    }
    last
}

fn check_options(label: &str, opts: DiffmapOptions) {
    let z = Zensim::new(ZensimProfile::latest());
    for (case_label, w, h, src, dst) in cases() {
        let src_img = RgbSlice::new(&src, w, h);
        let dst_img = RgbSlice::new(&dst, w, h);
        let pre = z.precompute_reference(&src_img).unwrap();
        let reference = z
            .compute_with_ref_and_diffmap(&pre, &dst_img, opts)
            .expect("reference compute failed");

        for &strip_rows in &[STRIP_ALIGNED, STRIP_OFFSET] {
            let mut sd = StreamingDiffmap::new(&z, &pre, opts);
            let last = drive_streaming(&mut sd, &dst_img, h, strip_rows);
            assert!(
                last.is_some(),
                "{label}/{case_label}/strip={strip_rows}: final push returned None"
            );
            let result = sd.finalize();

            let dscore = (result.score() - reference.score()).abs();
            assert!(
                dscore < 1e-4,
                "{label}/{case_label}/strip={strip_rows}: score drift {dscore} (streaming {} vs ref {})",
                result.score(),
                reference.score()
            );

            let max_pixel_diff = reference
                .diffmap()
                .iter()
                .zip(result.diffmap())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_pixel_diff < 1e-4,
                "{label}/{case_label}/strip={strip_rows}: max diffmap diff {max_pixel_diff}"
            );
            assert_eq!(result.diffmap().len(), w * h);
            assert_eq!(result.width(), w);
            assert_eq!(result.height(), h);
        }
    }
}

#[test]
fn parity_default_options() {
    check_options("default", DiffmapOptions::default());
}

#[test]
fn parity_balanced_weighting() {
    check_options(
        "balanced",
        DiffmapOptions {
            weighting: DiffmapWeighting::Balanced,
            ..Default::default()
        },
    );
}

#[test]
fn parity_with_masking_and_sqrt() {
    check_options(
        "masking_sqrt",
        DiffmapOptions {
            weighting: DiffmapWeighting::Trained,
            masking_strength: Some(4.0),
            sqrt: true,
            include_edge_mse: false,
            include_hf: false,
        },
    );
}

/// Sanity-check at 1920x1080 with a single representative case to ensure
/// the streaming path also matches at production sizes.
#[test]
fn parity_full_hd() {
    let z = Zensim::new(ZensimProfile::latest());
    let w = 1920;
    let h = 1080;
    let src = gen_value_noise(w, h, 42);
    let dst = distort_blur(&src, w, h, 2);
    let src_img = RgbSlice::new(&src, w, h);
    let dst_img = RgbSlice::new(&dst, w, h);
    let pre = z.precompute_reference(&src_img).unwrap();
    let opts = DiffmapOptions {
        weighting: DiffmapWeighting::Trained,
        include_edge_mse: true,
        ..Default::default()
    };
    let reference = z
        .compute_with_ref_and_diffmap(&pre, &dst_img, opts)
        .unwrap();

    for &strip_rows in &[STRIP_ALIGNED, STRIP_OFFSET] {
        let mut sd = StreamingDiffmap::new(&z, &pre, opts);
        let _ = drive_streaming(&mut sd, &dst_img, h, strip_rows);
        let result = sd.finalize();
        let dscore = (result.score() - reference.score()).abs();
        assert!(
            dscore < 1e-4,
            "1080p strip={strip_rows}: score drift {dscore}"
        );
        let max_pixel_diff = reference
            .diffmap()
            .iter()
            .zip(result.diffmap())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_pixel_diff < 1e-4,
            "1080p strip={strip_rows}: max diffmap diff {max_pixel_diff}"
        );
    }
}

#[test]
fn parity_edge_mse_and_hf() {
    check_options(
        "edge_mse_hf",
        DiffmapOptions {
            weighting: DiffmapWeighting::Trained,
            masking_strength: None,
            sqrt: false,
            include_edge_mse: true,
            include_hf: true,
        },
    );
}
