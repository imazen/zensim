//! Public-API regression gallery — exercises every visual entry point
//! in `diff_image::*` plus `generators` to detect output changes
//! between branches.
//!
//! Run with:
//!
//! ```text
//! cargo run -p zensim-regress --example montage_regression -- /tmp/baseline
//! ```
//!
//! Diff against another run:
//!
//! ```text
//! cargo run -p zensim-regress --example montage_diff -- A B
//! ```

use std::env;
use std::path::PathBuf;

use zensim_regress::Bitmap;
use zensim_regress::diff_image::{
    AnnotationText, MontageOptions, create_montage, create_structural_montage, generate_diff_image,
    generate_structural_diff, pixelate_upscale,
};
use zensim_regress::generators;

fn main() {
    let dir: PathBuf = env::args()
        .nth(1)
        .unwrap_or_else(|| "/tmp/montage_regression".to_string())
        .into();
    std::fs::create_dir_all(&dir).expect("create output dir");

    // Scene 1: same-dim, no annotations
    {
        let exp = gradient(96, 96);
        let act = with_center_shift(&exp, 30, -20);
        let m = MontageOptions::default().render(&exp, &act, &AnnotationText::empty());
        m.save(dir.join("01_samedim_plain.png")).unwrap();
    }

    // Scene 2: same-dim with full annotation + heatmap
    {
        let exp = gradient(96, 96);
        let act = with_center_shift(&exp, 60, -40);
        let mut ann = AnnotationText::empty().with_title("scene 2 — annotated with heatmap");
        ann.primary_lines = vec![
            ("FAIL".to_string(), [255, 80, 80, 255]),
            ("zdsim 0.18 > 0.01 FAIL".to_string(), [255, 80, 80, 255]),
            (
                "max_delta [0,60,40,0] > 4 FAIL".to_string(),
                [255, 80, 80, 255],
            ),
        ];
        ann.extra = "alpha: max delta 0  •  pixels_differing 34.2%".to_string();
        let m = MontageOptions::default().render(&exp, &act, &ann);
        m.save(dir.join("02_samedim_annotated.png")).unwrap();
    }

    // Scene 3: mismatched dimensions
    {
        let exp = gradient(96, 64);
        let act = gradient(72, 96);
        let m = MontageOptions::default().render(&exp, &act, &AnnotationText::empty());
        m.save(dir.join("03_mismatched_plain.png")).unwrap();
    }

    // Scene 4: mismatched dims with annotation
    {
        let exp = gradient(96, 64);
        let act = gradient(72, 96);
        let mut ann = AnnotationText::empty().with_title("scene 4 — mismatched dims annotated");
        ann.primary_lines = vec![(
            "DIMENSION MISMATCH 96x64 vs 72x96".to_string(),
            [255, 80, 80, 255],
        )];
        ann.extra = "compared on shared 96x96 canvas".to_string();
        let m = MontageOptions::default().render(&exp, &act, &ann);
        m.save(dir.join("04_mismatched_annotated.png")).unwrap();
    }

    // Scene 5: tiny image (triggers pixelate-upscale)
    {
        let exp = gradient(16, 16);
        let act = with_center_shift(&exp, 80, -50);
        let m = MontageOptions::default().render(&exp, &act, &AnnotationText::empty());
        m.save(dir.join("05_tiny_pixelate.png")).unwrap();
    }

    // Scene 6: custom labels + heatmap disabled
    {
        let exp = gradient(96, 96);
        let act = with_center_shift(&exp, 40, -30);
        let mut opts = MontageOptions::default();
        opts.expected_label = Some("REFERENCE-LONG-NAME".to_string());
        opts.actual_label = Some("CANDIDATE".to_string());
        opts.show_spatial_heatmap = false;
        let m = opts.render(&exp, &act, &AnnotationText::empty());
        m.save(dir.join("06_custom_labels.png")).unwrap();
    }

    // ── Additional scenes exercising the rest of the public API ──────

    // Scene 7: large image (256×256) — different from 5/6 in that it
    // does NOT trigger pixelate-upscale, so the panels render at 1:1.
    {
        let exp = gradient(256, 256);
        let act = with_center_shift(&exp, 20, -20);
        let m = MontageOptions::default().render(&exp, &act, &AnnotationText::empty());
        m.save(dir.join("07_large.png")).unwrap();
    }

    // Scene 8: PASS annotation (green primary lines).
    {
        let exp = gradient(96, 96);
        let act = with_center_shift(&exp, 1, 0);
        let mut ann = AnnotationText::empty().with_title("scene 8 — passing");
        ann.primary_lines = vec![
            ("PASS".to_string(), [80, 220, 80, 255]),
            ("zdsim 0.001 ≤ 0.01 ok".to_string(), [80, 220, 80, 255]),
        ];
        let m = MontageOptions::default().render(&exp, &act, &ann);
        m.save(dir.join("08_passing.png")).unwrap();
    }

    // Scene 9: title-only annotation (no primary_lines, no extra).
    {
        let exp = gradient(96, 96);
        let act = with_center_shift(&exp, 30, 0);
        let ann = AnnotationText::empty()
            .with_title("title-only annotation, very long: lorem ipsum dolor sit amet");
        let m = MontageOptions::default().render(&exp, &act, &ann);
        m.save(dir.join("09_title_only.png")).unwrap();
    }

    // Scene 10: amplification sweep — 1, 5, 10, 50 across the same
    // tiny pixel diff. Tests `MontageOptions::amplification`.
    {
        let exp = gradient(96, 96);
        let act = with_center_shift(&exp, 5, 0);
        let mut panels: Vec<Bitmap> = Vec::new();
        for amp in [1u8, 5, 10, 50] {
            let mut opts = MontageOptions::default();
            opts.amplification = amp;
            opts.show_spatial_heatmap = false;
            let mut ann = AnnotationText::empty();
            ann.primary_lines = vec![(format!("amp={amp}"), [200, 200, 200, 255])];
            panels.push(opts.render(&exp, &act, &ann));
        }
        let strip = create_montage(&panels.iter().collect::<Vec<_>>(), 8);
        strip.save(dir.join("10_amplification_sweep.png")).unwrap();
    }

    // Scene 11: standalone diff image (no montage frame). Just the raw
    // amplified pixel difference.
    {
        let exp = gradient(192, 96);
        let act = with_center_shift(&exp, 40, -20);
        let diff = generate_diff_image(&exp, &act, 10);
        diff.save(dir.join("11_diff_only.png")).unwrap();
    }

    // Scene 12: standalone structural diff (cyan = structure missing,
    // orange = added). Watermark-style: small bright shape only in `act`.
    {
        let exp = gradient(192, 96);
        let mut act = exp.clone();
        // Stamp a small white square that only exists in `act`.
        for y in 30..50 {
            for x in 80..120 {
                act.put_pixel(x, y, [255, 255, 255, 255]);
            }
        }
        let s_diff = generate_structural_diff(&exp, &act, 3, 6);
        s_diff.save(dir.join("12_structural_diff.png")).unwrap();
    }

    // Scene 13: 4-panel structural montage (exp / act / pixel diff /
    // structural diff).
    {
        let exp = gradient(120, 96);
        let mut act = with_center_shift(&exp, 30, -20);
        // Add a structural change visible to structural diff.
        for y in 20..30 {
            for x in 80..110 {
                act.put_pixel(x, y, [255, 255, 255, 255]);
            }
        }
        let m = create_structural_montage(&exp, &act, 8, 4, 3);
        m.save(dir.join("13_structural_montage.png")).unwrap();
    }

    // Scene 14: pixelate_upscale — turn an 8×8 image into a chunky 64×64
    // preview without filtering.
    {
        let small = Bitmap::from_fn(8, 8, |x, y| {
            [
                (x * 32) as u8,
                (y * 32) as u8,
                (((x + y) * 16) % 256) as u8,
                255,
            ]
        });
        let big = pixelate_upscale(&small, 64);
        big.save(dir.join("14_pixelate_upscale.png")).unwrap();
    }

    // Scene 15: a strip of generators::* outputs — one panel per
    // synthetic generator. Exercises the public surface that produces
    // RGBA byte buffers (which we then wrap into Bitmap for the strip).
    {
        let dim = 96u32;
        let mk = |bytes: Vec<u8>| Bitmap::from_rgba_slice(&bytes, dim, dim).expect("rgba dims");
        let panels = [
            mk(generators::gradient(dim, dim)),
            mk(generators::checkerboard(dim, dim, 8)),
            mk(generators::value_noise(dim, dim, 42)),
            mk(generators::mandelbrot(dim, dim)),
            mk(generators::color_blocks(dim, dim)),
        ];
        let strip = create_montage(&panels.iter().collect::<Vec<_>>(), 4);
        strip.save(dir.join("15_generators_strip.png")).unwrap();
    }

    // Scene 16: off_by_n diff — a controlled per-pixel offset to
    // test that the diff visualization picks up scattered noise
    // versus blocked changes.
    {
        let dim = 128u32;
        let exp_bytes = generators::gradient(dim, dim);
        let act_bytes = generators::off_by_n(&exp_bytes, 8, 3);
        let exp = Bitmap::from_rgba_slice(&exp_bytes, dim, dim).unwrap();
        let act = Bitmap::from_rgba_slice(&act_bytes, dim, dim).unwrap();
        let mut ann = AnnotationText::empty().with_title("scene 16 — off_by_n(8, every 3rd)");
        ann.extra = "should look like uniform speckle".to_string();
        let m = MontageOptions::default().render(&exp, &act, &ann);
        m.save(dir.join("16_off_by_n.png")).unwrap();
    }

    println!("Wrote 16 scenes to {}", dir.display());
}

fn gradient(w: u32, h: u32) -> Bitmap {
    Bitmap::from_fn(w, h, |x, y| {
        [
            ((x * 255) / w.max(1)) as u8,
            ((y * 255) / h.max(1)) as u8,
            (((x + y) * 255) / (w + h).max(1)) as u8,
            255,
        ]
    })
}

fn with_center_shift(src: &Bitmap, dg: i16, db: i16) -> Bitmap {
    let (w, h) = (src.width(), src.height());
    Bitmap::from_fn(w, h, |x, y| {
        let p = src.get_pixel(x, y);
        let in_center = x > w / 4 && x < 3 * w / 4 && y > h / 4 && y < 3 * h / 4;
        if in_center {
            [
                p[0],
                p[1].saturating_add_signed(dg.try_into().unwrap_or(0)),
                p[2].saturating_add_signed(db.try_into().unwrap_or(0)),
                255,
            ]
        } else {
            p
        }
    })
}
