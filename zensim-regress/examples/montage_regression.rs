//! Regression-comparison harness for the montage compositor.
//!
//! Renders six representative montages — same-dim plain, same-dim
//! annotated, mismatched-dim plain, mismatched-dim annotated, tiny
//! image (pixelate-upscale), and custom labels — into the directory
//! given by the first arg. Used to confirm the layout-module port
//! preserves visual output.
//!
//! ```text
//! cargo run -p zensim-regress --example montage_regression -- /tmp/baseline
//! ```

use std::env;
use std::path::PathBuf;

use image::{Rgba, RgbaImage};
use zensim_regress::diff_image::{AnnotationText, MontageOptions};

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

    println!("Wrote 6 scenes to {}", dir.display());
}

fn gradient(w: u32, h: u32) -> RgbaImage {
    RgbaImage::from_fn(w, h, |x, y| {
        Rgba([
            ((x * 255) / w.max(1)) as u8,
            ((y * 255) / h.max(1)) as u8,
            (((x + y) * 255) / (w + h).max(1)) as u8,
            255,
        ])
    })
}

fn with_center_shift(src: &RgbaImage, dg: i16, db: i16) -> RgbaImage {
    let (w, h) = (src.width(), src.height());
    RgbaImage::from_fn(w, h, |x, y| {
        let p = src.get_pixel(x, y);
        let in_center = x > w / 4 && x < 3 * w / 4 && y > h / 4 && y < 3 * h / 4;
        if in_center {
            Rgba([
                p[0],
                p[1].saturating_add_signed(dg.try_into().unwrap_or(0)),
                p[2].saturating_add_signed(db.try_into().unwrap_or(0)),
                255,
            ])
        } else {
            *p
        }
    })
}
