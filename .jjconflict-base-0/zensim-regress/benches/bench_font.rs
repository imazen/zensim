//! Benchmarks for the font-strip rendering pipeline.
//!
//! Covers the hot paths exercised by `MontageOptions::render` and the
//! gallery: short labels at small char_h (heatmap stats), single-line
//! titles at medium char_h (panel labels), wrapped paragraphs at small
//! char_h (annotation extra lines), and the multi-color `lines_fitted`
//! mode (annotation primary lines).
//!
//! Reflects the per-cell `PixelSlice::crop_view` + `StreamingResize`
//! pipeline: each call resamples the 96 glyph cells of the embedded
//! strip independently and then composites the requested string with
//! gamma-correct linear-light alpha blending.

use std::hint::black_box;

use zensim_regress::font;

const WHITE: [u8; 4] = [255, 255, 255, 255];
const BG: [u8; 4] = [40, 40, 40, 255];
const BG_RED: [u8; 4] = [150, 10, 10, 255];

fn main() {
    let result = zenbench::run(|suite| {
        suite.compare("render_text_height_label", |group| {
            for &(label, char_h) in &[("char_h_8", 8u32), ("char_h_16", 16), ("char_h_24", 24)] {
                group.bench(label, move |b| {
                    b.iter(move || {
                        let (buf, w, h) = font::render_text_height(
                            black_box("EXPECTED"),
                            black_box(WHITE),
                            black_box(BG),
                            black_box(char_h),
                        );
                        black_box((buf.len(), w, h))
                    })
                });
            }
        });

        suite.compare("render_text_height_paragraph", |group| {
            const PARA: &str = "the quick brown fox jumps over the lazy dog";
            for &(label, char_h) in &[("char_h_8", 8u32), ("char_h_12", 12), ("char_h_16", 16)] {
                group.bench(label, move |b| {
                    b.iter(move || {
                        let (buf, w, h) = font::render_text_height(
                            black_box(PARA),
                            black_box(WHITE),
                            black_box(BG),
                            black_box(char_h),
                        );
                        black_box((buf.len(), w, h))
                    })
                });
            }
        });

        suite.compare("render_text_wrapped", |group| {
            const PARA: &str = "the quick brown fox jumps over the lazy dog. \
                                pack my box with five dozen liquor jugs. \
                                the five boxing wizards jump quickly.";
            for &(label, max_w, char_h) in &[
                ("80px_8h", 80u32, 8u32),
                ("160px_12h", 160, 12),
                ("400px_8h", 400, 8),
                ("400px_16h", 400, 16),
            ] {
                group.bench(label, move |b| {
                    b.iter(move || {
                        let (buf, w, h) = font::render_text_wrapped(
                            black_box(PARA),
                            black_box(WHITE),
                            black_box(BG),
                            black_box(char_h),
                            black_box(max_w),
                        );
                        black_box((buf.len(), w, h))
                    })
                });
            }
        });

        suite.compare("render_lines_fitted_heatmap_cell", |group| {
            // Mirrors heatmap-cell text shape: 2-3 short lines, narrow
            // width, multi-color.
            for &(label, max_w) in &[("60px", 60u32), ("120px", 120u32), ("240px", 240u32)] {
                group.bench(label, move |b| {
                    b.iter(move || {
                        let red = [255, 80, 80, 255];
                        let yellow = [200, 200, 120, 255];
                        let gray = [170, 170, 170, 255];
                        let lines: Vec<(&str, [u8; 4])> = vec![
                            ("100% \u{0394}30", red),
                            ("zdsim:0.065", gray),
                            ("changed", yellow),
                        ];
                        let (buf, w, h) = font::render_lines_fitted(
                            black_box(&lines),
                            black_box(BG_RED),
                            black_box(max_w),
                        );
                        black_box((buf.len(), w, h))
                    })
                });
            }
        });
    });

    if let Err(e) = result.save("font_results.json") {
        eprintln!("Failed to save results: {e}");
    }
}
