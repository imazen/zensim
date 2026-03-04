//! Approximate mappings between zensim scores and other image quality metrics.
//!
//! Power-law fits calibrated on ~344k synthetic pairs across 6 codecs
//! (mozjpeg, zenjpeg, zenjpeg-xyb, zenwebp, zenavif, zenjxl),
//! quality levels q5–q80, and ~3600 source tiles from 817 images
//! (CID22 + CLIC2025 + Kodak + gb82-sc).
//! Individual images can deviate significantly from these medians.
//!
//! # Training results (344,336 pairs, 156 features, 10 restarts × 50 iters)
//!
//! | Target         | Embedded SROCC | Trained SROCC | PLCC   | KROCC  |
//! |----------------|----------------|---------------|--------|--------|
//! | GPU SSIM2      | 0.9925         | 0.9931        | 0.9878 | 0.9296 |
//! | DSSIM          | 0.9897         | 0.9923        | 0.9790 | 0.9291 |
//! | GPU Butteraugli| 0.8886         | 0.9360        | 0.7802 | 0.8014 |
//!
//! # Mapping fit accuracy (power-law residuals in zensim points)
//!
//! | Metric      | Points | MAE  | RMSE  | p50  | p90   | p99   |
//! |-------------|--------|------|-------|------|-------|-------|
//! | SSIM2       | 330k   | 1.48 | 2.23  | 0.98 | 3.33  | 8.16  |
//! | DSSIM       | 342k   | 1.75 | 2.69  | 1.10 | 4.02  | 10.04 |
//! | Butteraugli | 342k   | 7.03 | 10.22 | 4.78 | 15.97 | 37.30 |
//!
//! All functions use power-law fits and support the full useful range.
//! Extrapolation beyond the calibration range is clamped.

/// Approximate SSIMULACRA2 score from a zensim score.
///
/// Fitted from: `(100-zensim) = 0.7290 * (100-ssim2)^0.9545`
/// Calibrated on 330k synthetic pairs (6 codecs, q5–q80).
/// Median absolute error: ~1.0 zensim points.
pub fn zensim_to_ssim2(zensim: f64) -> f64 {
    if zensim >= 100.0 {
        return 100.0;
    }
    let zd = (100.0 - zensim).max(0.0);
    let sd = (zd / 0.728961).powf(1.0 / 0.954532);
    100.0 - sd
}

/// Approximate zensim score from an SSIMULACRA2 score.
///
/// Fitted from: `(100-zensim) = 0.7290 * (100-ssim2)^0.9545`
/// Calibrated on 330k synthetic pairs (6 codecs, q5–q80).
pub fn ssim2_to_zensim(ssim2: f64) -> f64 {
    if ssim2 >= 100.0 {
        return 100.0;
    }
    let sd = (100.0 - ssim2).max(0.0);
    let zd = 0.728961 * sd.powf(0.954532);
    (100.0 - zd).max(0.0)
}

/// Approximate butteraugli distance from a zensim score.
///
/// Fitted from: `zensim = 100 - 10.1832 * BA^0.5886`
/// Calibrated on 342k synthetic pairs (6 codecs, q5–q80).
/// Note: butteraugli correlates poorly with zensim (SROCC 0.89),
/// so this mapping has high error (MAE ~7 zensim points).
pub fn zensim_to_butteraugli(zensim: f64) -> f64 {
    if zensim >= 100.0 {
        return 0.0;
    }
    let zd = (100.0 - zensim).max(0.0);
    (zd / 10.183166).powf(1.0 / 0.588649)
}

/// Approximate zensim score from a butteraugli distance.
///
/// Fitted from: `zensim = 100 - 10.1832 * BA^0.5886`
/// Calibrated on 342k synthetic pairs. High variance (MAE ~7 points).
pub fn butteraugli_to_zensim(ba: f64) -> f64 {
    if ba <= 0.0 {
        return 100.0;
    }
    (100.0 - 10.183166 * ba.powf(0.588649)).max(0.0)
}

/// Approximate DSSIM value from a zensim score.
///
/// Fitted from: `zensim = 100 - 362.188 * dssim^0.4915`
/// Calibrated on 342k synthetic pairs (6 codecs, q5–q80).
/// Median absolute error: ~1.1 zensim points.
pub fn zensim_to_dssim(zensim: f64) -> f64 {
    if zensim >= 100.0 {
        return 0.0;
    }
    let zd = (100.0 - zensim).max(0.0);
    (zd / 362.188050).powf(1.0 / 0.491459)
}

/// Approximate zensim score from a DSSIM value.
///
/// Fitted from: `zensim = 100 - 362.188 * dssim^0.4915`
/// Calibrated on 342k synthetic pairs (6 codecs, q5–q80).
pub fn dssim_to_zensim(dssim: f64) -> f64 {
    if dssim <= 0.0 {
        return 100.0;
    }
    (100.0 - 362.188050 * dssim.powf(0.491459)).max(0.0)
}

/// Convert a zensim score to a zendissim value.
///
/// zendissim is a perceptual dissimilarity metric on the same scale as DSSIM:
/// - 0.0 = identical images
/// - higher values = more dissimilar
///
/// Uses the same power-law mapping as [`zensim_to_dssim`].
pub fn zensim_to_zendissim(zensim: f64) -> f64 {
    zensim_to_dssim(zensim)
}

/// Convert a zendissim value to a zensim score.
///
/// Inverse of [`zensim_to_zendissim`].
pub fn zendissim_to_zensim(zendissim: f64) -> f64 {
    dssim_to_zensim(zendissim)
}

/// Approximate libjpeg-turbo quality from a zensim score (natural images).
///
/// This is a rough inverse of the empirical median mapping from JPEG quality
/// to zensim score on CID22+CLIC2025. Individual images vary widely.
///
/// Returns a value in 0-100 range. Accuracy: ±5 quality units.
pub fn zensim_to_libjpeg_quality(zensim: f64) -> f64 {
    // Piecewise linear interpolation of the empirical median table
    const TABLE: [(f64, f64); 16] = [
        (30.9, 5.0),
        (52.9, 10.0),
        (64.2, 15.0),
        (70.0, 20.0),
        (73.4, 25.0),
        (76.3, 30.0),
        (79.7, 40.0),
        (82.0, 50.0),
        (83.8, 60.0),
        (86.0, 70.0),
        (87.1, 75.0),
        (88.5, 80.0),
        (89.9, 85.0),
        (91.6, 90.0),
        (93.7, 95.0),
        (95.4, 98.0),
    ];
    interp(&TABLE, zensim)
}

/// Approximate zensim score from a libjpeg-turbo quality (natural images).
///
/// Median zensim score across CID22+CLIC2025.
pub fn libjpeg_quality_to_zensim(quality: f64) -> f64 {
    const TABLE: [(f64, f64); 16] = [
        (5.0, 30.9),
        (10.0, 52.9),
        (15.0, 64.2),
        (20.0, 70.0),
        (25.0, 73.4),
        (30.0, 76.3),
        (40.0, 79.7),
        (50.0, 82.0),
        (60.0, 83.8),
        (70.0, 86.0),
        (75.0, 87.1),
        (80.0, 88.5),
        (85.0, 89.9),
        (90.0, 91.6),
        (95.0, 93.7),
        (98.0, 95.4),
    ];
    interp(&TABLE, quality)
}

/// Approximate zenjpeg quality (0-100) from a zensim score (natural images).
///
/// zenjpeg uses adaptive quantization + trellis, so its quality scale differs
/// from libjpeg-turbo. At low quality, zenjpeg produces much better results
/// per quality unit. At q40+, they converge within ~1 zensim point.
///
/// Calibrated with zenjpeg 0.6.1, YCbCr 4:2:0, auto_optimize=true.
pub fn zensim_to_zenjpeg_quality(zensim: f64) -> f64 {
    const TABLE: [(f64, f64); 21] = [
        (52.8, 0.0),
        (59.0, 5.0),
        (64.8, 10.0),
        (69.7, 15.0),
        (73.2, 20.0),
        (75.6, 25.0),
        (77.3, 30.0),
        (78.2, 35.0),
        (78.9, 40.0),
        (79.6, 45.0),
        (81.6, 50.0),
        (82.4, 55.0),
        (83.3, 60.0),
        (84.4, 65.0),
        (85.5, 70.0),
        (86.5, 75.0),
        (87.9, 80.0),
        (89.4, 85.0),
        (91.2, 90.0),
        (93.3, 95.0),
        (95.0, 100.0),
    ];
    interp(&TABLE, zensim)
}

/// Approximate zensim score from a zenjpeg quality (0-100, natural images).
///
/// Median across CID22+CLIC2025 (71 images).
/// Calibrated with zenjpeg 0.6.1, YCbCr 4:2:0, auto_optimize=true.
pub fn zenjpeg_quality_to_zensim(quality: f64) -> f64 {
    const TABLE: [(f64, f64); 21] = [
        (0.0, 52.8),
        (5.0, 59.0),
        (10.0, 64.8),
        (15.0, 69.7),
        (20.0, 73.2),
        (25.0, 75.6),
        (30.0, 77.3),
        (35.0, 78.2),
        (40.0, 78.9),
        (45.0, 79.6),
        (50.0, 81.6),
        (55.0, 82.4),
        (60.0, 83.3),
        (65.0, 84.4),
        (70.0, 85.5),
        (75.0, 86.5),
        (80.0, 87.9),
        (85.0, 89.4),
        (90.0, 91.2),
        (95.0, 93.3),
        (100.0, 95.0),
    ];
    interp(&TABLE, quality)
}

/// Piecewise linear interpolation. Clamps to table bounds.
fn interp(table: &[(f64, f64)], x: f64) -> f64 {
    if table.is_empty() {
        return 0.0;
    }
    if x <= table[0].0 {
        return table[0].1;
    }
    if x >= table[table.len() - 1].0 {
        return table[table.len() - 1].1;
    }
    for i in 1..table.len() {
        if x <= table[i].0 {
            let (x0, y0) = table[i - 1];
            let (x1, y1) = table[i];
            let t = (x - x0) / (x1 - x0);
            return y0 + t * (y1 - y0);
        }
    }
    table[table.len() - 1].1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_ssim2() {
        for z in [30.0, 50.0, 70.0, 80.0, 90.0, 95.0] {
            let s = zensim_to_ssim2(z);
            let z2 = ssim2_to_zensim(s);
            assert!(
                (z - z2).abs() < 0.1,
                "roundtrip failed: z={z} -> s={s} -> z2={z2}"
            );
        }
    }

    #[test]
    fn roundtrip_butteraugli() {
        for z in [40.0, 60.0, 70.0, 80.0, 90.0, 95.0] {
            let b = zensim_to_butteraugli(z);
            let z2 = butteraugli_to_zensim(b);
            assert!(
                (z - z2).abs() < 0.1,
                "roundtrip failed: z={z} -> b={b} -> z2={z2}"
            );
        }
    }

    #[test]
    fn roundtrip_dssim() {
        for z in [30.0, 50.0, 70.0, 80.0, 90.0, 95.0] {
            let d = zensim_to_dssim(z);
            let z2 = dssim_to_zensim(d);
            assert!(
                (z - z2).abs() < 0.1,
                "roundtrip failed: z={z} -> d={d} -> z2={z2}"
            );
        }
    }

    #[test]
    fn roundtrip_zendissim() {
        for z in [30.0, 50.0, 70.0, 80.0, 90.0, 95.0] {
            let d = zensim_to_zendissim(z);
            let z2 = zendissim_to_zensim(d);
            assert!(
                (z - z2).abs() < 0.1,
                "roundtrip failed: z={z} -> d={d} -> z2={z2}"
            );
        }
    }

    #[test]
    fn monotonicity() {
        // Higher zensim → higher ssim2
        let s80 = zensim_to_ssim2(80.0);
        let s90 = zensim_to_ssim2(90.0);
        assert!(s90 > s80);

        // Higher zensim → lower butteraugli
        let b80 = zensim_to_butteraugli(80.0);
        let b90 = zensim_to_butteraugli(90.0);
        assert!(b90 < b80);

        // Higher zensim → lower dssim
        let d80 = zensim_to_dssim(80.0);
        let d90 = zensim_to_dssim(90.0);
        assert!(d90 < d80);

        // Higher zensim → lower zendissim
        let zd80 = zensim_to_zendissim(80.0);
        let zd90 = zensim_to_zendissim(90.0);
        assert!(zd80 > zd90, "lower quality should have higher zendissim");
    }

    #[test]
    fn boundary_values() {
        assert_eq!(zensim_to_ssim2(100.0), 100.0);
        assert_eq!(ssim2_to_zensim(100.0), 100.0);
        assert_eq!(zensim_to_butteraugli(100.0), 0.0);
        assert_eq!(butteraugli_to_zensim(0.0), 100.0);
        assert_eq!(zensim_to_dssim(100.0), 0.0);
        assert_eq!(dssim_to_zensim(0.0), 100.0);
        assert_eq!(zensim_to_zendissim(100.0), 0.0);
        assert_eq!(zendissim_to_zensim(0.0), 100.0);
    }

    #[test]
    fn libjpeg_quality_mapping() {
        // Spot checks from empirical data
        let z50 = libjpeg_quality_to_zensim(50.0);
        assert!((z50 - 82.0).abs() < 1.0, "q50 should give ~82, got {z50}");

        let z90 = libjpeg_quality_to_zensim(90.0);
        assert!((z90 - 91.6).abs() < 1.0, "q90 should give ~91.6, got {z90}");

        // Roundtrip
        let q = zensim_to_libjpeg_quality(85.0);
        let z = libjpeg_quality_to_zensim(q);
        assert!((z - 85.0).abs() < 1.0, "roundtrip: 85 -> q={q} -> z={z}");
    }

    #[test]
    fn known_calibration_points() {
        // From 344k synthetic pairs (6 codecs): at zensim=82
        let s = zensim_to_ssim2(82.0);
        assert!(
            (s - 71.2).abs() < 3.0,
            "at zensim=82: expected ssim2~71.2, got {s}"
        );

        let b = zensim_to_butteraugli(82.0);
        assert!(
            (b - 2.6).abs() < 1.5,
            "at zensim=82: expected BA~2.6, got {b}"
        );

        let d = zensim_to_dssim(82.0);
        assert!(
            (d - 0.0022).abs() < 0.002,
            "at zensim=82: expected dssim~0.0022, got {d}"
        );
    }

    #[test]
    fn zenjpeg_quality_mapping() {
        // Spot checks from empirical medians
        let z50 = zenjpeg_quality_to_zensim(50.0);
        assert!(
            (z50 - 81.6).abs() < 1.0,
            "zenjpeg q50 should give ~81.6, got {z50}"
        );

        let z90 = zenjpeg_quality_to_zensim(90.0);
        assert!(
            (z90 - 91.2).abs() < 1.0,
            "zenjpeg q90 should give ~91.2, got {z90}"
        );

        // Roundtrip
        let q = zensim_to_zenjpeg_quality(85.0);
        let z = zenjpeg_quality_to_zensim(q);
        assert!((z - 85.0).abs() < 1.0, "roundtrip: 85 -> q={q} -> z={z}");

        // zenjpeg q0 should map to ~52.8 (not zero)
        let z0 = zenjpeg_quality_to_zensim(0.0);
        assert!(
            (z0 - 52.8).abs() < 0.1,
            "zenjpeg q0 should give ~52.8, got {z0}"
        );

        // zenjpeg q100 should map to ~95.0
        let z100 = zenjpeg_quality_to_zensim(100.0);
        assert!(
            (z100 - 95.0).abs() < 0.1,
            "zenjpeg q100 should give ~95.0, got {z100}"
        );
    }

    #[test]
    fn zenjpeg_vs_libjpeg_convergence() {
        // At high quality, zenjpeg and libjpeg converge to similar zensim scores.
        // At q90+, they should be within ~2 zensim points of each other.
        let lj90 = libjpeg_quality_to_zensim(90.0);
        let zj90 = zenjpeg_quality_to_zensim(90.0);
        assert!(
            (lj90 - zj90).abs() < 2.0,
            "at q90, libjpeg ({lj90}) and zenjpeg ({zj90}) should converge"
        );

        // At low quality, zenjpeg should be dramatically better
        let lj10 = libjpeg_quality_to_zensim(10.0);
        let zj10 = zenjpeg_quality_to_zensim(10.0);
        assert!(
            zj10 > lj10 + 5.0,
            "at q10, zenjpeg ({zj10}) should beat libjpeg ({lj10}) by >5 pts"
        );
    }
}
