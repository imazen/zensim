//! Approximate mappings between zensim scores and other image quality metrics.
//!
//! Piecewise linear interpolation of median metric values in zensim bins,
//! calibrated on ~344k synthetic pairs across 6 codecs
//! (mozjpeg, zenjpeg, zenjpeg-xyb, zenwebp, zenavif, zenjxl),
//! quality levels q5–q100, and ~3600 source tiles from 276 images.
//!
//! # Mapping accuracy (metric-space MAE via interpolation tables)
//!
//! | Metric      | Points | MAE (metric) | p50   | p90    | p99    |
//! |-------------|--------|--------------|-------|--------|--------|
//! | SSIM2       | 344k   | 4.7 pts      | 1.81  | 10.04  | 56.07  |
//! | DSSIM       | 344k   | 0.0021       | 0.0003| 0.003  | 0.039  |
//! | Butteraugli | 344k   | 1.64 dist    | 0.63  | 4.02   | 13.19  |
//!
//! All functions use piecewise linear interpolation and clamp beyond
//! the calibration range.

/// Approximate SSIMULACRA2 score from a zensim score.
///
/// Median SSIM2 values in zensim bins from 344k synthetic pairs
/// (6 codecs, q5–q100). Metric-space MAE: 4.7 SSIM2 points.
pub fn zensim_to_ssim2(zensim: f64) -> f64 {
    if zensim >= 100.0 {
        return 100.0;
    }
    const TABLE: [(f64, f64); 34] = [
        (5.0, 17.27),
        (10.0, 20.81),
        (15.0, 24.83),
        (20.0, 28.69),
        (25.0, 33.31),
        (30.0, 38.01),
        (35.0, 42.06),
        (40.0, 46.41),
        (45.0, 50.70),
        (50.0, 54.25),
        (52.0, 56.58),
        (54.0, 58.20),
        (56.0, 59.86),
        (58.0, 61.44),
        (60.0, 63.17),
        (62.0, 64.88),
        (64.0, 66.32),
        (66.0, 68.02),
        (68.0, 69.52),
        (70.0, 71.40),
        (72.0, 73.14),
        (74.0, 74.81),
        (76.0, 76.73),
        (78.0, 78.54),
        (80.0, 80.51),
        (82.0, 82.38),
        (84.0, 84.21),
        (86.0, 85.89),
        (88.0, 87.70),
        (90.0, 89.41),
        (92.0, 91.28),
        (94.0, 92.97),
        (96.0, 94.72),
        (98.0, 96.50),
    ];
    interp(&TABLE, zensim)
}

/// Approximate zensim score from an SSIMULACRA2 score.
///
/// Inverse of [`zensim_to_ssim2`]. Calibrated on 344k synthetic pairs.
pub fn ssim2_to_zensim(ssim2: f64) -> f64 {
    if ssim2 >= 100.0 {
        return 100.0;
    }
    const TABLE: [(f64, f64); 34] = [
        (17.27, 5.0),
        (20.81, 10.0),
        (24.83, 15.0),
        (28.69, 20.0),
        (33.31, 25.0),
        (38.01, 30.0),
        (42.06, 35.0),
        (46.41, 40.0),
        (50.70, 45.0),
        (54.25, 50.0),
        (56.58, 52.0),
        (58.20, 54.0),
        (59.86, 56.0),
        (61.44, 58.0),
        (63.17, 60.0),
        (64.88, 62.0),
        (66.32, 64.0),
        (68.02, 66.0),
        (69.52, 68.0),
        (71.40, 70.0),
        (73.14, 72.0),
        (74.81, 74.0),
        (76.73, 76.0),
        (78.54, 78.0),
        (80.51, 80.0),
        (82.38, 82.0),
        (84.21, 84.0),
        (85.89, 86.0),
        (87.70, 88.0),
        (89.41, 90.0),
        (91.28, 92.0),
        (92.97, 94.0),
        (94.72, 96.0),
        (96.50, 98.0),
    ];
    interp(&TABLE, ssim2)
}

/// Approximate butteraugli distance from a zensim score.
///
/// Median butteraugli values in zensim bins from 344k synthetic pairs
/// (6 codecs, q5–q100). Metric-space MAE: 1.6 BA distance units.
/// Note: butteraugli correlates poorly with zensim (SROCC ~0.89),
/// so this mapping has high variance.
pub fn zensim_to_butteraugli(zensim: f64) -> f64 {
    if zensim >= 100.0 {
        return 0.0;
    }
    const TABLE: [(f64, f64); 35] = [
        (5.0, 8.2421),
        (10.0, 7.9493),
        (15.0, 7.7553),
        (20.0, 7.4055),
        (25.0, 7.0157),
        (30.0, 6.5644),
        (35.0, 6.1886),
        (40.0, 5.7521),
        (45.0, 5.4422),
        (50.0, 5.1051),
        (52.0, 4.8197),
        (54.0, 4.5873),
        (56.0, 4.3262),
        (58.0, 4.3055),
        (60.0, 4.0261),
        (62.0, 3.9154),
        (64.0, 3.7726),
        (66.0, 3.5511),
        (68.0, 3.5063),
        (70.0, 3.3085),
        (72.0, 3.1613),
        (74.0, 2.9925),
        (76.0, 2.7094),
        (78.0, 2.4696),
        (80.0, 2.2480),
        (82.0, 2.0383),
        (84.0, 1.8193),
        (86.0, 1.6762),
        (88.0, 1.5714),
        (90.0, 1.2422),
        (92.0, 1.0073),
        (94.0, 0.7165),
        (96.0, 0.4050),
        (98.0, 0.2912),
        (99.0, 0.2344),
    ];
    interp(&TABLE, zensim)
}

/// Approximate zensim score from a butteraugli distance.
///
/// Inverse of [`zensim_to_butteraugli`]. Calibrated on 344k synthetic pairs.
pub fn butteraugli_to_zensim(ba: f64) -> f64 {
    if ba <= 0.0 {
        return 100.0;
    }
    const TABLE: [(f64, f64); 35] = [
        (0.2344, 99.0),
        (0.2912, 98.0),
        (0.4050, 96.0),
        (0.7165, 94.0),
        (1.0073, 92.0),
        (1.2422, 90.0),
        (1.5714, 88.0),
        (1.6762, 86.0),
        (1.8193, 84.0),
        (2.0383, 82.0),
        (2.2480, 80.0),
        (2.4696, 78.0),
        (2.7094, 76.0),
        (2.9925, 74.0),
        (3.1613, 72.0),
        (3.3085, 70.0),
        (3.5063, 68.0),
        (3.5511, 66.0),
        (3.7726, 64.0),
        (3.9154, 62.0),
        (4.0261, 60.0),
        (4.3055, 58.0),
        (4.3262, 56.0),
        (4.5873, 54.0),
        (4.8197, 52.0),
        (5.1051, 50.0),
        (5.4422, 45.0),
        (5.7521, 40.0),
        (6.1886, 35.0),
        (6.5644, 30.0),
        (7.0157, 25.0),
        (7.4055, 20.0),
        (7.7553, 15.0),
        (7.9493, 10.0),
        (8.2421, 5.0),
    ];
    interp(&TABLE, ba)
}

/// Approximate DSSIM value from a zensim score.
///
/// Median DSSIM values in zensim bins from 344k synthetic pairs
/// (6 codecs, q5–q100). Metric-space MAE: 0.0021.
pub fn zensim_to_dssim(zensim: f64) -> f64 {
    if zensim >= 100.0 {
        return 0.0;
    }
    const TABLE: [(f64, f64); 35] = [
        (5.0, 0.016200),
        (10.0, 0.014904),
        (15.0, 0.013534),
        (20.0, 0.012304),
        (25.0, 0.011101),
        (30.0, 0.009754),
        (35.0, 0.008635),
        (40.0, 0.007461),
        (45.0, 0.006497),
        (50.0, 0.005674),
        (52.0, 0.005170),
        (54.0, 0.004797),
        (56.0, 0.004438),
        (58.0, 0.004130),
        (60.0, 0.003794),
        (62.0, 0.003471),
        (64.0, 0.003172),
        (66.0, 0.002907),
        (68.0, 0.002650),
        (70.0, 0.002356),
        (72.0, 0.002098),
        (74.0, 0.001854),
        (76.0, 0.001588),
        (78.0, 0.001352),
        (80.0, 0.001119),
        (82.0, 0.000917),
        (84.0, 0.000725),
        (86.0, 0.000561),
        (88.0, 0.000405),
        (90.0, 0.000278),
        (92.0, 0.000156),
        (94.0, 0.000073),
        (96.0, 0.000021),
        (98.0, 0.000017),
        (99.0, 0.000012),
    ];
    interp(&TABLE, zensim)
}

/// Approximate zensim score from a DSSIM value.
///
/// Inverse of [`zensim_to_dssim`]. Calibrated on 344k synthetic pairs.
pub fn dssim_to_zensim(dssim: f64) -> f64 {
    if dssim <= 0.0 {
        return 100.0;
    }
    const TABLE: [(f64, f64); 35] = [
        (0.000012, 99.0),
        (0.000017, 98.0),
        (0.000021, 96.0),
        (0.000073, 94.0),
        (0.000156, 92.0),
        (0.000278, 90.0),
        (0.000405, 88.0),
        (0.000561, 86.0),
        (0.000725, 84.0),
        (0.000917, 82.0),
        (0.001119, 80.0),
        (0.001352, 78.0),
        (0.001588, 76.0),
        (0.001854, 74.0),
        (0.002098, 72.0),
        (0.002356, 70.0),
        (0.002650, 68.0),
        (0.002907, 66.0),
        (0.003172, 64.0),
        (0.003471, 62.0),
        (0.003794, 60.0),
        (0.004130, 58.0),
        (0.004438, 56.0),
        (0.004797, 54.0),
        (0.005170, 52.0),
        (0.005674, 50.0),
        (0.006497, 45.0),
        (0.007461, 40.0),
        (0.008635, 35.0),
        (0.009754, 30.0),
        (0.011101, 25.0),
        (0.012304, 20.0),
        (0.013534, 15.0),
        (0.014904, 10.0),
        (0.016200, 5.0),
    ];
    interp(&TABLE, dssim)
}

/// Convert a zensim score to a zendissim value.
///
/// zendissim is a perceptual dissimilarity metric on the same scale as DSSIM:
/// - 0.0 = identical images
/// - higher values = more dissimilar
///
/// Uses the same interpolation table as [`zensim_to_dssim`].
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
                (z - z2).abs() < 1.0,
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
                (z - z2).abs() < 1.0,
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
                (z - z2).abs() < 1.0,
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
                (z - z2).abs() < 1.0,
                "roundtrip failed: z={z} -> d={d} -> z2={z2}"
            );
        }
    }

    #[test]
    fn monotonicity() {
        // Higher zensim -> higher ssim2
        let s80 = zensim_to_ssim2(80.0);
        let s90 = zensim_to_ssim2(90.0);
        assert!(s90 > s80);

        // Higher zensim -> lower butteraugli
        let b80 = zensim_to_butteraugli(80.0);
        let b90 = zensim_to_butteraugli(90.0);
        assert!(b90 < b80);

        // Higher zensim -> lower dssim
        let d80 = zensim_to_dssim(80.0);
        let d90 = zensim_to_dssim(90.0);
        assert!(d90 < d80);

        // Higher zensim -> lower zendissim
        let zd80 = zensim_to_zendissim(80.0);
        let zd90 = zensim_to_zendissim(90.0);
        assert!(zd80 > zd90, "lower quality should have higher zendissim");
    }

    #[test]
    fn boundary_values() {
        // At zensim=100, ssim2=100
        assert_eq!(zensim_to_ssim2(100.0), 100.0);
        assert_eq!(ssim2_to_zensim(100.0), 100.0);

        // At zensim=100, BA=0
        assert_eq!(zensim_to_butteraugli(100.0), 0.0);
        assert_eq!(butteraugli_to_zensim(0.0), 100.0);

        // At zensim=100, dssim=0
        assert_eq!(zensim_to_dssim(100.0), 0.0);
        assert_eq!(dssim_to_zensim(0.0), 100.0);

        // zendissim same as dssim
        assert_eq!(zensim_to_zendissim(100.0), 0.0);
        assert_eq!(zendissim_to_zensim(0.0), 100.0);
    }

    #[test]
    fn known_calibration_points() {
        // From 344k synthetic pairs (6 codecs): median values at zensim=82
        let s = zensim_to_ssim2(82.0);
        assert!(
            (s - 82.38).abs() < 1.0,
            "at zensim=82: expected ssim2~82.38, got {s}"
        );

        let b = zensim_to_butteraugli(82.0);
        assert!(
            (b - 2.04).abs() < 0.5,
            "at zensim=82: expected BA~2.04, got {b}"
        );

        let d = zensim_to_dssim(82.0);
        assert!(
            (d - 0.000917).abs() < 0.0005,
            "at zensim=82: expected dssim~0.000917, got {d}"
        );
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
