//! Approximate mappings between zensim scores and other image quality metrics.
//!
//! Piecewise linear interpolation of median metric values in zensim bins,
//! calibrated on ~344k synthetic pairs across 6 codecs
//! (mozjpeg, zenjpeg, zenjpeg-xyb, zenwebp, zenavif, zenjxl),
//! quality levels q5–q80, and ~3600 source tiles from 817 images
//! (CID22 + CLIC2025 + Kodak + gb82-sc).
//! Individual images can deviate significantly from these medians.
//!
//! # Training results (344,336 pairs, 156 features, 10 restarts x 50 iters)
//!
//! | Target         | Embedded SROCC | Trained SROCC | PLCC   | KROCC  |
//! |----------------|----------------|---------------|--------|--------|
//! | GPU SSIM2      | 0.9925         | 0.9931        | 0.9878 | 0.9296 |
//! | DSSIM          | 0.9897         | 0.9923        | 0.9790 | 0.9291 |
//! | GPU Butteraugli| 0.8886         | 0.9360        | 0.7802 | 0.8014 |
//!
//! # Mapping accuracy (metric-space MAE via interpolation tables)
//!
//! | Metric      | Points | MAE (metric) | p50   | p90   | p99    |
//! |-------------|--------|--------------|-------|-------|--------|
//! | SSIM2       | 330k   | 2.39 pts     | 1.55  | 5.55  | 13.07  |
//! | DSSIM       | 342k   | 0.0008       | 0.0003| 0.002 | 0.007  |
//! | Butteraugli | 342k   | 1.48 dist    | 0.59  | 3.67  | 13.15  |
//!
//! Butteraugli mapping improved 1.7x over power-law via interpolation tables.
//!
//! All functions use piecewise linear interpolation and clamp beyond
//! the calibration range.

/// Approximate SSIMULACRA2 score from a zensim score.
///
/// Median SSIM2 values in 2-point zensim bins from 330k synthetic pairs
/// (6 codecs, q5–q80). Metric-space MAE: 2.4 SSIM2 points.
pub fn zensim_to_ssim2(zensim: f64) -> f64 {
    if zensim >= 100.0 {
        return 100.0;
    }
    // Medians from 330k pairs. z=25 bin has 103 pts (sparse tail).
    const TABLE: [(f64, f64); 28] = [
        (25.0, 3.43),
        (35.0, 4.74),
        (45.0, 9.26),
        (50.0, 15.77),
        (52.0, 20.46),
        (54.0, 23.38),
        (56.0, 27.07),
        (58.0, 30.65),
        (60.0, 34.33),
        (62.0, 37.55),
        (64.0, 41.19),
        (66.0, 44.45),
        (68.0, 48.03),
        (70.0, 51.30),
        (72.0, 54.77),
        (74.0, 58.12),
        (76.0, 61.22),
        (78.0, 64.53),
        (80.0, 67.86),
        (82.0, 71.13),
        (84.0, 74.48),
        (86.0, 77.90),
        (88.0, 81.29),
        (90.0, 84.42),
        (92.0, 87.58),
        (94.0, 90.77),
        (97.0, 93.26),
        (98.0, 95.06),
    ];
    interp(&TABLE, zensim)
}

/// Approximate zensim score from an SSIMULACRA2 score.
///
/// Inverse of [`zensim_to_ssim2`]. Calibrated on 330k synthetic pairs.
pub fn ssim2_to_zensim(ssim2: f64) -> f64 {
    if ssim2 >= 100.0 {
        return 100.0;
    }
    const TABLE: [(f64, f64); 28] = [
        (3.43, 25.0),
        (4.74, 35.0),
        (9.26, 45.0),
        (15.77, 50.0),
        (20.46, 52.0),
        (23.38, 54.0),
        (27.07, 56.0),
        (30.65, 58.0),
        (34.33, 60.0),
        (37.55, 62.0),
        (41.19, 64.0),
        (44.45, 66.0),
        (48.03, 68.0),
        (51.30, 70.0),
        (54.77, 72.0),
        (58.12, 74.0),
        (61.22, 76.0),
        (64.53, 78.0),
        (67.86, 80.0),
        (71.13, 82.0),
        (74.48, 84.0),
        (77.90, 86.0),
        (81.29, 88.0),
        (84.42, 90.0),
        (87.58, 92.0),
        (90.77, 94.0),
        (93.26, 97.0),
        (95.06, 98.0),
    ];
    interp(&TABLE, ssim2)
}

/// Approximate butteraugli distance from a zensim score.
///
/// Median butteraugli values in 2-point zensim bins from 342k synthetic pairs
/// (6 codecs, q5–q80). Metric-space MAE: 1.5 BA distance units.
/// Note: butteraugli correlates poorly with zensim (SROCC 0.89),
/// so this mapping has high variance.
pub fn zensim_to_butteraugli(zensim: f64) -> f64 {
    if zensim >= 100.0 {
        return 0.0;
    }
    const TABLE: [(f64, f64); 31] = [
        (5.0, 17.6416),
        (15.0, 16.2473),
        (25.0, 13.6851),
        (35.0, 11.2909),
        (45.0, 9.1998),
        (50.0, 8.3043),
        (52.0, 8.0261),
        (54.0, 7.8366),
        (56.0, 7.6134),
        (58.0, 7.2178),
        (60.0, 6.7846),
        (62.0, 6.4197),
        (64.0, 6.1438),
        (66.0, 5.8747),
        (68.0, 5.6214),
        (70.0, 5.3178),
        (72.0, 4.9365),
        (74.0, 4.6026),
        (76.0, 4.2882),
        (78.0, 3.9873),
        (80.0, 3.7655),
        (82.0, 3.3744),
        (84.0, 2.9971),
        (86.0, 2.5488),
        (88.0, 2.1368),
        (90.0, 1.7876),
        (92.0, 1.5249),
        (94.0, 1.0968),
        (97.0, 0.6390),
        (98.0, 0.3143),
        (99.0, 0.2344),
    ];
    interp(&TABLE, zensim)
}

/// Approximate zensim score from a butteraugli distance.
///
/// Inverse of [`zensim_to_butteraugli`]. Calibrated on 342k synthetic pairs.
pub fn butteraugli_to_zensim(ba: f64) -> f64 {
    if ba <= 0.0 {
        return 100.0;
    }
    // Reversed: BA ascending, zensim descending
    const TABLE: [(f64, f64); 31] = [
        (0.2344, 99.0),
        (0.3143, 98.0),
        (0.6390, 97.0),
        (1.0968, 94.0),
        (1.5249, 92.0),
        (1.7876, 90.0),
        (2.1368, 88.0),
        (2.5488, 86.0),
        (2.9971, 84.0),
        (3.3744, 82.0),
        (3.7655, 80.0),
        (3.9873, 78.0),
        (4.2882, 76.0),
        (4.6026, 74.0),
        (4.9365, 72.0),
        (5.3178, 70.0),
        (5.6214, 68.0),
        (5.8747, 66.0),
        (6.1438, 64.0),
        (6.4197, 62.0),
        (6.7846, 60.0),
        (7.2178, 58.0),
        (7.6134, 56.0),
        (7.8366, 54.0),
        (8.0261, 52.0),
        (8.3043, 50.0),
        (9.1998, 45.0),
        (11.2909, 35.0),
        (13.6851, 25.0),
        (16.2473, 15.0),
        (17.6416, 5.0),
    ];
    interp(&TABLE, ba)
}

/// Approximate DSSIM value from a zensim score.
///
/// Median DSSIM values in 2-point zensim bins from 342k synthetic pairs
/// (6 codecs, q5–q80). Metric-space MAE: 0.0008.
pub fn zensim_to_dssim(zensim: f64) -> f64 {
    if zensim >= 100.0 {
        return 0.0;
    }
    const TABLE: [(f64, f64); 31] = [
        (5.0, 0.064946),
        (15.0, 0.051281),
        (25.0, 0.038654),
        (35.0, 0.028333),
        (45.0, 0.021334),
        (50.0, 0.017508),
        (52.0, 0.015817),
        (54.0, 0.014503),
        (56.0, 0.013279),
        (58.0, 0.012092),
        (60.0, 0.011084),
        (62.0, 0.009997),
        (64.0, 0.008978),
        (66.0, 0.008056),
        (68.0, 0.007170),
        (70.0, 0.006318),
        (72.0, 0.005586),
        (74.0, 0.004802),
        (76.0, 0.004095),
        (78.0, 0.003470),
        (80.0, 0.002890),
        (82.0, 0.002370),
        (84.0, 0.001881),
        (86.0, 0.001424),
        (88.0, 0.001018),
        (90.0, 0.000692),
        (92.0, 0.000402),
        (94.0, 0.000174),
        (97.0, 0.000061),
        (98.0, 0.000018),
        (99.0, 0.000012),
    ];
    interp(&TABLE, zensim)
}

/// Approximate zensim score from a DSSIM value.
///
/// Inverse of [`zensim_to_dssim`]. Calibrated on 342k synthetic pairs.
pub fn dssim_to_zensim(dssim: f64) -> f64 {
    if dssim <= 0.0 {
        return 100.0;
    }
    // Reversed: dssim ascending, zensim descending
    const TABLE: [(f64, f64); 31] = [
        (0.000012, 99.0),
        (0.000018, 98.0),
        (0.000061, 97.0),
        (0.000174, 94.0),
        (0.000402, 92.0),
        (0.000692, 90.0),
        (0.001018, 88.0),
        (0.001424, 86.0),
        (0.001881, 84.0),
        (0.002370, 82.0),
        (0.002890, 80.0),
        (0.003470, 78.0),
        (0.004095, 76.0),
        (0.004802, 74.0),
        (0.005586, 72.0),
        (0.006318, 70.0),
        (0.007170, 68.0),
        (0.008056, 66.0),
        (0.008978, 64.0),
        (0.009997, 62.0),
        (0.011084, 60.0),
        (0.012092, 58.0),
        (0.013279, 56.0),
        (0.014503, 54.0),
        (0.015817, 52.0),
        (0.017508, 50.0),
        (0.021334, 45.0),
        (0.028333, 35.0),
        (0.038654, 25.0),
        (0.051281, 15.0),
        (0.064946, 5.0),
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
            (s - 71.13).abs() < 1.0,
            "at zensim=82: expected ssim2~71.13, got {s}"
        );

        let b = zensim_to_butteraugli(82.0);
        assert!(
            (b - 3.37).abs() < 0.5,
            "at zensim=82: expected BA~3.37, got {b}"
        );

        let d = zensim_to_dssim(82.0);
        assert!(
            (d - 0.00237).abs() < 0.0005,
            "at zensim=82: expected dssim~0.00237, got {d}"
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
