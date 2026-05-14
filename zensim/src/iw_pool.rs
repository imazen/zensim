//! Information-content-weighted spatial pooling (Wang & Li 2011, IW-SSIM).
//!
//! Replaces uniform spatial-mean pooling of per-pixel SSIM-like maps with
//! pooling weighted by the local information content of the reference
//! image. The motivation (from the source paper, IEEE TIP vol. 20 no. 5):
//! human observers concentrate attention on perceptually-informative
//! regions, so a metric that uniform-pools across the image dilutes
//! signal from salient regions with signal from flat / uninformative
//! regions. Weighting by per-region info content gives those regions more
//! influence on the aggregate score.
//!
//! ## Per-pixel weight estimator
//!
//! Wang 2011's exact form uses a Gaussian Scale Mixture model on per-band
//! wavelet coefficients. We use the practical approximation defended in
//! the same paper: **local variance of the reference image** is a
//! computationally cheap proxy with the same information-theoretic
//! direction (high variance = high information content per pixel).
//! Other candidate estimators (configurable via [`IwWeightKind`]):
//!
//! - [`IwWeightKind::LocalVariance`] (default) — variance in a 5×5 window.
//! - [`IwWeightKind::LocalGradL1`] — L1 norm of the gradient (∂x + ∂y).
//! - [`IwWeightKind::LocalGradL2`] — L2 norm of the gradient (√(∂x² + ∂y²)).
//!
//! All three return non-negative per-pixel weights. They are NOT
//! normalised here — the [`WeightedPool`] helper handles normalisation
//! at aggregate time.
//!
//! ## Status (V0_20a, 2026-05-14)
//!
//! Initial implementation. Designed for offline experimentation —
//! correctness over performance. Once the V0_20a sweep shows the
//! Wang 2011 paper claim is reproduced for our corpus, the pool
//! integration migrates into the SIMD streaming loop at
//! `streaming::process_scale_bands`.

/// Choice of per-pixel info-content estimator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IwWeightKind {
    /// Local variance in a square window (kernel size set by config).
    /// Default. Closest to Wang 2011's GSM scale parameter direction.
    LocalVariance,
    /// L1 norm of the gradient — `|∂x I| + |∂y I|`. Cheaper than variance,
    /// emphasises edges.
    LocalGradL1,
    /// L2 norm of the gradient — `√((∂x I)² + (∂y I)²)`.
    LocalGradL2,
}

impl Default for IwWeightKind {
    fn default() -> Self {
        Self::LocalVariance
    }
}

/// IW weight computation parameters.
#[derive(Debug, Clone, Copy)]
pub struct IwWeightConfig {
    /// Which estimator to use.
    pub kind: IwWeightKind,
    /// Half-width of the local window for `LocalVariance`. Window size
    /// is `(2k+1) × (2k+1)` — `kernel_half = 2` gives a 5×5 window.
    /// Ignored for gradient estimators.
    pub kernel_half: usize,
    /// Floor added to all weights so flat regions still contribute
    /// non-zero signal. Without it the weighted mean is undefined
    /// when the reference is locally constant. Default: 1e-3 of the
    /// dynamic range.
    pub weight_floor: f32,
}

impl Default for IwWeightConfig {
    fn default() -> Self {
        Self {
            kind: IwWeightKind::LocalVariance,
            kernel_half: 2,
            weight_floor: 1.0e-3,
        }
    }
}

/// Compute per-pixel info-content weights from the reference image.
/// Returns a buffer of length `width * height` containing non-negative
/// weights. Padding rows / columns (beyond `width × height`) are NOT
/// processed.
pub fn compute_iw_weights(
    ref_plane: &[f32],
    width: usize,
    height: usize,
    stride: usize,
    config: IwWeightConfig,
) -> Vec<f32> {
    assert!(
        ref_plane.len() >= stride * height,
        "ref_plane too short for stride={stride}×height={height}",
    );
    assert!(stride >= width, "stride={stride} < width={width}");

    let n = width * height;
    let mut weights = vec![0.0f32; n];

    match config.kind {
        IwWeightKind::LocalVariance => {
            compute_local_variance(ref_plane, width, height, stride, config.kernel_half, &mut weights);
        }
        IwWeightKind::LocalGradL1 => {
            compute_gradient(ref_plane, width, height, stride, GradNorm::L1, &mut weights);
        }
        IwWeightKind::LocalGradL2 => {
            compute_gradient(ref_plane, width, height, stride, GradNorm::L2, &mut weights);
        }
    }

    if config.weight_floor > 0.0 {
        let max_w = weights.iter().copied().fold(0.0f32, f32::max);
        let floor = config.weight_floor * max_w.max(1.0);
        for w in &mut weights {
            *w = w.max(floor);
        }
    }

    weights
}

fn compute_local_variance(
    plane: &[f32],
    width: usize,
    height: usize,
    stride: usize,
    half: usize,
    out: &mut [f32],
) {
    // Two-pass: sum and sum-of-squares in a (2h+1)² window for each
    // pixel; reflect-pad at the boundary.
    for y in 0..height {
        for x in 0..width {
            let y0 = y.saturating_sub(half);
            let y1 = (y + half).min(height - 1);
            let x0 = x.saturating_sub(half);
            let x1 = (x + half).min(width - 1);
            let n_w = ((y1 - y0 + 1) * (x1 - x0 + 1)) as f32;
            let mut s = 0.0f64;
            let mut s2 = 0.0f64;
            for yy in y0..=y1 {
                for xx in x0..=x1 {
                    let v = plane[yy * stride + xx] as f64;
                    s += v;
                    s2 += v * v;
                }
            }
            let mean = s / n_w as f64;
            let var = (s2 / n_w as f64 - mean * mean).max(0.0);
            out[y * width + x] = var as f32;
        }
    }
}

enum GradNorm {
    L1,
    L2,
}

fn compute_gradient(
    plane: &[f32],
    width: usize,
    height: usize,
    stride: usize,
    norm: GradNorm,
    out: &mut [f32],
) {
    for y in 0..height {
        let yu = if y == 0 { 0 } else { y - 1 };
        let yd = (y + 1).min(height - 1);
        for x in 0..width {
            let xl = if x == 0 { 0 } else { x - 1 };
            let xr = (x + 1).min(width - 1);
            let dx = plane[y * stride + xr] - plane[y * stride + xl];
            let dy = plane[yd * stride + x] - plane[yu * stride + x];
            out[y * width + x] = match norm {
                GradNorm::L1 => dx.abs() + dy.abs(),
                GradNorm::L2 => (dx * dx + dy * dy).sqrt(),
            };
        }
    }
}

/// Weighted spatial pool. Mirrors the unweighted `mean`, `L2`, `L4`
/// pools used in the basic feature block, but each pixel's contribution
/// is multiplied by the corresponding weight from
/// [`compute_iw_weights`]. Normalisation divides by the sum of weights
/// (which equals N for unit weights, recovering the standard mean).
pub struct WeightedPool;

impl WeightedPool {
    /// Weighted mean: `(Σ w_i v_i) / Σ w_i`. Returns 0 if weight sum
    /// is below 1e-12.
    pub fn mean(values: &[f32], weights: &[f32]) -> f64 {
        assert_eq!(values.len(), weights.len());
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in 0..values.len() {
            num += (values[i] as f64) * (weights[i] as f64);
            den += weights[i] as f64;
        }
        if den < 1e-12 { 0.0 } else { num / den }
    }

    /// Weighted L2 norm: `√((Σ w_i v_i²) / Σ w_i)`. Square-root taken
    /// after weighted mean of squares so units match the underlying
    /// signal.
    pub fn l2(values: &[f32], weights: &[f32]) -> f64 {
        assert_eq!(values.len(), weights.len());
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in 0..values.len() {
            num += (values[i] as f64).powi(2) * (weights[i] as f64);
            den += weights[i] as f64;
        }
        if den < 1e-12 { 0.0 } else { (num / den).sqrt() }
    }

    /// Weighted L4 norm: `((Σ w_i v_i⁴) / Σ w_i)^(1/4)`. Matches the
    /// basic feature block's L4 ("`ssim_4th`") which emphasises peak
    /// errors.
    pub fn l4(values: &[f32], weights: &[f32]) -> f64 {
        assert_eq!(values.len(), weights.len());
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in 0..values.len() {
            let v = values[i] as f64;
            num += v * v * v * v * (weights[i] as f64);
            den += weights[i] as f64;
        }
        if den < 1e-12 { 0.0 } else { (num / den).powf(0.25) }
    }
}

/// Per-channel IW-pooled SSIM block emitted in the V0_20a feature
/// extension. Mirrors the 6 "masked" features in the existing extended
/// profile but with the weight direction inverted (high info content
/// gets MORE weight, not less).
#[derive(Debug, Clone, Copy)]
pub struct IwSsimFeatures {
    /// Weighted mean of per-pixel SSIM(src, dst) at this scale & channel.
    pub iw_ssim_mean: f64,
    /// Weighted L2 of SSIM.
    pub iw_ssim_2nd: f64,
    /// Weighted L4 of SSIM (peak-emphasising, matches `ssim_4th`).
    pub iw_ssim_4th: f64,
    /// Weighted L4 of edge artifact map.
    pub iw_art_4th: f64,
    /// Weighted L4 of edge detail-lost map.
    pub iw_det_4th: f64,
    /// Weighted mean of (src-dst)².
    pub iw_mse: f64,
}

impl IwSsimFeatures {
    /// Number of features per call — matches `FEATURES_PER_CHANNEL_*_MASKED` in `metric.rs`.
    pub const FEATURES_PER_CALL: usize = 6;

    /// Flatten into the wire-order indices used by the trainer.
    pub fn as_array(&self) -> [f64; 6] {
        [
            self.iw_ssim_mean,
            self.iw_ssim_2nd,
            self.iw_ssim_4th,
            self.iw_art_4th,
            self.iw_det_4th,
            self.iw_mse,
        ]
    }

    /// Pool the supplied per-pixel maps with IW weights computed from
    /// the reference plane. All inputs must be the same length =
    /// `width * height`. `ref_plane` is the reference channel at this
    /// scale (used for weight computation); `ssim_map` / `art_map` /
    /// `det_map` / `mse_map` are pre-computed per-pixel diffmaps from
    /// the basic feature pipeline.
    ///
    /// Designed for the V0_20a sweep: take an existing reference +
    /// diffmap, weight by reference info-content, output 6 features.
    pub fn pool_from_maps(
        ref_plane: &[f32],
        width: usize,
        height: usize,
        stride: usize,
        ssim_map: &[f32],
        art_map: &[f32],
        det_map: &[f32],
        mse_map: &[f32],
        config: IwWeightConfig,
    ) -> Self {
        let weights = compute_iw_weights(ref_plane, width, height, stride, config);
        let n = width * height;
        assert_eq!(ssim_map.len(), n, "ssim_map size mismatch");
        assert_eq!(art_map.len(), n, "art_map size mismatch");
        assert_eq!(det_map.len(), n, "det_map size mismatch");
        assert_eq!(mse_map.len(), n, "mse_map size mismatch");
        Self {
            iw_ssim_mean: WeightedPool::mean(ssim_map, &weights),
            iw_ssim_2nd: WeightedPool::l2(ssim_map, &weights),
            iw_ssim_4th: WeightedPool::l4(ssim_map, &weights),
            iw_art_4th: WeightedPool::l4(art_map, &weights),
            iw_det_4th: WeightedPool::l4(det_map, &weights),
            iw_mse: WeightedPool::mean(mse_map, &weights),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_weights_recover_unweighted_mean() {
        let values: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let weights = vec![1.0f32; 16];
        let mean = WeightedPool::mean(&values, &weights);
        let expected: f64 = values.iter().map(|v| *v as f64).sum::<f64>() / 16.0;
        assert!((mean - expected).abs() < 1e-9);
    }

    #[test]
    fn weighted_mean_concentrates_on_high_weight() {
        // Two pixels: 0 and 10, weighted 1 and 9 → weighted mean = 90/10 = 9
        let mean = WeightedPool::mean(&[0.0, 10.0], &[1.0, 9.0]);
        assert!((mean - 9.0).abs() < 1e-9);
    }

    #[test]
    fn weighted_l2_handles_constant_signal() {
        // Constant 4.0 across all weights → L2 = 4.0 regardless of weights
        let values = vec![4.0f32; 8];
        let weights: Vec<f32> = (1..=8).map(|i| i as f32).collect();
        let l2 = WeightedPool::l2(&values, &weights);
        assert!((l2 - 4.0).abs() < 1e-9);
    }

    #[test]
    fn weighted_l4_emphasises_peaks() {
        // Pure peak: one value at 16, rest at 0 — L4 ≈ 16 · (w_peak / Σw)^(1/4)
        let values = vec![0.0, 0.0, 0.0, 16.0];
        let weights = vec![1.0, 1.0, 1.0, 1.0];
        let l4 = WeightedPool::l4(&values, &weights);
        // (Σ v⁴ / N)^(1/4) = (65536/4)^(1/4) = 16384^(1/4) ≈ 11.314
        let expected = (16384.0f64).powf(0.25);
        assert!((l4 - expected).abs() < 1e-6);
    }

    #[test]
    fn local_variance_zero_on_constant_plane() {
        let plane = vec![5.0f32; 64];
        let mut w = vec![0.0f32; 64];
        compute_local_variance(&plane, 8, 8, 8, 2, &mut w);
        // After variance pass: all zeros. The weight_floor in
        // compute_iw_weights would then raise it to floor*max_w but
        // here max_w=0 so floor*max=0; both are zero.
        for v in &w {
            assert!(v.abs() < 1e-6);
        }
    }

    #[test]
    fn local_variance_responds_to_edges() {
        // 4×4 with a step edge: left half = 0, right half = 10
        let plane: Vec<f32> = (0..16)
            .map(|i| if (i % 4) < 2 { 0.0 } else { 10.0 })
            .collect();
        let mut w = vec![0.0f32; 16];
        compute_local_variance(&plane, 4, 4, 4, 1, &mut w);
        // The middle two columns of each row span the edge — variance
        // should be > 0 there, and 0 in the homogeneous corners
        // (corner has only 0s or only 10s in its 3×3 neighborhood).
        assert!(w[1] > 0.0, "x=1 should see the edge");
        assert!(w[2] > 0.0, "x=2 should see the edge");
        assert!((w[0] - 0.0).abs() < 1e-6, "x=0 corner is homogeneous");
    }

    #[test]
    fn gradient_l1_picks_up_edges() {
        let plane: Vec<f32> = (0..16)
            .map(|i| if (i % 4) < 2 { 0.0 } else { 10.0 })
            .collect();
        let mut w = vec![0.0f32; 16];
        compute_gradient(&plane, 4, 4, 4, GradNorm::L1, &mut w);
        // Pixels at the edge boundary should have nonzero gradient
        // because the horizontal neighbours straddle the step.
        let edge_weight_l = w[1]; // x=1, just left of step
        let edge_weight_r = w[2]; // x=2, just right of step
        assert!(edge_weight_l > 0.0);
        assert!(edge_weight_r > 0.0);
    }

    #[test]
    fn pool_from_maps_uniform_weights_match_unweighted() {
        // Build a 4×4 reference, uniform — so weights are all `weight_floor·max=floor·0=0`,
        // which the floor raises to a uniform positive value, recovering
        // the unweighted pool.
        let ref_plane = vec![5.0f32; 16];
        let ssim = vec![0.8f32; 16];
        let art = vec![0.1f32; 16];
        let det = vec![0.1f32; 16];
        let mse = vec![0.01f32; 16];
        let f = IwSsimFeatures::pool_from_maps(
            &ref_plane, 4, 4, 4, &ssim, &art, &det, &mse, IwWeightConfig::default(),
        );
        assert!((f.iw_ssim_mean - 0.8).abs() < 1e-6);
        assert!((f.iw_ssim_2nd - 0.8).abs() < 1e-6);
        assert!((f.iw_ssim_4th - 0.8).abs() < 1e-6);
        assert!((f.iw_mse - 0.01).abs() < 1e-6);
    }

    #[test]
    fn pool_from_maps_emphasises_edge_regions() {
        // Reference: step edge (left=0, right=10). SSIM-map: errors only
        // at the edge column. With IW weighting (texture-emphasising),
        // the pool should report a LARGER error than uniform pooling
        // because the edge-pixel errors get more weight.
        let ref_plane: Vec<f32> = (0..16)
            .map(|i| if (i % 4) < 2 { 0.0 } else { 10.0 })
            .collect();
        // Inject error only at the edge column (x=1, where step happens).
        let mut ssim_err = vec![0.0f32; 16];
        for y in 0..4 {
            ssim_err[y * 4 + 1] = 1.0; // strong SSIM error at edge
        }
        let zero = vec![0.0f32; 16];
        let iw = IwSsimFeatures::pool_from_maps(
            &ref_plane, 4, 4, 4, &ssim_err, &zero, &zero, &zero, IwWeightConfig::default(),
        );
        // Unweighted: 4/16 = 0.25. IW-weighted: edge pixels (which have
        // high variance) get more weight → IW mean should be > 0.25.
        let unweighted_mean = ssim_err.iter().sum::<f32>() as f64 / 16.0;
        assert!(
            iw.iw_ssim_mean > unweighted_mean,
            "IW mean {} should exceed unweighted {}",
            iw.iw_ssim_mean,
            unweighted_mean,
        );
    }

    #[test]
    fn compute_iw_weights_applies_floor() {
        let plane = vec![0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let config = IwWeightConfig {
            kind: IwWeightKind::LocalVariance,
            kernel_half: 1,
            weight_floor: 0.01,
        };
        let w = compute_iw_weights(&plane, 4, 4, 4, config);
        let max_w = w.iter().copied().fold(0.0f32, f32::max);
        let min_w = w.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(max_w > 0.0);
        // Floor of 0.01 * max_w means min should be at least 0.01 of max
        assert!(min_w >= 0.01 * max_w - 1e-6, "min {min_w} should be at least 0.01·max {max_w}");
    }
}
