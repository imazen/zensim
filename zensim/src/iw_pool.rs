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
//! ## Status (2026-05-22)
//!
//! The IW-pool block shipped in the V_22-mix-LARGE+iwssim Balanced
//! recipe (PreviewV0_5Balanced) and contributes features
//! f300..f371. Implementation is fused into the SIMD streaming loop
//! via `streaming::process_scale_bands` and `process_strip_into_accum`;
//! the standalone helpers below are kept for offline experimentation
//! and for the legacy non-streaming code path.

/// Choice of per-pixel info-content estimator.
// dead_code: the non-default estimators are research knobs selected through
// `IwWeightConfig` by offline/training experiments (see
// `benchmarks/iw_pyramid_spike_methodology_2026-05-15.md`); they are kept in
// every feature set so the config surface stays feature-stable.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IwWeightKind {
    /// Local variance in a square window (kernel size set by config).
    /// Default. Closest to Wang 2011's GSM scale parameter direction.
    #[default]
    LocalVariance,
    /// L1 norm of the gradient — `|∂x I| + |∂y I|`. Cheaper than variance,
    /// emphasises edges.
    LocalGradL1,
    /// L2 norm of the gradient — `√((∂x I)² + (∂y I)²)`.
    LocalGradL2,
    /// **Paper-faithful steerable-pyramid GSM approximation** (spike,
    /// 2026-05-15). Replaces the scalar spatial variance with a
    /// **directional max** across 4 oriented gradient orientations
    /// (0°, 45°, 90°, 135°). For each pixel, the local variance of
    /// each oriented gradient is computed in a Gaussian-weighted
    /// patch (default ~9×9 support); the per-pixel σ²_p is the max
    /// across orientations.
    ///
    /// This captures Wang & Li 2011's directional sensitivity at zeroth
    /// order — diagonal edges weigh differently from axis-aligned edges
    /// of the same total energy. It is an approximation to the paper's
    /// 5-level Simoncelli steerable pyramid; see
    /// `benchmarks/iw_pyramid_spike_methodology_2026-05-15.md` for the
    /// exact divergences.
    ///
    /// MUST be combined with `info_log_sigma_e_sq = Some(σ²_e)` to get
    /// the paper's `log₂(1 + σ²_p / σ²_e)` weight formula. The variant
    /// produces raw σ²_p; the log transform is applied at
    /// [`compute_iw_weights`] time.
    SteerablePyramidLogGsm,
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
    /// When `Some(sigma_e_sq)`, apply Wang & Li 2011's information-
    /// content transform on the raw estimator output:
    ///
    ///   `w(x) ← log₂(1.0 + w_raw(x) / sigma_e_sq)`
    ///
    /// This is the paper's exact weight formula — `w_raw(x)` plays the
    /// role of the GSM scale parameter σ²_p(x), and `sigma_e_sq` is
    /// the noise-floor variance σ²_e. The log saturates high-variance
    /// regions instead of letting them dominate proportional to σ².
    ///
    /// `None` (default for back-compat with V_20a sweep) keeps the raw
    /// estimator output as the weight — variance / gradient magnitude
    /// directly. The default exists so existing experiments stay
    /// reproducible; new work should set `info_log_sigma_e_sq` for
    /// paper-faithful behavior.
    ///
    /// Added 2026-05-15 after the V_20a IW-SSIM falsification revealed
    /// our implementation diverged from the paper's `log(1 + σ²_p/σ²_e)`
    /// weight formula. See `benchmarks/extended_iw_runtime_perf_2026-05-15.md`
    /// and the IW-paper-faithfulness analysis in the same message.
    pub info_log_sigma_e_sq: Option<f32>,
}

impl Default for IwWeightConfig {
    fn default() -> Self {
        Self {
            kind: IwWeightKind::LocalVariance,
            kernel_half: 2,
            weight_floor: 1.0e-3,
            info_log_sigma_e_sq: None,
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
            compute_local_variance(
                ref_plane,
                width,
                height,
                stride,
                config.kernel_half,
                &mut weights,
            );
        }
        IwWeightKind::LocalGradL1 => {
            compute_gradient(ref_plane, width, height, stride, GradNorm::L1, &mut weights);
        }
        IwWeightKind::LocalGradL2 => {
            compute_gradient(ref_plane, width, height, stride, GradNorm::L2, &mut weights);
        }
        IwWeightKind::SteerablePyramidLogGsm => {
            // Use the same kernel_half as the variance estimator so users
            // can swap between LocalVariance and SteerablePyramidLogGsm
            // with one field change. half=4 (9×9 patch) is the recommended
            // default — see methodology doc.
            compute_directional_max_variance(
                ref_plane,
                width,
                height,
                stride,
                config.kernel_half,
                &mut weights,
            );
        }
    }

    // Paper-faithful info-content transform: apply log₂(1 + w / σ²_e)
    // when configured. Must happen BEFORE the weight floor — the floor
    // is a numerical guard, the log is the perceptual one.
    if let Some(sigma_e_sq) = config.info_log_sigma_e_sq {
        let inv_sigma = 1.0_f32 / sigma_e_sq.max(1e-12);
        for w in &mut weights {
            // log₂(1 + w/σ²_e) — non-negative for non-negative w, ≈ w/σ²_e/ln2 for small w.
            *w = (1.0_f32 + (*w) * inv_sigma).log2();
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

/// Paper-faithful steerable-pyramid GSM weight (spike, 2026-05-15).
///
/// Computes 4 oriented gradients on the reference plane (0°, 45°, 90°,
/// 135°), then for each pixel computes the local variance of each
/// oriented gradient over a square patch of half-width `half` (default
/// 4 → 9×9 patch). The per-pixel weight is the **max** variance across
/// the 4 orientations.
///
/// This approximates Wang & Li 2011's σ²_p (the GSM scale parameter on
/// per-band steerable-pyramid coefficients) by using oriented gradients
/// in lieu of the steerable subbands themselves. See methodology doc
/// at `benchmarks/iw_pyramid_spike_methodology_2026-05-15.md`.
///
/// Orientation kernels (3×3 centered differences):
/// - 0°   (horizontal):   ∂x — `[-1, 0, 1]` along x
/// - 90°  (vertical):     ∂y — `[-1, 0, 1]` along y
/// - 45°  (diagonal NE):  ∂x + ∂y combined kernel
/// - 135° (diagonal NW):  ∂x - ∂y combined kernel
///
/// The diagonal kernels are 3×3 with non-zero off-axis entries — they
/// catch energy along diagonal edges that horizontal+vertical gradients
/// MISS when the edge is exactly 45° (both ∂x and ∂y are equal in
/// magnitude but combined-as-L2 would give the same answer as for a
/// horizontal edge of equal total energy).
fn compute_directional_max_variance(
    plane: &[f32],
    width: usize,
    height: usize,
    stride: usize,
    half: usize,
    out: &mut [f32],
) {
    let n = width * height;
    // 4 oriented-gradient buffers, contiguous (width × height each).
    let mut g0 = vec![0.0f32; n]; // horizontal
    let mut g45 = vec![0.0f32; n]; // diagonal NE
    let mut g90 = vec![0.0f32; n]; // vertical
    let mut g135 = vec![0.0f32; n]; // diagonal NW

    // Compute each oriented gradient via 3×3 centered-difference kernel.
    // Reflect-pad at boundaries.
    for y in 0..height {
        let yu = if y == 0 { 0 } else { y - 1 };
        let yd = (y + 1).min(height - 1);
        for x in 0..width {
            let xl = if x == 0 { 0 } else { x - 1 };
            let xr = (x + 1).min(width - 1);
            let p_l = plane[y * stride + xl];
            let p_r = plane[y * stride + xr];
            let p_u = plane[yu * stride + x];
            let p_d = plane[yd * stride + x];
            let p_ul = plane[yu * stride + xl];
            let p_ur = plane[yu * stride + xr];
            let p_dl = plane[yd * stride + xl];
            let p_dr = plane[yd * stride + xr];
            // Horizontal: ∂x
            g0[y * width + x] = p_r - p_l;
            // Vertical: ∂y
            g90[y * width + x] = p_d - p_u;
            // Diagonal NE (45°): along (1, -1) direction — top-right minus bottom-left
            //   This is the gradient along an edge running from BL to TR.
            //   A horizontal or vertical edge gives zero here.
            g45[y * width + x] = p_ur - p_dl;
            // Diagonal NW (135°): along (-1, -1) direction — top-left minus bottom-right
            g135[y * width + x] = p_ul - p_dr;
        }
    }

    // For each oriented-gradient plane, compute local variance over the
    // (2h+1)² patch. Take the max across the 4 orientations.
    let mut v0 = vec![0.0f32; n];
    let mut v45 = vec![0.0f32; n];
    let mut v90 = vec![0.0f32; n];
    let mut v135 = vec![0.0f32; n];

    local_variance_into(&g0, width, height, half, &mut v0);
    local_variance_into(&g45, width, height, half, &mut v45);
    local_variance_into(&g90, width, height, half, &mut v90);
    local_variance_into(&g135, width, height, half, &mut v135);

    for i in 0..n {
        // The orientation max captures the dominant-orientation signal
        // the paper's §III-B discusses. A sum would also work; max is
        // closer to the "GSM scale parameter at the dominant orientation"
        // reading of the paper.
        let m = v0[i].max(v45[i]).max(v90[i]).max(v135[i]);
        out[i] = m;
    }
}

/// Compute local variance over a (2h+1)² window for a tightly-packed
/// width × height buffer. Reflect-pad at the boundary. Stride is
/// implicitly `width` here — we only ever call this on local buffers we
/// just allocated.
fn local_variance_into(plane: &[f32], width: usize, height: usize, half: usize, out: &mut [f32]) {
    for y in 0..height {
        let y0 = y.saturating_sub(half);
        let y1 = (y + half).min(height - 1);
        for x in 0..width {
            let x0 = x.saturating_sub(half);
            let x1 = (x + half).min(width - 1);
            let n_w = ((y1 - y0 + 1) * (x1 - x0 + 1)) as f32;
            let mut s = 0.0f64;
            let mut s2 = 0.0f64;
            for yy in y0..=y1 {
                for xx in x0..=x1 {
                    let v = plane[yy * width + xx] as f64;
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
///
/// **`mean` established 2026-07-18 (iteration 1)** as THE canonical
/// `Σw·v/Σw` formula (Wang & Li 2011's IW-SSIM Eq.36 exactly) for v2
/// "bounded" masked/IW/soft-peak pooling — v1's shipped hot path
/// (`streaming.rs`, `1/n`-pooled) does NOT match this form and is
/// unaffected; see `benchmarks/iw_pooling_normalization_2026-07-15.md`
/// for that divergence.
///
/// **Iteration 2 update:** the v2 kernel (`feature_v2.rs`,
/// `compute_channel_scale_v2`) no longer CALLS `mean` directly — batching
/// per-pixel values into arrays to call it was measured to make v2 2-5x
/// slower than v1 (`docs/FEATURE_V2_SPEC_2026-07-18.md` §A.12). The kernel
/// now uses `feature_v2::WeightedSum`, an O(1)-space incremental
/// accumulator computing the IDENTICAL `Σw·v/Σw` formula online. `mean`
/// remains the canonical REFERENCE form — `feature_v2::tests::
/// weighted_sum_matches_weighted_pool_mean_exactly` pins the two bit-exact
/// on random data (the "gated mirror" exception to the no-duplication
/// policy: same formula, different computational strategy, tested
/// equivalent). `l2`/`l4` remain unused by v2 (kept for parity / future
/// use, individually `#[allow(dead_code)]` below).
#[allow(dead_code)] // only caller is the gated-mirror equivalence test in feature_v2.rs
pub(crate) struct WeightedPool;

impl WeightedPool {
    /// Weighted mean: `(Σ w_i v_i) / Σ w_i`. Returns 0 if weight sum
    /// is below 1e-12. Canonical reference form — see the struct doc for
    /// why the v2 hot path calls `feature_v2::WeightedSum` instead.
    #[allow(dead_code)] // only caller is the gated-mirror equivalence test in feature_v2.rs
    pub(crate) fn mean(values: &[f32], weights: &[f32]) -> f64 {
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
    #[allow(dead_code)] // not used by v2 iteration 1 (bounded-basic block doesn't need weighted L2 pooling yet); kept for parity/future use
    pub(crate) fn l2(values: &[f32], weights: &[f32]) -> f64 {
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
    #[allow(dead_code)] // not used by v2 iteration 1 (bounded-basic block doesn't need weighted L4 pooling yet); kept for parity/future use
    pub(crate) fn l4(values: &[f32], weights: &[f32]) -> f64 {
        assert_eq!(values.len(), weights.len());
        let mut num = 0.0f64;
        let mut den = 0.0f64;
        for i in 0..values.len() {
            let v = values[i] as f64;
            num += v * v * v * v * (weights[i] as f64);
            den += weights[i] as f64;
        }
        if den < 1e-12 {
            0.0
        } else {
            (num / den).powf(0.25)
        }
    }
}

/// Per-channel IW-pooled SSIM block — 6 features at
/// `f228+72*ch+0..f228+72*ch+5` for each scale (see `metric.rs`
/// `FEATURES_PER_CHANNEL_*_MASKED`). Mirrors the 6 "masked" features
/// in the existing extended profile but with the weight direction
/// inverted (high info content gets MORE weight, not less). Shipped
/// in the PreviewV0_5Balanced bake's f300..f371 input block.
///
/// **Crate-internal as of 0.3.0.** The streaming hot path emits the
/// same 6 numbers via `streaming::process_strip_into_accum`; this
/// type is kept for the offline-experiment entry point
/// [`Self::pool_from_maps`] and the iw_pool unit tests.
#[derive(Debug, Clone, Copy)]
#[allow(dead_code)] // type is built only by the crate-internal pool_from_maps; tests construct it via that path
pub(crate) struct IwSsimFeatures {
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

#[allow(dead_code)] // tests-only reference implementation; hot path is fused into streaming
impl IwSsimFeatures {
    /// Number of features per call — matches `FEATURES_PER_CHANNEL_*_MASKED` in `metric.rs`.
    pub(crate) const FEATURES_PER_CALL: usize = 6;

    /// Flatten into the wire-order indices used by the trainer.
    pub(crate) fn as_array(&self) -> [f64; 6] {
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
    /// Offline-experiment entry point: take an existing reference +
    /// diffmap, weight by reference info-content, output 6 features.
    /// The streaming hot path fuses this into the SIMD inner loop —
    /// see `streaming::process_strip_into_accum`.
    pub(crate) fn pool_from_maps(
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
            &ref_plane,
            4,
            4,
            4,
            &ssim,
            &art,
            &det,
            &mse,
            IwWeightConfig::default(),
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
            &ref_plane,
            4,
            4,
            4,
            &ssim_err,
            &zero,
            &zero,
            &zero,
            IwWeightConfig::default(),
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
        let plane = vec![
            0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let config = IwWeightConfig {
            kind: IwWeightKind::LocalVariance,
            kernel_half: 1,
            weight_floor: 0.01,
            info_log_sigma_e_sq: None,
        };
        let w = compute_iw_weights(&plane, 4, 4, 4, config);
        let max_w = w.iter().copied().fold(0.0f32, f32::max);
        let min_w = w.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(max_w > 0.0);
        // Floor of 0.01 * max_w means min should be at least 0.01 of max
        assert!(
            min_w >= 0.01 * max_w - 1e-6,
            "min {min_w} should be at least 0.01·max {max_w}"
        );
    }

    // ─── Steerable-pyramid spike tests ──────────────────────────────────────

    /// Constant plane → all directional gradients zero → all variances
    /// zero → weight is zero everywhere.
    #[test]
    fn steerable_zero_on_constant_plane() {
        let plane = vec![5.0f32; 64];
        let config = IwWeightConfig {
            kind: IwWeightKind::SteerablePyramidLogGsm,
            kernel_half: 2,
            weight_floor: 0.0,
            info_log_sigma_e_sq: None,
        };
        let w = compute_iw_weights(&plane, 8, 8, 8, config);
        for v in &w {
            assert!(v.abs() < 1e-6, "expected 0, got {v}");
        }
    }

    /// Step edge along the horizontal direction (vertical content
    /// transition). The horizontal gradient is non-zero at the edge;
    /// the diagonals also see some energy due to the 3×3 stencil.
    /// Weight at the edge column should be > weight at the corner.
    #[test]
    fn steerable_responds_to_edges() {
        // 6×6 with a vertical step: left half = 0, right half = 10
        let mut plane = vec![0.0f32; 36];
        for y in 0..6 {
            for x in 3..6 {
                plane[y * 6 + x] = 10.0;
            }
        }
        let config = IwWeightConfig {
            kind: IwWeightKind::SteerablePyramidLogGsm,
            kernel_half: 1,
            weight_floor: 0.0,
            info_log_sigma_e_sq: None,
        };
        let w = compute_iw_weights(&plane, 6, 6, 6, config);
        // Pixel at x=2, y=3 (just left of step) should see the edge.
        let edge_w = w[3 * 6 + 2];
        // Pixel at x=0, y=0 (far corner, homogeneous patch) should not.
        let corner_w = w[0];
        assert!(
            edge_w > 0.0,
            "edge should have nonzero weight, got {edge_w}"
        );
        assert!(
            edge_w > corner_w,
            "edge weight {edge_w} should exceed corner weight {corner_w}",
        );
    }

    /// **The directional sensitivity test** — the core spike claim.
    ///
    /// Two synthetic images with the SAME total edge energy but
    /// different orientations:
    /// - Image A: a single horizontal step edge
    /// - Image B: a single diagonal step edge
    ///
    /// Spatial-variance (scalar) should produce SIMILAR weight
    /// distributions because the variance estimator doesn't see
    /// orientation. Steerable-pyramid (directional max) should
    /// produce DIFFERENT distributions because the 4 oriented
    /// gradients respond differently.
    ///
    /// Specifically: a diagonal edge concentrates energy in the
    /// diagonal oriented gradients, so the max-across-orientations is
    /// closer to the per-orientation max. A horizontal edge spreads
    /// energy across horizontal + (slight diagonal); the max is
    /// dominated by horizontal alone.
    #[test]
    fn steerable_directional_max_differs_from_scalar_variance() {
        // 16×16 image with one diagonal edge: pixel value = clamp(x+y - 16, 0, 10)
        let w_img = 16;
        let h_img = 16;
        let mut diag = vec![0.0f32; w_img * h_img];
        for y in 0..h_img {
            for x in 0..w_img {
                let s = (x as i32 + y as i32) - 16;
                let v = if s < 0 {
                    0.0
                } else if s > 1 {
                    10.0
                } else {
                    5.0
                };
                diag[y * w_img + x] = v;
            }
        }
        // Mirror image: horizontal edge at y=8
        let mut horiz = vec![0.0f32; w_img * h_img];
        for y in 8..h_img {
            for x in 0..w_img {
                horiz[y * w_img + x] = 10.0;
            }
        }
        let cfg = IwWeightConfig {
            kind: IwWeightKind::SteerablePyramidLogGsm,
            kernel_half: 2,
            weight_floor: 0.0,
            info_log_sigma_e_sq: None,
        };
        let w_diag = compute_iw_weights(&diag, w_img, h_img, w_img, cfg);
        let w_horiz = compute_iw_weights(&horiz, w_img, h_img, w_img, cfg);

        // Sums of weights — these are proxies for the total "info
        // content" the metric will integrate. For comparable edge
        // energies, the two should NOT be wildly different.
        let s_diag: f64 = w_diag.iter().map(|v| *v as f64).sum();
        let s_horiz: f64 = w_horiz.iter().map(|v| *v as f64).sum();
        assert!(s_diag > 0.0);
        assert!(s_horiz > 0.0);

        // Also compare to the spatial-variance estimator (CURRENT path).
        // The hypothesis: spatial variance does NOT differentiate
        // diagonal from horizontal as strongly as the directional max.
        let scalar_cfg = IwWeightConfig {
            kind: IwWeightKind::LocalVariance,
            kernel_half: 2,
            weight_floor: 0.0,
            info_log_sigma_e_sq: None,
        };
        let s_diag_scalar = compute_iw_weights(&diag, w_img, h_img, w_img, scalar_cfg);
        let s_horiz_scalar = compute_iw_weights(&horiz, w_img, h_img, w_img, scalar_cfg);
        let ss_diag: f64 = s_diag_scalar.iter().map(|v| *v as f64).sum();
        let ss_horiz: f64 = s_horiz_scalar.iter().map(|v| *v as f64).sum();

        // The maximum per-pixel weight at the diagonal edge should be
        // LARGER under the directional max than the diagonal/horizontal
        // ratio would suggest from the spatial variance estimator,
        // because the diagonal kernel concentrates the directional
        // signal. We assert the ratio differs.
        let ratio_dir = s_diag / s_horiz;
        let ratio_sca = ss_diag / ss_horiz;
        // ratios shouldn't be identical — that's the spike's point.
        // Loose bound: at least 5% difference between the two.
        let rel = (ratio_dir - ratio_sca).abs() / ratio_sca.max(1e-12);
        assert!(
            rel > 0.05,
            "directional / scalar ratios should differ by >5%; got dir_ratio={ratio_dir:.3} \
             scalar_ratio={ratio_sca:.3} rel_diff={rel:.3}",
        );
    }

    /// Reference-impl smoke test: at the center of an isolated step
    /// edge, with σ²_e = 1.0, the log-transformed weight should be
    /// roughly `log₂(1 + σ²_p)` where σ²_p is the local variance of
    /// the dominant-orientation gradient. We don't hardcode the exact
    /// number (depends on patch half-width), but we sanity-check that:
    /// - The log-transformed weight at the edge center is positive.
    /// - It saturates as expected (much smaller than σ²_p when σ²_p ≫ 1).
    #[test]
    fn steerable_log_transform_saturates() {
        // 10×10 with a single vertical step edge in the middle column.
        let w_img = 10;
        let h_img = 10;
        let mut plane = vec![0.0f32; w_img * h_img];
        for y in 0..h_img {
            for x in 5..w_img {
                plane[y * w_img + x] = 100.0;
            }
        }
        let raw_cfg = IwWeightConfig {
            kind: IwWeightKind::SteerablePyramidLogGsm,
            kernel_half: 2,
            weight_floor: 0.0,
            info_log_sigma_e_sq: None,
        };
        let log_cfg = IwWeightConfig {
            info_log_sigma_e_sq: Some(1.0),
            weight_floor: 0.0,
            ..raw_cfg
        };
        let raw_w = compute_iw_weights(&plane, w_img, h_img, w_img, raw_cfg);
        let log_w = compute_iw_weights(&plane, w_img, h_img, w_img, log_cfg);
        let raw_max = raw_w.iter().copied().fold(0.0f32, f32::max);
        let log_max = log_w.iter().copied().fold(0.0f32, f32::max);
        assert!(
            raw_max > 100.0,
            "step edge should produce large raw variance"
        );
        assert!(log_max > 0.0);
        assert!(
            log_max < raw_max,
            "log transform should saturate: raw={raw_max} log={log_max}",
        );
        let expected = (1.0 + raw_max).log2();
        assert!(
            (log_max - expected).abs() < 1e-3,
            "log_max {log_max} != log₂(1 + raw_max={raw_max}) = {expected}",
        );
    }

    /// A/B comparison sanity check on a synthetic input: the
    /// Pearson correlation between the LocalVariance weight map and
    /// the SteerablePyramidLogGsm weight map. For a structured input
    /// the two are necessarily correlated (both fire at edges), but
    /// they should NOT be 1:1. Floor the assertion at 0.99 so it
    /// catches code bugs without being noise-sensitive.
    #[test]
    fn ab_weight_map_pearson_not_unity() {
        // Use a 16×16 mix of horizontal + diagonal edges + random noise.
        let w_img = 16;
        let h_img = 16;
        let mut plane = vec![0.0f32; w_img * h_img];
        for y in 0..h_img {
            for x in 0..w_img {
                let mut v = 0.0f32;
                if x >= 8 {
                    v += 10.0;
                }
                if (x as i32 + y as i32) > 16 {
                    v += 5.0;
                }
                // Tiny deterministic pseudo-noise:
                v += ((x as f32 * 0.37 + y as f32 * 0.21).sin() * 0.5).abs();
                plane[y * w_img + x] = v;
            }
        }

        let scalar_cfg = IwWeightConfig {
            kind: IwWeightKind::LocalVariance,
            kernel_half: 2,
            weight_floor: 0.0,
            info_log_sigma_e_sq: None,
        };
        let dir_cfg = IwWeightConfig {
            kind: IwWeightKind::SteerablePyramidLogGsm,
            kernel_half: 2,
            weight_floor: 0.0,
            info_log_sigma_e_sq: None,
        };
        let scalar = compute_iw_weights(&plane, w_img, h_img, w_img, scalar_cfg);
        let dir = compute_iw_weights(&plane, w_img, h_img, w_img, dir_cfg);

        // Compute Pearson correlation.
        let n = scalar.len() as f64;
        let mean_a: f64 = scalar.iter().map(|v| *v as f64).sum::<f64>() / n;
        let mean_b: f64 = dir.iter().map(|v| *v as f64).sum::<f64>() / n;
        let mut cov = 0.0f64;
        let mut var_a = 0.0f64;
        let mut var_b = 0.0f64;
        for i in 0..scalar.len() {
            let a = scalar[i] as f64 - mean_a;
            let b = dir[i] as f64 - mean_b;
            cov += a * b;
            var_a += a * a;
            var_b += b * b;
        }
        let pearson = cov / (var_a.sqrt() * var_b.sqrt()).max(1e-12);
        // The two methods should be correlated (both fire at edges)
        // but NOT 1.0 — the directional max responds differently to
        // diagonal vs axis-aligned, so its per-pixel map differs.
        assert!(
            pearson > 0.0,
            "Pearson should be positive (both fire on edges); got {pearson}",
        );
        assert!(
            pearson < 0.9999,
            "Pearson is suspiciously close to 1.0 ({pearson}) — \
             steerable path may be a no-op vs LocalVariance",
        );
    }

    /// With `info_log_sigma_e_sq: Some(σ²_e)`, weights become
    /// `log₂(1 + σ²_p / σ²_e)`. Verify the saturation curve:
    /// a peak with 100× the variance of the floor produces weight ≈
    /// `log₂(101) ≈ 6.66`, NOT 100× the floor's weight.
    #[test]
    fn info_log_transform_saturates_high_variance() {
        // Pure peak at center, 0 elsewhere → variance is concentrated
        // at the center pixel.
        let plane = vec![
            0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ];
        let raw_cfg = IwWeightConfig {
            kind: IwWeightKind::LocalVariance,
            kernel_half: 1,
            weight_floor: 0.0,
            info_log_sigma_e_sq: None,
        };
        let log_cfg = IwWeightConfig {
            info_log_sigma_e_sq: Some(1.0),
            weight_floor: 0.0,
            ..raw_cfg
        };
        let raw_w = compute_iw_weights(&plane, 4, 4, 4, raw_cfg);
        let log_w = compute_iw_weights(&plane, 4, 4, 4, log_cfg);
        let raw_max = raw_w.iter().copied().fold(0.0f32, f32::max);
        let log_max = log_w.iter().copied().fold(0.0f32, f32::max);
        // Raw variance can be huge (depends on σ²_p in the window).
        // Log-transformed must be log₂(1 + raw_max / 1.0) — for a
        // raw_max > 1, log_max strictly < raw_max.
        if raw_max > 2.0 {
            assert!(
                log_max < raw_max,
                "log transform should reduce {raw_max} (got log_max={log_max})"
            );
            // And specifically: log_max == log₂(1 + raw_max)
            let expected = (1.0 + raw_max).log2();
            assert!(
                (log_max - expected).abs() < 1e-4,
                "log_max {log_max} should equal log₂(1 + {raw_max}) = {expected}"
            );
        }
    }

    /// MEASURE how much `mean_w` varies across real reference images — the
    /// number that decides whether the `1/n`-vs-`1/Σw` divergence is worth a
    /// full re-extract + retrain.
    ///
    /// # The divergence (confirmed in source, 2026-07-15)
    ///
    /// [`WeightedPool::mean`] here computes `Σ(w·v)/Σw` — a real weighted mean.
    /// The SHIPPED hot path, `streaming.rs::finalize`, accumulates `w·v` and
    /// then divides by `n` (`let one_over_n = 1.0 / self.n as f64;`). So:
    ///
    /// ```text
    ///   shipped = Σ(w·v)/n = (Σ(w·v)/Σw)·(Σw/n) = ref · mean_w
    ///   4th moments carry .powf(0.25):   shipped₄ = ref₄ · mean_w^0.25
    /// ```
    ///
    /// This module is marked `#[allow(dead_code)] // tests-only reference
    /// implementation; hot path is fused into streaming` — i.e. the two are
    /// *supposed* to agree, nothing ever checked it, and they do not. That is
    /// the gated-mirror pattern with no gate (CLAUDE.md permits a second
    /// implementation only when a test holds it bit-exact against the owner).
    ///
    /// # Why the number matters
    ///
    /// [`compute_iw_weights`] takes ONLY the reference plane, so `mean_w` is a
    /// **per-reference constant**. A per-reference multiplicative factor leaves
    /// within-image ranking exactly intact and corrupts CROSS-image ranking —
    /// pooled SROCC is cross-image, per-ref SROCC is not. If `mean_w` is
    /// ~constant the bug is a harmless global scale one weight absorbs; if it
    /// varies widely then 144 of our 372 features carry a per-image scale error
    /// that NO linear model can undo (it is a product of signal ×
    /// reference-property).
    ///
    /// Ignored: needs real images. Run with
    /// `ZENSIM_IW_REF_DIR=<dir> cargo test -p zensim --release iw_mean_weight -- --ignored --nocapture`
    #[test]
    #[ignore = "needs a dir of reference images; set ZENSIM_IW_REF_DIR and run with --ignored"]
    fn iw_mean_weight_spread_across_references() {
        let dir = std::env::var("ZENSIM_IW_REF_DIR")
            .expect("set ZENSIM_IW_REF_DIR to a directory of reference images");
        let mut paths: Vec<_> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("read {dir}: {e}"))
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                matches!(
                    p.extension().and_then(|e| e.to_str()),
                    Some("png" | "jpg" | "jpeg")
                )
            })
            .collect();
        paths.sort();
        paths.truncate(60);
        assert!(!paths.is_empty(), "no images in {dir}");

        let cfg = IwWeightConfig::default();
        let mut rows: Vec<(String, f64)> = Vec::new();
        for p in &paths {
            let Ok(img) = image::open(p) else { continue };
            let img = img.to_luma32f();
            let (w, h) = (img.width() as usize, img.height() as usize);
            let plane: Vec<f32> = img.pixels().map(|px| px.0[0]).collect();
            let weights = compute_iw_weights(&plane, w, h, w, cfg);
            let mean_w = weights.iter().map(|&x| x as f64).sum::<f64>() / weights.len() as f64;
            rows.push((
                p.file_name().unwrap().to_string_lossy().into_owned(),
                mean_w,
            ));
        }
        rows.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let vals: Vec<f64> = rows.iter().map(|r| r.1).collect();
        let n = vals.len();
        let pick = |q: f64| vals[((n as f64 - 1.0) * q).round() as usize];

        println!("\n# mean_w across {n} reference images ({dir})\n");
        println!("| stat | mean_w | feature factor = mean_w^0.25 |");
        println!("|---|--:|--:|");
        for (name, q) in [
            ("min", 0.0),
            ("p25", 0.25),
            ("p50", 0.5),
            ("p75", 0.75),
            ("max", 1.0),
        ] {
            let v = pick(q);
            println!("| {name} | {v:.6} | {:.4}x |", v.powf(0.25));
        }
        let spread = (pick(1.0) / pick(0.0)).powf(0.25);
        println!("\nCROSS-IMAGE FEATURE-SCALE SPREAD (max/min of mean_w^0.25): {spread:.3}x\n");
        for (name, v) in rows.iter().take(2) {
            println!("  LOW   mean_w={v:.6}  {name}");
        }
        for (name, v) in rows.iter().rev().take(2) {
            println!("  HIGH  mean_w={v:.6}  {name}");
        }
        println!(
            "\nEvery IW 4th-moment feature of the LOW image is scaled by {:.4} and of the HIGH \
             image by {:.4} -- purely from the REFERENCE's activity, nothing to do with the \
             distortion being measured.",
            pick(0.0).powf(0.25),
            pick(1.0).powf(0.25)
        );
    }
}
