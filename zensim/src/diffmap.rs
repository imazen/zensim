//! Per-pixel perceptual error map (diffmap) computation.
//!
//! Computes a multi-scale spatial error map from XYB-space planes using modified
//! SSIM (same as the main metric). The diffmap fuses SSIM error maps from all
//! pyramid scales, weighted by the profile's trained weights — coarser scales are
//! upsampled to full resolution and blended. This captures both fine-grained and
//! structural distortions.
//!
//! Designed for encoder quantization loops: the global zensim score tracks
//! convergence, while the diffmap tells the encoder WHERE to adjust quality.

use archmage::autoversion;

use crate::metric::{FEATURES_PER_CHANNEL_BASIC, config_from_params, validate_pair};
use crate::source::ImageSource;
use crate::streaming::PrecomputedReference;
use crate::{ZensimError, ZensimResult};

/// Channel weighting scheme for combining per-channel SSIM error into
/// the final diffmap.
///
/// The diffmap computes SSIM error independently on each XYB channel.
/// This enum controls how those three per-channel error values are combined
/// into a single per-pixel value.
///
/// # Which to use
///
/// - [`Trained`](Self::Trained) — best for encoder quant loops. Matches
///   the trained model's view of what matters: almost pure luminance.
/// - [`Balanced`](Self::Balanced) — useful for visualization or when you
///   want color errors to be visible in the map even if the model
///   doesn't weight them heavily at full resolution.
/// - [`Custom`](Self::Custom) — full control. Weights are normalized
///   to sum to 1.0.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum DiffmapWeighting {
    /// Per-scale weights derived from the profile's trained SSIM feature
    /// weights. Each scale gets its own channel weights (SSIM features per
    /// channel, normalized) and a blend weight proportional to the total
    /// weight mass at that scale.
    ///
    /// For the default profile (v0.1), scale 0 (full res) contributes ~6%
    /// of the blend, while scales 1-3 contribute ~28-35% each. At scale 0,
    /// channel weights are Y-dominant (~99.3% luminance). At coarser scales,
    /// color channels gain more influence.
    ///
    /// Automatically tracks the profile passed to [`crate::Zensim::new`].
    Trained,

    /// Moderate Y-dominant weights with visible color contribution:
    /// `[0.15, 0.70, 0.15]` (X, Y, B).
    ///
    /// Useful for visualization or when you want the diffmap to show
    /// color distortion even though the trained model doesn't weight
    /// it heavily for SSIM at scale 0.
    Balanced,

    /// Custom weights `[X, Y, B]`. Automatically normalized to sum to 1.0.
    /// All values must be non-negative.
    Custom([f32; 3]),

    /// **Research (`custom-profiles`): model-coherent weighting.** Per-feature
    /// sensitivity `s_k = ∂score/∂f_k` of the scalar model under test, laid out
    /// like the BASIC scored block (scale-major × 3 channels × 13 per-channel,
    /// the same layout `Trained` reads profile weights from). The diffmap then
    /// weights per-pixel signals by the MODEL'S own gradient instead of the
    /// static profile weights — making it (approximately) the scalar's spatial
    /// gradient. `|s_k|` is used (error polarity), with the {mean, 4th, 2nd}
    /// pooling triples folded per signal — identical arithmetic to `Trained`.
    ///
    /// Measured 2026-07-18 (`diffmap_block_coherence --bake`): the gradient
    /// ceiling (M2) is ≈1.0 for basic-input models (incl. LeakyReLU MLPs,
    /// which are piecewise-linear ⇒ locally exact); this weighting is the
    /// deployable approximation of that gradient.
    #[cfg(feature = "custom-profiles")]
    ModelSensitivity(&'static [f64]),
}

impl Default for DiffmapWeighting {
    /// Defaults to [`Trained`](Self::Trained).
    fn default() -> Self {
        Self::Trained
    }
}

/// Post-processing options for the diffmap signal.
///
/// These are applied after multi-scale fusion to shape the diffmap for
/// specific use cases. All options default to off (raw SSIM error signal).
#[derive(Debug, Clone, Copy, Default)]
pub struct DiffmapOptions {
    /// Channel weighting scheme (default: `Trained`).
    pub weighting: DiffmapWeighting,

    /// Apply contrast masking to suppress errors in complex/textured regions.
    ///
    /// When enabled, each diffmap value is divided by a local complexity mask
    /// derived from the source image's luminance variance:
    /// `mask = 1 + strength × local_variance(Y_src)`.
    ///
    /// This matches the perceptual intuition that errors in smooth areas
    /// (sky, gradients) are more visible than in textured areas (foliage, hair).
    ///
    /// Default: `None` (no masking). Typical values: 2.0–8.0.
    pub masking_strength: Option<f32>,

    /// Apply `sqrt` to diffmap values after masking.
    ///
    /// Compresses the dynamic range so that the diffmap is more linearly
    /// proportional to perceived distortion intensity. Similar to butteraugli's
    /// `sqrt(dc_masked + ac_masked)` combination.
    ///
    /// Default: `false`.
    pub sqrt: bool,

    /// Include edge artifact, edge detail loss, and MSE features in the diffmap.
    ///
    /// When `false` (default), the diffmap only includes SSIM error.
    /// When `true`, each pixel also accumulates:
    /// - **Edge artifact**: `max(0, residual_dst² − residual_src²)` — structure
    ///   added by distortion (ringing, blocking).
    /// - **Edge detail loss**: `max(0, residual_src² − residual_dst²)` — structure
    ///   removed by distortion (blurring, smoothing).
    /// - **MSE**: `(src − dst)²` — raw per-pixel squared difference.
    ///
    /// Feature contributions are weighted by the profile's trained weights for
    /// each feature type and channel. This makes the diffmap a more complete
    /// representation of the metric's per-pixel view.
    pub include_edge_mse: bool,

    /// Include high-frequency energy features in the diffmap.
    ///
    /// When `true`, each pixel also accumulates HF texture loss/gain signals:
    /// - **HF energy loss**: `max(0, (src−μ)² − (dst−μ)²)` — texture energy
    ///   removed by distortion (smoothing, quantization).
    /// - **HF magnitude loss**: `max(0, |src−μ| − |dst−μ|)` — texture magnitude
    ///   removed (L1 variant, more robust to outliers).
    /// - **HF energy gain**: `max(0, (dst−μ)² − (src−μ)²)` — texture energy
    ///   added by distortion (ringing, noise).
    ///
    /// These correspond to trained features 10–12 and help encoder quantization
    /// loops (especially JXL AC coefficient tuning) see where HF texture is
    /// being lost or gained. Requires mu1/mu2 storage (same as `include_edge_mse`).
    ///
    /// Default: `false`.
    pub include_hf: bool,

    /// **Research (`custom-profiles`): guided mass-conserving redistribution of
    /// coarse-scale contributions (E-JBU, 2026-07-30).** When `true`, each
    /// coarse-scale (s1–s3) contribution cell deposits its exact NN mass within
    /// its aligned `2^s × 2^s` footprint proportional to the full-res scale-0
    /// contribution plane (+ε) instead of replicating the value uniformly:
    /// `out(x,y) += cell_mass · g(x,y) / Σ_cell g`.
    ///
    /// Per-cell mass-conserving by construction: the pooled map total, the
    /// scalar score, and any grid-aligned block aggregation at ≥ the footprint
    /// size (8px) are unchanged (up to f32 summation order) — only
    /// sub-footprint localization sharpens. ε-fallback: a flat or all-zero
    /// scale-0 plane reproduces the current NN behavior exactly. O(N), diffmap
    /// render only — never on the scoring path.
    ///
    /// Default: `false` (current NN-replicate behavior).
    #[cfg(feature = "custom-profiles")]
    pub guided_coarse_redistribution: bool,
}

impl From<DiffmapWeighting> for DiffmapOptions {
    fn from(w: DiffmapWeighting) -> Self {
        Self {
            weighting: w,
            ..Default::default()
        }
    }
}

/// Per-pixel feature weights for a single channel in the diffmap.
///
/// When `art`, `det`, `mse` are all zero, only SSIM contributes (backward compatible).
/// HF features (`hf_loss`, `hf_mag`, `hf_gain`) add high-frequency texture sensitivity.
#[derive(Clone, Copy, Default)]
pub(crate) struct PixelFeatureWeights {
    pub ssim: f32,
    pub art: f32,
    pub det: f32,
    pub mse: f32,
    pub hf_loss: f32,
    pub hf_mag: f32,
    pub hf_gain: f32,
}

impl PixelFeatureWeights {
    /// Whether any edge/MSE features are active (need mu1/mu2 storage).
    pub fn needs_edge_mse(&self) -> bool {
        self.art != 0.0 || self.det != 0.0 || self.mse != 0.0
    }

    /// Whether any HF features are active (need mu1/mu2 storage).
    pub fn needs_hf(&self) -> bool {
        self.hf_loss != 0.0 || self.hf_mag != 0.0 || self.hf_gain != 0.0
    }
}

impl DiffmapWeighting {
    /// Return per-scale per-channel feature weights and scale blend weights.
    ///
    /// `per_scale_weights[s][c]` = feature weights for scale `s`, channel `c`.
    /// `scale_blend_weights[s]` = fraction of total weight mass at scale `s`.
    fn resolve_multiscale(
        self,
        profile_weights: &[f64],
        num_scales: usize,
        include_edge_mse: bool,
        include_hf: bool,
    ) -> (Vec<[PixelFeatureWeights; 3]>, Vec<f32>) {
        match self {
            Self::Trained => trained_multiscale_weights(
                profile_weights,
                num_scales,
                include_edge_mse,
                include_hf,
            ),
            // Model-coherent: SIGNED clamped fold of the model's own gradient
            // (v2 2026-07-18 — the abs-fold v1 mis-weighted positive-sensitivity
            // features like the winner MLP's hf_gain and scored WORSE than the
            // shipped map on that bake: median M3 0.446 vs M1 0.746).
            #[cfg(feature = "custom-profiles")]
            Self::ModelSensitivity(s) => {
                model_sensitivity_weights(s, num_scales, include_edge_mse, include_hf)
            }
            Self::Balanced => {
                let pw = PixelFeatureWeights {
                    ssim: 1.0,
                    ..Default::default()
                };
                let ch = [
                    PixelFeatureWeights { ssim: 0.15, ..pw },
                    PixelFeatureWeights { ssim: 0.70, ..pw },
                    PixelFeatureWeights { ssim: 0.15, ..pw },
                ];
                // For Balanced+edge_mse+hf, distribute equally among active feature groups
                let n_groups = 1 + include_edge_mse as usize * 3 + include_hf as usize * 3;
                let per_group = 1.0 / n_groups as f32;
                let ch = ch.map(|c| PixelFeatureWeights {
                    ssim: c.ssim * per_group,
                    art: if include_edge_mse {
                        c.ssim * per_group
                    } else {
                        0.0
                    },
                    det: if include_edge_mse {
                        c.ssim * per_group
                    } else {
                        0.0
                    },
                    mse: if include_edge_mse {
                        c.ssim * per_group
                    } else {
                        0.0
                    },
                    hf_loss: if include_hf { c.ssim * per_group } else { 0.0 },
                    hf_mag: if include_hf { c.ssim * per_group } else { 0.0 },
                    hf_gain: if include_hf { c.ssim * per_group } else { 0.0 },
                });
                let blend = 1.0 / num_scales as f32;
                (vec![ch; num_scales], vec![blend; num_scales])
            }
            Self::Custom(w) => {
                let nw = normalize_weights(w);
                let ch = [nw[0], nw[1], nw[2]].map(|s| {
                    let n_groups = 1 + include_edge_mse as usize * 3 + include_hf as usize * 3;
                    let per_group = 1.0 / n_groups as f32;
                    PixelFeatureWeights {
                        ssim: s * per_group,
                        art: if include_edge_mse { s * per_group } else { 0.0 },
                        det: if include_edge_mse { s * per_group } else { 0.0 },
                        mse: if include_edge_mse { s * per_group } else { 0.0 },
                        hf_loss: if include_hf { s * per_group } else { 0.0 },
                        hf_mag: if include_hf { s * per_group } else { 0.0 },
                        hf_gain: if include_hf { s * per_group } else { 0.0 },
                    }
                });
                let blend = 1.0 / num_scales as f32;
                (vec![ch; num_scales], vec![blend; num_scales])
            }
        }
    }
}

/// Derive per-scale per-feature channel weights and scale blend weights.
///
/// For each scale and channel, computes weights for SSIM, edge artifact,
/// edge detail loss, MSE, and HF features from the trained weight array.
///
/// When `include_edge_mse` is false, edge/MSE weights are 0.
/// When `include_hf` is false, HF weights are 0.
fn trained_multiscale_weights(
    weights: &[f64],
    num_scales: usize,
    include_edge_mse: bool,
    include_hf: bool,
) -> (Vec<[PixelFeatureWeights; 3]>, Vec<f32>) {
    const FPC: usize = FEATURES_PER_CHANNEL_BASIC;
    const FPS: usize = FPC * 3; // features per scale (basic only)

    let mut per_scale = Vec::with_capacity(num_scales);
    let mut scale_totals = Vec::with_capacity(num_scales);

    for s in 0..num_scales {
        let scale_base = s * FPS;
        // Per-channel feature weight sums
        let mut ssim_w = [0.0f64; 3];
        let mut art_w = [0.0f64; 3];
        let mut det_w = [0.0f64; 3];
        let mut mse_w = [0.0f64; 3];
        let mut hf_loss_w = [0.0f64; 3];
        let mut hf_mag_w = [0.0f64; 3];
        let mut hf_gain_w = [0.0f64; 3];
        let mut scale_total = 0.0f64;

        for c in 0..3 {
            let base = scale_base + c * FPC;
            if base + 2 < weights.len() {
                ssim_w[c] = weights[base].abs() + weights[base + 1].abs() + weights[base + 2].abs();
            }
            if include_edge_mse && base + 9 < weights.len() {
                art_w[c] =
                    weights[base + 3].abs() + weights[base + 4].abs() + weights[base + 5].abs();
                det_w[c] =
                    weights[base + 6].abs() + weights[base + 7].abs() + weights[base + 8].abs();
                mse_w[c] = weights[base + 9].abs();
            }
            if include_hf && base + 12 < weights.len() {
                hf_loss_w[c] = weights[base + 10].abs();
                hf_mag_w[c] = weights[base + 11].abs();
                hf_gain_w[c] = weights[base + 12].abs();
            }
            // Sum ALL features at this scale for blend weight
            for f in 0..FPC {
                if base + f < weights.len() {
                    scale_total += weights[base + f].abs();
                }
            }
        }

        // Normalize: all feature weights across all channels sum to 1.0
        let feat_total: f64 = ssim_w.iter().sum::<f64>()
            + art_w.iter().sum::<f64>()
            + det_w.iter().sum::<f64>()
            + mse_w.iter().sum::<f64>()
            + hf_loss_w.iter().sum::<f64>()
            + hf_mag_w.iter().sum::<f64>()
            + hf_gain_w.iter().sum::<f64>();

        let ch_weights = if feat_total > 0.0 {
            core::array::from_fn(|c| PixelFeatureWeights {
                ssim: (ssim_w[c] / feat_total) as f32,
                art: (art_w[c] / feat_total) as f32,
                det: (det_w[c] / feat_total) as f32,
                mse: (mse_w[c] / feat_total) as f32,
                hf_loss: (hf_loss_w[c] / feat_total) as f32,
                hf_mag: (hf_mag_w[c] / feat_total) as f32,
                hf_gain: (hf_gain_w[c] / feat_total) as f32,
            })
        } else {
            let eq = 1.0 / 3.0;
            [PixelFeatureWeights {
                ssim: eq,
                ..Default::default()
            }; 3]
        };

        per_scale.push(ch_weights);
        scale_totals.push(scale_total);
    }

    // Normalize scale blend weights
    let total: f64 = scale_totals.iter().sum();
    let blend = if total > 0.0 {
        scale_totals.iter().map(|&s| (s / total) as f32).collect()
    } else {
        let w = 1.0 / num_scales as f32;
        vec![w; num_scales]
    };

    (per_scale, blend)
}

/// Model-sensitivity weight mapping (`DiffmapWeighting::ModelSensitivity`).
///
/// Like [`trained_multiscale_weights`] but with a **signed** fold: the
/// {mean, 4th, 2nd} pooling triple of each signal is summed WITH sign and the
/// per-pixel weight is `−Σ s_k` — a score-oriented model has NEGATIVE
/// sensitivity on true-error signals (error ↑ ⇒ score ↓ ⇒ refining there
/// gains score → positive map weight), while a POSITIVE-sensitivity signal
/// (e.g. the winner MLP's winsorized `hf_gain`: artifact energy the model
/// REWARDS) gets a NEGATIVE map weight — blocks dominated by it rank below
/// zero, correctly predicting that refining them LOSES score. The map is
/// therefore signed; the linear fusion carries negatives through (masking /
/// sqrt options must stay off with this weighting). Normalization uses
/// |mass| so signs survive. The abs-fold of `Trained` is correct for profile
/// weight VECTORS (sign encodes distance polarity there) but wrong for
/// gradients — measured 2026-07-18: abs-fold scored WORSE than the shipped
/// map on the sign-mixed winner MLP (median M3 0.446 vs M1 0.746).
#[cfg(feature = "custom-profiles")]
fn model_sensitivity_weights(
    s: &[f64],
    num_scales: usize,
    include_edge_mse: bool,
    include_hf: bool,
) -> (Vec<[PixelFeatureWeights; 3]>, Vec<f32>) {
    const FPC: usize = FEATURES_PER_CHANNEL_BASIC;
    const FPS: usize = FPC * 3;
    // Signed triple fold; `get` = 0.0 beyond the slice (basic-only slices).
    let g = |i: usize| s.get(i).copied().unwrap_or(0.0);

    let mut per_scale = Vec::with_capacity(num_scales);
    let mut scale_totals = Vec::with_capacity(num_scales);
    for sc in 0..num_scales {
        let scale_base = sc * FPS;
        let mut w = [[0.0f64; 7]; 3]; // [channel][ssim,art,det,mse,hf_loss,hf_mag,hf_gain]
        for (c, wc) in w.iter_mut().enumerate() {
            let base = scale_base + c * FPC;
            wc[0] = -(g(base) + g(base + 1) + g(base + 2));
            if include_edge_mse {
                wc[1] = -(g(base + 3) + g(base + 4) + g(base + 5));
                wc[2] = -(g(base + 6) + g(base + 7) + g(base + 8));
                wc[3] = -g(base + 9);
            }
            if include_hf {
                wc[4] = -g(base + 10);
                wc[5] = -g(base + 11);
                wc[6] = -g(base + 12);
            }
        }
        let feat_total: f64 = w.iter().flat_map(|c| c.iter()).map(|v| v.abs()).sum();
        let ch_weights = if feat_total > 0.0 {
            core::array::from_fn(|c| PixelFeatureWeights {
                ssim: (w[c][0] / feat_total) as f32,
                art: (w[c][1] / feat_total) as f32,
                det: (w[c][2] / feat_total) as f32,
                mse: (w[c][3] / feat_total) as f32,
                hf_loss: (w[c][4] / feat_total) as f32,
                hf_mag: (w[c][5] / feat_total) as f32,
                hf_gain: (w[c][6] / feat_total) as f32,
            })
        } else {
            let eq = 1.0 / 3.0;
            [PixelFeatureWeights {
                ssim: eq,
                ..Default::default()
            }; 3]
        };
        per_scale.push(ch_weights);
        scale_totals.push(feat_total);
    }
    let total: f64 = scale_totals.iter().sum();
    let blend = if total > 0.0 {
        scale_totals.iter().map(|&t| (t / total) as f32).collect()
    } else {
        vec![1.0 / num_scales as f32; num_scales]
    };
    (per_scale, blend)
}

/// Reflect(mirror)-pad linear-f32 RGB planes up to at least `MIN_PYRAMID_DIM`
/// per dim, returning tightly-packed `(planes, padded_width, padded_height)`.
/// The planar-f32 analogue of `crate::metric::reflect_pad_to_min`; keeps the
/// original image in the top-left so the diffmap can be trimmed back.
fn reflect_pad_planes_f32(
    planes: [&[f32]; 3],
    w: usize,
    h: usize,
    stride: usize,
) -> ([Vec<f32>; 3], usize, usize) {
    use crate::metric::{MIN_PYRAMID_DIM, reflect_index};
    let (bw, bh) = (w.max(MIN_PYRAMID_DIM), h.max(MIN_PYRAMID_DIM));
    let mut out = [
        vec![0f32; bw * bh],
        vec![0f32; bw * bh],
        vec![0f32; bw * bh],
    ];
    for y in 0..bh {
        let sy = reflect_index(y, h);
        for x in 0..bw {
            let sx = reflect_index(x, w);
            let si = sy * stride + sx;
            let di = y * bw + x;
            out[0][di] = planes[0][si];
            out[1][di] = planes[1][si];
            out[2][di] = planes[2][si];
        }
    }
    (out, bw, bh)
}

/// Apply contrast masking to the diffmap using source luminance variance.
///
/// Divides each diffmap value by `1 + strength * local_variance(Y_src)`,
/// where local_variance is computed from the Y plane of the precomputed
/// reference at scale 0, using integral images for O(1) per-pixel variance.
fn apply_contrast_masking(
    diffmap: &mut [f32],
    precomputed: &PrecomputedReference,
    width: usize,
    height: usize,
    padded_width: usize,
    strength: f32,
) {
    let (ref planes, pw, ph) = precomputed.scales[0];
    debug_assert_eq!(pw, padded_width);
    debug_assert_eq!(ph, height);
    let y_plane = &planes[1]; // Y channel = luminance

    // Use a 5-pixel radius (11x11 window) matching the SSIM blur.
    let r = 5usize;

    // Build integral images for sum(Y) and sum(Y^2).
    // Layout: (width+1) x (height+1), with zero-padded top row and left column.
    let iw = width + 1;
    let ih = height + 1;
    let mut int_sum = vec![0.0f64; iw * ih];
    let mut int_sq = vec![0.0f64; iw * ih];

    for y in 0..height {
        let mut row_sum = 0.0f64;
        let mut row_sq = 0.0f64;
        for x in 0..width {
            let v = y_plane[y * padded_width + x] as f64;
            row_sum += v;
            row_sq += v * v;
            let idx = (y + 1) * iw + (x + 1);
            int_sum[idx] = row_sum + int_sum[idx - iw];
            int_sq[idx] = row_sq + int_sq[idx - iw];
        }
    }

    // Apply masking using integral image lookups for O(1) variance per pixel.
    let dims = [width, iw, r];
    for dy in 0..height {
        apply_masking_row(
            &mut diffmap[dy * width..(dy + 1) * width],
            &int_sum,
            &int_sq,
            dy,
            dims,
            height,
            strength,
        );
    }
}

/// Apply contrast masking for one row using precomputed integral images.
///
/// `dims` is packed as `[width, iw, r]` to keep param count low for autoversion.
///
/// The inner range `[r, width-r-1]` has a constant box size `(y1-y0)*(2r+1)`,
/// so its `inv_count` is hoisted out of the loop and the division becomes a
/// multiply. Edge ranges keep the full per-pixel computation.
#[autoversion]
fn apply_masking_row(
    dm_row: &mut [f32],
    int_sum: &[f64],
    int_sq: &[f64],
    dy: usize,
    dims: [usize; 3],
    height: usize,
    strength: f32,
) {
    let [width, iw, r] = dims;
    let y0 = dy.saturating_sub(r);
    let y1 = (dy + r + 1).min(height);
    let yh = y1 - y0;

    // Edge handling: for dx in [0, r) and [width-r, width), the box is clipped
    // by image boundaries so the count varies per dx. Beyond that, count is
    // constant — hoist its reciprocal and turn /count and /mask into multiplies.
    let edge_lo = r.min(width);
    let edge_hi = width.saturating_sub(r);

    let masked_div = |dm_val: &mut f32, x0: usize, x1: usize| {
        let tl = y0 * iw + x0;
        let tr = y0 * iw + x1;
        let bl = y1 * iw + x0;
        let br = y1 * iw + x1;
        let sum = int_sum[br] - int_sum[tr] - int_sum[bl] + int_sum[tl];
        let sq = int_sq[br] - int_sq[tr] - int_sq[bl] + int_sq[tl];
        let count = (yh * (x1 - x0)) as f64;
        let mean = sum / count;
        let variance = (sq / count - mean * mean).max(0.0) as f32;
        let mask = 1.0 + strength * variance;
        *dm_val /= mask;
    };

    // Left edge (x1 - x0 < 2r+1)
    for (dx, dm_val) in dm_row.iter_mut().enumerate().take(edge_lo) {
        let x0 = dx.saturating_sub(r);
        let x1 = (dx + r + 1).min(width);
        masked_div(dm_val, x0, x1);
    }

    // Interior: constant count = yh * (2r+1). Hoist inv_count out of the loop
    // and keep the box-query + mean + variance computation in f64 (the
    // box-corner subtractions can lose precision in f32 because the integral
    // image accumulates over the whole image).
    if edge_hi > edge_lo {
        let count = (yh * (2 * r + 1)) as f64;
        let inv_count = 1.0 / count;
        let row_y0 = y0 * iw;
        let row_y1 = y1 * iw;
        for (dx, dm_val) in dm_row.iter_mut().enumerate().take(edge_hi).skip(edge_lo) {
            let x0 = dx - r;
            let x1 = dx + r + 1;
            let tl = row_y0 + x0;
            let tr = row_y0 + x1;
            let bl = row_y1 + x0;
            let br = row_y1 + x1;
            let sum = int_sum[br] - int_sum[tr] - int_sum[bl] + int_sum[tl];
            let sq = int_sq[br] - int_sq[tr] - int_sq[bl] + int_sq[tl];
            let mean = sum * inv_count;
            let variance = (sq * inv_count - mean * mean).max(0.0) as f32;
            let mask = 1.0 + strength * variance;
            *dm_val *= 1.0 / mask;
        }
    }

    // Right edge
    for (dx, dm_val) in dm_row
        .iter_mut()
        .enumerate()
        .take(width)
        .skip(edge_hi.max(edge_lo))
    {
        let x0 = dx.saturating_sub(r);
        let x1 = (dx + r + 1).min(width);
        masked_div(dm_val, x0, x1);
    }
}

/// Element-wise sqrt, auto-vectorized.
#[autoversion]
fn sqrt_inplace(data: &mut [f32]) {
    for v in data.iter_mut() {
        *v = v.sqrt();
    }
}

fn normalize_weights(w: [f32; 3]) -> [f32; 3] {
    let sum = w[0] + w[1] + w[2];
    if sum > 0.0 {
        [w[0] / sum, w[1] / sum, w[2] / sum]
    } else {
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]
    }
}

/// Result containing both a zensim score and a per-pixel error map.
///
/// The diffmap is a multi-scale spatial error map using modified SSIM across
/// all three XYB channels. SSIM error maps are computed at each pyramid scale
/// (4 scales by default), then coarser scales are upsampled to full resolution
/// and blended with the profile's trained scale weights. This captures both
/// fine-grained detail loss (scale 0) and structural/low-frequency distortions
/// (scales 1-3).
///
/// It tells you WHERE perceptual error is concentrated, while the zensim score
/// tells you HOW MUCH total error there is.
///
/// # Diffmap signal
///
/// Each value is a weighted blend of per-scale SSIM errors:
///
/// ```text
/// per_channel_s = (1 − mean_term × structure_term).max(0)   [at scale s]
///   mean_term      = 1 − (μ_src − μ_dst)²
///   structure_term  = (2·σ_src_dst + C₂) / (σ²_src + σ²_dst + C₂)
///   C₂ = 0.0009
///
/// scale_error_s[i] = Σ_c w_c_s × per_channel_s_c   (channel-weighted)
/// diffmap[i] = Σ_s blend_s × upsample(scale_error_s)[i]   (scale-blended)
/// ```
///
/// With trained weights: scale 0 gets ~6%, scales 1-3 get ~28-35% each
/// (matching the trained model's emphasis on structural features).
///
/// - **Range**: `[0, +∞)`. Zero means perceptually identical at that location.
///   Values above ~0.5 indicate severe local distortion. On typical photo pairs,
///   most values fall in `[0, 0.3]`.
/// - **Spatial smoothing**: scale 0 reflects 11×11 neighborhoods, coarser scales
///   reflect progressively larger areas (22×22, 44×44, 88×88).
/// - **Color space**: computed in XYB (perceptual luma + chroma). Channel
///   combination weights are controlled by [`DiffmapWeighting`].
/// - **Not butteraugli distance**: the values are unitless SSIM error, not
///   butteraugli distance units. Use the global zensim score for overall quality.
///
/// # Layout
///
/// Row-major, `width × height` elements: `diffmap[y * width + x]`.
/// No padding — actual image dimensions, not SIMD-padded.
#[non_exhaustive]
pub struct DiffmapResult {
    result: ZensimResult,
    diffmap: Vec<f32>,
    width: usize,
    height: usize,
}

impl DiffmapResult {
    /// The full zensim comparison result (score, features, etc.).
    pub fn result(&self) -> &ZensimResult {
        &self.result
    }

    /// The zensim score (convenience shorthand for `result().score()`).
    pub fn score(&self) -> f64 {
        self.result.score()
    }

    /// Per-pixel perceptual error map. See [`DiffmapResult`] for signal
    /// definition, range, and layout.
    pub fn diffmap(&self) -> &[f32] {
        &self.diffmap
    }

    /// Consume this result and return the diffmap as an owned `Vec<f32>`,
    /// along with the zensim result and dimensions.
    pub fn into_parts(self) -> (ZensimResult, Vec<f32>, usize, usize) {
        (self.result, self.diffmap, self.width, self.height)
    }

    /// Image width (actual, not SIMD-padded).
    pub fn width(&self) -> usize {
        self.width
    }

    /// Image height.
    pub fn height(&self) -> usize {
        self.height
    }
}

impl crate::metric::Zensim {
    /// Compare with precomputed reference and return both score and per-pixel error map.
    ///
    /// The diffmap fuses SSIM error maps from all pyramid scales, weighted by the
    /// profile's trained weights. Coarser scales are upsampled to full resolution
    /// and blended. The zensim score uses the identical multi-scale pipeline.
    ///
    /// # Use case
    ///
    /// Encoder quantization loops: precompute the reference once, then in each iteration
    /// call this to get both a global quality score (for convergence) and a spatial error
    /// map (for per-block quant field adjustment).
    ///
    /// # Errors
    ///
    /// Returns [`ZensimError::ImageTooSmall`] if either dimension is zero.
    pub fn compute_with_ref_and_diffmap(
        &self,
        precomputed: &PrecomputedReference,
        distorted: &impl ImageSource,
        options: impl Into<DiffmapOptions>,
    ) -> Result<DiffmapResult, ZensimError> {
        let options = options.into();
        let params = self.profile().params();
        if distorted.width() == 0 || distorted.height() == 0 {
            return Err(ZensimError::ImageTooSmall);
        }
        crate::metric::validate_ref_match(precomputed, distorted)?;
        crate::metric::check_within_max_pixels(
            distorted.width(),
            distorted.height(),
            self.max_pixels(),
        )?;

        let width = distorted.width();
        let height = distorted.height();
        let config = config_from_params(params, self.parallel());
        let (per_scale_ch, scale_blend) = options.weighting.resolve_multiscale(
            params.weights,
            config.num_scales,
            options.include_edge_mse,
            options.include_hf,
        );

        // E-JBU (research): guided coarse-scale redistribution, opt-in via
        // `DiffmapOptions::guided_coarse_redistribution` (custom-profiles only;
        // hard-false otherwise so the default fold path is untouched).
        #[cfg(feature = "custom-profiles")]
        let guided = options.guided_coarse_redistribution;
        #[cfg(not(feature = "custom-profiles"))]
        let guided = false;

        // Fused: compute the full zensim score AND the multi-scale diffmap in a
        // single pipeline. Each scale's SSIM error is collected, then coarser
        // scales are upsampled and blended with trained scale weights.
        // Sub-64px distorted is reflect-padded to align with the (padded)
        // reference pyramid; the reflect-pad keeps the original image in the
        // top-left, so the diffmap trim below recovers it via `width`/`height`.
        let (result, mut diffmap_padded, padded_width) =
            if width < crate::metric::MIN_PYRAMID_DIM || height < crate::metric::MIN_PYRAMID_DIM {
                let d = crate::metric::reflect_pad_to_min(distorted);
                crate::streaming::compute_zensim_streaming_with_ref_and_diffmap(
                    precomputed,
                    &d,
                    &config,
                    params.weights,
                    &per_scale_ch,
                    &scale_blend,
                    guided,
                )
            } else {
                crate::streaming::compute_zensim_streaming_with_ref_and_diffmap(
                    precomputed,
                    distorted,
                    &config,
                    params.weights,
                    &per_scale_ch,
                    &scale_blend,
                    guided,
                )
            };
        let mut result = result.with_profile(self.profile());
        // Apply the profile's REAL scoring (bake forward + spline), exactly as
        // `compute()` does. FIX 2026-07-18: this path previously returned the
        // legacy V0_2 dot-product score for EVERY profile (apply_mlp_scoring
        // was never called), so encoder closed loops steering on
        // `DiffmapResult::score()` were tracking V0_2 — not the shipped
        // metric. Two paths scoring the same pair differently is a bug per the
        // parity contract; no-op for linear-weight legacy profiles
        // (`mlp_bytes: None` early-returns).
        crate::metric::apply_mlp_scoring_with_codec(
            &mut result,
            params,
            width as u32,
            height as u32,
            None,
        )?;

        // `precomputed.scales[0]` is `(padded_width, comp_h)`. For a ≥64px
        // distorted `comp_h == height` (original flow: trim to logical width,
        // then mask). For a reflect-padded sub-64px distorted the diffmap is
        // computed at the padded reference dims, so mask + sqrt there (the
        // masking reads the ref Y-plane at scale 0) then trim the original
        // top-left width×height (reflect-pad keeps the original there).
        let comp_h = precomputed.scales[0].2;
        let diffmap = if comp_h == height {
            let mut diffmap = if padded_width == width {
                diffmap_padded
            } else {
                let mut out = Vec::with_capacity(width * height);
                for y in 0..height {
                    out.extend_from_slice(
                        &diffmap_padded[y * padded_width..y * padded_width + width],
                    );
                }
                out
            };
            if let Some(strength) = options.masking_strength {
                apply_contrast_masking(
                    &mut diffmap,
                    precomputed,
                    width,
                    height,
                    padded_width,
                    strength,
                );
            }
            if options.sqrt {
                sqrt_inplace(&mut diffmap);
            }
            diffmap
        } else {
            if let Some(strength) = options.masking_strength {
                apply_contrast_masking(
                    &mut diffmap_padded,
                    precomputed,
                    padded_width,
                    comp_h,
                    padded_width,
                    strength,
                );
            }
            if options.sqrt {
                sqrt_inplace(&mut diffmap_padded);
            }
            let mut out = Vec::with_capacity(width * height);
            for y in 0..height {
                out.extend_from_slice(&diffmap_padded[y * padded_width..y * padded_width + width]);
            }
            out
        };

        Ok(DiffmapResult {
            result,
            diffmap,
            width,
            height,
        })
    }

    /// Compare planar linear RGB f32 against a precomputed reference, producing
    /// both a score and a per-pixel error map.
    ///
    /// `planes` are `[R, G, B]`, each with at least `stride * height` elements.
    /// `stride` is the number of f32 elements per row (≥ `width`).
    ///
    /// This avoids the interleave-to-RGBA overhead when the caller already has
    /// separate channel buffers in linear light (e.g., from an encoder's
    /// reconstruction pipeline).
    ///
    /// # Errors
    ///
    /// Returns [`ZensimError::ImageTooSmall`] if either dimension is zero.
    /// Returns [`ZensimError::DimensionMismatch`] if `width` / `height`
    /// differ from the precomputed reference's recorded dimensions.
    /// Returns [`ZensimError::ImageTooLarge`] if `width × height` exceeds
    /// the configured `max_pixels` cap or if `stride × height` overflows
    /// `usize`. Returns [`ZensimError::InvalidStride`] if `stride < width`.
    /// Returns [`ZensimError::InvalidDataLength`] if any plane is shorter
    /// than `stride × height`.
    pub fn compute_with_ref_and_diffmap_linear_planar(
        &self,
        precomputed: &PrecomputedReference,
        planes: [&[f32]; 3],
        width: usize,
        height: usize,
        stride: usize,
        options: impl Into<DiffmapOptions>,
    ) -> Result<DiffmapResult, ZensimError> {
        let options = options.into();
        let params = self.profile().params();
        if width == 0 || height == 0 {
            return Err(ZensimError::ImageTooSmall);
        }
        if precomputed.width() != width || precomputed.height() != height {
            return Err(ZensimError::DimensionMismatch);
        }
        crate::metric::check_within_max_pixels(width, height, self.max_pixels())?;
        if stride < width {
            return Err(ZensimError::InvalidStride);
        }
        // `stride * height` must fit in usize; reject overflow up front so
        // downstream `y * stride + x` arithmetic cannot wrap on 32-bit.
        let row_capacity = stride
            .checked_mul(height)
            .ok_or(ZensimError::ImageTooLarge)?;
        for plane in &planes {
            if plane.len() < row_capacity {
                return Err(ZensimError::InvalidDataLength);
            }
        }

        let config = config_from_params(params, self.parallel());
        let (per_scale_ch, scale_blend) = options.weighting.resolve_multiscale(
            params.weights,
            config.num_scales,
            options.include_edge_mse,
            options.include_hf,
        );

        // E-JBU (research): same opt-in as `compute_with_ref_and_diffmap`.
        #[cfg(feature = "custom-profiles")]
        let guided = options.guided_coarse_redistribution;
        #[cfg(not(feature = "custom-profiles"))]
        let guided = false;

        // Sub-64px planes are reflect-padded to the pyramid minimum (matching
        // the also-padded reference). The reflect-pad keeps the original in the
        // top-left, recovered by the trim below.
        let (result, mut diffmap_padded, padded_width) =
            if width < crate::metric::MIN_PYRAMID_DIM || height < crate::metric::MIN_PYRAMID_DIM {
                let (pp, pw, ph) = reflect_pad_planes_f32(planes, width, height, stride);
                let padded_width = crate::blur::simd_padded_width(pw);
                let (result, dm, _) =
                    crate::streaming::compute_zensim_streaming_with_ref_and_diffmap_linear_planar(
                        precomputed,
                        [&pp[0], &pp[1], &pp[2]],
                        pw,
                        ph,
                        pw,
                        &config,
                        params.weights,
                        &per_scale_ch,
                        &scale_blend,
                        guided,
                    );
                (result, dm, padded_width)
            } else {
                let padded_width = crate::blur::simd_padded_width(width);
                let (result, dm, _) =
                    crate::streaming::compute_zensim_streaming_with_ref_and_diffmap_linear_planar(
                        precomputed,
                        planes,
                        width,
                        height,
                        stride,
                        &config,
                        params.weights,
                        &per_scale_ch,
                        &scale_blend,
                        guided,
                    );
                (result, dm, padded_width)
            };
        let mut result = result.with_profile(self.profile());
        // Same real-scoring fix as `compute_with_ref_and_diffmap` (2026-07-18):
        // apply the profile's bake forward + spline; no-op for legacy profiles.
        crate::metric::apply_mlp_scoring_with_codec(
            &mut result,
            params,
            width as u32,
            height as u32,
            None,
        )?;

        // Mask + sqrt at the compute dims, then trim to the original (same
        // padded-vs-original branch as the RGB `compute_with_ref_and_diffmap`).
        let comp_h = precomputed.scales[0].2;
        let diffmap = if comp_h == height {
            let mut diffmap = if padded_width == width {
                diffmap_padded
            } else {
                let mut out = Vec::with_capacity(width * height);
                for y in 0..height {
                    out.extend_from_slice(
                        &diffmap_padded[y * padded_width..y * padded_width + width],
                    );
                }
                out
            };
            if let Some(strength) = options.masking_strength {
                apply_contrast_masking(
                    &mut diffmap,
                    precomputed,
                    width,
                    height,
                    padded_width,
                    strength,
                );
            }
            if options.sqrt {
                sqrt_inplace(&mut diffmap);
            }
            diffmap
        } else {
            if let Some(strength) = options.masking_strength {
                apply_contrast_masking(
                    &mut diffmap_padded,
                    precomputed,
                    padded_width,
                    comp_h,
                    padded_width,
                    strength,
                );
            }
            if options.sqrt {
                sqrt_inplace(&mut diffmap_padded);
            }
            let mut out = Vec::with_capacity(width * height);
            for y in 0..height {
                out.extend_from_slice(&diffmap_padded[y * padded_width..y * padded_width + width]);
            }
            out
        };

        Ok(DiffmapResult {
            result,
            diffmap,
            width,
            height,
        })
    }

    /// Compare two images and return both score and per-pixel error map.
    ///
    /// Convenience method that handles precomputation internally.
    /// For iterative use (encoder loops), prefer `precompute_reference` +
    /// `compute_with_ref_and_diffmap` to avoid re-converting the reference each time.
    ///
    /// # Errors
    ///
    /// Returns [`ZensimError`] if dimensions are mismatched or too small.
    pub fn compute_with_diffmap(
        &self,
        source: &impl ImageSource,
        distorted: &impl ImageSource,
        options: impl Into<DiffmapOptions>,
    ) -> Result<DiffmapResult, ZensimError> {
        validate_pair(source, distorted)?;
        let precomputed = self.precompute_reference(source)?;
        self.compute_with_ref_and_diffmap(&precomputed, distorted, options)
    }
}

#[cfg(test)]
mod tests {
    use super::DiffmapWeighting;
    use crate::{RgbSlice, ZensimProfile};

    #[test]
    fn test_diffmap_identical_images() {
        let pixels: Vec<[u8; 3]> = (0..64)
            .map(|i| [i as u8 * 4, 128, 255 - i as u8 * 4])
            .collect();
        let z = crate::Zensim::new(ZensimProfile::codec_target());
        let src = RgbSlice::new(&pixels, 8, 8);
        let result = z
            .compute_with_diffmap(&src, &src, DiffmapWeighting::default())
            .unwrap();
        assert_eq!(result.width(), 8);
        assert_eq!(result.height(), 8);
        assert_eq!(result.diffmap().len(), 64);
        // Identical images should have ~0 error everywhere.
        // Fused blur path has minor float precision differences (~3e-5), not exact zero.
        let max_err = result.diffmap().iter().copied().fold(0.0f32, f32::max);
        assert!(
            max_err < 1e-4,
            "max diffmap error for identical images: {max_err}"
        );
        // Score should be very high (small images may have minor artifacts from padding)
        assert!(result.score() > 95.0, "score: {}", result.score());
    }

    #[test]
    fn test_diffmap_localized_error() {
        // Create source: uniform gray
        let src_pixels: Vec<[u8; 3]> = vec![[128, 128, 128]; 16 * 16];
        // Create distorted: same, but with bright patch in top-left 4x4
        let mut dst_pixels = src_pixels.clone();
        for y in 0..4 {
            for x in 0..4 {
                dst_pixels[y * 16 + x] = [255, 255, 255];
            }
        }

        let z = crate::Zensim::new(ZensimProfile::codec_target());
        let src = RgbSlice::new(&src_pixels, 16, 16);
        let dst = RgbSlice::new(&dst_pixels, 16, 16);
        let result = z
            .compute_with_diffmap(&src, &dst, DiffmapWeighting::Balanced)
            .unwrap();
        let dm = result.diffmap();

        assert_eq!(dm.len(), 256);
        // The distorted region should have higher error than undistorted
        let mut distorted_sum = 0.0f32;
        for y in 0..4 {
            for x in 0..4 {
                distorted_sum += dm[y * 16 + x];
            }
        }
        let distorted_avg = distorted_sum / 16.0;

        let mut clean_sum = 0.0f32;
        for y in 8..12 {
            for x in 8..12 {
                clean_sum += dm[y * 16 + x];
            }
        }
        let clean_avg = clean_sum / 16.0;

        assert!(
            distorted_avg > clean_avg,
            "distorted region avg ({distorted_avg}) should exceed clean region avg ({clean_avg})"
        );
    }

    #[test]
    fn test_diffmap_masking_reduces_textured_error() {
        // Source: alternating bright/dark columns (textured)
        let mut src_pixels: Vec<[u8; 3]> = Vec::with_capacity(16 * 16);
        for y in 0..16 {
            for x in 0..16 {
                let v = if (x + y) % 2 == 0 { 200u8 } else { 60u8 };
                src_pixels.push([v, v, v]);
            }
        }
        // Distorted: shift all pixels by a fixed amount
        let dst_pixels: Vec<[u8; 3]> = src_pixels
            .iter()
            .map(|p| {
                [
                    p[0].saturating_add(30),
                    p[1].saturating_add(30),
                    p[2].saturating_add(30),
                ]
            })
            .collect();

        let z = crate::Zensim::new(ZensimProfile::codec_target());
        let src = RgbSlice::new(&src_pixels, 16, 16);
        let dst = RgbSlice::new(&dst_pixels, 16, 16);

        // Without masking
        let raw = z
            .compute_with_diffmap(&src, &dst, DiffmapWeighting::Balanced)
            .unwrap();
        let raw_max = raw.diffmap().iter().copied().fold(0.0f32, f32::max);

        // With masking
        let masked = z
            .compute_with_diffmap(
                &src,
                &dst,
                super::DiffmapOptions {
                    weighting: DiffmapWeighting::Balanced,
                    masking_strength: Some(4.0),
                    ..Default::default()
                },
            )
            .unwrap();
        let masked_max = masked.diffmap().iter().copied().fold(0.0f32, f32::max);

        // Masking should reduce error in textured content
        assert!(
            masked_max < raw_max,
            "masked max ({masked_max}) should be less than raw max ({raw_max})"
        );
        // Scores should match (masking is post-processing, doesn't affect score)
        assert!(
            (raw.score() - masked.score()).abs() < 0.01,
            "scores should match: raw {} vs masked {}",
            raw.score(),
            masked.score()
        );
    }

    #[test]
    fn test_diffmap_sqrt_compresses_range() {
        let src_pixels: Vec<[u8; 3]> = vec![[128, 128, 128]; 16 * 16];
        let mut dst_pixels = src_pixels.clone();
        for y in 0..4 {
            for x in 0..4 {
                dst_pixels[y * 16 + x] = [255, 255, 255];
            }
        }

        let z = crate::Zensim::new(ZensimProfile::codec_target());
        let src = RgbSlice::new(&src_pixels, 16, 16);
        let dst = RgbSlice::new(&dst_pixels, 16, 16);

        let raw = z
            .compute_with_diffmap(&src, &dst, DiffmapWeighting::Balanced)
            .unwrap();
        let raw_max = raw.diffmap().iter().copied().fold(0.0f32, f32::max);

        let sqrted = z
            .compute_with_diffmap(
                &src,
                &dst,
                super::DiffmapOptions {
                    weighting: DiffmapWeighting::Balanced,
                    sqrt: true,
                    ..Default::default()
                },
            )
            .unwrap();
        let sqrt_max = sqrted.diffmap().iter().copied().fold(0.0f32, f32::max);

        // sqrt(x) < x for x > 1, sqrt(x) > x for 0 < x < 1
        // For our diffmap values (typically < 1), sqrt expands small values but
        // the max should equal sqrt(raw_max) which for values < 1 is > raw_max
        // For values > 1, sqrt compresses. Either way, sqrt(max) != max.
        let expected_sqrt_max = raw_max.sqrt();
        assert!(
            (sqrt_max - expected_sqrt_max).abs() < 1e-5,
            "sqrt max ({sqrt_max}) should equal sqrt(raw_max) = {expected_sqrt_max}"
        );
    }

    #[test]
    fn test_diffmap_weighting_into_options() {
        // Verify backward compatibility: DiffmapWeighting converts to DiffmapOptions
        let z = crate::Zensim::new(ZensimProfile::codec_target());
        let pixels: Vec<[u8; 3]> = vec![[100, 150, 200]; 8 * 8];
        let src = RgbSlice::new(&pixels, 8, 8);

        // All three weighting variants should work via Into<DiffmapOptions>
        let _ = z
            .compute_with_diffmap(&src, &src, DiffmapWeighting::Trained)
            .unwrap();
        let _ = z
            .compute_with_diffmap(&src, &src, DiffmapWeighting::Balanced)
            .unwrap();
        let _ = z
            .compute_with_diffmap(&src, &src, DiffmapWeighting::Custom([0.3, 0.5, 0.2]))
            .unwrap();
    }

    #[test]
    fn test_diffmap_edge_mse_produces_valid_signal() {
        // Edge/MSE features should produce valid non-negative per-pixel values
        let src_pixels: Vec<[u8; 3]> = vec![[128, 128, 128]; 16 * 16];
        let mut dst_pixels = src_pixels.clone();
        // Create a sharp edge (high edge artifact)
        for y in 0..8 {
            for x in 0..16 {
                dst_pixels[y * 16 + x] = [200, 200, 200];
            }
        }

        let z = crate::Zensim::new(ZensimProfile::codec_target());
        let src = RgbSlice::new(&src_pixels, 16, 16);
        let dst = RgbSlice::new(&dst_pixels, 16, 16);

        let with_edge = z
            .compute_with_diffmap(
                &src,
                &dst,
                super::DiffmapOptions {
                    weighting: DiffmapWeighting::Balanced,
                    include_edge_mse: true,
                    ..Default::default()
                },
            )
            .unwrap();

        // All values should be non-negative
        assert!(
            with_edge.diffmap().iter().all(|&v| v >= 0.0),
            "all diffmap values should be non-negative"
        );
        // Should have signal (not all zeros)
        let max = with_edge.diffmap().iter().copied().fold(0.0f32, f32::max);
        assert!(max > 0.0, "max should be > 0 for distorted image");
        // Distorted half should have higher error than clean half
        let top_avg: f32 = with_edge.diffmap()[..128].iter().sum::<f32>() / 128.0;
        let bot_avg: f32 = with_edge.diffmap()[128..].iter().sum::<f32>() / 128.0;
        assert!(
            top_avg > bot_avg,
            "distorted region ({top_avg}) should exceed clean region ({bot_avg})"
        );
        // Scores should match (feature inclusion is diffmap-only)
        let ssim_only = z
            .compute_with_diffmap(&src, &dst, DiffmapWeighting::Balanced)
            .unwrap();
        assert!(
            (ssim_only.score() - with_edge.score()).abs() < 0.01,
            "scores should match: {} vs {}",
            ssim_only.score(),
            with_edge.score()
        );
    }

    #[test]
    fn test_diffmap_edge_mse_trained_weights() {
        // Trained weighting with edge/MSE should produce valid results
        let src_pixels: Vec<[u8; 3]> = (0..256).map(|i| [(i % 256) as u8, 128, 64]).collect();
        let mut dst_pixels = src_pixels.clone();
        for p in dst_pixels[..64].iter_mut() {
            p[0] = p[0].wrapping_add(40);
        }

        let z = crate::Zensim::new(ZensimProfile::codec_target());
        let src = RgbSlice::new(&src_pixels, 16, 16);
        let dst = RgbSlice::new(&dst_pixels, 16, 16);

        let result = z
            .compute_with_diffmap(
                &src,
                &dst,
                super::DiffmapOptions {
                    weighting: DiffmapWeighting::Trained,
                    include_edge_mse: true,
                    ..Default::default()
                },
            )
            .unwrap();

        assert_eq!(result.diffmap().len(), 256);
        // All values should be non-negative
        assert!(
            result.diffmap().iter().all(|&v| v >= 0.0),
            "all diffmap values should be non-negative"
        );
        // At least some values should be > 0 (we have distortion)
        let max = result.diffmap().iter().copied().fold(0.0f32, f32::max);
        assert!(max > 0.0, "max diffmap value should be > 0");
    }

    // The diffmap-vs-regular *score equality* invariant only holds for a
    // linear profile, where the diffmap is a spatial redistribution of the
    // same weighted feature distance. The MLP profile `A` scores the diffmap
    // path and the plain path through separate forward passes that diverge
    // (~2.5 units here), so this asserts against a reconstructed plain-linear
    // profile (the former `PreviewV0_2`, bit-identical via the builder
    // defaults). Gated on `custom-profiles`; CI runs it in `test-all-features`.
    #[cfg(feature = "custom-profiles")]
    #[test]
    fn test_diffmap_with_precomputed_ref() {
        use crate::profile::ProfileParams;
        use std::sync::OnceLock;
        let pixels: Vec<[u8; 3]> = (0..256).map(|i| [(i % 256) as u8, 128, 64]).collect();
        let mut dst = pixels.clone();
        // Perturb some pixels
        for p in dst[..32].iter_mut() {
            p[0] = p[0].wrapping_add(50);
        }

        static PARAMS: OnceLock<ProfileParams> = OnceLock::new();
        let z = crate::Zensim::new(ZensimProfile::Custom {
            params: PARAMS.get_or_init(|| ProfileParams::builder().build()),
            name: "zensim-preview-v0.2",
        });
        let src = RgbSlice::new(&pixels, 16, 16);
        let dst = RgbSlice::new(&dst, 16, 16);

        let precomputed = z.precompute_reference(&src).unwrap();
        let result = z
            .compute_with_ref_and_diffmap(&precomputed, &dst, DiffmapWeighting::Trained)
            .unwrap();

        assert_eq!(result.width(), 16);
        assert_eq!(result.height(), 16);
        // Score should match regular compute (linear profile invariant).
        let regular = z.compute_with_ref(&precomputed, &dst).unwrap();
        assert!(
            (result.score() - regular.score()).abs() < 0.01,
            "diffmap score {} vs regular score {}",
            result.score(),
            regular.score()
        );
    }

    /// Stress-test diffmap for NaN/Inf with adversarial inputs at realistic sizes.
    #[test]
    fn test_diffmap_no_nan() {
        let z = crate::Zensim::new(ZensimProfile::codec_target());
        let weightings = [
            DiffmapWeighting::Trained,
            DiffmapWeighting::Balanced,
            DiffmapWeighting::Custom([1.0, 0.0, 0.0]),
        ];
        let options_list = [
            super::DiffmapOptions::default(),
            super::DiffmapOptions {
                weighting: DiffmapWeighting::Trained,
                masking_strength: Some(4.0),
                sqrt: true,
                include_edge_mse: true,
                ..Default::default()
            },
            super::DiffmapOptions {
                weighting: DiffmapWeighting::Trained,
                include_hf: true,
                ..Default::default()
            },
            super::DiffmapOptions {
                masking_strength: Some(4.0),
                sqrt: true,
                include_edge_mse: true,
                include_hf: true,
                ..Default::default()
            },
        ];
        // Adversarial patterns: uniform, solid black, solid white, random-ish,
        // extreme contrast, near-identical
        #[allow(clippy::type_complexity)]
        let cases: Vec<(&str, usize, usize, Vec<[u8; 3]>, Vec<[u8; 3]>)> = vec![
            {
                let w = 64;
                let h = 64;
                let src = vec![[128, 128, 128]; w * h];
                let dst = vec![[128, 128, 128]; w * h];
                ("uniform_identical", w, h, src, dst)
            },
            {
                let w = 64;
                let h = 64;
                let src = vec![[0, 0, 0]; w * h];
                let dst = vec![[0, 0, 0]; w * h];
                ("black_identical", w, h, src, dst)
            },
            {
                let w = 64;
                let h = 64;
                let src = vec![[255, 255, 255]; w * h];
                let dst = vec![[255, 255, 255]; w * h];
                ("white_identical", w, h, src, dst)
            },
            {
                let w = 64;
                let h = 64;
                let src = vec![[0, 0, 0]; w * h];
                let dst = vec![[255, 255, 255]; w * h];
                ("black_vs_white", w, h, src, dst)
            },
            {
                let w = 128;
                let h = 128;
                let src: Vec<[u8; 3]> = (0..w * h)
                    .map(|i| {
                        let v = (i % 256) as u8;
                        [v, v, v]
                    })
                    .collect();
                let dst = src
                    .iter()
                    .map(|p| [p[0].wrapping_add(1), p[1], p[2]])
                    .collect();
                ("near_identical_128", w, h, src, dst)
            },
            {
                // Checkerboard: maximally adversarial for variance computation
                let w = 64;
                let h = 64;
                let src: Vec<[u8; 3]> = (0..w * h)
                    .map(|i| {
                        let x = i % w;
                        let y = i / w;
                        if (x + y) % 2 == 0 {
                            [0, 0, 0]
                        } else {
                            [255, 255, 255]
                        }
                    })
                    .collect();
                let dst = src
                    .iter()
                    .map(|p| {
                        [
                            p[0].saturating_add(10),
                            p[1].saturating_add(10),
                            p[2].saturating_add(10),
                        ]
                    })
                    .collect();
                ("checkerboard", w, h, src, dst)
            },
        ];

        for (label, w, h, src, dst) in &cases {
            let src_img = RgbSlice::new(src, *w, *h);
            let dst_img = RgbSlice::new(dst, *w, *h);
            for weighting in &weightings {
                let result = z
                    .compute_with_diffmap(&src_img, &dst_img, *weighting)
                    .unwrap();
                let nan_count = result.diffmap().iter().filter(|v| v.is_nan()).count();
                let inf_count = result.diffmap().iter().filter(|v| v.is_infinite()).count();
                assert!(
                    nan_count == 0 && inf_count == 0,
                    "{label}: {nan_count} NaN, {inf_count} Inf in diffmap (len={})",
                    result.diffmap().len()
                );
            }
            for options in &options_list {
                let result = z
                    .compute_with_diffmap(&src_img, &dst_img, *options)
                    .unwrap();
                let nan_count = result.diffmap().iter().filter(|v| v.is_nan()).count();
                let inf_count = result.diffmap().iter().filter(|v| v.is_infinite()).count();
                assert!(
                    nan_count == 0 && inf_count == 0,
                    "{label} (options): {nan_count} NaN, {inf_count} Inf in diffmap (len={})",
                    result.diffmap().len()
                );
            }
        }
    }

    /// E-JBU integration: guided coarse-scale redistribution is per-cell
    /// mass-conserving, so (a) the scalar score is bit-identical, (b) aligned
    /// 8px and 16px block sums are unchanged up to f32 summation order, (c) the
    /// map total is preserved — while (d) per-pixel values DO move (the path is
    /// engaged and changes sub-footprint localization only).
    #[cfg(feature = "custom-profiles")]
    #[test]
    fn test_guided_redistribution_block_invariant_pixels_move() {
        let (w, h) = (128usize, 96usize);
        let mut src = vec![[0u8, 0, 0]; w * h];
        let mut dst = vec![[0u8, 0, 0]; w * h];
        let mut lcg = 0x243F6A8885A308D3u64;
        let mut rnd = move || {
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((lcg >> 33) & 0x7) as i16 - 3 // -3..=4, deterministic
        };
        for y in 0..h {
            for x in 0..w {
                // Gradient + sine texture: gives the s0 guide real structure.
                let g = (x * 255 / w) as f32;
                let t = 24.0 * ((x as f32 * 0.7).sin() * (y as f32 * 0.5).cos());
                let v = (g + t).clamp(0.0, 255.0) as u8;
                src[y * w + x] = [v, v.saturating_add(8), v / 2 + 40];
                // Distortion: mild noise everywhere + blocky 8×8 artifacts in a
                // few spots (coarse-scale error mass with fine-scale texture).
                let n = rnd();
                let mut p = src[y * w + x];
                for c in &mut p {
                    *c = (*c as i16 + n).clamp(0, 255) as u8;
                }
                if (16..48).contains(&x) && (16..48).contains(&y) {
                    let q = ((x / 8) * 8 + (y / 8) * 8) as i16 % 48 - 24;
                    for c in &mut p {
                        *c = (*c as i16 + q).clamp(0, 255) as u8;
                    }
                }
                dst[y * w + x] = p;
            }
        }
        let z = crate::Zensim::new(ZensimProfile::codec_target());
        let s = RgbSlice::new(&src, w, h);
        let d = RgbSlice::new(&dst, w, h);
        let base_opts = super::DiffmapOptions {
            weighting: DiffmapWeighting::Balanced,
            include_edge_mse: true,
            include_hf: true,
            ..Default::default()
        };
        let on_opts = super::DiffmapOptions {
            guided_coarse_redistribution: true,
            ..base_opts
        };
        let a = z.compute_with_diffmap(&s, &d, base_opts).unwrap();
        let b = z.compute_with_diffmap(&s, &d, on_opts).unwrap();

        // (a) scalar score untouched (the option is render-only).
        assert_eq!(a.score(), b.score(), "score must be bit-identical");

        let (ma, mb) = (a.diffmap(), b.diffmap());
        assert_eq!(ma.len(), mb.len());
        // (c) total mass preserved.
        let (ta, tb) = (
            ma.iter().map(|&v| v as f64).sum::<f64>(),
            mb.iter().map(|&v| v as f64).sum::<f64>(),
        );
        let total_rel = (ta - tb).abs() / ta.abs().max(1e-12);
        assert!(total_rel < 1e-5, "total mass drift {total_rel:.3e}");
        // (b) aligned block sums unchanged for blocks ≥ the coarsest footprint.
        for block in [8usize, 16] {
            let (bx, by) = (w.div_ceil(block), h.div_ceil(block));
            let mut max_rel = 0.0f64;
            for byi in 0..by {
                for bxi in 0..bx {
                    let (mut sa, mut sb) = (0.0f64, 0.0f64);
                    for y in (byi * block)..((byi + 1) * block).min(h) {
                        for x in (bxi * block)..((bxi + 1) * block).min(w) {
                            sa += ma[y * w + x] as f64;
                            sb += mb[y * w + x] as f64;
                        }
                    }
                    max_rel = max_rel.max((sa - sb).abs() / sa.abs().max(1e-12));
                }
            }
            assert!(
                max_rel < 1e-4,
                "{block}px block-sum drift {max_rel:.3e} — redistribution leaked across aligned footprints"
            );
        }
        // (d) per-pixel localization DID change.
        let max_delta = ma
            .iter()
            .zip(mb.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let changed = ma
            .iter()
            .zip(mb.iter())
            .filter(|(x, y)| (**x - **y).abs() > 1e-9)
            .count();
        assert!(
            max_delta > 1e-6 && changed > ma.len() / 20,
            "guided redistribution changed too little (max Δ {max_delta:.3e}, {changed}/{} px)",
            ma.len()
        );
    }
}
