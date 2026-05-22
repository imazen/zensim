//! Buffer pool for reusable allocations across metric computation.

/// Pre-allocated buffers for metric computation, reused across scales.
pub(crate) struct ScaleBuffers {
    pub mul_buf: Vec<f32>,
    pub mu1: Vec<f32>,
    pub mu2: Vec<f32>,
    /// Holds blur(src² + dst²) for combined SSIM computation.
    pub sigma1_sq: Vec<f32>,
    pub sigma12: Vec<f32>,
    pub temp_blur: Vec<f32>,
    /// Local contrast masking weights (when masking enabled).
    pub mask: Vec<f32>,
    /// Information-content (IW) weights — texture-EMPHASISING counterpart
    /// to `mask`. **No longer written or read by the hot path as of
    /// 2026-05-22.** The IW weight (`1 + k_iw * blur(|src - mu|)`) is
    /// now computed inline at every consumer (SSIM, edge-diff, MSE) via
    /// the `*_iw_inline` SIMD kernels in `simd_ops.rs`, eliminating the
    /// per-pixel plane round-trip. Field retained to avoid gating
    /// `ScaleBuffers` through every constructor; allocation is cheap
    /// and unused buffers don't get touched. Safe to remove once a
    /// follow-up pass confirms no external consumer reads it.
    pub iw_weight: Vec<f32>,
    /// Strip-local H-blur of the current channel's source. Used as the
    /// per-channel "local mean" reference for the masked/IW activity
    /// computation (`activity = box_blur(|src - h_blur_src|)`). See
    /// `docs/PRINCIPLED_ACTIVITY.md`. Decouples the activity signal
    /// from cross-channel `bufs.mu1` reuse — every channel sees its
    /// own H-blurred source at all strip rows (inner + overlap).
    pub h_blur_src: Vec<f32>,
}

impl ScaleBuffers {
    pub fn new(size: usize) -> Self {
        Self {
            mul_buf: vec![0.0; size],
            mu1: vec![0.0; size],
            mu2: vec![0.0; size],
            sigma1_sq: vec![0.0; size],
            sigma12: vec![0.0; size],
            temp_blur: vec![0.0; size],
            mask: vec![0.0; size],
            iw_weight: vec![0.0; size],
            h_blur_src: vec![0.0; size],
        }
    }

    pub fn resize(&mut self, size: usize) {
        self.mul_buf.resize(size, 0.0);
        self.mu1.resize(size, 0.0);
        self.mu2.resize(size, 0.0);
        self.sigma1_sq.resize(size, 0.0);
        self.sigma12.resize(size, 0.0);
        self.temp_blur.resize(size, 0.0);
        self.mask.resize(size, 0.0);
        self.iw_weight.resize(size, 0.0);
        self.h_blur_src.resize(size, 0.0);
    }
}
