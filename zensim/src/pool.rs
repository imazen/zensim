//! Buffer pool for reusable allocations across metric computation.

/// Pre-allocated buffers for metric computation, reused across scales.
///
/// **Phase 2 Lever 3 (2026-05-22)**: the previously lazy-allocated
/// `h_blur_src` field has been deleted. The activity path now uses
/// `box_blur_h_into_abs_diff` which fuses the H-blur with the abs-diff
/// step — the H-blur plane is never materialized.
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
    }
}
