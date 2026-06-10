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
    /// Construct an empty `ScaleBuffers`. The first
    /// [`Self::ensure_capacity`] call grows all 7 buffers to the
    /// requested size; subsequent calls with smaller-or-equal sizes
    /// are no-ops. Use this when allocating per-rayon-worker scratch
    /// that gets reused across many bands within one process — first
    /// band on the worker pays the zero-fill once, subsequent bands
    /// reuse the existing allocation with no memset.
    pub fn empty() -> Self {
        Self {
            mul_buf: Vec::new(),
            mu1: Vec::new(),
            mu2: Vec::new(),
            sigma1_sq: Vec::new(),
            sigma12: Vec::new(),
            temp_blur: Vec::new(),
            mask: Vec::new(),
        }
    }

    /// Grow every buffer to at least `size` if it isn't already.
    /// Zero-fills new entries; existing entries are untouched. Cheap
    /// no-op when buffers are already large enough.
    pub fn ensure_capacity(&mut self, size: usize) {
        if self.mul_buf.len() < size {
            self.mul_buf.resize(size, 0.0);
            self.mu1.resize(size, 0.0);
            self.mu2.resize(size, 0.0);
            self.sigma1_sq.resize(size, 0.0);
            self.sigma12.resize(size, 0.0);
            self.temp_blur.resize(size, 0.0);
            self.mask.resize(size, 0.0);
        }
    }

    /// Legacy alias for [`Self::ensure_capacity`].
    pub fn resize(&mut self, size: usize) {
        self.ensure_capacity(size);
    }
}
