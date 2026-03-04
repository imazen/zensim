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
    /// Per-pixel SSIM error map (for extended percentile features).
    pub ssim_map: Vec<f32>,
    /// Per-pixel edge artifact map (for extended percentile features).
    pub art_map: Vec<f32>,
    /// Per-pixel edge detail_lost map (for extended percentile features).
    pub det_map: Vec<f32>,
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
            ssim_map: Vec::new(),
            art_map: Vec::new(),
            det_map: Vec::new(),
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
        // ssim_map/art_map/det_map are only allocated on demand by extended features
    }

    /// Ensure extended per-pixel map buffers are allocated.
    pub fn ensure_extended_maps(&mut self, size: usize) {
        self.ssim_map.resize(size, 0.0);
        self.art_map.resize(size, 0.0);
        self.det_map.resize(size, 0.0);
    }
}
