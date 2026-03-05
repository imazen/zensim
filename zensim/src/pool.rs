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
    /// Per-pixel SSIM error map (full-image path only).
    #[cfg(feature = "full_image")]
    pub ssim_map: Vec<f32>,
    /// Per-pixel edge artifact map (full-image path only).
    #[cfg(feature = "full_image")]
    pub art_map: Vec<f32>,
    /// Per-pixel edge detail_lost map (full-image path only).
    #[cfg(feature = "full_image")]
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
            #[cfg(feature = "full_image")]
            ssim_map: Vec::new(),
            #[cfg(feature = "full_image")]
            art_map: Vec::new(),
            #[cfg(feature = "full_image")]
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
    }

    /// Ensure extended per-pixel map buffers are allocated.
    #[cfg(feature = "full_image")]
    pub fn ensure_extended_maps(&mut self, size: usize) {
        self.ssim_map.resize(size, 0.0);
        self.art_map.resize(size, 0.0);
        self.det_map.resize(size, 0.0);
    }
}
