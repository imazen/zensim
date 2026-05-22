//! Buffer pool for reusable allocations across metric computation.

/// Pre-allocated buffers for metric computation, reused across scales.
///
/// `h_blur_src` is **lazy-allocated** — basic-only paths (no extended
/// or IW features) never touch it, paying no allocation cost. Callers
/// in the `need_activity` branch must invoke `ensure_h_blur_src(size)`
/// before any read/write to the field.
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
    /// Strip-local H-blur of the current channel's source. Used as the
    /// per-channel "local mean" reference for the masked/IW activity
    /// computation. **Lazy-allocated** — empty by default; call
    /// `ensure_h_blur_src(strip_n)` before first use. Allocating only
    /// when needed restores basic-path perf to the pre-2dab8f3 baseline
    /// at large image sizes (TLB/cache pressure relief).
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
            h_blur_src: Vec::new(),
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
        if !self.h_blur_src.is_empty() {
            self.h_blur_src.resize(size, 0.0);
        }
    }

    /// Lazy-grow `h_blur_src` to at least `size` elements. Called by the
    /// activity-needing path before first use; basic-only paths skip
    /// the allocation entirely.
    pub fn ensure_h_blur_src(&mut self, size: usize) {
        if self.h_blur_src.len() < size {
            self.h_blur_src.resize(size, 0.0);
        }
    }
}
