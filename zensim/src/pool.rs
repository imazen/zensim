//! Buffer pool for reusable allocations across metric computation.

/// Pre-allocated buffers for metric computation, reused across scales.
pub(crate) struct ScaleBuffers {
    pub temp: Vec<f32>,
    pub mul_buf: Vec<f32>,
    pub mu1: Vec<f32>,
    pub mu2: Vec<f32>,
    pub sigma1_sq: Vec<f32>,
    pub sigma2_sq: Vec<f32>,
    pub sigma12: Vec<f32>,
    pub temp_blur: Vec<f32>,
}

impl ScaleBuffers {
    pub fn new(size: usize) -> Self {
        Self {
            temp: vec![0.0; size],
            mul_buf: vec![0.0; size],
            mu1: vec![0.0; size],
            mu2: vec![0.0; size],
            sigma1_sq: vec![0.0; size],
            sigma2_sq: vec![0.0; size],
            sigma12: vec![0.0; size],
            temp_blur: vec![0.0; size],
        }
    }

    pub fn resize(&mut self, size: usize) {
        self.temp.resize(size, 0.0);
        self.mul_buf.resize(size, 0.0);
        self.mu1.resize(size, 0.0);
        self.mu2.resize(size, 0.0);
        self.sigma1_sq.resize(size, 0.0);
        self.sigma2_sq.resize(size, 0.0);
        self.sigma12.resize(size, 0.0);
        self.temp_blur.resize(size, 0.0);
    }
}
