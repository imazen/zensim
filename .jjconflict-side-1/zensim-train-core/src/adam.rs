//! Bit-exact port of `AdamState` from `zensim-validate/src/mlp_train.rs`.
//!
//! Two-layer MLP-specific layout: each of (w1, b1, w2, b2) gets its own
//! `g`/`m`/`v` arrays. The CubeCL refactor in Phase 2 will generalize
//! this to arbitrary layer counts; for now keeping the shape identical
//! to the existing trainer is the milestone.

pub(crate) struct AdamState {
    pub(crate) gw1: Vec<f64>,
    pub(crate) gb1: Vec<f64>,
    pub(crate) gw2: Vec<f64>,
    pub(crate) gb2: Vec<f64>,
    pub(crate) mw1: Vec<f64>,
    pub(crate) mb1: Vec<f64>,
    pub(crate) mw2: Vec<f64>,
    pub(crate) mb2: Vec<f64>,
    pub(crate) vw1: Vec<f64>,
    pub(crate) vb1: Vec<f64>,
    pub(crate) vw2: Vec<f64>,
    pub(crate) vb2: Vec<f64>,
    pub(crate) t: u64,
}

impl AdamState {
    pub(crate) fn new(nw1: usize, nb1: usize, nw2: usize, nb2: usize) -> Self {
        Self {
            gw1: vec![0.0; nw1],
            gb1: vec![0.0; nb1],
            gw2: vec![0.0; nw2],
            gb2: vec![0.0; nb2],
            mw1: vec![0.0; nw1],
            mb1: vec![0.0; nb1],
            mw2: vec![0.0; nw2],
            mb2: vec![0.0; nb2],
            vw1: vec![0.0; nw1],
            vb1: vec![0.0; nb1],
            vw2: vec![0.0; nw2],
            vb2: vec![0.0; nb2],
            t: 0,
        }
    }

    pub(crate) fn step(
        &mut self,
        w1: &mut [f64],
        b1: &mut [f64],
        w2: &mut [f64],
        b2: &mut [f64],
        lr: f64,
    ) {
        self.t += 1;
        let beta1: f64 = 0.9;
        let beta2: f64 = 0.999;
        let eps: f64 = 1e-8;
        let bc1 = 1.0 - beta1.powi(self.t as i32);
        let bc2 = 1.0 - beta2.powi(self.t as i32);

        let update = |w: &mut [f64], g: &mut [f64], m: &mut [f64], v: &mut [f64]| {
            for i in 0..w.len() {
                m[i] = beta1 * m[i] + (1.0 - beta1) * g[i];
                v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i];
                let m_hat = m[i] / bc1;
                let v_hat = v[i] / bc2;
                w[i] -= lr * m_hat / (v_hat.sqrt() + eps);
                g[i] = 0.0;
            }
        };
        update(w1, &mut self.gw1, &mut self.mw1, &mut self.vw1);
        update(b1, &mut self.gb1, &mut self.mb1, &mut self.vb1);
        update(w2, &mut self.gw2, &mut self.mw2, &mut self.vw2);
        update(b2, &mut self.gb2, &mut self.mb2, &mut self.vb2);
    }
}
