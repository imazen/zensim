//! P-Square (P²) quantile estimator (Jain & Chlamtac 1985).
//!
//! Single-pass online quantile estimation in O(1) memory per estimator and
//! O(1) work per sample. Five markers track the running estimate of the
//! desired quantile p ∈ (0, 1); the central marker `q[2]` is the
//! quantile estimate.
//!
//! Used in `crate::streaming::ScaleAccumulators` to replace L8-norm
//! near-worst-case approximations with TRUE percentile estimators
//! (p5/p50/p95). The L8 norm `(Σd⁸/N)^(1/8)` is dominated by extrema
//! but is blurred by lower-norm contributors; a true P² p95 carries
//! cleaner tail information for downstream MLP training.
//!
//! Reference: <https://www.cs.wustl.edu/~jain/papers/ftp/psqr.pdf>

/// Single P² quantile estimator. Maintains 5 markers + their desired
/// positions; samples update markers via piecewise-parabolic prediction.
///
/// Use one estimator per target quantile (e.g., p5, p50, p95 = three
/// estimators).
#[derive(Debug, Clone)]
pub struct P2Estimator {
    /// Target quantile (e.g., 0.05, 0.50, 0.95).
    p: f64,
    /// Sample count (also used as the position of the rightmost marker).
    n: usize,
    /// Marker positions (i.e., observation counts at each marker).
    np: [f64; 5],
    /// Marker desired positions (i.e., increment per sample).
    dn: [f64; 5],
    /// Marker heights (the actual quantile-tracked values).
    q: [f64; 5],
    /// Whether the estimator has buffered the initial 5 samples and is
    /// ready for online updates.
    initialized: bool,
    /// Initial sample buffer (used before `initialized`).
    init_buf: [f64; 5],
}

impl P2Estimator {
    /// Construct a new P² estimator for quantile `p` ∈ (0, 1).
    #[inline]
    pub fn new(p: f64) -> Self {
        debug_assert!(p > 0.0 && p < 1.0);
        Self {
            p,
            n: 0,
            np: [1.0, 1.0 + 2.0 * p, 1.0 + 4.0 * p, 3.0 + 2.0 * p, 5.0],
            dn: [0.0, p / 2.0, p, (1.0 + p) / 2.0, 1.0],
            q: [0.0; 5],
            initialized: false,
            init_buf: [0.0; 5],
        }
    }

    /// Add a sample. O(1) work after the first 5 samples (which buffer
    /// to seed the markers).
    #[inline]
    pub fn add(&mut self, x: f64) {
        if !self.initialized {
            self.init_buf[self.n] = x;
            self.n += 1;
            if self.n == 5 {
                // Seed: sort the first 5 samples into self.q. Marker
                // positions are 1..5 (the markers literally ARE the
                // first 5 sorted samples).
                let mut buf = self.init_buf;
                buf.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                self.q = buf;
                self.np = [1.0, 2.0, 3.0, 4.0, 5.0];
                self.initialized = true;
            }
            return;
        }

        // 1) Find cell k such that q[k] <= x < q[k+1]. Outer markers
        //    extend to cover x if it's outside the current range.
        let k: usize = if x < self.q[0] {
            self.q[0] = x;
            0
        } else if x < self.q[1] {
            0
        } else if x < self.q[2] {
            1
        } else if x < self.q[3] {
            2
        } else if x <= self.q[4] {
            3
        } else {
            self.q[4] = x;
            3
        };

        // 2) Increment positions of markers k+1..4 (the markers to the
        //    right of the cell). np[0] never increments.
        for i in (k + 1)..5 {
            self.np[i] += 1.0;
        }

        // 3) Recompute desired positions for all 5 markers.
        //    n = total samples seen so far (before incrementing for x).
        //    After this sample, the dataset has (n+1) elements, with
        //    marker 0 at position 1 and marker 4 at position n+1.
        let nf = (self.n + 1) as f64;
        self.dn[0] = 1.0;
        self.dn[1] = 1.0 + (nf - 1.0) * self.p / 2.0;
        self.dn[2] = 1.0 + (nf - 1.0) * self.p;
        self.dn[3] = 1.0 + (nf - 1.0) * (1.0 + self.p) / 2.0;
        self.dn[4] = nf;

        // 4) Adjust interior markers (1, 2, 3) per piecewise-parabolic
        //    or linear prediction. Canonical formula:
        //
        //    d = dn[i] - np[i]   (signed offset from desired)
        //    if (d >=  1 and np[i+1] - np[i]  > 1) ||
        //       (d <= -1 and np[i-1] - np[i]  < -1):
        //        d_sign = sign(d)
        //        parabolic = q[i] + d_sign / (np[i+1] - np[i-1]) * (
        //              (np[i] - np[i-1] + d_sign) * (q[i+1] - q[i]) / (np[i+1] - np[i])
        //            + (np[i+1] - np[i] - d_sign) * (q[i] - q[i-1]) / (np[i]   - np[i-1])
        //        )
        //        if q[i-1] < parabolic < q[i+1]: q[i] = parabolic
        //        else (linear fallback in direction of d_sign):
        //            q[i] = q[i] + d_sign * (q[i + d_sign as int] - q[i]) / (np[i + d_sign as int] - np[i])
        //        np[i] += d_sign
        for i in 1..4 {
            let d_signed = self.dn[i] - self.np[i];
            let np_next_diff = self.np[i + 1] - self.np[i];
            let np_prev_diff = self.np[i - 1] - self.np[i];
            if (d_signed >= 1.0 && np_next_diff > 1.0)
                || (d_signed <= -1.0 && np_prev_diff < -1.0)
            {
                let d_sign: f64 = if d_signed >= 0.0 { 1.0 } else { -1.0 };
                let qi = self.q[i];
                let qp = self.q[i + 1];
                let qm = self.q[i - 1];
                let npi = self.np[i];
                let npp = self.np[i + 1];
                let npm = self.np[i - 1];

                let parabolic = qi
                    + d_sign / (npp - npm)
                        * (((npi - npm + d_sign) * (qp - qi)) / (npp - npi)
                            + ((npp - npi - d_sign) * (qi - qm)) / (npi - npm));

                let new_q = if qm < parabolic && parabolic < qp {
                    parabolic
                } else if d_sign > 0.0 {
                    // Linear forward
                    qi + (qp - qi) / (npp - npi)
                } else {
                    // Linear backward
                    qi - (qm - qi) / (npm - npi)
                };

                self.q[i] = new_q;
                self.np[i] += d_sign;
            }
        }

        self.n += 1;
    }

    /// Return the current estimate of the target quantile.
    ///
    /// Before 5 samples have been added, returns the appropriate order
    /// statistic of the buffer; this is a small-sample fallback that
    /// keeps the estimator usable on degenerate inputs.
    ///
    /// The returned value is clamped to `[q[0], q[4]]` — the observed
    /// outer markers (i.e., the min and max observations seen so far).
    /// P²'s parabolic update can occasionally extrapolate the central
    /// marker outside the observed range when the distribution is
    /// heavily skewed or has many tied samples at a boundary (e.g.,
    /// SSIM-error d-values clamped to `[0, 1]` with most concentrated
    /// near 0 and a few near 1). Clamping to the outer-marker range
    /// keeps the estimate physically meaningful.
    #[inline]
    pub fn estimate(&self) -> f64 {
        if self.initialized {
            // Clamp to observed [min, max] = [q[0], q[4]] to guard
            // against parabolic-extrapolation artifacts in skewed
            // distributions.
            let lo = self.q[0];
            let hi = self.q[4];
            self.q[2].clamp(lo, hi)
        } else if self.n == 0 {
            0.0
        } else {
            // Pre-init: return appropriate order statistic of the buffer.
            let mut buf = [0.0f64; 5];
            buf[..self.n].copy_from_slice(&self.init_buf[..self.n]);
            buf[..self.n].sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            // For a target quantile p, pick rank closest to ceil(p * n).
            let rank = ((self.p * self.n as f64).ceil() as usize).saturating_sub(1).min(self.n - 1);
            buf[rank]
        }
    }

    /// Sample count seen so far.
    #[inline]
    pub fn count(&self) -> usize {
        self.n
    }

    /// Merge another estimator's samples into this one. The current P²
    /// algorithm has no exact merge; we approximate by reconstructing
    /// from the other's 5 markers (treating them as representative
    /// samples). This is used for cross-strip reduction in parallel
    /// extraction.
    pub fn merge(&mut self, other: &Self) {
        if !other.initialized {
            for i in 0..other.n {
                self.add(other.init_buf[i]);
            }
        } else {
            // Replay marker quantiles as proxies. Better than nothing for
            // strip-level reduction since per-strip n is small and the
            // markers span the distribution.
            self.add(other.q[0]);
            self.add(other.q[1]);
            self.add(other.q[2]);
            self.add(other.q[3]);
            self.add(other.q[4]);
        }
    }
}

impl Default for P2Estimator {
    /// Default-constructs a p=0.5 (median) estimator. Mostly here so
    /// `[P2Estimator; N]` literals work via `Default::default()`.
    fn default() -> Self {
        Self::new(0.5)
    }
}

/// A triple of P² estimators (p5, p50, p95) — fed by the same sample
/// stream, used together as a "shape of distribution" summary.
#[derive(Debug, Clone)]
pub struct P2Triplet {
    pub p5: P2Estimator,
    pub p50: P2Estimator,
    pub p95: P2Estimator,
}

impl P2Triplet {
    #[inline]
    pub fn new() -> Self {
        Self {
            p5: P2Estimator::new(0.05),
            p50: P2Estimator::new(0.50),
            p95: P2Estimator::new(0.95),
        }
    }

    #[inline]
    pub fn add(&mut self, x: f64) {
        self.p5.add(x);
        self.p50.add(x);
        self.p95.add(x);
    }

    /// Update with a batch of f32 samples (common case from SIMD lanes).
    #[inline]
    pub fn add_slice_f32(&mut self, xs: &[f32]) {
        for &x in xs {
            self.add(x as f64);
        }
    }

    #[inline]
    pub fn p5(&self) -> f64 {
        self.p5.estimate()
    }
    #[inline]
    pub fn p50(&self) -> f64 {
        self.p50.estimate()
    }
    #[inline]
    pub fn p95(&self) -> f64 {
        self.p95.estimate()
    }

    pub fn merge(&mut self, other: &Self) {
        self.p5.merge(&other.p5);
        self.p50.merge(&other.p50);
        self.p95.merge(&other.p95);
    }
}

impl Default for P2Triplet {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Quick-sort the slice and return its quantile (linear
    /// interpolation, same as numpy's `np.percentile(..., method='linear')`).
    fn exact_quantile(data: &mut [f64], p: f64) -> f64 {
        data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = data.len();
        if n == 0 {
            return 0.0;
        }
        if n == 1 {
            return data[0];
        }
        let idx = p * (n - 1) as f64;
        let lo = idx.floor() as usize;
        let hi = idx.ceil() as usize;
        let frac = idx - lo as f64;
        data[lo] + (data[hi] - data[lo]) * frac
    }

    #[test]
    fn test_uniform_distribution_p50() {
        // P² estimator on Uniform(0, 1) — p50 should converge to 0.5.
        let mut est = P2Estimator::new(0.5);
        let mut samples = Vec::with_capacity(10_000);
        let mut state = 0xcafe_babe_dead_beef_u64;
        for _ in 0..10_000 {
            // xorshift64
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let x = (state as f64) / (u64::MAX as f64);
            est.add(x);
            samples.push(x);
        }
        let exact = exact_quantile(&mut samples, 0.5);
        let p2 = est.estimate();
        let err = (exact - p2).abs();
        assert!(
            err < 0.01,
            "P² p50 error too large: exact={exact} p2={p2} err={err}"
        );
    }

    #[test]
    fn test_uniform_distribution_p95() {
        let mut est = P2Estimator::new(0.95);
        let mut samples = Vec::with_capacity(10_000);
        let mut state = 0x1234_5678_9abc_def0_u64;
        for _ in 0..10_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let x = (state as f64) / (u64::MAX as f64);
            est.add(x);
            samples.push(x);
        }
        let exact = exact_quantile(&mut samples, 0.95);
        let p2 = est.estimate();
        let err = (exact - p2).abs();
        assert!(err < 0.02, "P² p95 error too large: exact={exact} p2={p2} err={err}");
    }

    #[test]
    fn test_uniform_distribution_p5() {
        let mut est = P2Estimator::new(0.05);
        let mut samples = Vec::with_capacity(10_000);
        let mut state = 0xdead_face_b00b_c001_u64;
        for _ in 0..10_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let x = (state as f64) / (u64::MAX as f64);
            est.add(x);
            samples.push(x);
        }
        let exact = exact_quantile(&mut samples, 0.05);
        let p2 = est.estimate();
        let err = (exact - p2).abs();
        assert!(err < 0.02, "P² p5 error too large: exact={exact} p2={p2} err={err}");
    }

    #[test]
    fn test_normal_distribution_p50() {
        // Box-Muller approximation to standard normal; p50 → 0, p95 → ~1.645.
        let mut est_p50 = P2Estimator::new(0.5);
        let mut est_p95 = P2Estimator::new(0.95);
        let mut samples = Vec::with_capacity(10_000);
        let mut state = 0xfeed_face_dead_beef_u64;
        let mut spare: Option<f64> = None;
        for _ in 0..10_000 {
            let x = if let Some(s) = spare.take() {
                s
            } else {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let u1 = ((state as f64) / (u64::MAX as f64)).max(f64::EPSILON);
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                let u2 = (state as f64) / (u64::MAX as f64);
                let r = (-2.0 * u1.ln()).sqrt();
                let theta = 2.0 * std::f64::consts::PI * u2;
                spare = Some(r * theta.sin());
                r * theta.cos()
            };
            est_p50.add(x);
            est_p95.add(x);
            samples.push(x);
        }
        let exact_p50 = exact_quantile(&mut samples.clone(), 0.5);
        let exact_p95 = exact_quantile(&mut samples, 0.95);
        let err_p50 = (exact_p50 - est_p50.estimate()).abs();
        let err_p95 = (exact_p95 - est_p95.estimate()).abs();
        assert!(err_p50 < 0.05, "P² p50 normal error: exact={exact_p50} est={} err={err_p50}", est_p50.estimate());
        assert!(err_p95 < 0.10, "P² p95 normal error: exact={exact_p95} est={} err={err_p95}", est_p95.estimate());
    }

    #[test]
    fn test_few_samples_uses_fallback() {
        // <5 samples: estimator returns appropriate order statistic.
        let mut est = P2Estimator::new(0.5);
        est.add(3.0);
        // 1 sample: rank 0
        assert_eq!(est.estimate(), 3.0);
        est.add(1.0);
        // 2 samples: rank 0 (ceil(0.5*2)-1 = 0) → 1.0 after sort
        assert_eq!(est.estimate(), 1.0);
        est.add(2.0);
        // 3 samples: rank 1 (ceil(0.5*3)-1 = 1) → 2.0 after sort
        assert_eq!(est.estimate(), 2.0);
    }

    #[test]
    fn test_constant_input() {
        // All samples equal → estimate equals the constant.
        let mut est = P2Estimator::new(0.95);
        for _ in 0..200 {
            est.add(7.0);
        }
        assert_eq!(est.estimate(), 7.0);
    }

    #[test]
    fn test_triplet_basic() {
        let mut t = P2Triplet::new();
        let mut state = 0xb16b_00b5_dead_beef_u64;
        for _ in 0..5_000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let x = (state as f64) / (u64::MAX as f64);
            t.add(x);
        }
        // Uniform(0,1): p5 ≈ 0.05, p50 ≈ 0.5, p95 ≈ 0.95.
        assert!((t.p5() - 0.05).abs() < 0.02, "p5={}", t.p5());
        assert!((t.p50() - 0.50).abs() < 0.02, "p50={}", t.p50());
        assert!((t.p95() - 0.95).abs() < 0.02, "p95={}", t.p95());
    }

    #[test]
    fn test_triplet_merge_approximation() {
        // Merge is an approximation, but small error for two equal halves.
        let mut a = P2Triplet::new();
        let mut b = P2Triplet::new();
        let mut state = 0xface_u64;
        for i in 0..1000 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let x = (state as f64) / (u64::MAX as f64);
            if i % 2 == 0 {
                a.add(x);
            } else {
                b.add(x);
            }
        }
        a.merge(&b);
        // p50 should still be roughly 0.5; merge approximation widens band.
        assert!((a.p50() - 0.50).abs() < 0.10);
    }
}
