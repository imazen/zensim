//! Min-max monotone network (Sill 1998) — an EXPRESSIVE monotone-by-construction
//! head, the architecture chosen (2026-07-16) to break the consistent-dial ⟂
//! real-content tension: the positive-weight LeakyReLU MLP (`monotone_cbc`)
//! craters imazen-26 to ~0.03 when constrained, but a min-max of linear pieces is
//! a *universal approximator for monotone functions* (Sill), so it can (in
//! principle) fit the real-codec ssim2 surface while staying monotone in codec
//! quality — the thing a dial needs to be target-searchable.
//!
//! # The function
//!
//! `score(x) = min_g  max_h  (w[g][h] · x + b[g][h])`
//!
//! over `K` groups (the outer min) of `J` linear pieces each (the inner max),
//! with each weight vector `w[g][h]` SIGN-CONSTRAINED per the feature-sign mask:
//! `sign[f] · w[g][h][f] ≥ 0`. Because every linear piece is then monotone
//! non-decreasing in codec quality (a quality-increasing feature has `w ≥ 0`, a
//! quality-decreasing feature `w ≤ 0`), and max/min of monotone-non-decreasing
//! functions are monotone-non-decreasing, `score` is monotone in quality BY
//! CONSTRUCTION — regardless of the weight values. That is what
//! [`MinMaxMonotone::monotone_by_construction`] proves.
//!
//! Features with `sign[f] == 0` (the 72 "free" / sign-ambiguous features) are
//! dropped (`w == 0`) so monotonicity is exact; the min-max's inner/outer
//! structure is where the expressiveness that `monotone_strict` lost is meant to
//! come back.

/// A trained (or in-training) min-max monotone head.
#[derive(Clone, Debug)]
pub struct MinMaxMonotone {
    /// Outer min groups.
    pub k: usize,
    /// Inner max pieces per group.
    pub j: usize,
    pub n_features: usize,
    /// Row-major `[g][h][f]`, length `k*j*n_features`.
    pub w: Vec<f64>,
    /// Row-major `[g][h]`, length `k*j`.
    pub b: Vec<f64>,
    /// Per-feature monotone direction: `+1` increases with quality (`w ≥ 0`),
    /// `-1` decreases (`w ≤ 0`), `0` dropped (`w ≡ 0`). Length `n_features`.
    pub sign: Vec<f64>,
}

impl MinMaxMonotone {
    /// Index into `w` for group `g`, piece `h`, feature `f`.
    #[inline]
    fn wi(&self, g: usize, h: usize, f: usize) -> usize {
        (g * self.j + h) * self.n_features + f
    }

    /// One linear piece: `w[g][h] · x + b[g][h]`.
    #[inline]
    fn piece(&self, g: usize, h: usize, x: &[f64]) -> f64 {
        let base = (g * self.j + h) * self.n_features;
        let mut acc = self.b[g * self.j + h];
        for f in 0..self.n_features {
            acc += self.w[base + f] * x[f];
        }
        acc
    }

    /// Forward pass. Returns `(score, active_g, active_h)` — the argmin group and
    /// the argmax piece within it, which is where the sub-gradient flows.
    pub fn forward(&self, x: &[f64]) -> (f64, usize, usize) {
        debug_assert_eq!(x.len(), self.n_features);
        let mut best_min = f64::INFINITY;
        let (mut arg_g, mut arg_h) = (0usize, 0usize);
        for g in 0..self.k {
            // inner max over J pieces
            let mut best_max = f64::NEG_INFINITY;
            let mut hstar = 0usize;
            for h in 0..self.j {
                let v = self.piece(g, h, x);
                if v > best_max {
                    best_max = v;
                    hstar = h;
                }
            }
            // outer min over K groups
            if best_max < best_min {
                best_min = best_max;
                arg_g = g;
                arg_h = hstar;
            }
        }
        (best_min, arg_g, arg_h)
    }

    /// Project weights onto the sign constraint `sign[f] · w[f] ≥ 0` (and `w == 0`
    /// where `sign[f] == 0`). Applied after every optimizer step and at bake so
    /// the shipped bytes are monotone-by-construction, exactly like
    /// `monotone_cbc`'s per-step projection.
    pub fn project(&mut self) {
        for g in 0..self.k {
            for h in 0..self.j {
                for f in 0..self.n_features {
                    let i = self.wi(g, h, f);
                    let s = self.sign[f];
                    if s > 0.0 {
                        if self.w[i] < 0.0 {
                            self.w[i] = 0.0;
                        }
                    } else if s < 0.0 {
                        if self.w[i] > 0.0 {
                            self.w[i] = 0.0;
                        }
                    } else {
                        self.w[i] = 0.0;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random weights in [-scale, scale] (no rng dep).
    fn seeded(k: usize, j: usize, nf: usize, sign: Vec<f64>) -> MinMaxMonotone {
        let mut state = 0x9E37_79B9_7F4A_7C15u64;
        let mut nxt = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            ((state >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        let w: Vec<f64> = (0..k * j * nf).map(|_| nxt()).collect();
        let b: Vec<f64> = (0..k * j).map(|_| nxt()).collect();
        let mut m = MinMaxMonotone { k, j, n_features: nf, w, b, sign };
        m.project();
        m
    }

    /// The load-bearing property: after `project`, increasing any quality-
    /// increasing feature (or decreasing any quality-decreasing one) NEVER
    /// decreases the score — i.e. the score is monotone in codec quality by
    /// construction, for ARBITRARY weights. This is the whole reason to use
    /// min-max over the MLP.
    #[test]
    fn monotone_by_construction() {
        let nf = 12;
        // 8 quality-increasing (+1), 2 decreasing (-1), 2 dropped (0).
        let sign: Vec<f64> = (0..nf)
            .map(|f| match f % 6 {
                5 => 0.0,
                4 => -1.0,
                _ => 1.0,
            })
            .collect();
        let m = seeded(4, 3, nf, sign.clone());

        let mut state = 0xDEAD_BEEF_CAFE_BABEu64;
        let mut nxt = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        // 200 random base points × every feature × a positive quality step.
        for _ in 0..200 {
            let x: Vec<f64> = (0..nf).map(|_| nxt() * 4.0 - 2.0).collect();
            let (base, _, _) = m.forward(&x);
            for f in 0..nf {
                if sign[f] == 0.0 {
                    continue;
                }
                // move feature f in the QUALITY-increasing direction
                let mut xq = x.clone();
                xq[f] += sign[f] * (0.1 + nxt()); // +step for sign+1, −step for sign−1
                let (up, _, _) = m.forward(&xq);
                assert!(
                    up >= base - 1e-9,
                    "monotonicity violated: feat {f} sign {} moved quality-up but score {base} -> {up}",
                    sign[f]
                );
            }
        }
    }

    /// `project` zeroes dropped features and clamps signs.
    #[test]
    fn project_enforces_signs() {
        let sign = vec![1.0, -1.0, 0.0];
        let m = seeded(2, 2, 3, sign);
        for g in 0..m.k {
            for h in 0..m.j {
                assert!(m.w[m.wi(g, h, 0)] >= 0.0, "sign+1 feature must be >=0");
                assert!(m.w[m.wi(g, h, 1)] <= 0.0, "sign-1 feature must be <=0");
                assert_eq!(m.w[m.wi(g, h, 2)], 0.0, "sign0 feature must be dropped");
            }
        }
    }
}
