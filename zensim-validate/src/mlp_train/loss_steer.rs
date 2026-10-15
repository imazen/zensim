//! E34 K: exact activation-pattern repair residual. All rows here are already
//! standardized by the rank trainer's scale-only scaler. The image denominator
//! is evaluated once and detached; there is no label or finite-difference loss.
use super::{AdamState, SplitMix64, forward};

struct Repair {
    x: Vec<f64>,
    pre: Vec<f64>,
    hidden: Vec<f64>,
    gain: f64,
    residual: f64,
}

/// One image's eight sampled repairs, with all forwards evaluated before any
/// update. Gradients accumulate into the rank trainer's Adam state.
#[allow(clippy::too_many_arguments)]
fn image_gradient(
    base: &[f64],
    repairs: &[&[f64]],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    scale: f64,
    adam: &mut AdamState,
) -> Option<f64> {
    let n = base.len();
    let h = b1.len();
    let (y, pre, hidden) = forward(base, w1, b1, w2, b2, n, h, 0.0);
    let mut rows = Vec::with_capacity(repairs.len());
    for &x in repairs {
        let (yp, prep, hp) = forward(x, w1, b1, w2, b2, n, h, 0.0);
        let mut tangent = 0.;
        for j in 0..h {
            if pre[j] > 0. {
                let dz: f64 = (0..n).map(|i| w1[i * h + j] * (x[i] - base[i])).sum();
                tangent += w2[j] * dz;
            }
        }
        rows.push(Repair {
            x: x.to_vec(),
            pre: prep,
            hidden: hp,
            gain: yp - y,
            residual: (yp - y) - tangent,
        });
    }
    let denominator: f64 = rows.iter().map(|r| r.gain * r.gain).sum();
    assert!(denominator.is_finite(), "nonfinite K denominator");
    if denominator < 1e-10 {
        return None;
    }
    let mut loss = 0.;
    for r in rows {
        loss += scale * r.residual * r.residual / denominator;
        let dr = 2. * scale * r.residual / denominator;
        for j in 0..h {
            let a = f64::from(u8::from(pre[j] > 0.));
            let ap = f64::from(u8::from(r.pre[j] > 0.));
            let dz: f64 = (0..n).map(|i| w1[i * h + j] * (r.x[i] - base[i])).sum();
            adam.gw2[j] += dr * (r.hidden[j] - hidden[j] - a * dz);
            adam.gb1[j] += dr * w2[j] * (ap - a);
            // d r / d W1_ij = w2_j (a'_j-a_j) x'_i.
            // Activation masks are constant within each differentiable cell.
            for i in 0..n {
                adam.gw1[i * h + j] += dr * w2[j] * (ap - a) * r.x[i];
            }
        }
    }
    Some(loss)
}

/// The rank RNG is never borrowed here. One independent stream supplies the
/// registered 512 image draws and eight uniform repair draws per image.
#[allow(clippy::too_many_arguments)]
pub(super) fn epoch(
    images: &[super::steer_data::Image],
    rng: &mut SplitMix64,
    w1: &mut [f64],
    b1: &mut [f64],
    w2: &mut [f64],
    b2: &mut [f64],
    lambda: f64,
    lr: f64,
    pin: f64,
    adam: &mut AdamState,
) -> (f64, usize, [usize; 4]) {
    // Rejection sampling avoids modulo bias while consuming only K's stream.
    fn draw(rng: &mut SplitMix64, n: usize) -> usize {
        assert!(n > 0);
        let n = n as u64;
        let threshold = n.wrapping_neg() % n;
        loop {
            let x = rng.next_u64();
            if x >= threshold {
                return (x % n) as usize;
            }
        }
    }
    let (mut total, mut skipped, mut sizes) = (0., 0, [0; 4]);
    for _ in 0..512 {
        let image = &images[draw(rng, images.len())];
        let mut repairs = Vec::with_capacity(8);
        for _ in 0..8 {
            let k = draw(rng, image.repairs.len());
            repairs.push(image.repairs[k].as_slice());
            sizes[[8, 16, 32, 64]
                .iter()
                .position(|&s| s == image.sizes[k])
                .unwrap()] += 1;
        }
        match image_gradient(&image.base, &repairs, w1, b1, w2, b2, lambda / 512., adam) {
            None => skipped += 1,
            Some(loss) => {
                total += loss;
                adam.step(w1, b1, w2, b2, lr);
                super::nonneg_project(w2, b1, b2, Some(pin));
            }
        }
    }
    (total, skipped, sizes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_residual_gradient_matches_detached_finite_difference() {
        // Both activation crossings and unchanged masks. No point sits at a kink.
        let x = [0.3, 1.2];
        let repairs = [[1.4, 0.1], [0.1, 0.8]];
        let xp: Vec<&[f64]> = repairs.iter().map(|x| x.as_slice()).collect();
        let w1 = [1.2, -0.7, -0.8, 1.1];
        let b1 = [0.1, -0.2];
        let w2 = [-1.3, -0.6];
        let b2 = [100.];
        let mut adam = AdamState::new(4, 2, 2, 1);
        let loss = image_gradient(&x, &xp, &w1, &b1, &w2, &b2, 1., &mut adam).unwrap();
        // Independent scalar oracle, freezing D at the unperturbed network.
        let raw = |x: &[f64], w: &[f64], b: &[f64], v: &[f64]| -> f64 {
            100. + (0..2)
                .map(|j| v[j] * (b[j] + w[j] * x[0] + w[2 + j] * x[1]).max(0.))
                .sum::<f64>()
        };
        let y = raw(&x, &w1, &b1, &w2);
        let d: f64 = xp.iter().map(|p| (raw(p, &w1, &b1, &w2) - y).powi(2)).sum();
        let objective = |w: &[f64], b: &[f64], v: &[f64]| -> f64 {
            xp.iter()
                .map(|p| {
                    let tangent: f64 = (0..2)
                        .map(|j| {
                            let a = f64::from(u8::from(b[j] + w[j] * x[0] + w[2 + j] * x[1] > 0.));
                            v[j] * a * (w[j] * (p[0] - x[0]) + w[2 + j] * (p[1] - x[1]))
                        })
                        .sum();
                    (raw(p, w, b, v) - raw(&x, w, b, v) - tangent).powi(2) / d
                })
                .sum()
        };
        assert!((loss - objective(&w1, &b1, &w2)).abs() < 1e-12);
        for kind in 0..3 {
            let original = [&w1[..], &b1[..], &w2[..]][kind];
            for i in 0..original.len() {
                let eps = 1e-5;
                let (mut up, mut down) = (original.to_vec(), original.to_vec());
                up[i] += eps;
                down[i] -= eps;
                let f = |p: &[f64]| match kind {
                    0 => objective(p, &b1, &w2),
                    1 => objective(&w1, p, &w2),
                    _ => objective(&w1, &b1, p),
                };
                let fd = (f(&up) - f(&down)) / (2. * eps);
                let exact = [&adam.gw1[..], &adam.gb1[..], &adam.gw2[..]][kind][i];
                assert!(
                    (fd - exact).abs() < 1e-7,
                    "kind={kind} i={i} fd={fd} exact={exact}"
                );
            }
        }
        assert_eq!(adam.gb2, [0.]);
    }

    #[test]
    fn activation_gradient_matches_input_finite_differences() {
        let x = [0.3, 1.2];
        let w = [1.2, -0.7, -0.8, 1.1];
        let b = [0.1, -0.2];
        let v = [-1.3, -0.6];
        let (y, z, _) = forward(&x, &w, &b, &v, &[100.], 2, 2, 0.);
        let xp = [1.4, 0.1];
        let yp = forward(&xp, &w, &b, &v, &[100.], 2, 2, 0.).0;
        let mut exact_tangent = 0.;
        let mut fd_tangent = 0.;
        for i in 0..2 {
            let exact: f64 = (0..2)
                .filter(|&j| z[j] > 0.)
                .map(|j| v[j] * w[i * 2 + j])
                .sum();
            let (mut up, mut down) = (x, x);
            up[i] += 1e-5;
            down[i] -= 1e-5;
            let fd = (forward(&up, &w, &b, &v, &[100.], 2, 2, 0.).0
                - forward(&down, &w, &b, &v, &[100.], 2, 2, 0.).0)
                / 2e-5;
            assert!((exact - fd).abs() < 1e-8);
            exact_tangent += exact * (xp[i] - x[i]);
            fd_tangent += fd * (xp[i] - x[i]);
        }
        assert!(((yp - y - exact_tangent) - (yp - y - fd_tangent)).abs() < 1e-8);
        assert!(yp - y - exact_tangent < 0.);
    }

    #[test]
    fn zero_activation_tie_is_inactive() {
        let mut adam = AdamState::new(1, 1, 1, 1);
        let loss = image_gradient(
            &[0.],
            &[&[1.]],
            &[1.],
            &[0.],
            &[-1.],
            &[100.],
            1.,
            &mut adam,
        )
        .unwrap();
        assert_eq!(loss, 1.);
        assert_eq!(adam.gw1, [2.]);
        assert_eq!(adam.gb1, [2.]);
        assert_eq!(adam.gw2, [-2.]);
        assert_eq!(adam.gb2, [0.]);
    }

    #[test]
    fn epoch_has_registered_draw_budget_and_counts_degenerate_images() {
        let images = vec![super::super::steer_data::Image {
            base: vec![1., 2.],
            repairs: vec![vec![1., 2.]; 4],
            sizes: vec![8, 16, 32, 64],
        }];
        let (mut w1, mut b1, mut w2, mut b2) = (vec![1., 1.], vec![0.], vec![-1.], vec![100.]);
        let mut adam = AdamState::new(2, 1, 1, 1);
        let mut rng = SplitMix64::new(101 ^ 0xe340_7374_6565_7200);
        let (loss, skipped, sizes) = epoch(
            &images, &mut rng, &mut w1, &mut b1, &mut w2, &mut b2, 1., 0.001, 100., &mut adam,
        );
        assert_eq!(loss, 0.);
        assert_eq!(skipped, 512);
        assert_eq!(sizes.iter().sum::<usize>(), 4096);
        assert!(sizes.iter().all(|&n| n > 0));
        assert_eq!(adam.t, 0);
        assert_eq!(w1, [1., 1.]);
        assert_eq!(b2, [100.]);
    }

    #[test]
    fn zero_gain_is_countable_skip_without_gradient() {
        let mut adam = AdamState::new(2, 1, 1, 1);
        assert_eq!(
            image_gradient(
                &[1., 2.],
                &[&[1., 2.]],
                &[1., 1.],
                &[0.],
                &[-1.],
                &[100.],
                1.,
                &mut adam
            ),
            None
        );
        assert!(
            adam.gw1
                .iter()
                .chain(&adam.gw2)
                .chain(&adam.gb1)
                .chain(&adam.gb2)
                .all(|&g| g == 0.)
        );
    }
}
