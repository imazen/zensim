//! Fixed tau=1/8 activation on the admitted plain N path.
//! Legacy calls delegate unchanged; matrix accumulation retains its owner.
#![allow(clippy::too_many_arguments)]

pub(super) fn value(z: f64) -> f64 {
    if z <= 0. {
        0.
    } else if z < 0.125 {
        (4. * z) * z
    } else {
        z - 0.0625
    }
}
pub(super) fn slope(z: f64) -> f64 {
    if z <= 0. {
        0.
    } else if z < 0.125 {
        8. * z
    } else {
        1.
    }
}

pub(super) fn forward_finish(
    pre: &[f64],
    w2: &[f64],
    b2: &[f64],
    n: usize,
    alpha: f64,
    smooth: bool,
) -> (f64, Vec<f64>) {
    if !smooth {
        return crate::simd_mlp::forward_finish(pre, w2, b2, n, alpha);
    }
    let h: Vec<_> = pre.iter().map(|z| value(*z)).collect();
    let y = h.iter().zip(w2).fold(b2[0], |y, (h, w)| y + h * w);
    (y, h)
}
pub(super) fn forward(
    x: &[f64],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    nf: usize,
    nh: usize,
    alpha: f64,
    smooth: bool,
) -> (f64, Vec<f64>, Vec<f64>) {
    let (y, pre, h) = super::forward(x, w1, b1, w2, b2, nf, nh, alpha);
    if !smooth {
        return (y, pre, h);
    }
    let (y, h) = forward_finish(&pre, w2, b2, nh, alpha, true);
    (y, pre, h)
}
pub(super) fn forward_pair(
    xa: &[f64],
    xb: &[f64],
    w1: &[f64],
    b1: &[f64],
    w2: &[f64],
    b2: &[f64],
    nf: usize,
    nh: usize,
    alpha: f64,
    smooth: bool,
) -> (f64, Vec<f64>, Vec<f64>, f64, Vec<f64>, Vec<f64>) {
    let (ya, pa, ha, yb, pb, hb) = super::forward_pair(xa, xb, w1, b1, w2, b2, nf, nh, alpha);
    if !smooth {
        return (ya, pa, ha, yb, pb, hb);
    }
    let (ya, ha) = forward_finish(&pa, w2, b2, nh, alpha, true);
    let (yb, hb) = forward_finish(&pb, w2, b2, nh, alpha, true);
    (ya, pa, ha, yb, pb, hb)
}
pub(super) fn backprop_grad_head(
    pre: &[f64],
    h: &[f64],
    dl: f64,
    dh: &mut [f64],
    gb1: &mut [f64],
    w2: &[f64],
    gw2: &mut [f64],
    gb2: &mut [f64],
    nh: usize,
    alpha: f64,
    smooth: bool,
) {
    if !smooth {
        return crate::simd_mlp::backprop_grad_head(pre, h, dl, dh, gb1, w2, gw2, gb2, nh, alpha);
    }
    for j in 0..nh {
        gw2[j] += dl * h[j];
        dh[j] = (dl * w2[j]) * slope(pre[j]);
        gb1[j] += dh[j];
    }
    gb2[0] += dl;
}
pub(super) fn backprop_step(
    x: &[f64],
    pre: &[f64],
    h: &[f64],
    dl: f64,
    w1: &[f64],
    gw1: &mut [f64],
    gb1: &mut [f64],
    w2: &[f64],
    gw2: &mut [f64],
    gb2: &mut [f64],
    nf: usize,
    nh: usize,
    alpha: f64,
    smooth: bool,
) {
    if !smooth {
        return super::backprop_step(x, pre, h, dl, w1, gw1, gb1, w2, gw2, gb2, nf, nh, alpha);
    }
    let mut dh = vec![0.; nh];
    backprop_grad_head(pre, h, dl, &mut dh, gb1, w2, gw2, gb2, nh, alpha, true);
    for i in 0..nf {
        for j in 0..nh {
            gw1[i * nh + j] += x[i] * dh[j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn steer_hinge_closed_form_derivatives_and_legacy_bits() {
        for (z, h, d) in [
            (-1., 0., 0.),
            (0., 0., 0.),
            (0.03125, 0.00390625, 0.25),
            (0.0625, 0.015625, 0.5),
            (0.125, 0.0625, 1.),
            (1., 0.9375, 1.),
        ] {
            assert_eq!(value(z), h);
            assert_eq!(slope(z), d);
        }
        let (w1, b1, w2, b2) = ([1., -1.], [0., 0.], [-0.5, -0.25], [100.]);
        for x in [[-1.], [0.], [0.0625], [0.125], [1.]] {
            let a = super::super::forward(&x, &w1, &b1, &w2, &b2, 1, 2, 0.);
            let b = forward(&x, &w1, &b1, &w2, &b2, 1, 2, 0., false);
            assert_eq!(a, b);
            let (y, pre, h) = forward(&x, &w1, &b1, &w2, &b2, 1, 2, 0., true);
            assert!(y <= 100.);
            if x[0] == 0. {
                assert_eq!(y, 100.);
            }
            let mut gw1 = [0.; 2];
            let mut gb1 = [0.; 2];
            let mut gw2 = [0.; 2];
            let mut gb2 = [0.; 1];
            backprop_step(
                &x, &pre, &h, 1., &w1, &mut gw1, &mut gb1, &w2, &mut gw2, &mut gb2, 1, 2, 0., true,
            );
            for j in 0..2 {
                assert_eq!(gw1[j], x[0] * w2[j] * slope(pre[j]));
            }
        }
    }
}
