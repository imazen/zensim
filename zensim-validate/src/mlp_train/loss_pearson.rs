//! Within-leg pooled Pearson loss. Undefined constant batches contribute zero.

pub(super) fn loss_gradient(pred: &[f64], target: &[f64], weight: f64) -> (f64, Vec<f64>) {
    assert_eq!(pred.len(), target.len());
    let n = pred.len();
    let mut grad = vec![0.0; n];
    if n < 2 || weight == 0.0 {
        return (0.0, grad);
    }
    let pm = pred.iter().sum::<f64>() / n as f64;
    let tm = target.iter().sum::<f64>() / n as f64;
    let pp = pred.iter().map(|p| (p - pm).powi(2)).sum::<f64>();
    let tt = target.iter().map(|t| (t - tm).powi(2)).sum::<f64>();
    if pp <= 1e-24 || tt <= 1e-24 {
        return (0.0, grad);
    }
    let denominator = (pp * tt).sqrt();
    let rho = pred
        .iter()
        .zip(target)
        .map(|(p, t)| (p - pm) * (t - tm))
        .sum::<f64>()
        / denominator;
    for (i, g) in grad.iter_mut().enumerate() {
        *g = -weight * ((target[i] - tm) / denominator - rho * (pred[i] - pm) / pp);
    }
    (weight * (1.0 - rho), grad)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn gradient_matches_centered_finite_difference() {
        let p = [3.0, -2.0, 1.2, 8.0, 2.7];
        let t = [0.2, 0.8, -0.5, 0.9, 0.3];
        let (_, grad) = loss_gradient(&p, &t, 0.5);
        for i in 0..p.len() {
            let mut a = p;
            let mut b = p;
            a[i] += 1e-5;
            b[i] -= 1e-5;
            let numerical = (loss_gradient(&a, &t, 0.5).0 - loss_gradient(&b, &t, 0.5).0) / 2e-5;
            assert!((numerical - grad[i]).abs() < 1e-9);
        }
        assert!(grad.iter().sum::<f64>().abs() < 1e-12);
    }
    #[test]
    fn affine_invariance_and_constant_batches() {
        let p = [1.0, 3.0, 2.0, 8.0];
        let t = [0.0, 1.0, 0.5, 0.8];
        let scaled = p.map(|v| v * 4.0 + 71.0);
        assert!((loss_gradient(&p, &t, 0.5).0 - loss_gradient(&scaled, &t, 0.5).0).abs() < 1e-12);
        assert_eq!(loss_gradient(&[2.0; 4], &t, 0.5), (0.0, vec![0.0; 4]));
        assert_eq!(loss_gradient(&p, &[2.0; 4], 0.5), (0.0, vec![0.0; 4]));
    }
}
