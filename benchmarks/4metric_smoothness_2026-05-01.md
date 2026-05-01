# Smoothness analysis on synthetic q-sweeps

Per (source_path, codec) group, sorted by quality. Metrics:

- **violation_rate_mean**: % consecutive (q, q+) pairs where the model
  predicted MORE distance for the HIGHER quality (lower = better).
  Monotonicity violations are operationally bad: a user raising q
  expects a *better* quality score, not worse.
- **step_cv_mean**: stdev(|Δdist|) / mean(|Δdist|) per sweep
  (lower = more consistent step sizes → smoother gradient).
- **max_step_p95**: 95th-percentile of the largest single-step jump
  per sweep (lower = no nasty cliffs at random q).
- **ssim2_pearson_mean**: per-sweep |corr(predicted, -SSIM2)| —
  how closely each sweep tracks the ground-truth SSIM2 sweep
  for the same (source, codec). Higher = closer to SSIM2's behavior.

| metric | n_sweeps | violation_rate (mean / p95) | step_cv (mean / p95) | max_step (p50 / p95) | ssim2_pearson (mean / p10) |
|---|--:|--:|--:|--:|--:|
| V0_4-smooth | 20784 | 0.016 / 0.143 | 0.702 / 1.186 | 5.12 / 12.23 | 0.9900 / 0.9833 |
| V0_5 | 20784 | 0.017 / 0.167 | 0.818 / 1.398 | 20.45 / 63.66 | 0.9912 / 0.9811 |
| V0_4-smooth-konjnd-train | 20784 | 0.019 / 0.167 | 0.753 / 1.257 | 5.19 / 12.91 | 0.9932 / 0.9875 |
| V0_6 dct_hf | 20784 | 0.016 / 0.143 | 0.670 / 1.148 | 5.14 / 12.39 | 0.9860 / 0.9763 |

## Violation rate by codec (mean across sweeps)

| metric \ codec | mozjpeg-rs-420-e4 | zenavif-s5-e6 | zenjpeg-420-e2 | zenjpeg-420-xyb-e2 | zenjxl-e7 | zenwebp-default-m4 |
|---|--:|--:|--:|--:|--:|--:|
| V0_4-smooth | 0.001 | 0.001 | 0.001 | 0.007 | 0.002 | 0.084 |
| V0_5 | 0.002 | 0.001 | 0.001 | 0.007 | 0.003 | 0.093 |
| V0_4-smooth-konjnd-train | 0.004 | 0.002 | 0.004 | 0.011 | 0.008 | 0.089 |
| V0_6 dct_hf | 0.002 | 0.001 | 0.000 | 0.007 | 0.002 | 0.084 |
