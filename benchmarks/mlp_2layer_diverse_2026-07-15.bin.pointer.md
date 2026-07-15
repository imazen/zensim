# Pointer — 2-layer diverse MLP bake (2026-07-15)

The "best output" candidate from the optimal-blend search. **Not committed** (272 KB > 30 KB
git limit; regenerable). **Not shipped** — Profile B `include_bytes!` is untouched (user-gated).

- **Bake**: `/mnt/v/output/zensim/reports/b_negatives/mlp_2L_diverse_H128_2026-07-15.bin`
- **sha256**: `8898301955ac2d4035a27b2135cc9881095ad300e554382c90836e78809c41f9`  **md5**: `8c689610126071993559c85908910df2`  **size**: 271,900 B
- **arch**: 372→128→128→1 leaky, winsor_p99 clips + 18-knot ssim2 dial spline, f32
- **source npz**: `/mnt/v/output/zensim/reports/blend/blend_r3_3_r3-2L-H128.npz` (seed 13)
- **methodology**: `benchmarks/blend_2layer_methodology_2026-07-15.md`
- **verdict** (full panel + 10-band + dial): `/mnt/v/output/zensim/reports/b_negatives/verdict_2L_H128_2026-07-15.md`
- **dashboard**: `/mnt/v/output/zensim/dashboards/bandwise_dashboard_2026-07-15.html`

Held-out (baked, vs shipped B): CID22 0.8807 (+0.004), TID 0.8430 (+0.056), non-photo 0.9495
(+0.089), AIC-3 0.7865 (+0.009), AIC-4 0.8940 (+0.003); KADID 0.8169 (−0.003), KonJND 0.5086
(−0.038, G5 structural). Dial monotonicity 0.968 (G3 ✓). Regenerate via the reproduce block in
the methodology doc.
